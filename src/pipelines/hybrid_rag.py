"""Hybrid RAG Pipeline: Dense + BM25 + RRF → LLM 生成。

核心变化 vs NaiveRAGPipeline:
  - 把 query_points (单 Dense) 替换成 HybridRetriever (Dense + BM25 + RRF)
  - 其他(Prompt/LLM/引用格式)完全复用 Naive 的, 保证可比性

Day 9 评估时:
  - Naive 和 Hybrid 用同一套问题
  - 除检索器外所有变量不变 → 性能差异 100% 归因于 Hybrid 检索
"""
from dotenv import load_dotenv
load_dotenv()

from src.embeddings.bge import BGEEmbedder
from src.generators.llm import generate_answer
from src.retrievers.bm25_store import BM25Store
from src.retrievers.hybrid import HybridRetriever
from src.retrievers.qdrant_store import get_client


class HybridRAGPipeline:
    """Hybrid RAG: query → (dense + bm25 → rrf) → generate。

    比 NaiveRAGPipeline 多做的事情:
      - 构建时要加载 BM25 索引(从 Qdrant 全量拉 chunks)
      - 检索时两路并行, 用 RRF 融合
    """

    def __init__(
        self,
        collection_name: str = "financial_reports",
        top_k: int = 5,
        rrf_k: int = 60,
        recall_multiplier: int = 4,
    ):
        self.collection_name = collection_name
        self.top_k = top_k

        # 初始化共享资源
        self.embedder = BGEEmbedder()
        self.qdrant_client = get_client()

        # 构建 BM25 索引(从 Qdrant 全量加载 chunks)
        self.bm25_store = BM25Store()
        self._build_bm25_from_qdrant()

        # 组装 Hybrid 检索器
        self.retriever = HybridRetriever(
            qdrant_client=self.qdrant_client,
            bm25_store=self.bm25_store,
            embedder=self.embedder,
            collection_name=collection_name,
            k=rrf_k,
            recall_multiplier=recall_multiplier,
        )

    def _build_bm25_from_qdrant(self):
        """从 Qdrant scroll 全量 chunks, 构建 BM25 索引。

        为什么不从 pkl 加载?
            - 避免 pkl 和 Qdrant 数据不一致(增量更新时)
            - Qdrant 是 Source of Truth
            - 启动慢 3-5 秒可接受(生产环境只启动一次)
        """
        from src.loaders.base import Chunk

        all_points = []
        offset = None
        while True:
            response, offset = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=500,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            all_points.extend(response)
            if offset is None:
                break

        chunks = []
        for p in all_points:
            payload = dict(p.payload or {})
            content = payload.pop("content", "")
            chunk_type = payload.pop("chunk_type", "text")
            chunks.append(Chunk(
                content=content,
                chunk_type=chunk_type,
                metadata=payload,
            ))

        self.bm25_store.build(chunks)

    def query(
        self,
        question: str,
        filters: dict | None = None,
        top_k: int | None = None,
        show_sources: bool = True,
    ) -> dict:
        """执行一次完整的 Hybrid RAG 问答。

        Args:
            question: 用户问题
            filters: metadata 过滤
            top_k: 检索几条
            show_sources: 是否在返回中包含检索来源(含融合诊断信息)

        Returns:
            {
                "question": 原始问题,
                "answer": LLM 生成的答案,
                "sources": [
                    {
                        "rrf_score": RRF 分数,
                        "dense_rank": Dense 路排名(或 None),
                        "bm25_rank":  BM25 路排名(或 None),
                        "company": ..., "year": ..., "page": ..., "type": ...,
                        "preview": 前 100 字,
                    },
                    ...
                ]
            }
        """
        top_k = top_k or self.top_k

        # Step 1: Hybrid 检索
        search_results = self.retriever.search(
            query=question,
            top_k=top_k,
            filters=filters,
        )

        # Step 2: LLM 生成
        # 把融合后的结果转成 generate_answer 能处理的格式
        llm_input = [
            {
                "content": r["content"],
                "score": r["rrf_score"],
                "metadata": r["metadata"],
            }
            for r in search_results
        ]
        answer = generate_answer(question, llm_input)

        # 组装返回
        result: dict = {
            "question": question,
            "answer": answer,
        }

        if show_sources:
            result["sources"] = [
                {
                    "rrf_score": r["rrf_score"],
                    "dense_rank": r["dense_rank"],
                    "bm25_rank": r["bm25_rank"],
                    "dense_score": r["dense_score"],
                    "bm25_score": r["bm25_score"],
                    "company": r["metadata"].get("company", "?"),
                    "year": r["metadata"].get("year", "?"),
                    "page": r["metadata"].get("page", "?"),
                    "type": r["metadata"].get("chunk_type", "?"),
                    "preview": r["content"][:100],
                }
                for r in search_results
            ]

        return result