"""Naive RAG Pipeline: 最简单的 检索 → 生成 流程。

这是 baseline,后面 Advanced RAG 的所有优化都在这个基础上叠加。
"""
import os
from dotenv import load_dotenv
load_dotenv()

from src.embeddings.bge import BGEEmbedder
from src.generators.llm import generate_answer, format_context
from src.retrievers.qdrant_store import get_client, search_similar


class NaiveRAGPipeline:
    """Naive RAG: query → embed → retrieve → generate。

    没有查询改写、没有 Hybrid Search、没有 Reranker。
    这是 Day 5-7 优化的基线。
    """

    def __init__(
        self,
        collection_name: str = "financial_reports",
        top_k: int = 5,
    ):
        self.collection_name = collection_name
        self.top_k = top_k
        self.embedder = BGEEmbedder()
        self.qdrant_client = get_client()

    def query(
        self,
        question: str,
        filters: dict | None= None,
        top_k: int | None= None,
        show_sources: bool = True,
    ) -> dict:
        """执行一次完整的 RAG 问答。

        Args:
            question: 用户问题
            filters: 可选的 metadata 过滤(如 {"company": "贵州茅台"})
            top_k: 检索几条(默认用初始化时的值)
            show_sources: 是否在返回中包含检索来源

        Returns:
            {
                "question": 原始问题,
                "answer": LLM 生成的答案,
                "sources": [检索到的 chunks 信息],  # show_sources=True 时
            }
        """
        top_k = top_k or self.top_k

        # Step 1: 查询向量化
        q_vec = self.embedder.encode_query(question)

        # Step 2: 检索 Top-K
        search_results = search_similar(
            self.qdrant_client,
            q_vec.tolist(),
            self.collection_name,
            limit=top_k,
            filters=filters,
        )

        # Step 3: 生成答案
        answer = generate_answer(question, search_results)

        # 组装返回
        result : dict= {
            "question": question,
            "answer": answer,
        }

        if show_sources:
            result["sources"] = [
                {
                    "score": r["score"],
                    "company": r["metadata"].get("company", "?"),
                    "year": r["metadata"].get("year", "?"),
                    "page": r["metadata"].get("page", "?"),
                    "type": r["metadata"].get("chunk_type", "?"),
                    "preview": r["content"][:100],
                    "full_content": r["content"],   # Day 11 T1: RAGAS 用, 完整 chunk 不截断
                }
                for r in search_results
            ]

        return result