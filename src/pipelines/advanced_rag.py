"""Advanced RAG Pipeline: 集成 Multi-Query + HyDE + Router 的终极版。

支持 6 种 mode:
  - naive:        纯 Dense (Day 4 baseline)
  - hybrid:       Dense + BM25 + RRF (Day 5)
  - multi_query:  Hybrid + Multi-Query
  - hyde:         Hybrid + HyDE
  - full:         Hybrid + Multi-Query + HyDE
  - auto:         LLM Router 判断, 自动选择

设计要点:
  - 复用 Day 5 的 HybridRetriever 作为底层检索器
  - Multi-Query / HyDE 产生多个"探针", 每个探针跑 Hybrid
  - 多路结果用 RRF 再融合一次
  - 所有决策可追溯(routing_decision / probes 字段)
"""
from typing import Literal, Optional

from dotenv import load_dotenv
load_dotenv(override=True)

from src.embeddings.bge import BGEEmbedder
from src.generators.llm import generate_answer
from src.query_transformers.hyde import HyDEGenerator
from src.query_transformers.multi_query import MultiQueryRewriter
from src.query_transformers.router import QueryRouter
from src.retrievers.bm25_store import BM25Store
from src.retrievers.hybrid import HybridRetriever, rrf_fuse
from src.retrievers.qdrant_store import get_client


Mode = Literal["naive", "hybrid", "multi_query", "hyde", "full", "auto"]


class AdvancedRAGPipeline:
    """Advanced RAG: 支持多种模式的完整 pipeline。

    用法:
        rag = AdvancedRAGPipeline()
        
        # Auto 模式(推荐): LLM 决定用什么组合
        result = rag.query("紫金海外业务营收和风险", mode="auto")
        
        # 手动指定 mode
        result = rag.query(q, mode="hyde")
        result = rag.query(q, mode="full", filters={"company": "贵州茅台"})
    """

    def __init__(
        self,
        collection_name: str = "financial_reports",
        top_k: int = 5,
        recall_per_probe: int = 10,
    ):
        """
        Args:
            collection_name: Qdrant collection 名称
            top_k: 最终返回几个 chunks 给 LLM
            recall_per_probe: 每个检索探针召回几个(给后续融合留余地)
        """
        self.collection_name = collection_name
        self.top_k = top_k
        self.recall_per_probe = recall_per_probe

        # --- 共享资源 ---
        self.embedder = BGEEmbedder()
        self.qdrant_client = get_client()

        # --- BM25 索引 (从 Qdrant scroll 出全量 chunks 构建) ---
        self.bm25_store = BM25Store()
        self._build_bm25_from_qdrant()

        # --- 底层 Hybrid 检索器 (Day 5 的复用) ---
        self.hybrid_retriever = HybridRetriever(
            qdrant_client=self.qdrant_client,
            bm25_store=self.bm25_store,
            embedder=self.embedder,
            collection_name=collection_name,
        )

        # --- 查询转换器 ---
        self.router = QueryRouter()
        self.multi_query_rewriter = MultiQueryRewriter(
            num_queries=4, include_original=True
        )
        self.hyde_generator = HyDEGenerator()
        
         # ==== 🆕 Day 7: Reranker 相关 ====
        # 懒加载: 实际第一次 use_reranker=True 时才实例化
        self._reranker = None
        self._rerank_input_n = 20   # 召回给 Reranker 几条
        # 精排后保留多少条, 和 top_k 保持一致(默认 5)
    
    @property
    def reranker(self):
        """懒加载 Reranker (第一次用到才加载, 2 秒冷启动)."""
        if self._reranker is None:
            from src.rerankers.bge_reranker import BGEReranker
            print("[AdvancedRAG] 首次使用 Reranker, 正在加载 BGE-Reranker-base...")
            self._reranker = BGEReranker()
        return self._reranker


    def _build_bm25_from_qdrant(self):
        """从 Qdrant scroll 全量 chunks, 构建 BM25 索引。"""
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

    # ================================================================
    # 核心 query 方法
    # ================================================================

    def query(
        self,
        question: str,
        mode: Mode = "auto",
        filters: Optional[dict] = None,
        top_k: Optional[int] = None,
        # ==== 🆕 Day 7 新增 ====
        use_reranker: bool = False,
        rerank_input_n: Optional[int] = None,
    ) -> dict:
        """执行一次 Advanced RAG 查询。

        Args:
            question: 用户问题
            mode: 检索策略模式, 详见类 docstring
            filters: metadata 过滤条件
            top_k: 覆盖默认 top_k

        Returns:
            {
                "question": 原始问题,
                "answer": LLM 生成的答案,
                "mode_used": 实际使用的 mode (auto 会展开到具体 mode),
                "routing_decision": Router 的决策 (仅 auto 模式有),
                "probes": 用了哪些探针 (原查询/子查询/假答案),
                "sources": Top-K 检索结果
            }
        """
        top_k = top_k or self.top_k

        # ==== 1. 如果是 auto 模式, 先跑 Router 决策 ====
        routing_decision = None
        if mode == "auto":
            routing_decision = self.router.route(question)
            use_mq = routing_decision["use_multi_query"]
            use_hyde = routing_decision["use_hyde"]

            # 把 auto 展开成具体 mode
            if use_mq and use_hyde:
                mode_used = "full"
            elif use_mq:
                mode_used = "multi_query"
            elif use_hyde:
                mode_used = "hyde"
            else:
                mode_used = "hybrid"
        else:
            mode_used = mode

        # ==== 2. 根据 mode_used 准备"探针列表" ====
        # 每个探针是 (probe_type, probe_text) 的 tuple
        probes: list[tuple[str, str]] = [("original", question)]

        # 是否开 Multi-Query?
        if mode_used in ("multi_query", "full"):
            sub_queries = self.multi_query_rewriter.rewrite(question)
            # rewrite() 返回的第 0 个就是原查询, 这里跳过避免重复
            for i, sq in enumerate(sub_queries):
                if i == 0:
                    continue  # 原查询已在 probes 里
                probes.append((f"sub_query_{i}", sq))

        # 是否开 HyDE?
        if mode_used in ("hyde", "full"):
            fake_answer = self.hyde_generator.generate(question)
            probes.append(("hyde", fake_answer))


       # ==== 3. 根据 mode_used 执行检索 ====
        # 🆕 Day 7: 如果开 Reranker, 召回多捞几条 (rerank_input_n), 待精排后再截断
        recall_n = (rerank_input_n or self._rerank_input_n) if use_reranker else top_k

        if mode_used == "naive":
            search_results = self._naive_search(question, recall_n, filters)
        else:
            search_results = self._multi_probe_hybrid_search(
                probes, recall_n, filters, mode_used
            )

        # ==== 🆕 Day 7: Reranker 精排 ====
        rerank_info = None
        if use_reranker and search_results:
            recall_count = len(search_results)
            import time
            t0 = time.time()
            search_results = self.reranker.rerank(
                query=question,
                candidates=search_results,
                top_k=top_k,
            )
            rerank_elapsed_ms = (time.time() - t0) * 1000
            rerank_info = {
                "recall_n": recall_count,
                "final_k": len(search_results),
                "elapsed_ms": round(rerank_elapsed_ms, 1),
            }


        # ==== 4. LLM 生成答案 ====
        llm_input = [
            {
                "content": r["content"],
                "score": r.get("rrf_score", r.get("score", 0)),
                "metadata": r["metadata"],
            }
            for r in search_results
        ]
        answer = generate_answer(question, llm_input)

        # ==== 5. 组装返回 ====
        return {
            "question": question,
            "answer": answer,
            "mode_used": mode_used,
            "routing_decision": routing_decision,
            "rerank_info": rerank_info, # 🆕
            "probes": [
                {"type": p_type, "text": p_text[:100]}  # 截断展示
                for p_type, p_text in probes
            ],
            "sources": [
                {
                    "score": r.get("rrf_score", r.get("score", 0)),
                    "rerank_score": r.get("rerank_score"),     # 🆕
                    "rerank_rank": r.get("rerank_rank"),       # 🆕
                    "company": r["metadata"].get("company", "?"),
                    "year": r["metadata"].get("year", "?"),
                    "page": r["metadata"].get("page", "?"),
                    "type": r["metadata"].get("chunk_type", "?"),
                    "preview": r["content"][:100],
                }
                for r in search_results
            ],
        }

    # ================================================================
    # 内部辅助方法
    # ================================================================

    def _naive_search(
        self,
        question: str,
        top_k: int,
        filters: Optional[dict],
    ) -> list[dict]:
        """纯 Dense 检索 (Day 4 方式)。"""
        from src.retrievers.qdrant_store import search_similar

        q_vec = self.embedder.encode_query(question)
        return search_similar(
            self.qdrant_client,
            q_vec.tolist(),
            self.collection_name,
            limit=top_k,
            filters=filters,
        )

    def _multi_probe_hybrid_search(
        self,
        probes: list[tuple[str, str]],
        top_k: int,
        filters: Optional[dict],
        mode_used: str,
    ) -> list[dict]:
        """多探针 Hybrid 检索 + 二次 RRF 融合。

        对 probes 列表里每一个探针独立跑 Hybrid, 然后把所有结果
        用 RRF 再融合一次, 取 Top-K。

        当 probes 只有 1 个(如 mode=hybrid)时, 直接返回 Hybrid 结果。
        """
        # 如果只有一个探针, 直接单路 Hybrid
        if len(probes) == 1:
            _, probe_text = probes[0]
            # 对 HyDE 的探针, 用 encode 而不是 encode_query (不加前缀)
            # 但这里只有 original 一个, 走标准 Hybrid
            return self.hybrid_retriever.search(
                query=probe_text,
                top_k=top_k,
                filters=filters,
            )

        # 多探针: 每个探针独立跑 Hybrid, 召回多一些
        all_probe_results: list[list[dict]] = []
        for probe_type, probe_text in probes:
            if probe_type == "hyde":
                # HyDE 探针特殊处理: 用陈述句 embed, 不加查询前缀
                results = self._hybrid_search_with_statement(
                    probe_text, self.recall_per_probe, filters
                )
            else:
                # 原查询或子查询, 走标准 Hybrid
                results = self.hybrid_retriever.search(
                    query=probe_text,
                    top_k=self.recall_per_probe,
                    filters=filters,
                )
            all_probe_results.append(results)

        # 二次 RRF 融合: 把多路 Hybrid 结果再融合一次
        fused = self._rrf_fuse_multi_probes(all_probe_results, top_k)
        return fused

    def _hybrid_search_with_statement(
        self,
        statement: str,
        limit: int,
        filters: Optional[dict],
    ) -> list[dict]:
        """针对 HyDE 陈述句的特殊 Hybrid 检索。

        关键区别: Dense embedding 用 encode() 不是 encode_query(),
                 因为陈述句不是查询, 不加 BGE 前缀。
        """
        from src.retrievers.bm25_store import BM25Store  # noqa: F401
        from src.retrievers.qdrant_store import search_similar

        # Dense: 用 encode 不加前缀
        d_vec = self.embedder.encode(statement)
        dense_results = search_similar(
            self.qdrant_client,
            d_vec.tolist(),
            self.collection_name,
            limit=limit,
            filters=filters,
        )

        # BM25: 用假答案做分词检索 (其实 BM25 不care句式)
        bm25_results = self.bm25_store.search(
            query=statement,
            limit=limit,
            filters=filters,
        )

        # 两路 RRF 融合
        return rrf_fuse(dense_results, bm25_results, k=60, top_k=limit)

    def _rrf_fuse_multi_probes(
        self,
        all_probe_results: list[list[dict]],
        top_k: int,
        k: int = 60,
    ) -> list[dict]:
        """对多个探针的 Hybrid 结果做二次 RRF 融合。

        每个 probe 的 Top-K 都当作一路, 用 content 去重, RRF 排序。
        """
        # content → 累积的 RRF 分数 + 保留最后一次的 metadata
        candidates: dict[str, dict] = {}

        for probe_results in all_probe_results:
            for rank, r in enumerate(probe_results, start=1):
                key = r["content"]
                rrf_contribution = 1.0 / (k + rank)

                if key in candidates:
                    candidates[key]["rrf_score"] += rrf_contribution
                    candidates[key]["probe_hits"] += 1
                else:
                    candidates[key] = {
                        "content": r["content"],
                        "metadata": r["metadata"],
                        "rrf_score": rrf_contribution,
                        "probe_hits": 1,
                    }

        # 按 RRF 分数排序, 取 Top-K
        ranked = sorted(
            candidates.values(),
            key=lambda x: -x["rrf_score"],
        )[:top_k]

        return ranked