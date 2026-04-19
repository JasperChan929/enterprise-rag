# src/rerankers/bge_reranker.py
"""BGE Cross-Encoder Reranker.

用于 RAG 两阶段架构中的第二阶段(精排)。
对上游召回返回的 Top-N 候选,用 Cross-Encoder 重新打分,输出 Top-K。

核心设计:
  - 单例式加载: 模型只加载一次, 后续复用
  - batch 预测: 一次调用算完所有 (query, doc) 对, 充分利用 CPU
  - 保留原始元数据: 只改 rank 和 score, 不动 content/metadata
"""
from __future__ import annotations

import time
from typing import Optional

from sentence_transformers import CrossEncoder


class BGEReranker:
    """BGE Cross-Encoder Reranker (CPU 友好).

    用法:
        reranker = BGEReranker()

        # 对 Hybrid 返回的 20 条做精排, 取 Top-5
        reranked = reranker.rerank(
            query="招行不良贷款率",
            candidates=hybrid_results,   # list of dict with "content" key
            top_k=5,
        )
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        max_length: int = 512,
        verbose: bool = False,
    ):
        """初始化 Reranker.

        Args:
            model_name: HuggingFace 模型名, 默认 bge-reranker-base
            max_length: query + doc 拼接后最大 token 数, 超过会截断
            verbose: 是否打印加载耗时和打分细节
        """
        self.model_name = model_name
        self.verbose = verbose

        t0 = time.time()
        self.model = CrossEncoder(model_name, max_length=max_length)
        if verbose:
            print(f"[BGEReranker] 加载 {model_name} 耗时 {time.time()-t0:.2f}s")


    def score(self, query: str, doc: str) -> float:
     """对单个 (query, doc) 对打分.
    Returns:
        相关度分数. 
        bge-reranker-base/large 返回 sigmoid 后概率 (范围 0~1, 越大越相关).
        bge-reranker-v2-m3 返回原始 logits (范围约 -10~+10, 越大越相关).
        无论哪种, 降序排即可.
    """

     return float(self.model.predict([(query, doc)])[0])
    
    
    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """对候选列表做精排.

        Args:
            query: 用户原始查询
            candidates: 候选列表, 每条是 dict, 必须有 "content" 键.
                        通常是 HybridRetriever.search() 或
                        AdvancedRAGPipeline 里的 _multi_probe_hybrid_search() 返回
            top_k: 精排后保留几条. None 表示全部返回(仅重排, 不截断)

        Returns:
            重排后的候选列表, 每条增加两个字段:
                - rerank_score: Cross-Encoder 打的分
                - rerank_rank:  精排后的排名(从 1 开始)
            原有字段(content/metadata/rrf_score 等)完全保留.
        """
        if not candidates:
            return []

        # 1. 构造 (query, doc) pair
        pairs = [(query, c["content"]) for c in candidates]

        # 2. batch 预测 (CrossEncoder 内部会自动组 batch)
        t0 = time.time()
        scores = self.model.predict(pairs)
        elapsed = time.time() - t0

        if self.verbose:
            print(f"[BGEReranker] 对 {len(pairs)} 条打分耗时 {elapsed*1000:.0f}ms"
                  f" ({elapsed/len(pairs)*1000:.1f}ms/条)")

        # 3. 把分数塞回 candidate, 保留所有原始字段
        scored = []
        for c, s in zip(candidates, scores):
            c_new = dict(c)          # 浅拷贝, 避免改到原 dict
            c_new["rerank_score"] = float(s)
            scored.append(c_new)

        # 4. 按 rerank_score 降序排
        scored.sort(key=lambda x: -x["rerank_score"])

        # 5. 标记 rank
        for i, c in enumerate(scored, start=1):
            c["rerank_rank"] = i

        # 6. 按需截断
        if top_k is not None:
            scored = scored[:top_k]

        return scored