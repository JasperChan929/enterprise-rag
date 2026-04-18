"""Hybrid Retriever: Dense + BM25 + RRF 融合。

核心流程:
  1. 同一查询同时跑 Dense (Qdrant) 和 BM25 (内存索引)
  2. 两路各召回 Top-N (N 大于最终 K, 给融合留空间)
  3. 取并集, 用 RRF 公式重新打分
  4. 输出 Top-K, 带可解释的元数据(两路各自的排名)

为什么 RRF 分数必须 >0:
  1/(k+rank) 永远正, 所以能直接 -score 排序。
  不像 BM25 可能 0 分, Dense 可能微负(虽然实际不会)。
"""
from typing import Optional

from qdrant_client import QdrantClient

from src.embeddings.bge import BGEEmbedder
from src.retrievers.bm25_store import BM25Store
from src.retrievers.qdrant_store import search_similar


# ============================================================
# RRF 融合函数
# ============================================================

def rrf_fuse(
    dense_results: list[dict],
    bm25_results: list[dict],
    k: int = 60,
    top_k: int = 5,
) -> list[dict]:
    """对 Dense 和 BM25 的结果列表做 RRF 融合。

    Args:
        dense_results: Dense 检索结果, 每项包含 content/score/metadata
        bm25_results:  BM25 检索结果, 同上
        k: RRF 常数, 默认 60
        top_k: 返回几个最终结果

    Returns:
        融合后的 Top-K, 每项额外带:
          - dense_rank: 该文档在 Dense 路的排名(1-indexed), None 表示未进 Top-N
          - bm25_rank:  该文档在 BM25 路的排名, None 表示未进 Top-N
          - dense_score: Dense 原始分数(余弦相似度)
          - bm25_score:  BM25 原始分数
          - rrf_score:   最终融合分数

    融合逻辑:
        用 content 作为去重 key(同一 chunk 在两路可能都出现)。
        如果有 chunk_id/point_id 更好, 但我们的 BM25 用 chunk_index,
        Dense 用 UUID, 不统一, 所以用 content 兜底。
    """
    # 用 content 作为唯一标识(字符串哈希, 足够区分)
    # candidate 结构: content → {source_info, dense_info, bm25_info}
    candidates: dict[str, dict] = {}

    # ---- 处理 Dense 结果 ----
    for rank, r in enumerate(dense_results, start=1):
        key = r["content"]
        candidates[key] = {
            "content": r["content"],
            "metadata": r["metadata"],
            "dense_rank": rank,
            "dense_score": r["score"],
            "bm25_rank": None,
            "bm25_score": None,
        }

    # ---- 处理 BM25 结果, 合并到 candidates ----
    for rank, r in enumerate(bm25_results, start=1):
        key = r["content"]
        if key in candidates:
            # 两路都有 → 补充 BM25 信息
            candidates[key]["bm25_rank"] = rank
            candidates[key]["bm25_score"] = r["score"]
        else:
            # 只在 BM25 出现
            candidates[key] = {
                "content": r["content"],
                "metadata": r["metadata"],
                "dense_rank": None,
                "dense_score": None,
                "bm25_rank": rank,
                "bm25_score": r["score"],
            }

    # ---- 计算 RRF 分数 ----
    for info in candidates.values():
        score = 0.0
        if info["dense_rank"] is not None:
            score += 1.0 / (k + info["dense_rank"])
        if info["bm25_rank"] is not None:
            score += 1.0 / (k + info["bm25_rank"])
        info["rrf_score"] = score

    # ---- 排序取 Top-K ----
    ranked = sorted(
        candidates.values(),
        key=lambda x: -x["rrf_score"],
    )[:top_k]

    return ranked


# ============================================================
# HybridRetriever 封装
# ============================================================

class HybridRetriever:
    """Hybrid 检索器: Dense + BM25 + RRF。

    用法:
        retriever = HybridRetriever(
            qdrant_client=client,
            bm25_store=bm25_store,
            embedder=embedder,
        )
        results = retriever.search("招商银行不良贷款率", filters={"company": "招商银行"})

    设计要点:
        - recall_multiplier: 两路各召回 top_k * multiplier 个, 给融合留余地
        - k: RRF 常数, 默认 60
    """

    def __init__(
        self,
        qdrant_client: QdrantClient,
        bm25_store: BM25Store,
        embedder: BGEEmbedder,
        collection_name: str = "financial_reports",
        k: int = 60,
        recall_multiplier: int = 4,
    ):
        self.qdrant_client = qdrant_client
        self.bm25_store = bm25_store
        self.embedder = embedder
        self.collection_name = collection_name
        self.k = k
        self.recall_multiplier = recall_multiplier

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """混合检索: Dense + BM25 → RRF 融合 → Top-K。

        Args:
            query: 用户查询
            top_k: 最终返回几个
            filters: metadata 过滤(如 {"company": "贵州茅台"})

        Returns:
            融合结果列表, 每项包含 rrf_score / dense_rank / bm25_rank 等
        """
        recall_k = top_k * self.recall_multiplier

        # ---- Dense 路 ----
        q_vec = self.embedder.encode_query(query)
        dense_results = search_similar(
            self.qdrant_client,
            q_vec.tolist(),
            self.collection_name,
            limit=recall_k,
            filters=filters,
        )

        # ---- BM25 路 ----
        bm25_results = self.bm25_store.search(
            query,
            limit=recall_k,
            filters=filters,
        )

        # ---- RRF 融合 ----
        fused = rrf_fuse(
            dense_results,
            bm25_results,
            k=self.k,
            top_k=top_k,
        )

        return fused