"""scripts/debug_topk.py - Day 9 Task 1 bug 诊断脚本

用途: 验证 retriever 在不同 top_k / rrf_k / recall_mult 下是否真的行为不同.
如果 Top-5 列表在参数变化时一模一样, 说明参数没生效 (脚本 27 有 bug).

用 U4 招行作测试样本, 因为 Day 8 有明确 Top-20 基线可对比.
"""
import pickle
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from src.embeddings.bge import BGEEmbedder
from src.retrievers.bm25_store import BM25Store
from src.retrievers.hybrid import HybridRetriever
from src.retrievers.qdrant_store import get_client

# 常量 (和脚本 27 对齐)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
COLLECTION = "financial_reports"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
CHUNKS_CACHE = DOCS_DIR / f"day09-chunks_size{CHUNK_SIZE}_overlap{CHUNK_OVERLAP}.pkl"

# U4 查询 (和脚本 27 QUERIES[3] 对齐)
QUERY_TEXT = "招商银行2025年不良贷款率"
COMPANY_FILTER = "招商银行"


def setup_retriever(rrf_k: int, recall_mult: int):
    """和脚本 27 的 setup_retriever 逻辑一致."""
    if not CHUNKS_CACHE.exists():
        print("⚠️  chunks 缓存不存在, 先跑 scripts/27_audit_retrieval_params.py --stage=draft")
        raise SystemExit(1)

    with CHUNKS_CACHE.open("rb") as f:
        chunks = pickle.load(f)

    bm25_store = BM25Store()
    bm25_store.build(chunks)

    embedder = BGEEmbedder()
    client = get_client()

    return HybridRetriever(
        qdrant_client=client,
        bm25_store=bm25_store,
        embedder=embedder,
        collection_name=COLLECTION,
        k=rrf_k,
        recall_multiplier=recall_mult,
    )


# ============================================================
# 诊断主逻辑
# ============================================================

filters = {"company": COMPANY_FILTER}

print("=" * 70)
print(f"测试查询: U4 — {QUERY_TEXT}")
print(f"过滤条件: {filters}")
print("=" * 70)

# --- 测试 1: top_k 扫描 ---
print("\n### 测试 1: top_k 是否生效 (固定 rrf_k=60, recall_mult=4) ###\n")
r = setup_retriever(rrf_k=60, recall_mult=4)
for tk in [3, 5, 10, 15]:
    res = r.search(QUERY_TEXT, top_k=tk, filters=filters)
    pages = [x["metadata"]["page"] for x in res]
    print(f"  top_k={tk:3d}: Top-{tk} 页号 = {pages}")

print("\n  预期: top_k=3 的 Top-3 应是 top_k=5 Top-5 的前 3 名前缀.")
print("        如果不是 → top_k 参数有问题.")

# --- 测试 2: rrf_k 扫描 ---
print("\n\n### 测试 2: rrf_k 是否生效 (固定 top_k=5, recall_mult=4) ###\n")
for rk in [10, 30, 60, 100]:
    r = setup_retriever(rrf_k=rk, recall_mult=4)
    res = r.search(QUERY_TEXT, top_k=5, filters=filters)
    pages = [x["metadata"]["page"] for x in res]
    rrf_scores = [round(x.get("rrf_score", 0), 4) for x in res]
    print(f"  rrf_k={rk:3d}: Top-5 页号 = {pages}")
    print(f"           rrf_scores = {rrf_scores}")

print("\n  预期: 不同 rrf_k 下 Top-5 页号和 rrf_scores 应不同 (k=10 vs k=100 差别最大).")
print("        如果 Top-5 页号完全一样 → rrf_k 没生效.")

# --- 测试 3: recall_mult 扫描 ---
print("\n\n### 测试 3: recall_mult 是否生效 (固定 top_k=5, rrf_k=60) ###\n")
for rm in [2, 4, 8]:
    r = setup_retriever(rrf_k=60, recall_mult=rm)
    res = r.search(QUERY_TEXT, top_k=5, filters=filters)
    pages = [x["metadata"]["page"] for x in res]
    print(f"  recall_mult={rm}: Top-5 页号 = {pages}")

print("\n  预期: recall_mult 影响候选池大小, Top-5 应有变化但不一定剧烈.")

print("\n" + "=" * 70)
print("跑完把 3 组输出贴回给 Claude 做诊断.")
print("=" * 70)