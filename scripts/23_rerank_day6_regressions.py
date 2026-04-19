# scripts/23_rerank_day6_regressions.py
"""复现并尝试修复 Day 6 的三个宽召回稀释案例.

Day 6 发现:
  - 招行不良贷款率: Full 模式把 p.45 (整体 0.93%) 挤出 Top-5
  - 紫金海外风险:   Full 模式把 p.59 (地缘政治详解) 挤出 Top-3
  - 茅台营收:       Full 延迟比 HyDE 高 40%, 结果没更好

Day 7 假设: 
  Cross-Encoder Reranker 能识别"精信号", 把被稀释的金页拎回 Top-K.

对比实验:
  对每个案例, 跑 full 模式两次:
    (A) use_reranker=False  → 复现 Day 6 现象
    (B) use_reranker=True   → 验证 Reranker 修复效果
  对比两次的 Top-5 页面变化, 以及 LLM 答案质量.
"""
from dotenv import load_dotenv
load_dotenv(override=True)

import time
from src.pipelines.advanced_rag import AdvancedRAGPipeline


# ========== Day 6 明确失败的三个案例 ==========
CASES = [
    {
        "name": "招行不良贷款率 (Day 6 丢了 p.45 整体 0.93%)",
        "query": "招商银行2024年不良贷款率是多少?",
        "filters": {"company": "招商银行"},
        "expected_page": 45,       # Day 6 被挤出的金页
        "expected_signal": "0.93%",
    },
    {
        "name": "紫金海外业务 (Day 6 丢了 p.59 地缘政治详解)",
        "query": "紫金矿业海外业务的营收和风险",
        "filters": {"company": "紫金矿业"},
        "expected_page": 59,
        "expected_signal": "地缘政治",
    },
    {
        "name": "茅台营收 (Day 6 Full 结果不如单开 HyDE)",
        "query": "贵州茅台2023年的营业收入是多少?",
        "filters": {"company": "贵州茅台"},
        "expected_page": 56,       # Day 5 曾挤出的关键数字页
        "expected_signal": "营业收入",
    },
]


def run_one(rag, case, use_reranker: bool):
    """跑一次 full 模式, 返回关键信息."""
    t0 = time.time()
    result = rag.query(
        question=case["query"],
        mode="full",
        filters=case["filters"],
        top_k=5,
        use_reranker=use_reranker,
        rerank_input_n=20,
    )
    elapsed = time.time() - t0

    # 找金页的排名 (1-indexed, 没找到返回 None)
    golden_rank = None
    for i, s in enumerate(result["sources"], start=1):
        if s["page"] == case["expected_page"]:
            golden_rank = i
            break

    return {
        "answer": result["answer"],
        "pages": [s["page"] for s in result["sources"]],
        "golden_rank": golden_rank,
        "elapsed": elapsed,
        "rerank_info": result.get("rerank_info"),
        "sources": result["sources"],
    }


def print_comparison(case, no_rr, with_rr):
    print("\n" + "=" * 70)
    print(f"案例: {case['name']}")
    print(f"Query: {case['query']}")
    print(f"金页: p.{case['expected_page']} (应含 '{case['expected_signal']}')")
    print("=" * 70)

    # ---- A: 无 Reranker ----
    print(f"\n【A】full 模式, 无 Reranker:")
    print(f"  耗时: {no_rr['elapsed']:.2f}s")
    print(f"  Top-5 页: {no_rr['pages']}")
    print(f"  金页 p.{case['expected_page']} 排名: "
          f"{'Rank ' + str(no_rr['golden_rank']) if no_rr['golden_rank'] else '❌ 未进 Top-5'}")
    print(f"  答案: {no_rr['answer'][:200]}...")

    # ---- B: 有 Reranker ----
    print(f"\n【B】full 模式, 带 Reranker (N=20 → Top-5):")
    print(f"  耗时: {with_rr['elapsed']:.2f}s (精排耗时: "
          f"{with_rr['rerank_info']['elapsed_ms']:.0f}ms)")
    print(f"  Top-5 页: {with_rr['pages']}")
    print(f"  金页 p.{case['expected_page']} 排名: "
          f"{'Rank ' + str(with_rr['golden_rank']) if with_rr['golden_rank'] else '❌ 未进 Top-5'}")
    print(f"  答案: {with_rr['answer'][:200]}...")

    # ---- C: 诊断结论 ----
    print(f"\n【诊断】")
    if no_rr["golden_rank"] is None and with_rr["golden_rank"] is not None:
        print(f"  ✅ Reranker 成功救援! 金页从 '被挤出' 拎回 Rank {with_rr['golden_rank']}")
    elif (no_rr["golden_rank"] is not None and with_rr["golden_rank"] is not None
          and with_rr["golden_rank"] < no_rr["golden_rank"]):
        print(f"  ✅ Reranker 提升排名: Rank {no_rr['golden_rank']} → {with_rr['golden_rank']}")
    elif with_rr["golden_rank"] == no_rr["golden_rank"]:
        print(f"  🟡 排名不变 (可能本来就在 Top-5 里, 或精排没区分出差异)")
    elif no_rr["golden_rank"] is not None and with_rr["golden_rank"] is None:
        print(f"  ❌ Reranker 反而把金页挤掉了!  需要深入诊断")
    else:
        print(f"  ⚠️ 两次都没捞到金页 (召回层问题, 不是精排问题)")

    print(f"  延迟代价: +{(with_rr['elapsed']-no_rr['elapsed'])*1000:.0f}ms")


def main():
    print("初始化 AdvancedRAGPipeline...")
    rag = AdvancedRAGPipeline()

    # 预热一次 Reranker (懒加载触发)
    print("\n预热 Reranker...")
    _ = rag.reranker

    # 跑三个案例
    for case in CASES:
        print(f"\n\n{'#'*70}")
        print(f"# 开始测试: {case['name']}")
        print(f"{'#'*70}")

        no_rr = run_one(rag, case, use_reranker=False)
        with_rr = run_one(rag, case, use_reranker=True)
        print_comparison(case, no_rr, with_rr)

    print("\n\n" + "=" * 70)
    print("所有测试完成")
    print("=" * 70)


if __name__ == "__main__":
    main()