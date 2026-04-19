"""Day 7 闭环验证: 招行案例是'数据边界'还是'Reranker 失败'?

背景:
  Day 7 案例 1 发现, query 问 '招行 2024 年不良贷款率',
  无 Reranker 组召回 p.45 (但 LLM 拒答, 因 p.45 是 2025 年数据),
  带 Reranker 组把 p.45 挤掉.
  
  初判: Reranker 精读出时间错位, 把 2025 年数据降权是正确行为.
  需要验证: 如果 query 改成 2025 年, 两组是否都能答对 0.93%?

逻辑:
  - 两组都答对 '0.93%'  → 证明 p.45 是 2025 数据, 原案例是数据边界
  - 两组都拒答          → Qdrant 里数据有问题, 需重新审视
"""
from dotenv import load_dotenv
load_dotenv(override=True)

import time
from src.pipelines.advanced_rag import AdvancedRAGPipeline


def run_one(rag, use_reranker: bool) -> dict:
    """跑一次 full 模式查询, 返回关键信息.
    
    Args:
        rag: AdvancedRAGPipeline 实例
        use_reranker: 是否启用 Cross-Encoder 精排
            - False: 复现 Day 6 行为 (Hybrid + Multi-Query + HyDE, Top-5 直出)
            - True:  召回 20 条 → Reranker 精排 → 取 Top-5
    
    Returns:
        {
            "answer": LLM 生成的答案字符串,
            "pages": Top-5 的页码列表,
            "elapsed": 端到端耗时(秒),
        }
    """
    t0 = time.time()
    result = rag.query(
        question="招商银行2025年不良贷款率是多少?",  # 关键: 问 2025 年, 对准 p.45 的真实年份
        mode="full",                                  # 用 Full 模式保持和 Day 6/23 同口径
        filters={"company": "招商银行"},             # 只查招行, 避免跨公司噪声
        top_k=5,                                      # LLM 最终看到 5 条
        use_reranker=use_reranker,
        rerank_input_n=20,                            # 精排开启时召回 20 条给 Reranker
    )
    return {
        "answer": result["answer"],
        "pages": [s["page"] for s in result["sources"]],
        "elapsed": time.time() - t0,
    }


def main():
    rag = AdvancedRAGPipeline()
    _ = rag.reranker  # 预热 Reranker, 避免第一次调用算入 A 组耗时

    print("\n" + "=" * 70)
    print("闭环验证: Query 改为 '2025 年' 后, 两组应该都答对")
    print("=" * 70)

    print("\n【A】无 Reranker:")
    a = run_one(rag, use_reranker=False)
    print(f"  Top-5 页: {a['pages']}")
    print(f"  耗时: {a['elapsed']:.2f}s")
    print(f"  答案: {a['answer'][:250]}")

    print("\n【B】有 Reranker:")
    b = run_one(rag, use_reranker=True)
    print(f"  Top-5 页: {b['pages']}")
    print(f"  耗时: {b['elapsed']:.2f}s")
    print(f"  答案: {b['answer'][:250]}")

    print("\n【诊断】")
    print("  两组都明确答出 '0.93%' → 证明 p.45 是 2025 年数据")
    print("  → Day 7 案例 1 性质: 数据边界, 不是 Reranker 失败")


if __name__ == "__main__":
    main()