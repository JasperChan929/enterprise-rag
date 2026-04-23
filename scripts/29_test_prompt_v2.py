# scripts/29_test_prompt_v2.py
"""Day 10 T2: TD-8-2 Prompt v2 验证 (U3 + U4 肉眼对比).

================================================================================
验证目标
================================================================================
新 RAG_SYSTEM_PROMPT 加了规则 5/6 (多口径处理) 后:
  - U4 招行不良率 (Day 8 多口径拒答典型) 是否从拒答变为作答
  - U3 国电南自 (Day 9 已会区分子公司/集团) 答案结构是否更清晰

借助 src.evaluation.answer_check.is_refusal 做拒答态硬判定 (Day 9 D9 教训).

================================================================================
样本边界 (TD-9-1 延续)
================================================================================
2 样本 × 1 次 = 2 次调用. 是"信号"不是"结论", Day 11 RAGAS 大样本评估为准.

================================================================================
用法
================================================================================
uv run python scripts/29_test_prompt_v2.py

================================================================================
🛑 简化版自我攻击 (按 Day 10 Prompt 改动规模 scale)
================================================================================
- FM: LLM 过度触发多口径规则, 列出无关口径 → 接受, 记录到 day10-summary
- FN: U4 仍拒答 → 进 Day 10 T3 登记, 作为"否定指令遵循度弱"案例
- 无法验证的假设: Prompt 真的生效 (vs LLM 随机) → 不做 3 次抽样, 1 次即可
"""
from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(override=True)

from src.evaluation.answer_check import is_refusal
from src.pipelines.advanced_rag import AdvancedRAGPipeline


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
DAY09_RESULTS = DOCS_DIR / "day09-l1-diagnosis-results.json"


# ============================================================
# 样本定义 (保持 Day 8/9 原 query 措辞, 便于新旧答案公平对比)
# ============================================================

CASES = [
    {
        "id": "U3",
        "question": "国电南自2024年归属于上市公司股东的净利润",
        "company_filter": "国电南自",
        "baseline_desc": (
            "Day 9 Phase 2 auto+reranker: 正确区分子公司/集团, "
            "明确说'未提供集团口径'. 已经不错, T2 验证新 Prompt 是否让结构更清晰."
        ),
    },
    {
        "id": "U4",
        "question": "招商银行2025年不良贷款率",
        "company_filter": "招商银行",
        "baseline_desc": (
            "Day 8 审计结论: Top-5 含 3 个口径 (本公司 0.93% / 集团 0.94% / "
            "消费信贷 1.52%), LLM 拒答. T2 期望: 列出 3 口径并优先本公司 0.93%."
        ),
    },
]


# ============================================================
# 工具
# ============================================================

def load_u3_day9_answer() -> str:
    """从 day09-l1-diagnosis-results.json 取 U3 的 auto+reranker 答案作基线."""
    if not DAY09_RESULTS.exists():
        return "(未找到 day09 结果文件)"
    data = json.loads(DAY09_RESULTS.read_text(encoding="utf-8"))
    for entry in data.get("phase2", []):
        if (
            entry.get("query_id") == "U3"
            and entry.get("mode") == "auto"
            and entry.get("use_reranker") is True
        ):
            return entry.get("answer_preview", "(未取到 answer_preview 字段)")
    return "(day09 results 里没有 U3 auto+reranker 的记录)"


# ============================================================
# 主逻辑
# ============================================================

def main():
    print("=" * 78)
    print("Day 10 T2: Prompt v2 验证 (U3 + U4 肉眼对比)")
    print("=" * 78)

    print("\n初始化 AdvancedRAGPipeline + 预热 Reranker...")
    rag = AdvancedRAGPipeline()
    _ = rag.reranker

    u3_day9 = load_u3_day9_answer()

    for case in CASES:
        print(f"\n\n{'#' * 78}")
        print(f"# {case['id']}: {case['question']}")
        print(f"{'#' * 78}")

        # --- 旧基线 ---
        print(f"\n【旧基线】")
        print(f"  {case['baseline_desc']}")
        if case['id'] == 'U3':
            truncated = u3_day9[:400] + ('...' if len(u3_day9) > 400 else '')
            print(f"\n  Day 9 原答案:\n  {truncated}")

        # --- 新跑 ---
        print(f"\n【新 Prompt v2 跑一次 (auto + reranker)】")
        result = rag.query(
            question=case['question'],
            mode="auto",
            filters={"company": case['company_filter']},
            use_reranker=True,
        )

        new_answer = result['answer']
        is_ref, pattern = is_refusal(new_answer)

        print(f"\n  mode_used: {result.get('mode_used')}")
        print(f"  is_refusal: {is_ref}" + (f" (pattern: '{pattern}')" if is_ref else ""))
        print(f"\n  新答案 (完整):\n{new_answer}")

        # --- Top-5 来源 ---
        print(f"\n  Top-5 来源页:")
        for i, src in enumerate(result['sources'][:5], 1):
            company = src.get('company', '?')
            year = src.get('year', '?')
            page = src.get('page', '?')
            chunk_type = src.get('chunk_type', '?')
            print(f"    [{i}] {company} {year}年报 第{page}页 ({chunk_type})")

    print(f"\n\n{'=' * 78}")
    print("T2 完成. 请肉眼判断 (记到 day10-summary.md):")
    print("  U3: 新答案结构是否比旧答案更清晰 (已找到 X / 未找到 Y 的格式)?")
    print("  U4: 新答案是否列出 3 个口径 (本公司/集团/消费信贷)?")
    print(f"{'=' * 78}")


if __name__ == "__main__":
    main()