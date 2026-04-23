"""Day 11 T2/T3: RAGAS 主评估脚本.

===============================================================================
作用
===============================================================================
跑 11 条 U × 3 mode × 3 metric = 99 条评估记录, 产出:
- docs/day11-ragas-results.jsonl   (每行 1 条 (U, mode) 的完整 trace, T3 人工抽检用)
- docs/day11-ragas-summary.md      (聚合分数表 + 发现)
- 终端打印进度条 + 告警 (retrieved 为空 / is_refusal 触发 / Faithfulness NaN 等)

===============================================================================
设计决策
===============================================================================
- 3 mode 选择: hybrid / hyde / full+reranker  (Day 10 §8.1 定)
- 3 metric: Faithfulness + ResponseRelevancy + LLMContextPrecisionWithoutReference
  (决策点 ③ 方案 3, 都不需要 reference)
- 每条结果**立即落盘** (JSONL append 模式), 中途崩溃也不丢数据. 这是 Day 9
  Task 2 "30 次调用中途 4 次失败要重跑" 教训的直接应用.
- is_refusal 指标并行跑, 标记 TD-10-3 FN-2 污染度字段. Day 12 决策就看这个数.

===============================================================================
跑法
===============================================================================
    uv run python scripts/31_ragas_eval.py

    # 或指定只跑部分 U (smoke):
    uv run python scripts/31_ragas_eval.py --queries U1 U6

    # 或指定只跑部分 mode:
    uv run python scripts/31_ragas_eval.py --modes hybrid

===============================================================================
预估时长
===============================================================================
- 11 U × hybrid (~10s) = ~2 min
- 11 U × hyde (~20s) = ~4 min
- 11 U × full+reranker (~50s, 含 23s rerank) = ~10 min
- judge 调用: 每 (U, mode) 约 3 metric × 3-5 次 = 10-20 次 judge, ×33 = 约 500 次
  * gpt-4o-mini 经中转约 1-3 秒/次 = 10-30 min
- 总计: 25-50 分钟. 配预算 1 小时跑完有余
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import warnings
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(override=True)


# =============================================================================
# 配置
# =============================================================================
ROOT_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT_DIR / "docs"
EVAL_QUERIES_PATH = DOCS_DIR / "day11-eval-queries.json"
RESULTS_JSONL_PATH = DOCS_DIR / "day11-ragas-results.jsonl"
SUMMARY_MD_PATH = DOCS_DIR / "day11-ragas-summary.md"

# 3 mode 配置
MODES = [
    {"name": "hybrid", "mode": "hybrid", "use_reranker": False},
    {"name": "hyde", "mode": "hyde", "use_reranker": False},
    {"name": "full+reranker", "mode": "full", "use_reranker": True},
]
METRIC_KEYS = ("faithfulness", "answer_relevancy", "context_precision")

# =============================================================================
# 加载评估集
# =============================================================================
def load_queries(filter_ids: list[str] | None = None) -> list[dict]:
    with open(EVAL_QUERIES_PATH, encoding="utf-8") as f:
        data = json.load(f)
    queries = data["queries"]
    if filter_ids:
        queries = [q for q in queries if q["id"] in filter_ids]
        if not queries:
            raise ValueError(f"没有匹配的 U ID: {filter_ids}")
    return queries


# =============================================================================
# 构造 RAGAS metrics (Collections API)
# =============================================================================
def build_metrics(llm):
    from openai import AsyncOpenAI
    from ragas.embeddings.base import embedding_factory
    from ragas.metrics.collections import (
        Faithfulness,
        AnswerRelevancy,
        ContextPrecisionWithoutReference,
    )
    import os

    # AnswerRelevancy 内部 embedding 跑同步, 用同步 OpenAI client
    embedding_client = AsyncOpenAI(
        api_key=os.environ["OPENAI_JUDGE_API_KEY"],
        base_url=os.environ["OPENAI_JUDGE_BASE_URL"],
    )
    embeddings = embedding_factory(
        "openai",
        model="text-embedding-3-small",
        client=embedding_client,
        interface="modern",
    )

    return {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),  # type: ignore[arg-type]
        "context_precision": ContextPrecisionWithoutReference(llm=llm),
    }


# =============================================================================
# 单 (U, mode) 评估
# =============================================================================
async def evaluate_one(query: dict, mode_cfg: dict, metrics: dict) -> dict:
    """跑一条 U × mode, 返回完整记录."""
    from src.evaluation.pipeline_adapter import eval_run
    from src.evaluation.answer_check import is_refusal

    qid = query["id"]
    qtext = query["text"]
    company = query["company_filter"]
    mode_name = mode_cfg["name"]

    print(f"  ⏳ {qid} × {mode_name} ... ", end="", flush=True)

    # 1. 跑 pipeline
    t0 = time.time()
    try:
        pipe_out = eval_run(
            question=qtext,
            mode=mode_cfg["mode"],
            use_reranker=mode_cfg["use_reranker"],
            filters={"company": company},
        )
        pipe_elapsed = time.time() - t0
    except Exception as e:
        pipe_elapsed = time.time() - t0
        print(f"❌ pipeline 失败 ({pipe_elapsed:.1f}s): {type(e).__name__}: {e}")
        return {
            "qid": qid, "mode": mode_name,
            "error": f"pipeline: {type(e).__name__}: {e}",
            "pipe_elapsed_s": round(pipe_elapsed, 1),
        }

    answer = pipe_out["answer"]
    contexts = pipe_out["retrieved_contexts"]

    # 2. is_refusal
    refusal_hit, refusal_pattern = is_refusal(answer)

    # 3. 3 个 metric. 各接受参数不同:
    #   Faithfulness / ContextPrecisionWithoutReference: user_input + response + retrieved_contexts
    #   AnswerRelevancy: user_input + response (不要 retrieved_contexts)
    scores: dict = {k: None for k in METRIC_KEYS}
    metric_errors: dict = {}
    for mname, scorer in metrics.items():
        try:
            if mname == "answer_relevancy":
                result = await scorer.ascore(
                    user_input=qtext,
                    response=answer,
                )
            else:
                result = await scorer.ascore(
                    user_input=qtext,
                    response=answer,
                    retrieved_contexts=contexts,
                )
            score = result.value if hasattr(result, "value") else float(result)
            scores[mname] = score
        except Exception as e:
            metric_errors[mname] = f"{type(e).__name__}: {e}"

    total_elapsed = time.time() - t0

    # 4. 进度打印
    score_str = " ".join(
        f"{m[:4]}={scores[m]:.2f}" if scores[m] is not None else f"{m[:4]}=ERR"
        for m in METRIC_KEYS
    )
    refusal_str = "🚫拒答" if refusal_hit else "✅作答"
    print(f"{refusal_str} {score_str} ({total_elapsed:.1f}s)")

    return {
        "qid": qid,
        "mode": mode_name,
        "query_text": qtext,
        "query_type": query.get("query_type"),
        "known_td_link": query.get("known_td_link"),
        "answer": answer,
        "answer_length": len(answer),
        "is_refusal": refusal_hit,
        "refusal_pattern": refusal_pattern,
        "top_pages": pipe_out["top_pages"],
        "rerank_info": pipe_out.get("rerank_info"),
        "contexts_count": len(contexts),
        "contexts_total_chars": sum(len(c) for c in contexts),
        "scores": scores,
        "metric_errors": metric_errors if metric_errors else None,
        "pipe_elapsed_s": round(pipe_elapsed, 1),
        "total_elapsed_s": round(total_elapsed, 1),
    }


# =============================================================================
# 结果聚合 (简易版, T3 人工抽检才是主要分析手段)
# =============================================================================
def summarize(all_results: list[dict]) -> str:
    """生成 markdown 聚合表. 按 mode × metric 计算 mean / median / refusal_rate."""
    from statistics import mean, median

    lines = ["# Day 11 RAGAS 聚合结果 (自动生成)", ""]
    lines.append(f"- 总记录数: {len(all_results)}")
    lines.append(f"- 生成时间: Day 11 T2 自动生成")
    lines.append("")
    lines.append("## mode × metric 平均分")
    lines.append("")
    lines.append("| mode | faithfulness | answer_relevancy | context_precision | refusal_rate |")
    lines.append("|---|---|---|---|---|")

    for mode_name in ["hybrid", "hyde", "full+reranker"]:
        rows = [r for r in all_results if r.get("mode") == mode_name and "scores" in r]
        if not rows:
            lines.append(f"| {mode_name} | - | - | - | - |")
            continue

        def agg(metric):
            vals = [r["scores"][metric] for r in rows if r["scores"].get(metric) is not None]
            return f"{mean(vals):.3f} (n={len(vals)})" if vals else "-"

        refusal_rate = sum(1 for r in rows if r.get("is_refusal")) / len(rows)
        lines.append(
            f"| {mode_name} | {agg('faithfulness')} | {agg('answer_relevancy')} "
            f"| {agg('context_precision')} | {refusal_rate:.0%} ({sum(1 for r in rows if r.get('is_refusal'))}/{len(rows)}) |"
        )

    lines.append("")
    lines.append("## 按 query_type 拆分 (faithfulness 平均)")
    lines.append("")
    lines.append("| query_type | hybrid | hyde | full+reranker |")
    lines.append("|---|---|---|---|")

    qtypes: list[str] = sorted({
        r["query_type"] for r in all_results
        if r.get("query_type") is not None
    })
    for qt in qtypes:
        row = [qt]
        for mode_name in ["hybrid", "hyde", "full+reranker"]:
            vals = [
                r["scores"]["faithfulness"]
                for r in all_results
                if r.get("mode") == mode_name
                and r.get("query_type") == qt
                and r.get("scores", {}).get("faithfulness") is not None
            ]
            row.append(f"{mean(vals):.3f}" if vals else "-")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("## TD-10-3 FN-2 候选 (is_refusal=True 但答案较长)")
    lines.append("")
    lines.append("定义: is_refusal 命中但 answer_length > 200 字符 — 疑似'拒答句+半作答'")
    lines.append("")
    lines.append("| qid | mode | answer_length | faithfulness | refusal_pattern |")
    lines.append("|---|---|---|---|---|")

    fn2_candidates = [
        r for r in all_results
        if r.get("is_refusal") and r.get("answer_length", 0) > 200
    ]
    if not fn2_candidates:
        lines.append("| - | - | - | - | - |")
        lines.append("")
        lines.append("*无 FN-2 候选. 或 is_refusal 和答案长度强相关 (纯拒答都很短).*")
    else:
        for r in fn2_candidates:
            faith = r.get("scores", {}).get("faithfulness")
            faith_str = f"{faith:.3f}" if faith is not None else "-"
            lines.append(
                f"| {r['qid']} | {r['mode']} | {r['answer_length']} | "
                f"{faith_str} | {r.get('refusal_pattern', '-')} |"
            )
        lines.append("")
        lines.append(f"*共 {len(fn2_candidates)} 条疑似 FN-2. T3 人工抽检核对, Day 12 决策修 is_refusal 用*")

    lines.append("")
    lines.append("## 明细参见")
    lines.append("")
    lines.append(f"- `docs/day11-ragas-results.jsonl` - 每行 1 条 (U, mode) 完整 trace")
    lines.append(f"- T3 人工抽检重点看: is_refusal=True 的条目 + faithfulness < 0.5 的条目")

    return "\n".join(lines)


# =============================================================================
# 主流程
# =============================================================================
async def main(filter_queries: list[str] | None, filter_modes: list[str] | None):
    print("=" * 70)
    print("Day 11 T2: RAGAS 评估主脚本")
    print("=" * 70)
    print()

    # ---- 构造 judge ----
    from src.evaluation.judge_config import build_judge_llm, describe_judge
    print(describe_judge())
    llm = build_judge_llm()
    metrics = build_metrics(llm)
    print(f"✅ {len(metrics)} 个 metric 就绪: {list(metrics.keys())}")
    print()

    # ---- 加载 queries / modes ----
    queries = load_queries(filter_queries)
    modes_to_run = [m for m in MODES if (not filter_modes or m["name"] in filter_modes)]
    total = len(queries) * len(modes_to_run)
    print(f"评估任务: {len(queries)} U × {len(modes_to_run)} mode = {total} 条")
    print(f"  U:     {[q['id'] for q in queries]}")
    print(f"  mode:  {[m['name'] for m in modes_to_run]}")
    print()

    # ---- 清空旧结果 JSONL (允许重跑) ----
    if RESULTS_JSONL_PATH.exists():
        print(f"⚠️  覆盖旧 results: {RESULTS_JSONL_PATH}")
        RESULTS_JSONL_PATH.unlink()

    # ---- 主循环: U outer, mode inner. 按 U 分组打印更易看 ----
    all_results = []
    t0 = time.time()
    for qi, query in enumerate(queries, 1):
        print(f"▶ [{qi}/{len(queries)}] {query['id']}: {query['text']}")
        for mode_cfg in modes_to_run:
            record = await evaluate_one(query, mode_cfg, metrics)
            all_results.append(record)

            # 立即落盘 (崩溃也不丢数据)
            with open(RESULTS_JSONL_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print()

    total_elapsed = time.time() - t0
    print(f"✅ 评估完成, 总耗时 {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")

    # ---- 生成 summary ----
    print(f"生成 summary → {SUMMARY_MD_PATH}")
    summary = summarize(all_results)
    with open(SUMMARY_MD_PATH, "w", encoding="utf-8") as f:
        f.write(summary)

    print()
    print("=" * 70)
    print(summary)
    print("=" * 70)
    print()
    print(f"明细: {RESULTS_JSONL_PATH}")
    print(f"聚合: {SUMMARY_MD_PATH}")
    print()
    print("下一步 (T3): 读 day11-ragas-results.jsonl, 人工抽检以下条目:")
    print("  - is_refusal=True 的条目 → 看是否'拒答句+半作答', 估计 FN-2 占比")
    print("  - faithfulness<0.5 的条目 → 看是 judge 判错还是 RAG 真答错")
    print("  - scores 有 None 的 → 看是 judge 解析失败还是 metric 不适用")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", nargs="+", help="只跑指定 U ID, 如 --queries U1 U6")
    parser.add_argument("--modes", nargs="+",
                        choices=["hybrid", "hyde", "full+reranker"],
                        help="只跑指定 mode")
    args = parser.parse_args()

    asyncio.run(main(args.queries, args.modes))
