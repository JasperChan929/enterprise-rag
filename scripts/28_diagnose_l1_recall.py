"""Day 9 Task 2: L1 召回诊断 + AdvancedRAG 验证.

==============================================================================
1. 任务身份溯源 (偏好 7)
==============================================================================
完整身份: Day 9 Task 2 "L1 召回诊断". Day 8 第 8.2 节预告, Day 9 Task 1 发现
         U3/U6/U5 三个死穴样本后细化为 "用 Day 6-7 现成武器验证能否救死穴".

历史出处:
  - Day 8 笔记 8.2 节首次登记为 Task 2
  - Day 9 Task 1 跑完发现 U3 9 金页 top_k=15 仅 1 个命中, U6 top_k=10 全 0
    → 确认 U3/U6/U5 是死穴, Task 2 的核心价值从"三路分工"升级为"救药验证"
  - Day 9 Task 2 设计反复 3 次 (第 1 版漏看 Multi-Query 已做, 第 2 版漏看
    Reranker 已集成, 第 3 版确认 /mnt/project 不是权威源后定稿)

当前用途:
  - Phase 1: 读 day09-3way-cache.json 做三路分工诊断 (快, 10 秒)
  - Phase 2: 对 U3/U5/U6 跑 5 mode × 2 reranker = 30 次 AdvancedRAG (2-4 min)
  - Phase 3: 紫金专项 (砍了, 留 Day 10)

当前状态: Day 9 Task 2 今天跑, 产出 docs/day09-l1-diagnosis.md

==============================================================================
2. 自我攻击清单 (D7 新规则含第 6 步参数耦合检查)
==============================================================================

FP1: Phase 1 "某路 0 金页" 可能是金页标注过严
  - day09-gold-pages.final.json 是 grep + 人工复核, 可能漏标了相关页
  - 缓解: 对两路都 0 的查询, 额外看 Top-20 有没有 "金页 ±2 页", 提示可能漏标

FP2: Phase 2 "LLM 答对" 判断靠什么 (主观判定是不能接受的)
  - 缓解: 硬标准 = 答案里含"金页对应关键数字或短语"
    U3: "582,259,599.61" / "5.82亿" / "归母 3.41亿" (对应 p.7/11/137)
    U5: 含 "地缘政治" + "海外项目" 的应对措施 (对应 p.59/60)
    U6: "423,701,834" / "4237亿" / "17.04%" (对应 p.11/116)
  - 这个标准硬编码在 ANSWER_CHECK 里

FN1: Reranker 可能反而把 Top-5 搞糟
  - Day 7 发现 Reranker 在数据边界场景把金页挤掉过 (招行 2024 案例)
  - 缓解: 报告同时记录 use_reranker=False/True 的命中排名变化

FN2: Auto 模式 Router 可能每次选错
  - 缓解: 记录每次 routing_decision, 看 Router 实际选了什么 mode

FP6 (D7 参数耦合检查):
  - AdvancedRAGPipeline() 默认参数: top_k=5, recall_per_probe=10
  - Reranker 参数: rerank_input_n=20 (只在 use_reranker=True 生效)
  - 这些参数彼此独立, 不耦合 → Phase 2 结论适用"默认参数配置"
  - 报告首尾标注这个约束

预估结果 (偏差大要警觉):
  - U6 宁德营收: 5 种 mode 应至少 3 种答对 (简单数字查询)
    -> 如果都答不对, 有 bug 要查
  - U3 国电南自: multi_query 或 full 应能救 (Multi-Query 展开"净利润"变体)
  - U5 紫金海外: full + Reranker 应能救 (HyDE 生成风险类假答案接近 p.59)
  - 如果 U3/U5 full + Reranker 都救不回 → 真 L4 数据边界或金页标错

==============================================================================
3. 样本边界声明 (TD-9-1)
==============================================================================
本实验只覆盖 3 个死穴样本 × 5 mode × 2 reranker = 30 次.
结论是"信号"不是"统计证据", Day 11 大样本评估后才能下系统结论.
===============================================================================
"""
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv(override=True)

import json
import time
from pathlib import Path
from typing import Optional

from src.pipelines.advanced_rag import AdvancedRAGPipeline


# ============================================================================
# 配置区
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"

CACHE_FILE = DOCS_DIR / "day09-3way-cache.json"
GOLD_FILE = DOCS_DIR / "day09-gold-pages.final.json"
REPORT_FILE = DOCS_DIR / "day09-l1-diagnosis.md"

# Phase 2 要验证的死穴
DEAD_QUERIES = ["U3", "U5", "U6"]
MODES = ["hybrid", "multi_query", "hyde", "full", "auto"]

# 答案正确性硬标准 (按 FP2 设计)
#   只要答案字符串含列表里任一关键字 → 算答对
#   多关键字是 OR 关系, 只要命中一个就算
ANSWER_CHECK = {
    "U3": [
        "582,259,599",      # 准确净利润数字 (含分隔符)
        "5.82亿",             # 亿元口径
        "5.82 亿",
        "3.41亿",             # 归母净利润
        "3.41 亿",
        "归母净利润",
    ],
    "U5": [
        # U5 定性查询, 要求同时含 "地缘政治"+"海外" 相关词 (任一组合)
        # 用分开检查: answer 要同时含"地缘政治" AND ("海外"或"风险")
        # 见 check_answer() 函数特判
        "__U5_SPECIAL__",
    ],
    "U6": [
        "423,701,834",       # 准确营收
        "4237亿",             # 亿元口径
        "4237 亿",
        "4,237",              # 可能的千分位
        "17.04%",             # 同比增长率
    ],
}


# ============================================================================
# 工具函数
# ============================================================================

def check_answer(query_id: str, answer: str) -> tuple[bool, str]:
    """检查 LLM 答案是否含金页关键信息.

    Returns:
        (是否答对, 命中的关键字或原因)
    """
    if query_id == "U5":
        # U5 定性查询: 答案需同时提到"地缘政治"或"海外" + 风险/应对
        has_geopolitical = "地缘政治" in answer
        has_overseas = "海外" in answer
        has_risk = any(w in answer for w in ["风险", "应对", "社区", "治理"])
        if (has_geopolitical or has_overseas) and has_risk:
            return True, "含'海外/地缘政治' + 风险/应对"
        return False, f"缺关键维度 (海外/地缘政治={has_geopolitical or has_overseas}, 风险={has_risk})"

    # U3 / U6: 任一关键字命中就算
    keywords = ANSWER_CHECK.get(query_id, [])
    for kw in keywords:
        if kw in answer:
            return True, f"含 '{kw}'"
    return False, "关键数字未出现"


def get_gold_pages(query_id: str, gold_data: dict) -> set[int]:
    """从 final.json 取某 query 的金页集."""
    for q in gold_data["queries"]:
        if q["id"] != query_id:
            continue
        gold = set()
        if q.get("is_empty_gold"):
            return gold
        for c in q.get("candidates", []):
            if c.get("confirmed") is True:
                gold.add(c["page"])
        for extra in q.get("extra_gold_pages", []):
            if isinstance(extra, list) and len(extra) == 2:
                gold.add(extra[1])
        return gold
    return set()


def count_hits(pages: list[int], gold_pages: set[int]) -> tuple[int, Optional[int]]:
    """统计金页命中数 + 首次命中排名 (1-indexed).

    注意: 同一个 page 在 pages 里可能出现多次 (同页多 chunk),
          但命中数只算一次 (用 set 去重).
    """
    # 去重但保序: 取每个 page 首次出现的 rank
    seen_pages = {}  # page -> first_rank
    for rank, page in enumerate(pages, start=1):
        if page not in seen_pages:
            seen_pages[page] = rank

    # 统计命中
    hit_pages = set(seen_pages.keys()) & gold_pages
    hits = len(hit_pages)
    first_rank = min(
        (seen_pages[p] for p in hit_pages),
        default=None,
    )
    return hits, first_rank


# ============================================================================
# Phase 1: 三路分工诊断 (从 cache 读)
# ============================================================================

def run_phase_1(cache_data: dict, gold_data: dict, rag: AdvancedRAGPipeline) -> list[dict]:
    """Phase 1: 三路分工诊断.

    读 day09-3way-cache.json 取 Dense/BM25 Top-20 (脚本 27 已存).
    Hybrid Top-20 需要现场跑一次 (脚本 27 只存了 Top-5, 这是 Task 1 的遗漏,
    登记为小瑕疵, 不必修 27, Phase 1 自己补跑就好).
    """
    print("\n" + "=" * 70)
    print("Phase 1: 三路分工诊断 (U1-U6, 默认参数 top_k=5/rrf_k=60/recall_mult=4)")
    print("=" * 70)

    # cache 结构: {"per_query": [{"query_id", "dense_top20", "bm25_top20", ...}]}
    per_query_map = {p["query_id"]: p for p in cache_data.get("per_query", [])}

    # 为了得到 Hybrid Top-20, 用 AdvancedRAG 的 hybrid_retriever 直接跑
    # (比重开 HybridRAGPipeline 省时, AdvancedRAG 已经有现成 retriever)
    hybrid_retriever = rag.hybrid_retriever

    results = []
    for qid in ["U1", "U2", "U3", "U4", "U5", "U6"]:
        if qid not in per_query_map:
            print(f"  ⚠️ {qid} 不在 cache, 跳过")
            continue

        gold_pages = get_gold_pages(qid, gold_data)
        gold_count = len(gold_pages)

        entry = per_query_map[qid]

        if gold_count == 0:
            print(f"\n  {qid}: 空集 (L4), 跳过分析")
            results.append({
                "id": qid,
                "gold_count": 0,
                "is_empty": True,
            })
            continue

        # Dense / BM25 Top-20 (从 cache)
        dense_pages = [c["page"] for c in entry.get("dense_top20", [])]
        bm25_pages = [c["page"] for c in entry.get("bm25_top20", [])]

        # Hybrid Top-20 (现场跑)
        # 从 entry 里取 query_text 和 filters (cache 自带)
        query_text = entry.get("query_text", "")
        filters = entry.get("filters")
        hybrid_top20_raw = hybrid_retriever.search(
            query=query_text,
            top_k=20,
            filters=filters,
        )
        hybrid_pages = [r["metadata"].get("page", -1) for r in hybrid_top20_raw]

        d_hits, d_rank = count_hits(dense_pages, gold_pages)
        b_hits, b_rank = count_hits(bm25_pages, gold_pages)
        h_hits, h_rank = count_hits(hybrid_pages, gold_pages)
        h5_hits, _ = count_hits(hybrid_pages[:5], gold_pages)

        print(f"\n  {qid} (金页 {gold_count} 个):")
        print(f"    Dense  Top-20: 命中 {d_hits}/{gold_count}, 首次排名 {d_rank}")
        print(f"    BM25   Top-20: 命中 {b_hits}/{gold_count}, 首次排名 {b_rank}")
        print(f"    Hybrid Top-20: 命中 {h_hits}/{gold_count}, 首次排名 {h_rank}")
        print(f"    Hybrid Top-5 : 命中 {h5_hits}/{gold_count}")

        # 诊断归类
        if d_hits == 0 and b_hits == 0:
            diag = "L1 双路盲 (检索根因)"
        elif (d_hits > 0 or b_hits > 0) and h5_hits == 0:
            diag = "RRF 融合问题 (单路有但 Top-5 没进)"
        elif h5_hits > 0:
            diag = "检索 OK (L2 LLM 层或 L2a 口径)"
        else:
            diag = "未知组合"

        print(f"    诊断: {diag}")

        results.append({
            "id": qid,
            "gold_count": gold_count,
            "dense_hits": d_hits, "dense_rank": d_rank,
            "bm25_hits": b_hits, "bm25_rank": b_rank,
            "hybrid_hits": h_hits, "hybrid_rank": h_rank,
            "hybrid_top5_hits": h5_hits,
            "diagnosis": diag,
        })

    return results


# ============================================================================
# Phase 2: AdvancedRAG 模式矩阵
# ============================================================================

def run_phase_2(gold_data: dict, rag: AdvancedRAGPipeline) -> list[dict]:
    """Phase 2: 死穴救援验证.

    对 U3/U5/U6 跑 5 mode × 2 reranker = 30 次 AdvancedRAG.
    记录: 金页命中数 + LLM 答对与否 + 耗时 + Router 决策 (auto) + Reranker info.
    """
    print("\n\n" + "=" * 70)
    print("Phase 2: 死穴救援 (U3/U5/U6 × 5 mode × 2 reranker = 30 次)")
    print("=" * 70)

    results = []

    # 从 gold_data 里取查询文本和公司过滤
    query_map = {}
    for q in gold_data["queries"]:
        query_map[q["id"]] = {
            "text": q["text"],
            "company": q["company_filter"],
            "gold": get_gold_pages(q["id"], gold_data),
        }

    total = len(DEAD_QUERIES) * len(MODES) * 2
    done = 0

    for qid in DEAD_QUERIES:
        if qid not in query_map:
            print(f"  ⚠️ {qid} 不在 gold_data, 跳过")
            continue

        q = query_map[qid]
        print(f"\n--- {qid}: {q['text']} (金页 {len(q['gold'])} 个) ---")

        for mode in MODES:
            for use_reranker in [False, True]:
                done += 1
                t0 = time.time()
                try:
                    result = rag.query(
                        question=q["text"],
                        mode=mode,
                        filters={"company": q["company"]},
                        top_k=5,
                        use_reranker=use_reranker,
                        rerank_input_n=20,
                    )
                    elapsed = time.time() - t0

                    # 取 Top-5 页号
                    pages = [s["page"] for s in result["sources"] if isinstance(s.get("page"), int)]
                    hits, first_rank = count_hits(pages, q["gold"])

                    correct, reason = check_answer(qid, result["answer"])

                    # Router 选了什么 (仅 auto)
                    routing = result.get("routing_decision")
                    actual_mode = result.get("mode_used", mode)

                    # Reranker 实际开启情况
                    rerank_info = result.get("rerank_info")
                    rerank_ms = rerank_info["elapsed_ms"] if rerank_info else None

                    rr_label = "RR-ON" if use_reranker else "RR-OFF"
                    print(f"  [{done:2d}/{total}] {mode:11s} {rr_label:6s} "
                          f"hits={hits}/{len(q['gold'])} "
                          f"首位={first_rank} "
                          f"答对={'✅' if correct else '❌'} "
                          f"耗时={elapsed:.1f}s"
                          + (f" rerank={rerank_ms:.0f}ms" if rerank_ms else ""))

                    results.append({
                        "query_id": qid,
                        "mode": mode,
                        "use_reranker": use_reranker,
                        "actual_mode": actual_mode,
                        "routing_decision": routing,
                        "top5_pages": pages,
                        "gold_hits": hits,
                        "gold_total": len(q["gold"]),
                        "first_rank": first_rank,
                        "answer_correct": correct,
                        "answer_reason": reason,
                        "answer_preview": result["answer"][:200],
                        "elapsed_s": round(elapsed, 2),
                        "rerank_ms": rerank_ms,
                    })

                except Exception as e:
                    print(f"  [{done:2d}/{total}] {mode} rr={use_reranker} 失败: {e}")
                    results.append({
                        "query_id": qid,
                        "mode": mode,
                        "use_reranker": use_reranker,
                        "error": str(e),
                    })

    return results


# ============================================================================
# 报告生成
# ============================================================================

def write_report(phase1: list[dict], phase2: list[dict]):
    """把 Phase 1 + 2 结果写成 markdown 报告."""
    lines = []
    lines.append("# Day 9 Task 2: L1 召回诊断 + AdvancedRAG 验证\n")
    lines.append(f"> 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("> ")
    lines.append("> **样本边界声明 (TD-9-1)**: 本实验 Phase 1 含 6 样本, Phase 2")
    lines.append("> 含 3 死穴 × 5 mode × 2 reranker = 30 次. 结论是'信号'非'统计证据',")
    lines.append("> 仅适用默认参数配置 (top_k=5, recall_per_probe=10, rerank_input_n=20).")
    lines.append("> Day 11 大样本评估后才能下系统结论.\n")

    # ---- Phase 1 ----
    lines.append("## Phase 1: 三路分工诊断\n")
    lines.append("| 查询 | 金页 | Dense Top-20 | BM25 Top-20 | Hybrid Top-20 | Hybrid Top-5 | 诊断 |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in phase1:
        if r.get("is_empty"):
            lines.append(f"| {r['id']} | 空集 | — | — | — | — | L4 跳过 |")
        else:
            lines.append(
                f"| {r['id']} | {r['gold_count']} | "
                f"{r['dense_hits']}/{r['gold_count']} (R{r['dense_rank']}) | "
                f"{r['bm25_hits']}/{r['gold_count']} (R{r['bm25_rank']}) | "
                f"{r['hybrid_hits']}/{r['gold_count']} (R{r['hybrid_rank']}) | "
                f"{r['hybrid_top5_hits']}/{r['gold_count']} | "
                f"{r['diagnosis']} |"
            )

    # ---- Phase 2 ----
    lines.append("\n## Phase 2: 死穴救援矩阵\n")
    for qid in DEAD_QUERIES:
        q_results = [r for r in phase2 if r["query_id"] == qid and "error" not in r]
        if not q_results:
            continue
        lines.append(f"### {qid}\n")
        lines.append("| Mode | Reranker | Top-5 金页命中 | 首位 | 答对? | 耗时 |")
        lines.append("|---|---|---|---|---|---|")
        for r in q_results:
            rr = "ON" if r["use_reranker"] else "OFF"
            correct = "✅" if r["answer_correct"] else "❌"
            hits_str = f"{r['gold_hits']}/{r['gold_total']}"
            rank_str = str(r['first_rank']) if r['first_rank'] else "-"
            # 如果是 auto 模式, 加 Router 决策
            mode_str = r['mode']
            if r['mode'] == 'auto' and r['actual_mode']:
                mode_str = f"auto → {r['actual_mode']}"
            lines.append(
                f"| {mode_str} | {rr} | {hits_str} | {rank_str} | {correct} | {r['elapsed_s']}s |"
            )
        lines.append("")

        # 找"最佳救援" = 答对 + 命中最高
        best = sorted(q_results, key=lambda x: (
            -(1 if x.get("answer_correct") else 0),
            -(x.get("gold_hits") or 0)
        ))[:1]
        if best:
            b = best[0]
            if b.get("answer_correct"):
                rr = "ON" if b["use_reranker"] else "OFF"
                lines.append(f"**{qid} 救援结论**: ✅ `{b['mode']} + reranker={rr}` 救回, 答对.")
            else:
                lines.append(f"**{qid} 救援结论**: ❌ 5 mode × 2 reranker 全部答错. 可能是 L4 数据边界或金页标注错误.")
            lines.append("")

    # ---- 对 Day 10 的影响 ----
    lines.append("\n## 对 Day 10 开工的影响\n")
    # 简要总结救援情况
    救回 = []
    没救 = []
    for qid in DEAD_QUERIES:
        q_results = [r for r in phase2 if r["query_id"] == qid and "error" not in r]
        if any(r.get("answer_correct") for r in q_results):
            救回.append(qid)
        else:
            没救.append(qid)

    if 救回:
        lines.append(f"**救回的死穴**: {', '.join(救回)}")
        lines.append(f"  → Day 10 开工第一步: 跑 `AdvancedRAGPipeline(mode='auto')` 作为新 baseline, 确认 Router 选的是正确 mode")
    if 没救:
        lines.append(f"\n**未救回的死穴**: {', '.join(没救)}")
        lines.append(f"  → Day 10 需要追加分析: 是金页标错, 还是真 L4, 还是 Prompt 工程能救")

    lines.append("\n## 下一步 (Day 10)")
    lines.append("- 基于 Phase 2 结果, 决定 TD-8-2 Prompt 工程的优先级")
    lines.append("- 如 auto mode 已能救大部分死穴, Day 10 转向 Router 校准")
    lines.append("- 如未救回, Day 10 按原计划进 TD-8-2 Prompt 工程主战场")

    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n✅ 报告已写入 {REPORT_FILE}")


# ============================================================================
# 主入口
# ============================================================================

def main():
    # 读数据
    if not CACHE_FILE.exists():
        print(f"❌ 缺 {CACHE_FILE}, 需先跑脚本 27 --stage=scan")
        return
    if not GOLD_FILE.exists():
        print(f"❌ 缺 {GOLD_FILE}, 需先跑脚本 27 --stage=draft + 人工复核")
        return

    print("加载 day09-3way-cache.json ...")
    cache_data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))

    print("加载 day09-gold-pages.final.json ...")
    gold_data = json.loads(GOLD_FILE.read_text(encoding="utf-8"))

    # 初始化 AdvancedRAG (Phase 1 也要用, 因为要现场跑 Hybrid Top-20)
    print("\n初始化 AdvancedRAGPipeline (含 BM25 build, 约 5 秒)...")
    rag = AdvancedRAGPipeline()

    # Phase 1 (快)
    phase1 = run_phase_1(cache_data, gold_data, rag)

    # 暂停确认 (D7 元规则: 凌晨分阶段, 小范围验证)
    print("\n\n" + "=" * 70)
    print("Phase 1 完成. 是否继续 Phase 2? (30 次 AdvancedRAG 调用, 2-4 min)")
    print("=" * 70)
    input("  按 Enter 继续, Ctrl+C 取消: ")

    # Phase 2 (慢, 30 次 LLM)
    print("\n预热 Reranker (首次加载 2 秒)...")
    _ = rag.reranker  # 触发懒加载

    phase2 = run_phase_2(gold_data, rag)

    # 写报告
    write_report(phase1, phase2)

    # 存完整结果为 JSON (供 Day 10 复用)
    results_json = DOCS_DIR / "day09-l1-diagnosis-results.json"
    results_json.write_text(
        json.dumps({"phase1": phase1, "phase2": phase2}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"✅ 完整结果 JSON: {results_json}")
    print("\n Task 2 跑完.")


if __name__ == "__main__":
    main()