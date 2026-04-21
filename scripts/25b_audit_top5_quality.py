# scripts/25b_audit_top5_quality.py
"""Day 8 - 脚本 25b: L2 子类型细化 (L2a vs L2b)

===============================================================================
背景 (为什么需要这个脚本)
===============================================================================
脚本 25 的结果显示 U2/U3/U4 三个案例都被判为 L2 (LLM 作答问题), 推翻了
Day 7 的 L4/L0 先验判定. 但 "L2" 这个大类粒度太粗, 里面混了两种情况:

  L2a (纯 LLM 问题): Top-5 里证据完整, LLM 还是拒答
                     → 修法: Prompt 工程 (CoT / 放松规则 / metadata 注入)
                     → 成本低

  L2b (排序问题):    Top-5 里证据碎片, 完整证据在 Top-6~20
                     → 修法: Reranker 条件开启 / 调 RRF / 扩 Top-N
                     → 成本中

这两种修法不一样, 错判会浪费时间. 必须区分.

===============================================================================
判定方法: "自足 chunk" 概念
===============================================================================

"自足 chunk": 对于一个特定问题, chunk 里同时包含 3 个要素:
  1. 主体 (公司): 由 metadata.company 保证, 不需要 chunk 文本里有
  2. 时间锚点: chunk 文本里有目标年份 (如 "2024" "2025")
  3. 核心数据: chunk 文本里有直接答案 (如 "0.93%" "X 万元净利润")

三要素齐全 → 自足 chunk (LLM 一看就能答)
缺 1-2 要素 → 碎片 chunk (LLM 需要推理组合)

判定逻辑:
  Top-5 里有自足 chunk → L2a (Prompt 问题)
  Top-5 全是碎片 chunk + Top-6~20 有自足 chunk → L2b (排序问题)
  Top-20 都没自足 chunk → 重审 (可能是 chunking 问题, 或关键词检查太严)

===============================================================================
为什么只审 U3/U4, 不审 U2
===============================================================================
U2 (宁德磷酸铁锂) 的问题不是"没数据", 是"抽象词语义鸿沟":
  用户问 "磷酸铁锂产品情况", 年报只列了 "神行电池 / Pro 电池 / 钠新乘用车电池"
  这种具体产品. "情况" 这个抽象词 LLM 无法对应到产品列表.

  这是 L2 里的第 3 种子类型 (可以叫 L2c "语义粒度错配"), 和 L2a/L2b 不一样.
  需要的修法是 Multi-Query (改写成更具体的子查询), Day 6 已经做了.
  所以 U2 跳过, 不在 Top-5 审计范围内.
"""
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv(override=True)

import re

from src.embeddings.bge import BGEEmbedder
from src.loaders.base import Chunk
from src.retrievers.bm25_store import BM25Store
from src.retrievers.hybrid import HybridRetriever
from src.retrievers.qdrant_store import get_client


# ============================================================================
# 配置: U3 和 U4 的自足 chunk 要素定义
# ============================================================================
#
# 每个案例指定:
#   year_anchors:  时间锚点关键词 (任一命中算"时间锚点存在")
#   data_signals:  核心数据关键词 (任一命中算"核心数据存在")
#   data_regex:    可选的正则模式 (比 keyword 更灵活, 能匹配数字)
#
# 设计原则:
#   year_anchors 要严 (不能匹配到无关年份, 如 U4 里 "2024 年对比"
#                      对应的是"同比数据"不是"本期"):
#     → 用 "2025 年" 而不是 "2025"
#   data_signals 要多样 (覆盖同一数据的不同写法):
#     → U3 净利润写法多: "净利润 X 亿" / "净利润 X 万元" / 纯数字
# ============================================================================

AUDIT_CASES = [
    {
        "id": "U3",
        "name": "国电南自 2024 净利润",
        "query": "国电南自2024年净利润",
        "company": "国电南自",
        "year": 2024,
        # U3 的时间锚点: 必须明确是 2024 年 (不能是 2023 同比数据)
        # "报告期" / "本年度" 也算锚点, 但要求 chunk 里另有 2024 的线索
        "year_anchors": ["2024年", "2024 年", "本报告期", "本期", "报告期末"],
        # 核心数据: 净利润的直接数字
        # 注意: "归属于上市公司股东的净利润" 这种长短语不算"数据"只算"主题",
        # 真正的数据是后面跟的数字. 所以用 regex 抓"净利润"+数字的模式
        "data_signals": ["净利润", "归属于上市公司股东", "归母"],
        # 正则: 匹配 "净利润 XXX,XXX,XXX.XX" 这种金融年报标准数字格式
        # 或 "净利润 X.XX 亿元" 这种简写
        # ⚠️ 这个 regex 只是判定"是否含数字形式的净利润", 不要求数字大小对
        "data_regex": re.compile(
            r"(净利润|归属.+?股东|归母.{0,5}净利润)"
            r"[\s\S]{0,20}?"                        # 中间任意字符 (跨列跨行)
            r"[\d,]{4,}\.?\d*"                      # 大数字 (万元/元级)
            r"|"                                     # 或者
            r"净利润[\s\S]{0,10}?\d+\.\d+\s*(亿|万)"  # "净利润 X.XX 亿/万"
        ),
    },
    {
        "id": "U4",
        "name": "招行 2025 不良贷款率",
        "query": "招商银行2025年不良贷款率",
        "company": "招商银行",
        "year": 2025,
        "year_anchors": ["2025年", "2025 年", "报告期末", "本期末", "本报告期"],
        # U4 的核心数据: 直接的百分比数字
        "data_signals": ["不良贷款率"],
        "data_regex": re.compile(
            r"不良贷款率[\s\S]{0,20}?\d+\.\d+\s*%"   # "不良贷款率 X.XX%"
        ),
    },
]


# ============================================================================
# 工具: 判定一个 chunk 是"自足"还是"碎片"
# ============================================================================

def check_chunk_completeness(content: str, case: dict) -> dict:
    """检查 chunk 的 3 要素完整性.

    为什么返回 dict 而不是 bool:
      调试时需要知道"缺哪个要素". 如果只返回 True/False, 看报告时
      无法快速判断"是缺年份还是缺数据".

    参数:
        content: chunk 的完整内容
        case:    案例配置 (含 year_anchors / data_signals / data_regex)

    返回:
        {
            "is_self_contained": bool,    # 是否 3 要素齐全
            "has_year": bool,             # 有时间锚点?
            "has_data": bool,             # 有核心数据?
            "matched_year_kws": [str],    # 命中的年份锚点
            "matched_data_kws": [str],    # 命中的数据关键词
            "regex_match": str|None,      # regex 匹配到的片段 (前 60 字)
        }
    """
    # 时间锚点: 任一关键词命中即算有
    matched_year = [kw for kw in case["year_anchors"] if kw in content]
    has_year = len(matched_year) > 0

    # 核心数据: 两个条件任一满足算有
    #   1. 关键词 + regex 同时命中 (严格)
    #   2. 只 regex 命中 (有时 keyword 在长表格里被切开, 但 regex 能跨行)
    matched_data_kws = [kw for kw in case["data_signals"] if kw in content]
    regex_match = case["data_regex"].search(content)

    # 判定核心数据是否存在: 要求 "关键词命中" AND "regex 命中"
    # 为什么双重条件: 避免 regex 把无关数字识别成净利润
    # 比如 "研发投入 1,234,567.89" 被匹配成净利润就是假阳性
    has_data = bool(matched_data_kws) and regex_match is not None

    is_self_contained = has_year and has_data

    return {
        "is_self_contained": is_self_contained,
        "has_year": has_year,
        "has_data": has_data,
        "matched_year_kws": matched_year,
        "matched_data_kws": matched_data_kws,
        "regex_match": regex_match.group(0)[:60] if regex_match else None,
    }


# ============================================================================
# 工具: 分析一个 case 的 Top-20
# ============================================================================

def analyze_top20(retriever: HybridRetriever, case: dict) -> dict:
    """对一个案例跑 Top-20 并做完整性分析.

    返回:
        {
            "top20": [                          # 20 个 chunk 的详细信息
                {
                    "rank": int,
                    "page": int,
                    "chunk_type": str,
                    "content_preview": str,      # 前 200 字, 给人看
                    "completeness": dict,        # check_chunk_completeness 的返回
                },
                ...
            ],
            "self_contained_ranks": [int],      # 自足 chunk 的排名列表
        }
    """
    results = retriever.search(
        query=case["query"],
        top_k=20,
        filters={"company": case["company"], "year": case["year"]},
    )

    top20_analyzed = []
    self_contained_ranks = []

    for rank, r in enumerate(results, start=1):
        content = r["content"]
        completeness = check_chunk_completeness(content, case)

        top20_analyzed.append({
            "rank": rank,
            "page": r["metadata"].get("page", "?"),
            "chunk_type": r["metadata"].get("chunk_type", "?"),
            "content_preview": content[:200].replace("\n", " "),
            "content_full": content,   # 完整内容, Top-5 打印时会用
            "completeness": completeness,
        })

        if completeness["is_self_contained"]:
            self_contained_ranks.append(rank)

    return {
        "top20": top20_analyzed,
        "self_contained_ranks": self_contained_ranks,
    }


# ============================================================================
# 工具: 判定 L2a 还是 L2b
# ============================================================================

def classify_l2_subtype(analysis: dict) -> dict:
    """根据自足 chunk 的分布判定 L2a / L2b / 需要重审.

    判定规则:
      - Top-5 有自足 chunk → L2a (LLM 看到完整证据还拒答 → Prompt 问题)
      - Top-5 全是碎片 + Top-6~20 有自足 → L2b (排序问题, Reranker 可救)
      - Top-20 全是碎片 → 重审 (可能自足 chunk 定义太严, 或真 chunking 问题)

    为什么用 "Top-5 有自足" 而不是 "Top-1 是自足":
      LLM 是看整个 Top-5 作答, 不是只看 Top-1. 只要 Top-5 内有一个自足 chunk,
      LLM 理论上应该能答对. 答不对就是 LLM 层问题.

    参数:
        analysis: analyze_top20 的返回

    返回:
        {
            "subtype": "L2a" | "L2b" | "需要重审",
            "reason": 说明,
            "self_in_top5": [int],
            "self_in_top6_20": [int],
        }
    """
    ranks = analysis["self_contained_ranks"]
    self_in_top5 = [r for r in ranks if r <= 5]
    self_in_top6_20 = [r for r in ranks if 6 <= r <= 20]

    if self_in_top5:
        return {
            "subtype": "L2a",
            "reason": f"Top-5 有自足 chunk (Rank {self_in_top5}) → LLM 看到完整证据却拒答",
            "self_in_top5": self_in_top5,
            "self_in_top6_20": self_in_top6_20,
        }
    elif self_in_top6_20:
        return {
            "subtype": "L2b",
            "reason": f"Top-5 全碎片, 自足 chunk 在 Top-6~20 (Rank {self_in_top6_20}) → 排序问题",
            "self_in_top5": [],
            "self_in_top6_20": self_in_top6_20,
        }
    else:
        return {
            "subtype": "需要重审",
            "reason": "Top-20 都没有自足 chunk. 可能: (1) 自足定义太严, (2) 真 chunking 问题",
            "self_in_top5": [],
            "self_in_top6_20": [],
        }


# ============================================================================
# 报告打印
# ============================================================================

def print_case_detail(case: dict, analysis: dict, subtype_verdict: dict):
    """打印单个案例的详细分析报告."""
    print("\n" + "=" * 90)
    print(f"案例 {case['id']}: {case['name']}")
    print(f"查询: {case['query']}")
    print(f"自足 chunk 定义:")
    print(f"  年份锚点: {case['year_anchors']}")
    print(f"  核心数据关键词: {case['data_signals']}")
    print(f"  Regex: {case['data_regex'].pattern[:80]}...")
    print("=" * 90)

    # ========== 全 Top-20 完整性概览 ==========
    print(f"\n【Top-20 完整性概览】")
    print(f"{'Rank':<6} {'Page':<6} {'Type':<8} {'Year':<6} {'Data':<6} {'Self-Contained?':<18} Preview")
    print("-" * 110)

    for item in analysis["top20"]:
        c = item["completeness"]
        year_mark = "✅" if c["has_year"] else "❌"
        data_mark = "✅" if c["has_data"] else "❌"
        self_mark = "🟢 YES" if c["is_self_contained"] else "⚪ NO"
        preview = item["content_preview"][:60]
        print(f"{item['rank']:<6} p.{item['page']:<4} {item['chunk_type']:<8} "
              f"{year_mark:<6} {data_mark:<6} {self_mark:<18} {preview}")

    # ========== Top-5 深度展开 ==========
    print(f"\n【Top-5 深度展开 (完整 content)】")
    for item in analysis["top20"][:5]:
        c = item["completeness"]
        print(f"\n--- Rank {item['rank']} (p.{item['page']}, {item['chunk_type']}) ---")
        print(f"  自足? {'🟢 YES' if c['is_self_contained'] else '⚪ NO'}"
              f" (年份: {'有' if c['has_year'] else '无'}, 数据: {'有' if c['has_data'] else '无'})")
        if c["matched_year_kws"]:
            print(f"  命中年份锚点: {c['matched_year_kws']}")
        if c["regex_match"]:
            print(f"  命中数据片段: {c['regex_match']}")
        # 打印完整 content, 让人肉眼判断
        content_display = item["content_full"].replace("\n", " ")
        if len(content_display) > 400:
            content_display = content_display[:400] + "..."
        print(f"  完整内容: {content_display}")

    # ========== 判定 ==========
    print(f"\n【L2 子类型判定】")
    print(f"  Top-5 自足 chunk 排名: {subtype_verdict['self_in_top5'] or '无'}")
    print(f"  Top-6~20 自足 chunk 排名: {subtype_verdict['self_in_top6_20'] or '无'}")
    print(f"  子类型: {subtype_verdict['subtype']}")
    print(f"  理由: {subtype_verdict['reason']}")

    # 给出针对性修复建议
    print(f"\n【Day 10 修复方向】")
    if subtype_verdict["subtype"] == "L2a":
        print(f"  → Prompt 工程 (低成本)")
        print(f"    1. 加 CoT 引导: '先识别每个候选的时间锚点, 再抽数据'")
        print(f"    2. metadata 注入: Prompt 拼入 '本次查询的公司=X, 年份=Y'")
        print(f"    3. 放松拒答规则: '部分证据可用时给出可能答案并标注'")
    elif subtype_verdict["subtype"] == "L2b":
        print(f"  → 排序优化 (中成本)")
        print(f"    1. 开 Reranker 条件开启 (Router 判这类查询开)")
        print(f"    2. 调 RRF k 让精信号更突出")
        print(f"    3. top_k 从 5 扩到 7-10")
    else:
        print(f"  → 需要重审自足定义, 或查是否真 L0 问题")


def print_summary(results: list[dict]):
    """打印汇总."""
    print("\n\n" + "=" * 90)
    print("📊 L2 子类型汇总")
    print("=" * 90)
    print(f"\n{'ID':<4} {'案例':<22} {'自足@Top-5':<16} {'自足@Top6-20':<16} {'子类型':<12}")
    print("-" * 90)
    for r in results:
        case = r["case"]
        v = r["subtype"]
        t5 = str(v["self_in_top5"]) if v["self_in_top5"] else "无"
        t620 = str(v["self_in_top6_20"]) if v["self_in_top6_20"] else "无"
        print(f"{case['id']:<4} {case['name']:<22} {t5:<16} {t620:<16} {v['subtype']:<12}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 90)
    print("Day 8 — L2 子类型细化 (脚本 25b: U3/U4 Top-5 证据完整性)")
    print("=" * 90)

    # ---- 初始化 (和脚本 25 一样, 代码抄过来) ----
    print("\n[初始化] 加载 BGE + 构建 BM25...")
    client = get_client()
    embedder = BGEEmbedder()

    all_points = []
    offset = None
    while True:
        response, offset = client.scroll(
            collection_name="financial_reports", limit=500,
            offset=offset, with_payload=True, with_vectors=False,
        )
        all_points.extend(response)
        if offset is None:
            break

    chunks = []
    for p in all_points:
        payload = dict(p.payload or {})
        content = payload.pop("content", "")
        chunk_type = payload.pop("chunk_type", "text")
        chunks.append(Chunk(content=content, chunk_type=chunk_type, metadata=payload))

    bm25_store = BM25Store()
    bm25_store.build(chunks)

    retriever = HybridRetriever(
        qdrant_client=client, bm25_store=bm25_store, embedder=embedder,
    )
    print(f"[初始化完成] chunks: {len(chunks)}")

    # ---- 跑 U3 U4 ----
    results = []
    for case in AUDIT_CASES:
        print(f"\n\n{'#' * 90}")
        print(f"# 分析 {case['id']}: {case['name']}")
        print(f"{'#' * 90}")

        analysis = analyze_top20(retriever, case)
        subtype_verdict = classify_l2_subtype(analysis)
        print_case_detail(case, analysis, subtype_verdict)

        results.append({
            "case": case,
            "analysis": analysis,
            "subtype": subtype_verdict,
        })

    print_summary(results)


if __name__ == "__main__":
    main()