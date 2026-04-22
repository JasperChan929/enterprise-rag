# scripts/26_audit_table_fragment.py
"""Day 8 - 脚本 26 (重写 v2, Day 8 收工时改名 v3): 量化 TD-8-1 表格切断 chunk

===============================================================================
命名说明 (Day 8 收工时的修正)
===============================================================================
本脚本最初叫 26_audit_chunking_fragment.py, 内部用"碎片 chunk"指代量化目标.
Day 8 收工时发现"碎片 chunk"这个词在中文里语义过于宽泛, 实际可指至少 5 种
chunker 切断形态:
  1. 表格切断 (本脚本量化的)
  2. 指代切断 ("本公司"无公司名)
  3. 时间锚点切断 ("报告期末"无年份)
  4. 口径说明切断 (数字 chunk 和口径定义 chunk 分离)
  5. 叙事切断 ("因此..." 但原因在前一 chunk)

为避免命名混淆 + 后续其他切断形态登记时撞名:
  文件名: 26_audit_chunking_fragment.py → 26_audit_table_fragment.py
  概念名: "碎片 chunk" → "表格切断 chunk"

其他形态登记为 TD-8-4, 由 Day 9 量化.

===============================================================================
🛑 自我攻击环节 (发脚本前必做, 给调用者看的质量保证)
===============================================================================

目标: 抓"利润表/收入表/资产负债表等财务数据表的表头被 chunker 切断,
     数字部分在下一个 chunk"的情况. U3 国电南自 p.138 是教科书样例.

【考虑过的假阳性场景】

FP-1: 叙述文字谈财务趋势, 天然无数字
  例: "营业收入稳健增长, 净利润保持双位数增速"
  → 本脚本能过滤 (要求末尾含"单位:"/"币种:"/列名列表特征)

FP-2: 财务名词解释/定义
  例: "净利润: 指企业当期利润总额减除所得税费用后的金额"
  → 本脚本能过滤 (无表头特征)

FP-3: 章节标题或目录
  例: "第三节 经营情况讨论与分析"
  → 本脚本能过滤 (无表头特征)

【考虑过的假阴性场景】

FN-1: 表头没有"单位:"/"币种:"显式声明, 只有列名
  例: "本公司 2025 年 2024 年同比 金额 金额 变化率"
  → 本脚本会漏抓, 接受. Day 9 视数据再决定是否补规则.

FN-2: 表头 + 数字在同一 chunk, 但数字被截断 (chunk 中间被切)
  → 本脚本会漏抓 (我们的规则看"完全无数字"), 接受.

【预期结果范围】
  全库 6881 chunks 里, 表格切断 chunk 数应该 < 100 (约 1-2%).
  如果 > 500 → 规则太松, 需要重审
  如果 < 20  → 规则太严或真的只是 U3 个案

【实际结果】
  跑完后真碎片 = 25 个 (4.5% of 含财务关键词 chunk), 在预期范围内.

===============================================================================
判定规则
===============================================================================
一个 chunk 是 "表格切断 chunk" 必须同时满足:
  1. 含财务关键词 (净利润/营业收入/营业成本/毛利率/...)
  2. 不含任何"像数字的东西" (\d{4,} / \d+\.\d+ / 千分位都没)
  3. 具有表头特征之一:
     a. chunk 里含 "单位:" (如 "单位: 元" / "单位: 万元")
     b. chunk 里含 "币种:" (如 "币种: 人民币")
     c. chunk 末尾 80 字里有连续列名 (2 个以上财务名词连排, 空格分隔)

===============================================================================
设计上的节制
===============================================================================
- 输出 3 个层次: 全局 / 公司分布 / 样例
- 不尝试定位配对 chunk (那个复杂度留给 Day 10)
"""
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv(override=True)

import re
from collections import defaultdict

from src.retrievers.qdrant_store import get_client


# ============================================================================
# 规则定义
# ============================================================================

# 条件 1 的财务关键词
FINANCIAL_KEYWORDS = [
    "净利润",
    "营业收入",
    "营业成本",
    "营业利润",
    "毛利率",
    "净资产",
    "总资产",
    "归属于上市公司股东",
    "归母",
    "基本每股收益",
    "每股净资产",
    "净资产收益率",
]

# 条件 2 的数字 pattern
# \d{4,}: 4+ 位连续数字 (覆盖亿/万级)
# \d+\.\d+: 小数 (百分比/比率)
# \d{1,3}(?:,\d{3})+: 千分位 (如 123,456,789)
NUMBER_PATTERN = re.compile(
    r"\d{4,}|\d+\.\d+|\d{1,3}(?:,\d{3})+"
)

# 条件 3a 和 3b 的表头特征关键词
# 这些词几乎只在表格上下文出现, 是"表头刚结束, 即将开始数据行"的信号
UNIT_MARKERS = [
    "单位:元",
    "单位:万元",
    "单位:亿元",
    "单位:千元",
    "单位: 元",
    "单位: 万元",
    "单位: 亿元",
    "单位:人民币元",
    "单位:人民币万元",
    "币种:人民币",
    "币种:美元",
    "币种: 人民币",
    "币种: 美元",
    # 中文全角冒号版本 (很多 PDF 抽出来是全角)
    "单位:元",
    "单位:万元",
    "单位:亿元",
    "单位:千元",
    "单位:人民币元",
    "单位:人民币万元",
    "币种:人民币",
    "币种:美元",
]


def has_financial_keyword(content: str) -> list[str]:
    """条件 1 检测."""
    return [kw for kw in FINANCIAL_KEYWORDS if kw in content]


def has_number(content: str) -> bool:
    """条件 2 检测."""
    return bool(NUMBER_PATTERN.search(content))


def has_unit_marker(content: str) -> list[str]:
    """条件 3a/3b 检测: 是否含"单位:"或"币种:"标识."""
    return [m for m in UNIT_MARKERS if m in content]


def has_column_name_list(content: str, tail_chars: int = 80) -> bool:
    """条件 3c 检测: chunk 末尾是否是连续列名列表.

    策略:
      取 chunk 最后 80 字, 看里面有没有 2 个以上不同财务关键词.

    为什么是"末尾": 表头通常放在 chunk 最后, 紧跟着下一个 chunk 的数字.
    为什么 80 字: 太长会误判 (叙述段里也可能多次出现关键词),
                  太短又漏掉宽表头. 80 字是经验值.

    ⚠️ 这是启发式, 不保证 100% 精确.
    """
    tail = content[-tail_chars:] if len(content) > tail_chars else content
    matched = set(kw for kw in FINANCIAL_KEYWORDS if kw in tail)
    # 要求至少 2 个不同关键词同时出现在末尾
    return len(matched) >= 2


def is_table_fragment_chunk(content: str) -> dict:
    """完整判定: 是不是"表格切断 chunk".

    返回:
        {
            "is_table_fragment": bool,
            "kws":               [str, ...],  # 命中的财务关键词
            "why":               str,         # 判定理由 (供调试/样例展示)
        }
    """
    kws = has_financial_keyword(content)

    # 条件 1 不满足 → 不是表格切断
    if not kws:
        return {"is_table_fragment": False, "kws": [], "why": "无财务关键词"}

    # 条件 2 不满足 → 有数字, 不是表格切断
    if has_number(content):
        return {"is_table_fragment": False, "kws": kws, "why": "含数字, 非表格切断"}

    # 条件 3 判定表头特征
    unit_markers = has_unit_marker(content)
    col_list = has_column_name_list(content)

    if unit_markers:
        return {
            "is_table_fragment": True,
            "kws": kws,
            "why": f"表头特征: 含 {unit_markers}",
        }
    if col_list:
        return {
            "is_table_fragment": True,
            "kws": kws,
            "why": "表头特征: 末尾列名列表",
        }

    # 条件 1 + 条件 2 满足但条件 3 不满足 → 可能是叙述文字, 不算
    return {
        "is_table_fragment": False,
        "kws": kws,
        "why": "有关键词无数字, 但无表头特征 (可能是叙述文字, 排除)",
    }


# ============================================================================
# 扫描 + 统计
# ============================================================================

def scan_and_analyze(client):
    """scroll 全库, 边扫边分类."""
    total = 0
    with_financial_kw = 0
    with_kw_no_num = 0        # 满足条件 1 + 条件 2 的 (潜在表格切断池)
    table_fragments = 0       # 三条件全满足的真表格切断

    by_company = defaultdict(lambda: {
        "total": 0, "with_kw": 0, "with_kw_no_num": 0, "table_fragment": 0,
    })

    table_fragment_samples = []

    offset = None
    while True:
        response, offset = client.scroll(
            collection_name="financial_reports",
            limit=500,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for p in response:
            payload = p.payload or {}
            content = payload.get("content", "")
            company = payload.get("company", "?")
            year = payload.get("year", "?")
            page = payload.get("page", "?")
            chunk_type = payload.get("chunk_type", "?")

            total += 1
            key = f"{company}_{year}"
            by_company[key]["total"] += 1

            kws = has_financial_keyword(content)
            if kws:
                with_financial_kw += 1
                by_company[key]["with_kw"] += 1

                # 潜在表格切断: 满足条件 1 + 条件 2
                if not has_number(content):
                    with_kw_no_num += 1
                    by_company[key]["with_kw_no_num"] += 1

            result = is_table_fragment_chunk(content)
            if result["is_table_fragment"]:
                table_fragments += 1
                by_company[key]["table_fragment"] += 1

                if len(table_fragment_samples) < 10:  # 存 10 个样例
                    table_fragment_samples.append({
                        "company": company,
                        "year": year,
                        "page": page,
                        "chunk_type": chunk_type,
                        "matched_kws": result["kws"],
                        "why": result["why"],
                        "content": content[:250].replace("\n", " "),
                    })

        if offset is None:
            break

    return {
        "total": total,
        "with_financial_kw": with_financial_kw,
        "with_kw_no_num": with_kw_no_num,
        "table_fragments": table_fragments,
        "by_company": dict(by_company),
        "samples": table_fragment_samples,
    }


# ============================================================================
# 报告打印
# ============================================================================

def print_report(result: dict):
    total = result["total"]
    with_kw = result["with_financial_kw"]
    potential = result["with_kw_no_num"]
    frag = result["table_fragments"]

    print("=" * 80)
    print("Day 8 — TD-8-1 表格切断 chunk 全库量化 (脚本 26 v3)")
    print("=" * 80)

    print(f"\n【全局统计 - 漏斗视图】")
    print(f"  Level 1: 总 chunks                         {total}")
    print(f"  Level 2: 含财务关键词 (条件 1)             {with_kw} ({with_kw/total*100:.1f}%)")
    print(f"  Level 3: 关键词 + 无数字 (条件 1+2)        {potential}")
    print(f"           └ 这是'潜在表格切断池', 含叙述文字")
    print(f"  Level 4: + 表头特征 (条件 1+2+3) = 真表格切断 {frag}")
    print(f"           └ 占总 chunks: {frag/total*100:.2f}%")
    if with_kw:
        print(f"           └ 占含关键词 chunk: {frag/with_kw*100:.2f}%")
    print(f"           └ 从潜在池过滤掉 {potential-frag} 个叙述文字假阳性")

    print(f"\n【按公司_年份分布】 (按表格切断数降序)")
    print(f"{'公司_年份':<24} {'总数':>8} {'含关键词':>10} {'潜在':>8} {'表格切断':>10} {'切断率':>10}")
    print("-" * 80)

    sorted_items = sorted(
        result["by_company"].items(),
        key=lambda kv: -kv[1]["table_fragment"],
    )
    for key, stats in sorted_items:
        t = stats["total"]
        w = stats["with_kw"]
        p = stats["with_kw_no_num"]
        f = stats["table_fragment"]
        rate = f / w * 100 if w > 0 else 0
        print(f"{key:<24} {t:>8} {w:>10} {p:>8} {f:>10} {rate:>9.1f}%")

    print(f"\n【表格切断样例】 (最多 10 个)")
    for i, s in enumerate(result["samples"], start=1):
        print(f"\n  样例 {i}: {s['company']} {s['year']} p.{s['page']} [{s['chunk_type']}]")
        print(f"    判定理由: {s['why']}")
        print(f"    命中关键词: {s['matched_kws']}")
        print(f"    内容: {s['content']}")

    print(f"\n【判定 + Day 10 建议】")
    frag_rate_in_kw = frag / with_kw * 100 if with_kw else 0

    if frag_rate_in_kw >= 15:
        sev = "🔥 严重 (全局性结构问题)"
        rec = ("Day 10 必修. 首选 Prompt metadata 注入 (低成本), "
               "Day 11+ 考虑表格感知 chunker (中成本)")
    elif frag_rate_in_kw >= 5:
        sev = "⚠️  中度 (影响特定公司或查询类型)"
        rec = "Day 10 metadata 注入可缓解. 视评估集表现决定是否深修"
    else:
        sev = "✅ 轻微 (U3 级个案, 非全局问题)"
        rec = ("TD-8-1 降级为 P3 个案. 不做专项修复, "
               "Day 10 Prompt 工程顺带处理即可")

    print(f"  含财务关键词的 chunk 中, 表格切断比例 = {frag_rate_in_kw:.1f}% → {sev}")
    print(f"  建议: {rec}")

    print(f"\n【⚠️ 重要范围说明】")
    print(f"  本脚本只量化了 chunker 切断的 1/5 形态 (表格切断).")
    print(f"  其他 4 种切断形态 (指代/时间锚点/口径说明/叙事) 登记为 TD-8-4,")
    print(f"  由 Day 9 量化. TD-8-1 的 P3 判定仅基于本脚本覆盖的形态, ")
    print(f"  全量切断形态量化后 TD-8-1 优先级可能需要重新评估.")


def main():
    print("[启动] 连接 Qdrant 并 scroll 全库...")
    client = get_client()
    result = scan_and_analyze(client)
    print_report(result)


if __name__ == "__main__":
    main()