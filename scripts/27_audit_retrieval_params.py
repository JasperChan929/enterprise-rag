"""scripts/27_audit_retrieval_params.py

Day 9 Task 1: 检索层 3 个核心参数的扫描审计
(top_k / RRF k / recall_multiplier)

====================================================================
设计决策与自我攻击清单 (Day 8 脚本 26 v1 教训的延续)
====================================================================

本脚本的产出会直接影响 Day 10 开工的第二件事(是否调参数).
所以设计阶段必须先做完整的自我攻击, 防止"跑完了才发现结论不可信".

--- 核心概念 (按 Day 8 日记终义) ---
金页 (Golden Page):
    (公司, 页号) 二元组, 表示"按理应该被检索到的 PDF 页".
    不是"查询答案", 是"含答案的原文位置".
    本脚本金页通过 "PDF grep + Top-50 反向找漏 + 人工复核" 方式生成,
    存在漏报/误报误差, 是"工程近似"不是"金标准".
    命中率绝对值不具外推价值, 只能用于"参数间相对差异"比较.

金页命中率:
    Top-K 召回里出现的金页数 / 金页总数
    U1 比亚迪汽车毛利率是金页空集(数据真不存在),
    必须从命中率均值里剔除, 只算延迟.

--- 3 个假阳性 (误把不最优当最优) ---

FP1: top_k=5 看起来最优, 可能只是 6 样本恰好都在 Top-5 内命中
缓解: 报告强制带 "top_k=3 → top_k=5 增量" 列,
      增量 < 10% 判定要降级为 "top_k=3 可能已够用, 扩大样本验证"

FP2: RRF k=60 最优可能因为 hybrid.py 有 hardcode 耦合
已排除: 读了 hybrid.py:rrf_fuse (第 87-94 行), 公式是纯 1/(k+rank),
        没有归一化/加权/裁剪, 3 个参数都是显式构造参数.

FP3: recall_mult=4 时命中率 95% 可能全部来自 Dense 单路
缓解: 每次检索顺手导出 Dense/BM25 单路 Top-20,
      报告里看"金页是哪一路救的", 避免幻觉

--- 2 个假阴性 (误把最优当不最优) ---

FN1: U1 金页空集拉低整体均值
缓解: 命中率统计明确"5 个有效查询 (U1 排除)", U1 只算延迟

FN2: 某参数救了死穴样本但被均值淹没
缓解: 报告必带"逐查询展开表", 单样本显著异常 (Δ > 30%) 单独标注

--- 预估结果范围 (跑完必须和实际对比) ---

top_k:       预估 5 维持 或 上调 7 (增量 <10% → 维持 5)
RRF k:       预估 60 维持 (论文推荐值)
recall_mult: 预估 4 维持 或 升到 6 (U5 紫金死穴值得试 6)

如果实际结果和预估差距巨大 (如某参数当前值最差):
    警惕是脚本写错了, 先回头核 hybrid.py 调用 / 金页标注 / 指标计算,
    不要立刻信结果. 这是 Day 8 脚本 26 v1 教训写成的硬规则.

====================================================================
用法
====================================================================

两阶段分开跑 (Day 9 Task 1 Q1=C** 方案):

Stage 1 - 金页候选生成 (~30 秒, 全自动):
    uv run python scripts/27_audit_retrieval_params.py --stage=draft

    产出: docs/day09-gold-pages.draft.json
    操作: 打开该文件, 对每个 candidate 把 confirmed: null 改成 true/false,
          U1 这种无答案查询把 is_empty_gold 改成 true,
          改完后另存为 day09-gold-pages.final.json

Stage 2 - 参数扫描 + 报告 (~6 分钟):
    uv run python scripts/27_audit_retrieval_params.py --stage=scan

    产出: docs/day09-param-audit.txt (主报告)
          docs/day09-3way-cache.json (Task 2 复用)

====================================================================
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

import pdfplumber

from src.chunking.recursive import chunk_documents
from src.embeddings.bge import BGEEmbedder
from src.loaders.pdf_loader import load_pdf
from src.retrievers.bm25_store import BM25Store
from src.retrievers.hybrid import HybridRetriever, rrf_fuse
from src.retrievers.qdrant_store import get_client, search_similar


# ============================================================
# 常量
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DOCS_DIR = PROJECT_ROOT / "docs"

COLLECTION = "financial_reports"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

# 缓存文件 (chunk_size/overlap 写进文件名防呆)
CHUNKS_CACHE = DOCS_DIR / f"day09-chunks_size{CHUNK_SIZE}_overlap{CHUNK_OVERLAP}.pkl"

# 金页相关
DRAFT_JSON = DOCS_DIR / "day09-gold-pages.draft.json"
FINAL_JSON = DOCS_DIR / "day09-gold-pages.final.json"

# 输出
REPORT_TXT = DOCS_DIR / "day09-param-audit.txt"
THREEWAY_CACHE = DOCS_DIR / "day09-3way-cache.json"


# ============================================================
# Phase 0: 6 查询定义
# ============================================================

# 每个查询:
#   id: U1-U6, 和 Day 8 笔记对齐
#   text: 发给 RAG 的查询文本
#   company_filter: 过滤条件. None 表示不过滤 (跨公司查询)
#                   大多数查询应该过滤, 否则金页定义会被其他公司污染
#   keywords: PDF grep 用的关键词列表 (OR 匹配)
#             这些关键词是"找候选金页"用的, 不要求精确, 粗召即可
#             人工复核时会把误报候选标 false
#   pdf_name: 对应的 PDF 文件名 (用于 grep). None 表示跨公司查询不做 grep
#
# 关键词设计原则 (按偏好 3):
#   - 包含"指标名" + "已知数字片段" (如果 Day 8 审计里提过)
#   - 包含"同义说法" (比如"营业收入"也可能写作"主营业务收入"/"总收入")
#   - 不用太泛的词 (比如单纯"毛利" 会命中几十页目录/制度)

QUERIES = [
    {
        "id": "U1",
        # ==========================================================
        # U1 知识点溯源 (偏好 7)
        # ==========================================================
        # 完整身份:  比亚迪 2025 年汽车业务毛利率查询, Day 8 定案 L4 (数据真缺失)
        # 历史出处:
        #   - Day 5: 首次翻车, LLM 拒答
        #   - Day 7 TD-7-x: 判 L4 "年报未披露汽车业务专项成本"
        #   - Day 8 脚本 25 审计: 3 步法 PDF grep + Qdrant scroll 双阴性 →
        #     L4 定案 (docs/day08-audit-25.txt)
        #   - Day 8 笔记 4.1 节: 标注为"Day 7 唯一判对, 但属于幸存者偏差"
        # 当前用途:
        #   - Day 9 Task 1 用作"空集金页"测试, 验证脚本对金页空集的剔除处理 (FN1)
        #   - 命中率统计必须排除它, 只进延迟统计
        # 当前状态: L4 已定案, 不可修 (年报只披露汽车业务收入, 无单独成本口径)
        # ==========================================================
        "text": "比亚迪汽车业务2025年的毛利率是多少",
        "company_filter": "比亚迪",
        "keywords": ["汽车业务毛利率", "汽车毛利率", "汽车业务", "毛利率"],
        "pdf_name": "002594_比亚迪_2025年年度报告.pdf",
        # 预期金页: 空集 (is_empty_gold=true)
    },
    {
        "id": "U2",
        # ==========================================================
        # U2 知识点溯源 (偏好 7)
        # ==========================================================
        # 完整身份:  宁德时代 2025 磷酸铁锂电池产品情况查询, Day 8 从 L4 推翻为 L2
        # 历史出处:
        #   - Day 5-6: LLM 自述 "未做磷酸铁锂专项披露", 判 L4 数据边界
        #   - Day 8 脚本 25 审计: PDF grep 命中 3 页 (p.14/17/21), Qdrant scroll
        #     3 chunks 全召回, Top-20 Rank 5/16 命中 → 数据完整 + 检索 OK →
        #     推翻 L4, 判 L2 (LLM 作答问题)
        #   - Day 8 笔记 4.2 节: 真实根因 = LLM 对"产品情况"这种抽象词对应能力弱,
        #     年报有"磷酸铁锂电池"+具体产品名 (神行/Pro/钠新) 但没有叫
        #     "磷酸铁锂产品情况"的章节, LLM 找不到字面对应就拒答.
        #     归为 L2 的第 3 种子类型 L2c "语义粒度错配" (Day 8 未展开细分)
        # 当前用途:
        #   - Day 9 Task 1 验证不同参数下语义粒度错配类查询的表现稳定性
        #   - 预期金页在 Top-5 范围波动不大 (Rank 5 贴近边界, 参数敏感)
        # 当前状态: L2c, 修法 = Day 10 Multi-Query 改写成具体产品型号
        # ==========================================================
        "text": "宁德时代磷酸铁锂产品2025年的情况",
        "company_filter": "宁德时代",
        "keywords": ["磷酸铁锂", "磷酸铁", "LFP"],
        "pdf_name": "300750_宁德时代_2025年年度报告.pdf",
        # 预期金页: (宁德时代, 14), (宁德时代, 17), (宁德时代, 21) - 来自 Day 8 审计
    },
    {
        "id": "U3",
        # ==========================================================
        # U3 知识点溯源 (偏好 7)
        # ==========================================================
        # 完整身份:  国电南自 2024 年净利润查询, Day 8 从 L4 推翻为 L2b (排序问题)
        # 历史出处:
        #   - Day 5: LLM 自述 "参考资料未提供 2024 年净利润", 判 L4 数据链断裂
        #     (当时推测 loader 丢了利润表)
        #   - Day 8 脚本 25 审计: PDF grep 21 页命中, Qdrant 27 chunks 命中,
        #     Top-20 Rank 1/2/9/15/18 多个命中 → 数据完整 + 检索 OK →
        #     推翻 L4 (docs/day08-audit-25.txt)
        #   - Day 8 脚本 25b: Top-5 全碎片, 自足 chunk 在 Rank 15/18 → 判 L2b
        #   - Day 8 笔记 4.3 节: 真实根因 = RRF 排序把真金页压在 Rank 15/18,
        #     Top-5 被子公司/附注类 chunk 占满. p.138 "母公司利润表"表头在 Rank 2,
        #     但真正数字在 p.139, p.139 未进 Top-20 (TD-8-1 表格切断样例)
        # 当前用途:
        #   - Day 9 Task 1 关键观察样本: top_k 从 5 扩到 10/15 能不能让 Rank 15
        #     的自足 chunk 进视野
        #   - RRF k 变化可能改变 Top-5 名单, 若能把 Rank 15 的自足 chunk 推上来
        #     则是 L2b 可调参缓解的证据
        # 当前状态: L2b, 修法 = Reranker 条件开启 或 扩 top_k, Day 10 验证
        # ==========================================================
        "text": "国电南自2024年度净利润是多少",
        "company_filter": "国电南自",
        "keywords": ["净利润", "归属于母公司股东的净利润", "净利", "归属于上市公司股东的净利润"],
        "pdf_name": "600268_国电南自_2024年年度报告.pdf",
        # 预期金页: 含净利润数字的利润表页 (p.138/139 附近), 具体待 draft 阶段定
    },
    {
        "id": "U4",
        # ==========================================================
        # U4 知识点溯源 (偏好 7) - 最有戏剧性的一例
        # ==========================================================
        # 完整身份:  招商银行 2025 不良贷款率查询, Day 8 从 L0 推翻为 L2a (多口径错位)
        # 历史出处 (推翻链条):
        #   - Day 4-5: Naive RAG 拒答, 判表格数据缺失 (TD-4-1, 后推翻)
        #   - Day 5: Hybrid 能答出 0.93% 本公司 + 0.94% 集团, 证明数据在库
        #   - Day 7: 发现 p.45 chunk 只写"报告期末 0.93%"没写"2025 年",
        #     推理 chunker 切走年份锚点, 判 L0 → 登记 TD-7-1
        #   - Day 8 脚本 25 审计: PDF grep 11 页命中, Qdrant 14 chunks 命中,
        #     Top-20 Rank 1/3/5 多命中 → 排除 L0/L4 (docs/day08-audit-25.txt)
        #   - Day 8 脚本 25b: Top-5 有 3 个自足 chunk → 判 L2a
        #   - Day 8 笔记 4.4 节: **真正根因 = Top-5 有 3 个不同口径数字
        #     (0.93% 本公司 / 0.94% 集团 / 1.52% 消费信贷), LLM 不知道选哪个**,
        #     不是 chunker 切了年份.
        #     Top-2 (p.5 董事长致辞) 明确写"2025 年度报告" → LLM 能推断时间
        #   - Day 8 技术债重估:
        #     - TD-7-1 归档 (归因错误, 不删保留认知过程)
        #     - TD-7-3 (Reranker 不判口径) 升级为 TD-8-2 (多口径错位) P1,
        #       Day 10 Prompt 工程主战场
        # 当前用途:
        #   - Day 9 Task 1 验证: 不同 RRF k 下 Top-5 会不会改变 3 口径的排序分布
        #   - 如果某 RRF k 能把单口径 (如 0.93% 本公司) 稳定推到 Top-1, 可能缓解
        #     多口径错位 (但主解药是 Day 10 Prompt 工程, 调参只是辅助)
        # 当前状态: L2a 已定案, TD-8-2 待 Day 10 修
        # ==========================================================
        "text": "招商银行2025年不良贷款率",
        "company_filter": "招商银行",
        "keywords": ["不良贷款率", "0.93%", "0.94%", "1.52%"],
        "pdf_name": "600036_招商银行_2025年年度报告.pdf",
        # 预期金页: (招商银行, 45) 本公司, (招商银行, 31) 集团, (招商银行, 50) 消费信贷 - Day 8 审计确认
    },
    {
        "id": "U5",
        # Day 6-7 原案: 紫金 p.59 (海外业务风险披露页) 被 Full 模式挤出 Top-3
        # 金页内容含 "地缘政治风险 / 海外项目不确定性 / 应对措施"
        # Day 9 Task 1 引入这个查询是为了验证不同 top_k/RRF k/recall_mult 下能否捞回 p.59
        "text": "紫金矿业海外业务的营收和风险",
        "company_filter": "紫金矿业",
        "keywords": ["地缘政治", "海外项目", "海外业务", "海外"],
        "pdf_name": "601899_紫金矿业_2025年年度报告.pdf",
        # 预期金页: (紫金矿业, 59)
    },
    {
        "id": "U6",
        # Day 9 新增, 作为"正常成功案例对照基线"
        # ------------------------------------------------------------------
        # 为什么加对照样本: 5 个查询里 1 个空集 + 4 个困难样本,
        # 没有基准锚点, 参数变化信号会被噪声淹没. 需要一个"正常 100%"的
        # 参照来判断"参数是否改坏了常规情况".
        #
        # 为什么选宁德营收不选茅台营收 (Day 9 决策 D4):
        #   - 维度 1 流程严谨: 忠实 Day 8 规划的"宁德产品情况" 相关领域 → 选宁德
        #   - 维度 2 实验严谨: 两者都是已知 3 路都能命中的成功案例 → 持平
        #   - 维度 3 分布均衡: 茅台能多覆盖 1 家公司 (白酒) → 选茅台
        #   - 维度 4 数据版本一致性: 宁德是 2025 和其他查询同版本, 茅台是 2023 → 选宁德
        #
        # 最终选宁德: 维度 1 + 维度 4 > 维度 3. 用户选 A (维持宁德) 2026-04-21
        #
        # 已知局限: 宁德在 6 查询里占 2 个 (U2 + U6), 单公司权重 33% 偏高.
        # 如果 Day 11 大样本评估时发现这造成偏倚, 届时可调整.
        # ------------------------------------------------------------------
        "text": "宁德时代2025年营业收入是多少",
        "company_filter": "宁德时代",
        "keywords": ["营业收入", "主营业务收入", "总收入"],
        "pdf_name": "300750_宁德时代_2025年年度报告.pdf",
    },
]


# ============================================================
# Phase 1: 金页候选生成 (Stage = draft)
# ============================================================

def grep_pdf_for_candidates(query_cfg: dict) -> list[dict]:
    """对一个查询的 PDF 做 grep, 输出候选页列表.

    复用 Day 8 脚本 25 的 grep_pdf 逻辑, 改造输出格式.

    为什么这么做 (按偏好 1):
        金页是"含答案的 PDF 页". grep 是"找候选"的最快方式,
        再配合 Top-50 反向找漏 + 人工复核, 得到近似 ground truth.

    边界处理:
        - PDF 不存在 → 返回空列表 (不报错, 让调用方知道这个查询没候选)
        - 某页无文本 (扫描图片页) → 跳过
        - 多关键词命中同一页 → 只记录一次, 关键词列表聚合

    Returns:
        [
            {
                "company": "招商银行",
                "page": 45,
                "source": "grep",
                "matched_keywords": ["不良贷款率", "0.93%"],
                "excerpt": "...前后 60 字上下文...",
                "confirmed": None
            },
            ...
        ]
    """
    pdf_name = query_cfg.get("pdf_name")
    if not pdf_name:
        return []

    pdf_path = DATA_RAW_DIR / pdf_name
    if not pdf_path.exists():
        print(f"   ⚠️  PDF 不存在: {pdf_path}")
        return []

    # 从文件名解析公司名 (跟 pdf_loader.parse_filename 同规则)
    # 文件名格式: {stock_code}_{company}_{year}年年度报告.pdf
    import re
    match = re.match(r"\d{6}_(.+?)_\d{4}年年度报告", pdf_path.stem)
    company = match.group(1) if match else "未知"

    keywords = query_cfg["keywords"]
    candidates = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if not text.strip():
                continue

            matched = [kw for kw in keywords if kw in text]
            if not matched:
                continue

            # 抽第一个命中关键词的上下文 (前后 60 字)
            # 60 比 Day 8 的 40 宽一点, 因为这里上下文是给人看的, 多点信息方便判断
            first_kw = matched[0]
            idx = text.find(first_kw)
            start = max(0, idx - 60)
            end = min(len(text), idx + len(first_kw) + 60)
            excerpt = text[start:end].replace("\n", " ").strip()

            candidates.append({
                "company": company,
                "page": page_num,
                "source": "grep",
                "matched_keywords": matched,
                "excerpt": excerpt,
                "confirmed": None,  # 三态: null 待判 / true 是金页 / false 不是
            })

    return candidates


def topk50_back_check(
    query_cfg: dict,
    retriever: HybridRetriever,
    existing_candidates: list[dict],
) -> list[dict]:
    """Top-50 反向找漏: 用当前 Hybrid 跑一次 Top-50, 看有没有 grep 没标的页.

    意图:
        grep 的主要失败模式是漏报 (金页里没有关键词字面).
        借助检索系统, 可能能找到"grep 没标但检索系统认为相关"的页.
        注意这是"启发式补强", 不保证找齐所有漏报.

    实现:
        1. 用查询文本跑 top_k=50 的 Hybrid 检索 (带 company filter)
        2. 过滤掉已经在 existing_candidates 里的 (company, page)
        3. 剩下的作为"top50_back"来源的候选, confirmed=null

    为什么 top_k=50 不是 100: 50 足够宽, 再大人工复核成本翻倍, 边际收益低.

    Returns: 新增的候选列表 (和 existing_candidates 合并时用 (company, page) 去重)
    """
    filters = None
    if query_cfg.get("company_filter"):
        filters = {"company": query_cfg["company_filter"]}

    # 用默认参数跑一次 Top-50
    # top_k=50, RRF k=60 默认, recall_mult=4 默认
    # 注: 这里用的是当前 retriever 的配置, RRF k/recall_mult 是当前默认值
    #     这对"找漏候选"没问题, 因为不同参数下的 Top-50 重合度很高
    results = retriever.search(query_cfg["text"], top_k=50, filters=filters)

    # 已知 (company, page) 集合
    known = {(c["company"], c["page"]) for c in existing_candidates}

    new_candidates = []
    for r in results:
        meta = r["metadata"]
        company = meta.get("company", "未知")
        page = meta.get("page", -1)

        if (company, page) in known:
            continue

        # excerpt 从 chunk content 里取前 120 字
        content = r.get("content", "")
        excerpt = content[:120].replace("\n", " ").strip()
        if len(content) > 120:
            excerpt += "..."

        new_candidates.append({
            "company": company,
            "page": page,
            "source": "top50_back",
            "matched_keywords": [],  # Top-50 找的不是关键词匹配, 是语义
            "excerpt": excerpt,
            "rrf_score": round(r.get("rrf_score", 0), 4),
            "confirmed": None,
        })
        known.add((company, page))

    return new_candidates


def run_stage_draft() -> None:
    """Stage 1: 生成金页候选 JSON. 全自动, 不需要人工介入."""
    print("=" * 60)
    print("Stage 1: 金页候选生成 (draft)")
    print("=" * 60)

    # --- Setup: 需要 retriever 做 Top-50 反向找漏 ---
    retriever, _, _ = setup_retriever(rrf_k=60, recall_mult=4)

    # --- 逐查询生成候选 ---
    result = {
        "_instructions": (
            "对每个 candidate, 把 confirmed 改成 true(是金页) 或 false(不是). "
            "如果知道 grep 和 top50 都漏掉的金页, 加到 extra_gold_pages 数组里, "
            "格式 [[公司名, 页号], ...]. "
            "U1 这种真没答案的查询, 把 is_empty_gold 改成 true. "
            "改完后把文件另存为 day09-gold-pages.final.json"
        ),
        "_generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "queries": [],
    }

    for q in QUERIES:
        print(f"\n🔎 {q['id']} — {q['text']}")

        # Step 1: PDF grep
        grep_candidates = grep_pdf_for_candidates(q)
        print(f"   grep 候选: {len(grep_candidates)} 页")

        # Step 2: Top-50 反向找漏
        back_candidates = topk50_back_check(q, retriever, grep_candidates)
        print(f"   top50 新增: {len(back_candidates)} 页")

        all_candidates = grep_candidates + back_candidates

        result["queries"].append({
            "id": q["id"],
            "text": q["text"],
            "company_filter": q.get("company_filter"),
            "keywords": q["keywords"],
            "candidates": all_candidates,
            "extra_gold_pages": [],  # 人工补充格式: [["招商银行", 50], ...]
            "is_empty_gold": False,
            "empty_gold_reason": "",
        })

    # --- 写 JSON ---
    DRAFT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with DRAFT_JSON.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 金页候选已写入: {DRAFT_JSON}")
    print(f"\n下一步:")
    print(f"  1. 打开 {DRAFT_JSON}")
    print(f"  2. 对每个 candidate 把 confirmed: null 改为 true/false")
    print(f"  3. 对 U1 这种无答案查询, 把 is_empty_gold 改为 true")
    print(f"  4. 另存为 {FINAL_JSON.name}")
    print(f"  5. 跑: uv run python scripts/27_audit_retrieval_params.py --stage=scan")


# ============================================================
# Phase 1.5: 读 final.json 校验 (Stage = scan 的入口)
# ============================================================

def load_final_gold_pages() -> dict:
    """读 final.json 并校验.

    校验规则:
        1. 文件必须存在 (否则提示跑 draft 再手工改)
        2. 每个查询的每个 candidate 必须 confirmed 不为 null
           (null 意味着没改, 拒绝跑实验)
        3. is_empty_gold=true 的查询必须有 empty_gold_reason
        4. extra_gold_pages 格式必须是 [[str, int], ...]

    Returns:
        {
            "U1": {
                "text": "...",
                "company_filter": "比亚迪",
                "gold_pages": [],  # 整理后的 (company, page) 列表, 为空说明 is_empty_gold
                "is_empty_gold": True,
                "empty_gold_reason": "Day 8 定案 L4 数据缺失",
            },
            "U4": {
                "text": "...",
                "gold_pages": [("招商银行", 45), ("招商银行", 31), ("招商银行", 50)],
                "is_empty_gold": False,
                ...
            }
        }
    """
    if not FINAL_JSON.exists():
        print(f"❌ 找不到 {FINAL_JSON}")
        print(f"   请先跑 --stage=draft, 然后手工改候选并另存为 final.json")
        sys.exit(1)

    with FINAL_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)

    processed = {}
    errors = []

    for q in data["queries"]:
        qid = q["id"]
        is_empty = q.get("is_empty_gold", False)

        # 校验 1: confirmed 必须不是 null
        unconfirmed = [
            f"{c['company']} p.{c['page']}"
            for c in q.get("candidates", [])
            if c.get("confirmed") is None
        ]
        if unconfirmed:
            errors.append(
                f"{qid}: 还有 {len(unconfirmed)} 个候选未确认 (confirmed=null): "
                f"{unconfirmed[:3]}{'...' if len(unconfirmed) > 3 else ''}"
            )
            continue

        # 校验 2: is_empty_gold=true 必须有 reason
        if is_empty and not q.get("empty_gold_reason", "").strip():
            errors.append(f"{qid}: is_empty_gold=true 但 empty_gold_reason 为空")

        # 整理 gold_pages
        gold_pages = []
        # 来自 candidates 里 confirmed=true 的
        for c in q.get("candidates", []):
            if c.get("confirmed") is True:
                gold_pages.append((c["company"], c["page"]))
        # 来自 extra_gold_pages 的 (人工补充)
        for extra in q.get("extra_gold_pages", []):
            if not (isinstance(extra, list) and len(extra) == 2):
                errors.append(f"{qid}: extra_gold_pages 格式错, 应该是 [[str, int], ...]")
                continue
            gold_pages.append((extra[0], int(extra[1])))

        # 去重
        gold_pages = list(set(gold_pages))

        # 校验 3: is_empty_gold=false 但 gold_pages 为空 → 矛盾
        if not is_empty and not gold_pages:
            errors.append(
                f"{qid}: is_empty_gold=false 但没有任何 confirmed=true 的候选, "
                f"也没有 extra_gold_pages. 如果真是空集请把 is_empty_gold 改成 true"
            )

        processed[qid] = {
            "text": q["text"],
            "company_filter": q.get("company_filter"),
            "gold_pages": gold_pages,
            "is_empty_gold": is_empty,
            "empty_gold_reason": q.get("empty_gold_reason", ""),
        }

    if errors:
        print("❌ final.json 校验失败:")
        for e in errors:
            print(f"   - {e}")
        sys.exit(1)

    # 打印摘要
    print("✅ final.json 校验通过")
    for qid, info in processed.items():
        if info["is_empty_gold"]:
            print(f"   {qid}: 空集 ({info['empty_gold_reason']})")
        else:
            print(f"   {qid}: {len(info['gold_pages'])} 金页 {info['gold_pages']}")

    return processed


# ============================================================
# Phase 2: 参数扫描
# ============================================================

def setup_retriever(rrf_k: int, recall_mult: int) -> tuple[HybridRetriever, BGEEmbedder, BM25Store]:
    """初始化 retriever. 带 chunks 缓存加速二次跑.

    注意: Qdrant 不重建 (假设已经 ingest 过).
    只需要重新加载 chunks 用于 BM25 构建 + embedder.

    chunks 缓存:
        文件名含 chunk_size/overlap, 改参数会自动失效.
        但如果你手动改了 chunker 的实现(而不是参数), 删掉 pkl 重跑.
    """
    # Step 1: 加载 chunks (走缓存)
    if CHUNKS_CACHE.exists():
        print(f"📦 从缓存读 chunks: {CHUNKS_CACHE.name}")
        with CHUNKS_CACHE.open("rb") as f:
            chunks = pickle.load(f)
        print(f"   {len(chunks)} chunks")
    else:
        print(f"🔨 首次运行, 从 PDF 重建 chunks...")
        pdf_files = sorted(DATA_RAW_DIR.glob("*.pdf"))
        print(f"   找到 {len(pdf_files)} 份 PDF")

        chunks = []
        for pdf_path in pdf_files:
            raw = load_pdf(pdf_path)
            ck = chunk_documents(raw, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            chunks.extend(ck)
            print(f"   {pdf_path.name}: {len(ck)} chunks")

        CHUNKS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with CHUNKS_CACHE.open("wb") as f:
            pickle.dump(chunks, f)
        print(f"   总 {len(chunks)} chunks, 已缓存")

    # Step 2: BM25 构建
    bm25_store = BM25Store()
    bm25_store.build(chunks)

    # Step 3: Embedder
    embedder = BGEEmbedder()

    # Step 4: Qdrant client
    client = get_client()

    # Step 5: Retriever
    retriever = HybridRetriever(
        qdrant_client=client,
        bm25_store=bm25_store,
        embedder=embedder,
        collection_name=COLLECTION,
        k=rrf_k,
        recall_multiplier=recall_mult,
    )

    return retriever, embedder, bm25_store


def single_search_3way(
    query_cfg: dict,
    retriever: HybridRetriever,
    top_k: int,
) -> dict:
    """跑一次查询, 同时导出 Hybrid + Dense 单路 + BM25 单路.

    这样设计的好处 (按 Task 1 Q3=D):
        1. Task 1 只用 hybrid 结果
        2. Task 2 直接复用这个 cache 里的 dense_top20 / bm25_top20
        3. 一次检索用三次, 避免 Task 2 重跑

    返回结构:
        {
            "query_id": "U4",
            "query_text": "...",
            "filters": {"company": "招商银行"} or None,
            "hybrid_topk": [...],        # 最终 top_k, 含 dense_rank/bm25_rank
            "dense_top20": [...],        # Dense 单路 Top-20 (固定 20)
            "bm25_top20": [...],         # BM25 单路 Top-20
            "elapsed_ms": float,
        }
    """
    filters = None
    if query_cfg.get("company_filter"):
        filters = {"company": query_cfg["company_filter"]}

    t0 = time.time()

    # Hybrid (正常走 retriever.search)
    hybrid_topk = retriever.search(query_cfg["text"], top_k=top_k, filters=filters)

    t_mid = time.time()

    # Dense 单路 Top-20 (固定 20, 给 Task 2 用)
    q_vec = retriever.embedder.encode_query(query_cfg["text"])
    dense_top20 = search_similar(
        retriever.qdrant_client,
        q_vec.tolist(),
        retriever.collection_name,
        limit=20,
        filters=filters,
    )

    # BM25 单路 Top-20
    bm25_top20 = retriever.bm25_store.search(query_cfg["text"], limit=20, filters=filters)

    t1 = time.time()

    # 瘦身 3 路结果 (excerpt 只留 (company, page, score), 节省 JSON 体积)
    def slim(r: dict) -> dict:
        m = r.get("metadata", {})
        out = {
            "company": m.get("company", "未知"),
            "page": m.get("page", -1),
            "chunk_type": m.get("chunk_type", "text"),
        }
        # 区分不同 score 字段
        if "rrf_score" in r:
            out["rrf_score"] = round(r["rrf_score"], 4)
            out["dense_rank"] = r.get("dense_rank")
            out["bm25_rank"] = r.get("bm25_rank")
        if "score" in r:
            out["score"] = round(r["score"], 4)
        return out

    return {
        "query_id": query_cfg["id"],
        "query_text": query_cfg["text"],
        "filters": filters,
        "hybrid_topk": [slim(r) for r in hybrid_topk],
        "dense_top20": [slim(r) for r in dense_top20],
        "bm25_top20": [slim(r) for r in bm25_top20],
        "elapsed_ms": round((t1 - t0) * 1000, 1),
        "hybrid_only_ms": round((t_mid - t0) * 1000, 1),
    }


def scan_param(
    param_name: str,
    values: list[int],
    fixed_top_k: int,
    fixed_rrf_k: int,
    fixed_recall_mult: int,
    retriever_cache: dict,
) -> list[dict]:
    """扫描一个参数, 其他两个固定.

    性能优化: retriever 只在 rrf_k 或 recall_mult 改变时重建.
    top_k 不影响 retriever 内部状态 (见 HybridRetriever.search 签名),
    所以 top_k 扫描可以复用同一个 retriever.

    retriever_cache: {(rrf_k, recall_mult): retriever} 由调用方维护.

    返回:
        [{param_value, per_query_results}, ...]
    """
    out = []
    for v in values:
        # 构造本轮的 (top_k, rrf_k, recall_mult)
        if param_name == "top_k":
            tk, rk, rm = v, fixed_rrf_k, fixed_recall_mult
        elif param_name == "rrf_k":
            tk, rk, rm = fixed_top_k, v, fixed_recall_mult
        elif param_name == "recall_mult":
            tk, rk, rm = fixed_top_k, fixed_rrf_k, v
        else:
            raise ValueError(f"unknown param: {param_name}")

        print(f"   扫 {param_name}={v} (top_k={tk}, rrf_k={rk}, recall_mult={rm})")

        # 复用 retriever: 只在 (rrf_k, recall_mult) 没见过时才 setup
        cache_key = (rk, rm)
        if cache_key not in retriever_cache:
            print(f"      (首次 rrf_k={rk} recall_mult={rm}, 构建 retriever)")
            retriever, _, _ = setup_retriever(rrf_k=rk, recall_mult=rm)
            retriever_cache[cache_key] = retriever
        else:
            retriever = retriever_cache[cache_key]

        per_query = []
        for q in QUERIES:
            r = single_search_3way(q, retriever, top_k=tk)
            per_query.append(r)

        out.append({
            "param_name": param_name,
            "param_value": v,
            "fixed": {"top_k": tk, "rrf_k": rk, "recall_mult": rm},
            "per_query": per_query,
        })

    return out


# ============================================================
# Phase 3: 指标计算 + 报告
# ============================================================

def compute_hit_rate(
    hybrid_results: list[dict],
    gold_pages: list[tuple],
    top_k: int,
) -> Optional[float]:
    """金页命中率 = Top-K 里出现的金页数 / 金页总数.

    Returns:
        None 如果 gold_pages 为空 (U1 空集)
        否则 0.0 ~ 1.0
    """
    if not gold_pages:
        return None

    gold_set = set(gold_pages)
    topk_pages = {(r["company"], r["page"]) for r in hybrid_results[:top_k]}
    hit = len(gold_set & topk_pages)
    return hit / len(gold_set)


def compute_avg_rank(
    hybrid_results: list[dict],
    gold_pages: list[tuple],
    top_k: int,
) -> Optional[float]:
    """金页平均 Rank. 没命中记 top_k+1.

    Returns:
        None 如果 gold_pages 为空
    """
    if not gold_pages:
        return None

    gold_set = set(gold_pages)
    ranks = []
    for gp in gold_set:
        found = False
        for rank, r in enumerate(hybrid_results[:top_k], start=1):
            if (r["company"], r["page"]) == gp:
                ranks.append(rank)
                found = True
                break
        if not found:
            ranks.append(top_k + 1)
    return sum(ranks) / len(ranks)


def format_param_table(scan_results: list[dict], gold_map: dict, param_name: str) -> str:
    """格式化一个参数扫描的报告段.

    输出结构:
        === top_k 参数扫描 ===

        均值视角 (5 个有效查询, U1 因空集排除):
        <表格>

        逐查询展开:
        <表格>

        异常提醒:
        - <按规则扫出的>
    """
    lines = []
    lines.append(f"=== {param_name} 参数扫描 ===\n")

    # --- 均值视角 ---
    lines.append("均值视角 (U1/其他空集查询从命中率剔除):")
    header = f"{param_name:<14} {'均值命中率':<10} {'均值Rank':<10} {'均值延迟ms':<12}"
    lines.append(header)
    lines.append("-" * len(header))

    per_value_stats = []  # 给"逐查询展开"和"异常提醒"用
    for entry in scan_results:
        value = entry["param_value"]
        per_query = entry["per_query"]
        tk = entry["fixed"]["top_k"]

        hit_rates = []
        avg_ranks = []
        latencies = [r["elapsed_ms"] for r in per_query]

        for r in per_query:
            qid = r["query_id"]
            gold_info = gold_map.get(qid, {})
            gold_pages = gold_info.get("gold_pages", [])
            if gold_info.get("is_empty_gold", False):
                continue  # 跳过空集查询

            hr = compute_hit_rate(r["hybrid_topk"], gold_pages, tk)
            ar = compute_avg_rank(r["hybrid_topk"], gold_pages, tk)
            if hr is not None:
                hit_rates.append(hr)
                avg_ranks.append(ar)

        mean_hr = sum(hit_rates) / len(hit_rates) if hit_rates else 0
        mean_ar = sum(avg_ranks) / len(avg_ranks) if avg_ranks else 0
        mean_lat = sum(latencies) / len(latencies) if latencies else 0

        lines.append(
            f"{str(value):<14} {mean_hr*100:>6.1f}%   {mean_ar:>6.2f}     {mean_lat:>7.1f}"
        )

        per_value_stats.append({
            "value": value,
            "tk": tk,
            "hit_rates": hit_rates,
            "per_query": per_query,
        })

    # --- 逐查询展开 ---
    lines.append("\n逐查询展开 (命中率 %):")
    # 表头
    header = "查询".ljust(6) + "".join(f"{param_name}={s['value']}".ljust(12) for s in per_value_stats)
    lines.append(header)
    lines.append("-" * len(header))

    # 每个查询一行
    for q in QUERIES:
        qid = q["id"]
        gold_info = gold_map.get(qid, {})

        if gold_info.get("is_empty_gold", False):
            row = f"{qid}".ljust(6) + "".join("[空集]".ljust(12) for _ in per_value_stats)
        else:
            gold_pages = gold_info.get("gold_pages", [])
            row = f"{qid}".ljust(6)
            for s in per_value_stats:
                r = next(r for r in s["per_query"] if r["query_id"] == qid)
                hr = compute_hit_rate(r["hybrid_topk"], gold_pages, s["tk"])
                row += f"{(hr or 0)*100:>6.1f}%     "
        lines.append(row)

    # --- 异常提醒 (FN2 对抗) ---
    lines.append("\n异常提醒:")
    alerts = []

    # 规则 1: 某查询在所有参数值下都 0% → 死穴候选
    for q in QUERIES:
        qid = q["id"]
        gold_info = gold_map.get(qid, {})
        if gold_info.get("is_empty_gold", False):
            continue
        gold_pages = gold_info.get("gold_pages", [])

        all_hrs = []
        for s in per_value_stats:
            r = next(r for r in s["per_query"] if r["query_id"] == qid)
            hr = compute_hit_rate(r["hybrid_topk"], gold_pages, s["tk"])
            if hr is not None:
                all_hrs.append(hr)

        if all_hrs and max(all_hrs) == 0:
            alerts.append(f"- {qid} 在所有 {param_name} 下都 0%, 可能是 L1 召回根因 → Task 2 重点")

    # 规则 2: 某查询在不同参数值下 Δ > 30% → 单样本敏感
    for q in QUERIES:
        qid = q["id"]
        gold_info = gold_map.get(qid, {})
        if gold_info.get("is_empty_gold", False):
            continue
        gold_pages = gold_info.get("gold_pages", [])

        all_hrs = []
        for s in per_value_stats:
            r = next(r for r in s["per_query"] if r["query_id"] == qid)
            hr = compute_hit_rate(r["hybrid_topk"], gold_pages, s["tk"])
            if hr is not None:
                all_hrs.append(hr)

        if all_hrs and (max(all_hrs) - min(all_hrs)) > 0.3:
            alerts.append(
                f"- {qid} 对 {param_name} 敏感: {min(all_hrs)*100:.0f}% → {max(all_hrs)*100:.0f}%, "
                f"单样本极值差 > 30% (FN2 候选, 留意)"
            )

    if not alerts:
        alerts.append("- (无显著异常)")

    lines.extend(alerts)

    return "\n".join(lines)


def write_report(
    topk_scan: list[dict],
    rrfk_scan: list[dict],
    recallmult_scan: list[dict],
    gold_map: dict,
) -> None:
    """写主报告到 docs/day09-param-audit.txt."""
    lines = []

    # 页首免责声明 (TD-9-1)
    lines.append("=" * 70)
    lines.append("Day 9 Task 1: 检索层参数扫描审计")
    lines.append("=" * 70)
    lines.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("⚠️ 样本边界声明 (TD-9-1):")
    lines.append("   本实验 6 样本, 金页通过 PDF grep + Top-50 反向找漏 + 人工复核 生成,")
    lines.append("   存在漏报/误报误差, 是工程近似不是金标准.")
    lines.append("   命中率绝对值不具外推价值, 结论限定在参数间相对差异维度.")
    lines.append("   需 Day 11 构造 30+ 代表性测试集做 RAGAS 评估才能下系统结论.")
    lines.append("")

    # 金页标注一览
    lines.append("=== 金页标注一览 ===")
    for qid in ["U1", "U2", "U3", "U4", "U5", "U6"]:
        info = gold_map.get(qid, {})
        if info.get("is_empty_gold", False):
            lines.append(f"  {qid}: [空集] {info.get('empty_gold_reason', '')}")
        else:
            lines.append(f"  {qid}: {info.get('gold_pages', [])}")
    lines.append("")

    # 三个参数扫描段
    lines.append(format_param_table(topk_scan, gold_map, "top_k"))
    lines.append("")
    lines.append(format_param_table(rrfk_scan, gold_map, "rrf_k"))
    lines.append("")
    lines.append(format_param_table(recallmult_scan, gold_map, "recall_mult"))
    lines.append("")

    # 页尾再来一次边界声明 (故意重复, 防止翻笔记只看结尾)
    lines.append("=" * 70)
    lines.append("⚠️ 结论使用须知 (重复, 故意):")
    lines.append("   6 样本 + grep 近似金页, 本报告仅作 Day 10 决策的信号参考,")
    lines.append("   不是 Day 11 大样本评估结论的替代. 参数调整前应先结合")
    lines.append("   Task 2 (L1 召回诊断) 和 Day 11 RAGAS 评估综合判断.")
    lines.append("=" * 70)

    REPORT_TXT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_TXT.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n✅ 主报告已写入: {REPORT_TXT}")


def write_3way_cache(
    topk_scan: list[dict],
    rrfk_scan: list[dict],
    recallmult_scan: list[dict],
) -> None:
    """写 Dense/BM25/Hybrid 三路缓存到 JSON, 给 Task 2 复用.

    只保留"当前默认参数"那组 (top_k=5, rrf_k=60, recall_mult=4),
    因为 Task 2 要的是"在当前参数下 3 路怎么分工",
    不需要所有参数组合.
    """
    # 找默认组: top_k=5 scan 里 value=5 那一项
    default_entry = next(
        (e for e in topk_scan if e["param_value"] == 5),
        None,
    )
    if default_entry is None:
        print("⚠️  写 3way-cache 时找不到 top_k=5 的扫描结果, 跳过")
        return

    cache = {
        "_note": "Task 1 产出的三路缓存, 对应默认参数 (top_k=5, rrf_k=60, recall_mult=4). Task 2 直接复用",
        "_generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fixed_params": default_entry["fixed"],
        "per_query": default_entry["per_query"],
    }

    with THREEWAY_CACHE.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    print(f"✅ 3 路缓存已写入: {THREEWAY_CACHE} (Task 2 可复用)")


def run_stage_scan() -> None:
    """Stage 2: 读 final.json, 跑 84 次检索, 出报告."""
    print("=" * 60)
    print("Stage 2: 参数扫描 + 报告")
    print("=" * 60)

    # Step 1: 读金页
    gold_map = load_final_gold_pages()

    # Step 2: 扫三个参数 (用 retriever_cache 避免重复重建)
    # setup 一次 (rrf_k=60, recall_mult=4) 就能覆盖 top_k 扫描的全部 5 组
    # rrf_k 扫描会新建 3 个 retriever (10, 30, 100, 60 已存在)
    # recall_mult 扫描会新建 4 个 retriever (2, 3, 6, 8, 4 已存在)
    # 合计 setup 8 次, 而不是 14 次
    retriever_cache = {}

    print("\n🔍 扫描 top_k ∈ {3, 5, 7, 10, 15} ...")
    topk_scan = scan_param("top_k", [3, 5, 7, 10, 15],
                           fixed_top_k=5, fixed_rrf_k=60, fixed_recall_mult=4,
                           retriever_cache=retriever_cache)

    print("\n🔍 扫描 rrf_k ∈ {10, 30, 60, 100} ...")
    rrfk_scan = scan_param("rrf_k", [10, 30, 60, 100],
                           fixed_top_k=5, fixed_rrf_k=60, fixed_recall_mult=4,
                           retriever_cache=retriever_cache)

    print("\n🔍 扫描 recall_mult ∈ {2, 3, 4, 6, 8} ...")
    recallmult_scan = scan_param("recall_mult", [2, 3, 4, 6, 8],
                                 fixed_top_k=5, fixed_rrf_k=60, fixed_recall_mult=4,
                                 retriever_cache=retriever_cache)

    # Step 3: 写报告
    write_report(topk_scan, rrfk_scan, recallmult_scan, gold_map)

    # Step 4: 写 3 路缓存 (Task 2 用)
    write_3way_cache(topk_scan, rrfk_scan, recallmult_scan)


# ============================================================
# main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Day 9 Task 1: 参数扫描审计")
    parser.add_argument(
        "--stage",
        choices=["draft", "scan"],
        required=True,
        help="draft: 生成金页候选 JSON; scan: 读 final.json 跑参数扫描"
    )
    args = parser.parse_args()

    t0 = time.time()
    if args.stage == "draft":
        run_stage_draft()
    elif args.stage == "scan":
        run_stage_scan()

    elapsed = time.time() - t0
    print(f"\n⏱️  总耗时: {elapsed:.1f}s")


if __name__ == "__main__":
    main()