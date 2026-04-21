# scripts/25_audit_data_missing.py
"""Day 8 - 脚本 25: 核心"数据缺失"结论的统一审计.

===============================================================================
背景 (为什么需要这个脚本)
===============================================================================
Day 4-7 做实验时, 遇到 LLM "答不出来" 的情况, 我们通过 LLM 的自述判定它是哪一层的问题.
比如:
  - 比亚迪汽车毛利率拒答 → LLM 说 "年报未披露汽车业务专项成本" → 判 L4 (数据缺失)
  - 国电南自 2024 净利润拒答 → LLM 说 "参考资料未提及" → 判 L4

但 "LLM 说没有" ≠ "真的没有". 可能实际情况是:
  - PDF 里真没写 → L4
  - PDF 写了但 chunker 切走了 → L0
  - 切好入库了但检索没捞到 → L1
  - 捞到了但 LLM 读漏了 → L2

4 种情况症状都是"拒答", 但根因和修法完全不同:
  - L0 修 chunker (加 metadata / 换语义切分)
  - L1 修检索 (扩召回池 / 加 Reranker)
  - L2 修 LLM 层 (Prompt 工程 / 换更强模型)
  - L4 不可修 (换数据源)

所以 Day 8 要用脚本证据重新验证 Day 4-7 的 4 个关键判定.

===============================================================================
审计的 4 个案例
===============================================================================
  U1 比亚迪汽车业务毛利率 → Day 7 先验: L4
  U2 宁德磷酸铁锂产品    → Day 5/6 先验: L4
  U3 国电南自 2024 净利润 → Day 5 先验: L4
  U4 招行 p.45 不良贷款率 → Day 7 先验: L0 (TD-7-1)

(U4 不是 L4 嫌疑, 是 L0 已定案案例. 放进来是为了验证 L0 判定也靠谱.)

===============================================================================
审计方法: 3 步法 + 非对称判定
===============================================================================

判定哲学: "宽进严出"
  - 要说"数据存在"(排除 L4): 只要找到一条证据就够 (grep 命中一次即可)
  - 要说"数据不存在"(定案 L4): 多路证据都阴性才能下结论

这样设计是因为 "找不到" 比 "找到" 弱得多:
  - 关键词换说法 (毛利率 vs 毛利润率)
  - PDF 抽取丢字
  - chunker 切词
  任何一个原因都会导致假阴性.

Step 1: 读 PDF 原文, 多关键词 grep
  命中 → 排除 L4 (PDF 里白纸黑字有)
  没中 → L4 嫌疑, 进入 Step 2

Step 2: 扫 Qdrant (按 company+year 过滤), 同样 grep 关键词
  命中 + Step 1 没中 → Qdrant 有但 PDF grep 漏了 (通常是关键词覆盖不全)
  命中 + Step 1 命中 → 数据完整进入检索系统, 进 Step 3
  没中 + Step 1 命中 → L0 定案! PDF 有但 chunker 切走
  没中 + Step 1 没中 → L4 定案! PDF 真没有

Step 3: 跑 Hybrid 检索 Top-20, 看金页在不在 Top-20
  在 → L2 (检索层 OK, 问题在 LLM 作答)
  不在 → L1 (检索层捞不到)

===============================================================================
关键工程决策
===============================================================================
1. PDF grep 用 pdfplumber (和 src/loaders/pdf_loader.py 同一工具)
   → 审计工具 = 生产工具, 避免工具差异导致假阳性/假阴性

2. Qdrant scroll 加 company+year 双重过滤
   → 避免跨公司/跨年污染 (U3 审 2024 净利润不能让 2023 数据干扰)

3. Step 3 用 HybridRetriever 不用 AdvancedRAGPipeline
   → AdvancedRAG 含 Multi-Query+HyDE 有 LLM 随机性, 审计要可复现

4. Step 3 用 Top-20 不是 Top-5
   → Top-20 告诉你 "金页在不在召回池", Top-5 只能告诉你 "在不在最终候选"
   → 这个区分决定 Day 9+ 的修法 (扩召回 vs 调 RRF/Reranker)
"""
from __future__ import annotations

# 必须先加载环境变量 (DeepSeek API key / HuggingFace mirror 等)
# override=True 是为了让 .env 能覆盖 shell 里已设的变量
from dotenv import load_dotenv
load_dotenv(override=True)

import time
from pathlib import Path

import pdfplumber
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from src.embeddings.bge import BGEEmbedder
from src.loaders.base import Chunk
from src.retrievers.bm25_store import BM25Store
from src.retrievers.hybrid import HybridRetriever
from src.retrievers.qdrant_store import get_client


# ============================================================================
# 配置区: 4 个审计案例的参数
# ============================================================================
#
# 每个案例含 7 个字段:
#   id:                案例编号 (U1/U2/U3/U4), 用于日志和报告
#   name:              人类可读的名称
#   query:             Day 4-7 用过的真实查询
#   company/year:      Qdrant filter 参数, 定位到特定年报
#   pdf_name:          原始 PDF 文件名 (data/raw/ 下必须存在)
#   keywords:          多关键词列表, 命中任何一个算命中 (OR 关系)
#   prior_conclusion:  Day 4-7 的先验判定 (用于最后对比)
#
# 关键词设计原则:
#   - 覆盖同义表述 (毛利率 / 毛利润率 / 综合毛利)
#   - 包含关键数字 (如 U4 "0.93%" 直接在 PDF 就定位 L0 还是 L2)
#   - 不宜过宽 (如只写 "毛利" 会匹配到集团合并毛利, 失去针对性)
# ============================================================================

# 项目约定: PDF 原文统一放在 data/raw/
DATA_RAW_DIR = Path("data/raw")

CASES = [
    {
        "id": "U1",
        "name": "比亚迪汽车业务毛利率",
        "query": "比亚迪2025年汽车业务毛利率",
        "company": "比亚迪",
        "year": 2025,
        "pdf_name": "002594_比亚迪_2025年年度报告.pdf",
        "keywords": [
            # 最精确匹配: 年报如果披露了一定含这个短语
            "汽车业务毛利率",
            # 正式说法的变体: 有些年报写得更全
            "汽车及相关产品毛利率",
            # 简写: 一些年报会省
            "汽车毛利率",
            # 只含"毛利"不含"率": 有些表格结构是 "毛利: xxx, 毛利率: xx%"
            # 但这里要求 "汽车" + "毛利" 一起出现才算,否则会污染
            "汽车业务毛利",
            # 年报标准结构 "按行业分析" 的行业毛利率一览表
            "分行业毛利率",
        ],
        "prior_conclusion": "L4 数据缺失 (年报未按汽车业务单独披露成本)",
    },
    {
        "id": "U2",
        "name": "宁德磷酸铁锂产品",
        "query": "宁德时代2025年磷酸铁锂电池产品情况",
        "company": "宁德时代",
        "year": 2025,
        "pdf_name": "300750_宁德时代_2025年年度报告.pdf",
        "keywords": [
            "磷酸铁锂",
            "LFP",         # 磷酸铁锂英文缩写, 技术章节常用
            "铁锂电池",    # 口语化简称
        ],
        "prior_conclusion": "L4 数据边界 (未披露磷酸铁锂专项数据)",
    },
    {
        "id": "U3",
        "name": "国电南自 2024 净利润",
        "query": "国电南自2024年净利润",
        "company": "国电南自",
        "year": 2024,
        "pdf_name": "600268_国电南自_2024年年度报告.pdf",
        "keywords": [
            "净利润",
            # 年报正式表述, 更长的版本
            "归属于上市公司股东的净利润",
            # 常用简写
            "归母净利润",
            # 另一种正式说法
            "归属于母公司股东的净利润",
        ],
        "prior_conclusion": "L4 数据链断裂 (推测 loader 丢了利润表)",
    },
    {
        "id": "U4",
        "name": "招行 2025 不良贷款率",
        "query": "招商银行2025年不良贷款率",
        "company": "招商银行",
        "year": 2025,
        "pdf_name": "600036_招商银行_2025年年度报告.pdf",
        "keywords": [
            "不良贷款率",
            "0.93%",   # 本公司口径
            "0.94%",   # 集团口径, Day 5 两个数都见过
        ],
        "prior_conclusion": "L0 chunking 边界 (TD-7-1 切走年份锚点)",
    },
]


# ============================================================================
# Step 1: PDF 原文 grep
# ============================================================================

def grep_pdf(case: dict) -> dict:
    """Step 1: 在原始 PDF 里逐页 grep 关键词.

    为什么这么做:
      这一步的目的是 "PDF 原文里到底写没写过相关内容".
      和 Step 2 Qdrant scroll 的区别是: PDF 是"入库前的数据", Qdrant 是"入库后".
      两者对比就能区分 L0 (chunker 切丢了) 和 L4 (真没数据).

    实现细节:
      - 用 pdfplumber 而不是 pypdf, 因为 pdf_loader.py 用的就是 pdfplumber,
        保证 grep 的文本和入库时看到的文本抽取方式一致
        (虽然入库时还做了 clean_text, 这一步我们不做,
         因为我们想看 "原始 PDF 文本是否提及")
      - 逐页处理, 这样能记录 "第几页命中", 方便后续和 Qdrant 的 page metadata 对齐
      - 抽取前后 40 字的上下文, 用于报告展示 (不是为了算法判定, 是给人看的)

    参数:
        case: 案例配置 (含 pdf_name / keywords 等)

    返回:
        {
            "pdf_exists": bool,       # PDF 文件是否存在
            "total_pages": int,       # PDF 总页数
            "hits": [                 # 命中页面列表
                {
                    "page": int,                    # 页号
                    "matched_keywords": [str, ...], # 该页命中的关键词
                    "excerpt": str,                 # 前后 40 字上下文
                },
                ...
            ],
            "elapsed": float,         # 读 PDF 耗时 (秒)
        }

    边界情况:
      - PDF 文件不存在 → 返回 pdf_exists=False, 其他字段为空
        (classify() 会走"两步法退化", 不会让整个脚本崩溃)
      - 某页无文本 (如扫描图片页) → 跳过该页
    """
    pdf_path = DATA_RAW_DIR / case["pdf_name"]

    # 边界: PDF 文件不存在时的优雅退化
    # 返回特殊结构让 classify() 能识别这种情况, 判定时会注明 "PDF 不可用"
    if not pdf_path.exists():
        return {
            "pdf_exists": False,
            "total_pages": 0,
            "hits": [],
            "elapsed": 0.0,
        }

    t0 = time.time()
    hits = []

    # ⚠️ pdfplumber.open 是上下文管理器, 用 with 保证文件句柄正确释放
    # 对大 PDF (比亚迪年报 450+ 页) 尤其重要, 否则可能泄漏文件描述符
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)

        # 逐页处理. 注意: page_num 从 1 开始 (和 PDF 阅读器一致),
        # 但 Python 索引从 0 开始, 所以用 enumerate(..., start=1)
        for page_num, page in enumerate(pdf.pages, start=1):
            # extract_text() 可能返回 None (纯图片页) 或空字符串
            # 用 `or ""` 统一成空字符串, 避免后面 None.strip() 报错
            text = page.extract_text() or ""
            if not text.strip():
                continue  # 跳过空页

            # 多关键词 OR 匹配: 只要命中任何一个关键词就算命中
            # 这里用简单的 `in` 操作, 不用 regex
            # 原因: 关键词已经设计成清晰的字符串, 不需要模糊匹配
            matched = [kw for kw in case["keywords"] if kw in text]

            if not matched:
                continue  # 这一页没有任何命中, 跳过

            # 抽第一个命中关键词的上下文 (前后 40 字)
            # 为什么只抽第一个: 避免报告太长, 第一个一般就能说明问题
            # 为什么 40 字: 够显示一个句子的片段, 但不会把报告撑爆
            first_kw = matched[0]
            idx = text.find(first_kw)
            start = max(0, idx - 40)
            end = min(len(text), idx + len(first_kw) + 40)
            # 把换行替换成空格, 避免报告里出现断行影响可读性
            excerpt = text[start:end].replace("\n", " ").strip()

            hits.append({
                "page": page_num,
                "matched_keywords": matched,
                "excerpt": excerpt,
            })

    elapsed = time.time() - t0

    return {
        "pdf_exists": True,
        "total_pages": total_pages,
        "hits": hits,
        "elapsed": elapsed,
    }


# ============================================================================
# Step 2: Qdrant scroll 全库扫描 (按公司+年份过滤)
# ============================================================================

def scan_qdrant(
    client: QdrantClient,
    case: dict,
    collection: str = "financial_reports",
) -> dict:
    """Step 2: 扫 Qdrant 里该公司该年份的全部 chunk, 客户端 grep 关键词.

    为什么这么做:
      这一步看的是 "检索系统能看到的数据", 对应 "入库后 chunker 切完的结果".
      和 Step 1 对比就能知道 chunker 有没有丢信息.

    为什么用 scroll 而不是 query_points:
      - query_points 是向量检索, 返回 Top-K 最相近的
      - scroll 是 "把符合 filter 的全部点都拉出来", 不做相关度排序
      - 我们要做的是 "全量 grep", 所以 scroll 正是我们需要的 API

    为什么加 company + year 双重 filter:
      - 不加 company 会跨公司污染. 比如审 U1 比亚迪毛利率时,
        不过滤会把茅台的毛利率也算进来, 导致误判 "库里有数据"
      - 不加 year 会跨年污染. 审 U3 2024 净利润时,
        库里还有国电南自 2023 数据, 不过滤会把 2023 的数据算进来

    参数:
        client:     Qdrant 客户端
        case:       案例配置
        collection: Qdrant collection 名, 默认和 Day 4 入库时一致

    返回:
        {
            "total_scanned": int,  # 该公司该年份的总 chunk 数
            "hits": [              # 命中的 chunk 列表
                {
                    "page":             int,    # 页号 (来自 payload metadata)
                    "chunk_type":       str,    # "text" 或 "table"
                    "matched_keywords": [str, ...],
                    "content":          str,    # 完整 chunk 内容
                    "point_id":         str,    # Qdrant 唯一 ID, 方便后续溯源
                },
                ...
            ],
            "elapsed": float,
        }
    """
    t0 = time.time()

    # 构造 Qdrant filter: company + year 双重过滤 (AND 关系)
    # Filter(must=[...]) 表示 "必须同时满足所有条件"
    # FieldCondition 是单个字段的条件
    # MatchValue 是精确等值匹配 (相当于 SQL 的 `field = value`)
    qdrant_filter = Filter(must=[
        FieldCondition(key="company", match=MatchValue(value=case["company"])),
        FieldCondition(key="year", match=MatchValue(value=case["year"])),
    ])

    # scroll 一次最多拉 500 条, 需要循环拉完
    # Qdrant 的 scroll API 设计:
    #   第一次调用返回 (points, next_offset)
    #   传 offset=next_offset 继续拉
    #   next_offset=None 表示拉完了
    all_points = []
    offset = None  # 第一次 scroll 不传 offset
    while True:
        response, offset = client.scroll(
            collection_name=collection,
            scroll_filter=qdrant_filter,   # ⚠️ 参数名是 scroll_filter 不是 filter
            limit=500,                      # 一批最多 500 个
            offset=offset,
            with_payload=True,              # 要 payload (里面有 content 和 metadata)
            with_vectors=False,             # 不要向量 (省内存, 我们只做文本 grep)
        )
        all_points.extend(response)
        if offset is None:
            break  # 拉完了

    # 客户端 grep: 把关键词匹配放在 Python 侧做
    # 为什么不用 Qdrant 的 full-text 搜索: Qdrant 对中文全文搜索的支持不强,
    # 且我们的关键词匹配逻辑简单 (纯 `in` 操作), 客户端做反而可控
    hits = []
    for p in all_points:
        # ⚠️ payload 可能是 None (虽然我们入库时都带了 payload)
        # 用 `p.payload or {}` 做防御
        content = (p.payload or {}).get("content", "")

        # 多关键词 OR 匹配
        matched = [kw for kw in case["keywords"] if kw in content]
        if not matched:
            continue

        hits.append({
            "page": (p.payload or {}).get("page", "?"),
            "chunk_type": (p.payload or {}).get("chunk_type", "?"),
            "matched_keywords": matched,
            "content": content,
            "point_id": str(p.id),
        })

    elapsed = time.time() - t0

    return {
        "total_scanned": len(all_points),
        "hits": hits,
        "elapsed": elapsed,
    }


# ============================================================================
# Step 3: Top-20 召回检查 (HybridRetriever)
# ============================================================================

def check_recall(
    retriever: HybridRetriever,
    case: dict,
    step2_hits: list[dict],
    top_k: int = 20,
) -> dict:
    """Step 3: 用 Hybrid 检索 Top-20, 看 Step 2 命中的 chunk 是否进入召回池.

    为什么这么做:
      Step 2 告诉我们 "库里有这些包含关键词的 chunk".
      但库里有不代表检索能捞到 —— 可能向量嵌入质量差/关键词权重低.
      Step 3 用真实 Hybrid 检索跑一次, 看 Step 2 找到的 chunk 能不能被召回.

      - 召回到了 → L2 (检索层 OK, 问题在 LLM)
      - 召回不到 → L1 (检索层本身有问题)

    为什么用 top_k=20 不是 5:
      Top-5 是给 LLM 的最终候选. 如果金页不在 Top-5 但在 Top-20,
      说明 "召回池够了, 但 RRF 排序没把它推上来" → 可以用 Reranker 救.
      如果金页不在 Top-20, 说明 "召回池根本没捞到" → 必须扩召回池或改检索算法.

      这两种情况的修法完全不同, 必须用 Top-20 才能区分.

    为什么用 HybridRetriever 不用 AdvancedRAGPipeline:
      AdvancedRAG 会跑 Multi-Query (LLM 改写 4 个子查询再检索) 和 HyDE
      (LLM 生成假答再检索). 这些 LLM 步骤每次结果不同, 审计结果不可复现.
      HybridRetriever 只做 Dense + BM25 + RRF, 确定性输出.

    参数:
        retriever:   已构建好的 Hybrid 检索器
        case:        案例配置
        step2_hits:  Step 2 找到的 chunk 列表, 用于和 Top-20 做匹配
        top_k:       检索返回几个, 默认 20

    返回:
        {
            "top20_pages": [int, ...],            # Top-20 全部页面列表
            "golden_hits_in_top20": [int, ...],   # Step 2 命中的 chunk 在 Top-20 的排名 (1-indexed)
            "first_hit_rank": Optional[int],      # 第一个命中 chunk 的排名
            "elapsed": float,
        }
    """
    t0 = time.time()

    # 跑 Hybrid 检索. 注意 filters 参数格式和 HybridRetriever.search 一致
    results = retriever.search(
        query=case["query"],
        top_k=top_k,
        filters={
            "company": case["company"],
            "year": case["year"],
        },
    )

    elapsed = time.time() - t0

    # 匹配策略: 用 content 精确相等来判断 "Top-20 里的某个 chunk 是 Step 2 命中的吗"
    #
    # 为什么不用 page 匹配: page 级匹配会误判. 同一页可能有多个 chunk
    # (比如 p.45 有两个 chunk: chunk A 含 "0.93%", chunk B 不含).
    # Step 2 找到的是 chunk A, 但如果 Top-20 里只有 chunk B, 用 page 匹配会误判 "命中".
    #
    # 为什么不用 point_id: HybridRetriever 返回的结构里不带 point_id,
    # 只带 content 和 metadata. 改接口返回 point_id 是下一步优化, 今天用 content 够用.
    #
    # ⚠️ 局限: content 完全相等要求 Qdrant 返回的和 Step 2 存的字符串一字不差.
    # 这个假设对当前代码成立 (都是直接从 payload 取), 但未来改了接口可能静默失效.
    # 面试被问 "你的审计有什么局限" 时可以提这个点.
    step2_contents = {h["content"] for h in step2_hits}

    golden_hits_in_top20 = []
    for rank, r in enumerate(results, start=1):
        if r["content"] in step2_contents:
            golden_hits_in_top20.append(rank)

    first_hit_rank = golden_hits_in_top20[0] if golden_hits_in_top20 else None

    return {
        "top20_pages": [r["metadata"].get("page", "?") for r in results],
        "golden_hits_in_top20": golden_hits_in_top20,
        "first_hit_rank": first_hit_rank,
        "elapsed": elapsed,
    }


# ============================================================================
# 归因逻辑 (核心!)
# ============================================================================

def classify(step1: dict, step2: dict, step3: dict) -> dict:
    """按非对称判定规则归类.

    这是整个脚本的"大脑". 前面三步收集证据, 这里统一做判定.

    判定逻辑 (完整决策树):

        if PDF 不可用:
            # 只能两步法 (降级处理)
            Qdrant 没命中 → L4_suspect (嫌疑, 但无法排除 L0)
            Qdrant 命中 + 召回不到 → L1
            Qdrant 命中 + 召回到 → L2

        else (PDF 可用, 三步法):
            (PDF 没中, Qdrant 没中) → L4 定案 (双阴性, 硬证据)
            (PDF 中,   Qdrant 没中) → L0 (chunker 丢了)
            (PDF 没中, Qdrant 中)   → 关键词覆盖不全导致 PDF grep 漏,
                                      但数据在库是硬证据, 降级为 L1/L2
            (PDF 中,   Qdrant 中)   → 数据完整, 看召回:
                召回到 → L2
                没召回 → L1

    为什么叫"非对称":
      PDF grep 命中 = 硬证据排除 L4 (白纸黑字在 PDF 里)
      PDF grep 未命中 = 嫌疑但不定案 (可能是关键词覆盖不全)
      这两个方向的证据权重不对等, 所以判定规则也不对等.

    参数:
        step1/step2/step3: 前三步的返回值

    返回:
        {
            "layer":   "L0" | "L1" | "L2" | "L4" | "L4_suspect" | "unknown",
            "reason":  一句话说明判定依据,
        }
    """
    # 把"是否命中"提炼成布尔值, 后面判定更清晰
    s1_hit = len(step1.get("hits", [])) > 0
    s2_hit = len(step2.get("hits", [])) > 0
    s3_hit = step3.get("first_hit_rank") is not None

    # ========================================================================
    # 分支 A: PDF 不可用 → 两步法降级
    # ========================================================================
    if not step1.get("pdf_exists"):
        if not s2_hit:
            return {
                "layer": "L4_suspect",
                "reason": "PDF 不可用 + Qdrant 未命中 → L4 嫌疑但无法排除 L0",
            }
        elif not s3_hit:
            return {"layer": "L1", "reason": "数据在库但 Top-20 未召回"}
        else:
            return {"layer": "L2", "reason": "数据在库且召回到, LLM 作答问题"}

    # ========================================================================
    # 分支 B: PDF 可用 → 三步法完整判定
    # ========================================================================

    # 情况 1: PDF + Qdrant 都没命中 → L4 定案 (双阴性硬证据)
    if not s1_hit and not s2_hit:
        return {
            "layer": "L4",
            "reason": "PDF grep + Qdrant 双阴性 → 数据真不存在",
        }

    # 情况 2: PDF 有但 Qdrant 没 → L0 定案 (chunker 丢了信息)
    # 这是我们最关心的案例之一, 对应 Day 7 TD-7-1 的结构性问题
    if s1_hit and not s2_hit:
        return {
            "layer": "L0",
            "reason": "PDF 有但 Qdrant 没有 → loader/chunker 切走了",
        }

    # 情况 3: PDF 没但 Qdrant 有
    # 这个组合看似矛盾, 实际可能性:
    # - PDF grep 的关键词表覆盖不全 (比如有个同义说法没写进来)
    # - chunker 把附近的词拼在一起形成了 PDF grep 看不到的组合
    # 不管哪种, 数据在库是硬证据, 不算 L4
    if not s1_hit and s2_hit:
        if not s3_hit:
            return {
                "layer": "L1",
                "reason": "Qdrant 有但 Top-20 未召回 (PDF grep 漏因关键词覆盖不全)",
            }
        else:
            return {
                "layer": "L2",
                "reason": "Qdrant 有且召回到, LLM 作答问题 (PDF grep 漏因关键词覆盖)",
            }

    # 情况 4: PDF 和 Qdrant 都有 → 数据完整, 看召回
    # (到这里 s1_hit 和 s2_hit 必然都是 True)
    if not s3_hit:
        return {
            "layer": "L1",
            "reason": "数据在 PDF + 在库, 但 Top-20 未召回",
        }
    return {
        "layer": "L2",
        "reason": "数据在 PDF + 在库 + 召回到, LLM 作答问题",
    }


# ============================================================================
# 报告打印
# ============================================================================

def print_case_report(case: dict, step1: dict, step2: dict, step3: dict, verdict: dict):
    """打印单个案例的完整审计报告.

    为什么单独做一个函数:
      main() 里循环 4 个案例, 每个案例都要打印. 抽成函数避免重复,
      也方便将来改打印格式.

    格式设计:
      - 4 段式: 案例头 → Step 1 结果 → Step 2 结果 → Step 3 结果 → 判定
      - 每段有明确的 section 分隔符 (===)
      - emoji 作为视觉快速标记 (✅ ❌ ⚠️ 🔥)
    """
    print("\n" + "=" * 80)
    print(f"案例 {case['id']}: {case['name']}")
    print(f"查询: {case['query']}")
    print(f"Day 7 先验结论: {case['prior_conclusion']}")
    print("=" * 80)

    # ==================== Step 1 结果 ====================
    print(f"\n[Step 1] PDF grep ({case['pdf_name']})")
    if not step1.get("pdf_exists"):
        print(f"  ⚠️  PDF 文件不存在, 跳过 Step 1")
    else:
        print(f"  耗时: {step1['elapsed']:.1f}s, 总页数: {step1['total_pages']}")
        if step1["hits"]:
            print(f"  ✅ 命中 {len(step1['hits'])} 页, 样例 (前 3):")
            for h in step1["hits"][:3]:
                kw_str = "/".join(h["matched_keywords"])
                print(f"     p.{h['page']} [{kw_str}]")
                print(f"       ...{h['excerpt']}...")
        else:
            print(f"  ❌ 全 PDF 均未命中关键词 {case['keywords']}")

    # ==================== Step 2 结果 ====================
    print(f"\n[Step 2] Qdrant scroll (company={case['company']}, year={case['year']})")
    print(f"  耗时: {step2['elapsed']:.1f}s, 扫描 chunks: {step2['total_scanned']}")
    if step2["hits"]:
        print(f"  ✅ 命中 {len(step2['hits'])} 个 chunks, 样例 (前 3):")
        for h in step2["hits"][:3]:
            kw_str = "/".join(h["matched_keywords"])
            content_preview = h["content"][:80].replace("\n", " ")
            print(f"     p.{h['page']} [{h['chunk_type']}] [{kw_str}]")
            print(f"       {content_preview}...")
    else:
        print(f"  ❌ Qdrant 该公司该年份全部 chunks 均未命中")

    # ==================== Step 3 结果 ====================
    print(f"\n[Step 3] Top-20 召回检查 (HybridRetriever, 无 Reranker)")
    print(f"  耗时: {step3['elapsed']:.2f}s")
    print(f"  Top-20 页面: {step3['top20_pages']}")
    if step3["first_hit_rank"]:
        print(f"  ✅ Step 2 命中的 chunk 首次出现在 Rank {step3['first_hit_rank']}")
        if len(step3["golden_hits_in_top20"]) > 1:
            print(f"     全部排名: {step3['golden_hits_in_top20']}")
    else:
        if step2["hits"]:
            print(f"  ❌ Step 2 命中的 chunk 均未进入 Top-20 → 召回问题")
        else:
            print(f"  ⚠️  Step 2 未命中, 无法做召回检查")

    # ==================== 最终判定 ====================
    print(f"\n[判定] {verdict['layer']}")
    print(f"  理由: {verdict['reason']}")
    print(f"  vs Day 7 先验: {case['prior_conclusion']}")

    # 从先验结论字符串里提取层代号 ("L4 数据缺失..." → "L4")
    day7_layer = case["prior_conclusion"].split(" ")[0]

    if verdict["layer"] == day7_layer:
        print(f"  ✅ 与先验结论一致")
    else:
        print(f"  🔥 推翻先验! Day 7 猜 {day7_layer}, 实际 {verdict['layer']}")


def print_summary(all_results: list[dict]):
    """打印 4 个案例的汇总表 + 按层分布统计."""
    print("\n\n" + "=" * 80)
    print("📊 Day 8 审计汇总")
    print("=" * 80)
    print(f"\n{'ID':<4} {'公司':<10} {'Day 7 先验':<15} {'实际判定':<10} {'一致?':<10}")
    print("-" * 80)

    for r in all_results:
        case = r["case"]
        day7 = case["prior_conclusion"].split(" ")[0]
        actual = r["verdict"]["layer"]
        match = "✅" if day7 == actual else "🔥 推翻"
        print(f"{case['id']:<4} {case['company']:<10} {day7:<15} {actual:<10} {match:<10}")

    # 按 4 层光谱分布统计
    print(f"\n\n按 4 层光谱分布:")
    from collections import Counter
    layer_dist = Counter(r["verdict"]["layer"] for r in all_results)
    for layer, count in sorted(layer_dist.items()):
        print(f"  {layer}: {count}")


# ============================================================================
# Main
# ============================================================================

def main():
    """主流程: 初始化共享资源 → 循环跑 4 案例 → 打印汇总."""
    print("=" * 80)
    print("Day 8 — 核心'数据缺失'结论审计 (脚本 25)")
    print("=" * 80)

    # ------------------------------------------------------------------------
    # 初始化共享资源
    #
    # 为什么一次性初始化:
    #   BGE 加载 ~95MB + BM25 从 Qdrant 构建 ~10 秒, 每个案例独立加载太慢.
    #   改成共享后 4 个案例跑完就多付 1 次初始化成本.
    # ------------------------------------------------------------------------
    print("\n[初始化] 加载 BGE + 构建 BM25 索引 (和 AdvancedRAG 一致)...")
    client = get_client()
    embedder = BGEEmbedder()

    # 从 Qdrant scroll 全部点, 构建 BM25 索引
    # 这段代码和 advanced_rag.py 里的 _build_bm25_from_qdrant 一样,
    # 这里抄过来是为了让脚本独立可跑 (不依赖 AdvancedRAGPipeline 整体初始化)
    all_points = []
    offset = None
    while True:
        response, offset = client.scroll(
            collection_name="financial_reports",
            limit=500,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        all_points.extend(response)
        if offset is None:
            break

    # 把 Qdrant 的 point 转成 Chunk 对象 (BM25Store.build 要 Chunk 列表)
    chunks = []
    for p in all_points:
        payload = dict(p.payload or {})
        content = payload.pop("content", "")
        chunk_type = payload.pop("chunk_type", "text")
        # 注意: 剩下的 payload 就是 metadata (company/year/page 等)
        chunks.append(Chunk(
            content=content,
            chunk_type=chunk_type,
            metadata=payload,
        ))

    bm25_store = BM25Store()
    bm25_store.build(chunks)

    retriever = HybridRetriever(
        qdrant_client=client,
        bm25_store=bm25_store,
        embedder=embedder,
    )

    print(f"\n[初始化完成] 总 chunks: {len(chunks)}")

    # ------------------------------------------------------------------------
    # 循环跑 4 个案例
    # ------------------------------------------------------------------------
    all_results = []
    for case in CASES:
        print(f"\n\n{'#' * 80}")
        print(f"# 跑案例 {case['id']}: {case['name']}")
        print(f"{'#' * 80}")

        # 3 步依次执行. Step 3 需要 Step 2 的结果, 所以顺序不能换.
        step1 = grep_pdf(case)
        step2 = scan_qdrant(client, case)
        step3 = check_recall(retriever, case, step2["hits"])
        verdict = classify(step1, step2, step3)

        print_case_report(case, step1, step2, step3, verdict)

        all_results.append({
            "case": case,
            "step1": step1,
            "step2": step2,
            "step3": step3,
            "verdict": verdict,
        })

    # ------------------------------------------------------------------------
    # 打印汇总
    # ------------------------------------------------------------------------
    print_summary(all_results)


if __name__ == "__main__":
    main()