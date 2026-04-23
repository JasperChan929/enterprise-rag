# Enterprise-RAG 项目概览 (Day 1-8)

> 最后更新: Day 8 结束
> 项目状态: 23 天路线图的 Day 8 已完成, 进入 Phase 1 末端, Day 9 继续审计日 2

---

## 📌 一句话定位

**企业金融年报 RAG 问答系统**. 面向 AI 应用工程师 / LLM 工程师岗位的
找工作级项目, 强调**端到端闭环 + 数据驱动决策 + 诊断方法论**, 而不是炫技功能堆叠.

---

## 🎯 项目目标

- **业务目标**: 对 8 家 A 股上市公司年报 (6881 个 chunks) 做智能问答, 覆盖
  白酒/银行/新能源/有色/水电/电力设备 6 个行业
- **技术目标**: 完整跑通 Naive RAG → Hybrid → Advanced → Reranker 四代架构,
  每一代都有实证数据
- **面试目标**: 在二三面能讲出 "方法论认知升级" 的深度故事, 不只是 "我做了什么"

---

## 🗓️ 23 天路线图 (Day 8 更新版)

| 阶段 | 天数 | 主题 | 状态 |
|---|---|---|---|
| 基础建设 | Day 1-4 | 数据加载 + 切分 + 向量化 + Naive RAG 跑通 | ✅ 完成 |
| 检索升级 | Day 5-7 | Hybrid + Multi-Query/HyDE + Reranker | ✅ 完成 |
| **Phase 1 诊断地基** | Day 8-9 | 结论审计 (Day 8 完成, Day 9 进行中) | 🟡 进行中 |
| Phase 2 评估基础 | Day 10-11 | Prompt 工程 + 30 条评估集 + RAGAS | ⏸ 待开始 |
| Phase 3 全量评估 | Day 12 | 3 pipeline 评估 + 瓶颈定位 | ⏸ 待开始 |
| Phase 4 精准优化 | Day 13-16 | 主瓶颈 2 天 + 次瓶颈 1 天 + 回归消融 1 天 | ⏸ 待开始 |
| Phase 5 工程化 | Day 17-18 | 成本/错误分析 + 流式响应/Agent 思考 | ⏸ 待开始 |
| Phase 6 系统化 | Day 19-21 | FastAPI + Streamlit + 鲁棒/部署 | ⏸ 待开始 |
| Phase 7 交付 | Day 22-23 | README + 博客 + 面试话术 | ⏸ 待开始 |

**路线图变更记录**:
- Day 8 发现 Day 4-7 诊断 75% 判错, **Prompt 工程从"锦上添花"升级为"Day 10 主线"**
- TD-8-2 多口径错位 (原 TD-7-3) 从 P2 升 P1
- TD-8-1 (原"财务表切断", 已改名"表格切断 chunk") 全库量化 4.5%, 判 P3 个案级
- **Day 8 收工时发现样本边界问题**: 75% 推翻率基于 4 翻车样例, **Day 10 必须用 30+ 代表性测试集验证才能下"系统性"结论**
- **新增 TD-8-4 (其他 chunker 切断形态待量化)**, Day 9 起步; chunker 全量切断量化后可能升级 TD-8-1
- 语义 chunking 推到 Day 22+ (如果时间允许)

---

## 📂 代码结构 (Day 8 末)

```
enterprise-rag/
├── src/
│   ├── loaders/                    # Day 2 产出
│   │   ├── base.py                 # Chunk 数据结构 (dataclass)
│   │   └── pdf_loader.py           # pdfplumber + 分流 + 去重 + 清洗
│   ├── chunking/
│   │   └── recursive.py            # 递归字符切分器 (chunk_size=400, overlap=50)
│   ├── embeddings/
│   │   └── bge.py                  # BGE-small-zh-v1.5 (512 维, 单例封装)
│   ├── retrievers/
│   │   ├── qdrant_store.py         # Qdrant 操作 (建库/入库/检索/过滤)
│   │   ├── bm25_store.py           # BM25Okapi + jieba 分词
│   │   ├── financial_dict.py       # 金融术语词典 (175 个)
│   │   └── hybrid.py               # Dense + BM25 + RRF 融合 (k=60)
│   ├── rerankers/                  # Day 7 产出
│   │   └── bge_reranker.py         # BGE-Reranker-base (Cross-Encoder)
│   ├── query_transformers/         # Day 6 产出
│   │   ├── multi_query.py          # LLM 改写 4 个子查询
│   │   ├── hyde.py                 # LLM 生成假答案做检索
│   │   └── router.py               # LLM 判断是否开 MQ/HyDE
│   ├── generators/
│   │   └── llm.py                  # DeepSeek-Chat + Prompt 模板
│   └── pipelines/
│       ├── naive_rag.py            # Day 4: Dense → LLM
│       ├── hybrid_rag.py           # Day 5: Hybrid → LLM
│       └── advanced_rag.py         # Day 6-7: Multi-Query + HyDE + Router + Reranker
├── scripts/                        # 28 个实验脚本 (Day 2-8)
│   ├── 01-05: Day 2 PDF loader & chunker
│   ├── 06-10: Day 3 BGE embedding
│   ├── 11-13: Day 4 Naive RAG 端到端
│   ├── (Day 5-6 没编号, 散在模块内)
│   ├── 21-24: Day 7 Reranker 实验
│   └── 25-26: Day 8 审计脚本  ⬅ 本期新增
│       ├── 25_audit_data_missing.py       # 4 案例 3 步法审计
│       ├── 25b_audit_top5_quality.py      # L2 子类型细化
│       └── 26_audit_chunking_fragment.py  # TD-8-1 量化
├── docs/
│   ├── day02-summary.md ~ day07-summary.md  # Day 2-7 笔记
│   ├── day08-summary.md                     # ⬅ 本期新增, 8 段完整笔记
│   ├── day08-audit-{25,25b,26}.txt          # ⬅ 本期新增, 审计输出数据
│   └── project-overview_1-8.md              # 本文档
├── data/
│   ├── raw/                        # 8 份年报 PDF (不提交 git)
│   └── processed/                  # 向量化结果 (不提交 git)
├── qdrant_storage/                 # Qdrant Docker 持久化 (不提交 git)
├── main.py
├── pyproject.toml
├── .env                            # LLM API key (不提交 git)
└── .gitignore
```

---

## 🧠 4 代架构演进

### Day 4 — Naive RAG (baseline)
```
问题 → BGE.encode_query → Qdrant 向量检索 Top-5 → 组 Prompt → DeepSeek → 答案
```
- 最简形态, 单路 Dense 检索
- **已知局限**: 对专业术语区分弱 (如 "营业收入" vs "利息收入")
- 基线答对率: **1/4 (25%)** (4 测试查询)

### Day 5 — Hybrid RAG
```
问题 → Dense + BM25 并行检索 → RRF 融合 (k=60) → Top-5 → DeepSeek
```
- BM25 补 Dense 在专业术语上的弱点
- **改善**: 7 个查询中 4 个改善 (57%), 1 个回归, 2 个持平
- **新问题**: 宽召回稀释 (TD-5-2)

### Day 6 — Advanced RAG
```
问题 → Router (判断是否 MQ/HyDE)
     → Multi-Query 改写 4 个 / HyDE 生成假答 / 原问题
     → 每路独立 Hybrid 检索
     → 二次 RRF 融合 → Top-5 → DeepSeek
```
- 6 种检索模式: naive / hybrid / multi_query / hyde / full / auto
- **改善**: 4 个查询中 3 个改善 (75%, 拒答率 50% → 25%)
- **新问题**: Full 模式宽召回稀释加重 (TD-6-5)

### Day 7 — Advanced RAG + Reranker
```
... → Top-20 候选 → BGE-Reranker-base 精排 → Top-5 → DeepSeek
```
- Cross-Encoder 对召回做精排
- **实证**: 9 组 A/B 仅 2 组答案有可见提升 (不是银弹)
- 延迟代价: +5~15 秒
- **关键认知**: Reranker 价值分 4 层 (排名救援 / 补证提升 / 无变化 / 数据锁死)

### Day 8 — 结论审计 (不是新架构, 是方法论升级)
- 不改代码, 不增功能
- 用脚本证据重新审计 Day 4-7 的推测性结论
- **4 案例 3 个推翻**, 暴露 "LLM 自述诊断" 的系统性偏差
- 产出: **5 层瓶颈模型** + **非对称判定方法论** + **自足 chunk 概念**

---

## 🏗️ 5 层瓶颈模型 (Day 8 升级版)

Day 7 提出 4 层, Day 8 把 L2 细化为 L2a/L2b, 升级为 5 层:

```
┌─────────────────────────────────────────────────────────────────┐
│ L0 Chunking (最上游)                                            │
│   PDF 切块时丢信息. 例: 表头和数据被切成两段 (TD-8-1, 4.5%).    │
│   修法: metadata 注入 / 语义 chunking                           │
├─────────────────────────────────────────────────────────────────┤
│ L1 召回 (Dense/BM25/Hybrid/Reranker)                            │
│   金页没进 Top-20. 例: 紫金 p.59 地缘政治页.                    │
│   修法: 扩召回池 / Reranker / 调 RRF                            │
├─────────────────────────────────────────────────────────────────┤
│ L2a LLM 认知问题 ⬅ Day 8 新增                                   │
│   Top-5 有自足 chunk 但 LLM 仍拒答 (如多口径错位).              │
│   例: U4 招行看到 0.93/0.94/1.52% 三个口径无法选择.             │
│   修法: Prompt 工程 (CoT / 口径引导 / 放松拒答规则)             │
├─────────────────────────────────────────────────────────────────┤
│ L2b 排序问题 ⬅ Day 8 新增                                       │
│   Top-5 全碎片, 自足 chunk 在 Top-6~20.                         │
│   例: U3 国电南自自足 chunk 在 Rank 15/18.                      │
│   修法: Reranker 条件开启 / 调 RRF k / 扩 top_k                 │
├─────────────────────────────────────────────────────────────────┤
│ L4 数据缺失 (年报本来就没写)                                    │
│   例: U1 比亚迪 2025 年报未按汽车业务拆分成本.                  │
│   修法: 不可修 (换数据源)                                       │
└─────────────────────────────────────────────────────────────────┘
```

**核心诊断工具**: "自足 chunk" 概念
- 要素 1: 主体 (公司) — 靠 metadata.company 保证
- 要素 2: 时间锚点 — chunk 文本里有年份 或 "报告期末" 等
- 要素 3: 核心数据 — chunk 文本里有直接答案

三要素齐全 → LLM 一看就能答, 缺 1-2 要素 → 需要推理组合

---

## 📚 核心方法论 (Day 8 沉淀)

### 1. 非对称判定 (宽进严出)
不同方向结论需要的证据强度不同:
- 排除 L4 (说"数据存在"): 1 条证据即可
- 定案 L4 (说"数据不存在"): 多路阴性才定

**原理**: 证伪 L4 代价几秒, 错判 L4 代价几天. 默认立场要向"数据可能存在"倾斜.

### 2. LLM 自述不作诊断证据
3 个不可靠原因:
- "不确定"和"真没有"表述相同, 无法区分
- 拒答偏好导致系统性错报为 L4
- 工程师读 LLM 自述会本能往"上游归因", 系统性偏向 L4

**规则**: P1/P2 技术债必须有脚本证据. LLM 自述只作"嫌疑线索".

### 3. 脚本规则设计自我攻击清单
写规则后、跑脚本前必做:
1. 列 3 个假阳性场景
2. 列 2 个假阴性场景
3. 对每种标注: 接受 / 加约束过滤
4. 预估结果范围
5. 把前 4 项写进脚本注释顶部

**Day 8 反面教材**: 脚本 26 v1 因为没做自我攻击, 把叙述文字误当碎片.

### 4. 技术债优先级决策原则
- P1 = 对系统影响大 (**不是**修起来难易)
- 归因错误的技术债归档保留, 不删除
- 架构升级不是唯一选项, 先思考轻量方案 (Prompt 工程往往够用)

---

## 📊 关键数据规模

### 数据层
| 项目 | 数值 |
|---|---|
| 年报数量 | 8 份 (6 行业) |
| PDF 总页数 | ~2000 页 |
| Qdrant total points | 6881 chunks |
| text chunks | 5220 (75.9%) |
| table chunks | 1661 (24.1%) |
| 向量维度 | 512 (BGE-small-zh) |
| 距离度量 | Cosine |

### 公司分布
| 公司 | 年份 | chunks | 表格占比 |
|---|---|---|---|
| 贵州茅台 | 2023 | 493 | 46.2% |
| 宁德时代 | 2025 | 726 | 47.5% |
| 招商银行 | 2025 | 1388 | **1.0%** ⚠️ (TD-5-1) |
| 比亚迪 | 2025 | 728 | 16.2% |
| 紫金矿业 | 2025 | 1234 | 9.0% |
| 长江电力 | 2024 | 730 | 37.3% |
| 国电南自 | 2023 | 761 | 36.2% |
| 国电南自 | 2024 | 821 | 36.2% |

### Day 8 审计结果 (最新)

**4 案例审计矩阵** (Day 8 核心产出):
| ID | 案例 | Day 7 先验 | Day 8 实测 | 推翻? |
|---|---|---|---|---|
| U1 | 比亚迪汽车毛利率 | L4 | L4 | ✅ 一致 |
| U2 | 宁德磷酸铁锂产品 | L4 | L2 | 🔥 推翻 |
| U3 | 国电南自 2024 净利润 | L4 | L2b | 🔥 推翻 |
| U4 | 招行 2025 不良贷款率 | L0 | L2a | 🔥 推翻 |

**推翻率**: 3/4 = 75%
**方向规律**: 所有推翻都落到 L2, 系统性地把 L2 误判为 L0/L4

**TD-8-1 财务表切断全库量化**:
- 总 chunks 6881
- 含财务关键词 552 (8.0%)
- 真碎片 25 (4.5% of 含关键词)
- 最严重: 宁德时代 22.6%; 最低: 招行 0%
- **判定**: P3 个案级, 非全局问题

---

## 🐛 技术债总图 (Day 8 终版)

### P1 Day 10 必修
- **TD-8-2** (多口径错位, 原 TD-7-3 升级) — Prompt 工程主战场
- **TD-8-3** (Prompt 是否注入 metadata 待审) — Day 10 开工前 30 分钟必做

### P2 Day 11+ 评估后决策
- TD-7-2 (Reranker 延迟) — Day 9 审计
- TD-7-4 (条件开 Reranker)
- **TD-8-4 (其他 chunker 切断形态待量化)** — Day 9 起步, 量化后可能影响 TD-8-1 优先级
- TD-5-1 (招行表格抽取率 1%)
- TD-6-3 (Router 校准不足)

### P3 个案级, 不专项修
- TD-8-1 (表格切断 chunk, 4.5%) — Day 10 Prompt 顺带处理。**注: 仅基于 1/5 chunker 切断形态量化, 待 TD-8-4 全量后重审**
- TD-2-1 (49 个超长表格 BGE 截断)
- TD-2-2 / 2-3 / 6-1 / 6-2 / 6-4 / 6-5 (持续观察)

### 📕 历史归档 (Day 8 推翻)
- **TD-7-1** (chunking 丢年份锚点) — 归因错误, 真问题是 TD-8-2 多口径
- TD-4-1 (招行表格数据缺失, Day 5 已推翻)

---

## 🎤 面试话术素材 (Day 8 沉淀)

**Q1**: 你项目最重要的方法论认知升级是什么?
→ Day 8 从 "LLM 自述诊断" 升级为 "脚本证据 + 非对称判定", 揭示 75% 判错率

**Q2**: 你项目最反直觉的发现是什么?
→ Reranker 和 chunker 价值都比教科书小, 真正瓶颈在 Prompt 层 (LLM 时代特征)

**Q3**: 你改过自己的结论吗?
→ Day 7 TD-7-1 (chunking 丢年份) 被 Day 8 脚本推翻, 真问题是多口径错位

完整话术见 `day08-summary.md` 第 5 节.

---

## 🧪 技术栈

| 类别 | 选型 | 版本/参数 |
|---|---|---|
| 开发环境 | Windows + Git Bash + Python + uv | Python 3.12 |
| LLM | DeepSeek-Chat | temperature=0.1 |
| Embedding | BGE-small-zh-v1.5 | 512 维, CPU |
| 向量库 | Qdrant (Docker) | localhost:6333, Cosine |
| BM25 | BM25Okapi + jieba | 自定义金融词典 175 词 |
| Reranker | BGE-Reranker-base | CPU, max_length=512 |
| PDF 解析 | pdfplumber | 分流 + 伪表格过滤 |
| 评估 | RAGAS (计划) | Day 10-11 接入 |
| 部署 | FastAPI + Streamlit (计划) | Day 18-19 |

---

## 🚀 下一步 (Day 9)

**主题**: 结论审计日 2 — 参数合理性 + L1 召回机制

**Task 清单**:
- Task 0 (开工前 30 min): TD-8-3 验证 Prompt 是否注入 metadata
- Task 1 (3h): 脚本 27 参数审计 (top_k / RRF k / recall_multiplier)
- Task 2 (2h): 脚本 28 L1 召回深度诊断 (Dense/BM25/Hybrid 三路对比)
- Task 3 (1h): 架构缺口盘点
- Task 4 (1.5h): day09-summary.md 8 段笔记

**Day 9 结束应回答**:
1. 检索参数是否最优?
2. Dense 和 BM25 谁主导?
3. Prompt 是否拼了 metadata?
4. 架构缺口有哪些, 紧迫性如何?

---

## 📖 文档索引

| 文档 | 描述 |
|---|---|
| `docs/day02-summary.md` | Day 2: PDF 加载 + 递归切分 |
| `docs/day03-summary.md` | Day 3: BGE embedding |
| `docs/day04-summary.md` | Day 4: Naive RAG 端到端 |
| `docs/day05-summary.md` | Day 5: Hybrid (BM25 + RRF) |
| `docs/day06-summary.md` | Day 6: Multi-Query + HyDE + Router |
| `docs/day07-summary.md` | Day 7: Cross-Encoder Reranker |
| `docs/day08-summary.md` ⭐ | **Day 8: 结论审计日 1, 方法论升级** |
| `docs/project-overview_1-8.md` | 本文档, Day 1-8 完整概览 |

---

**笔记使用原则** (Day 8 新增):
1. 引用任何概念时带一句简短提醒, 不假设读者记得 (如 "TD-7-1 (Day 7 登记的 chunking 边界问题, Day 8 已归档)")
2. 每个关键结论带 "发现过程" 或 "证据链", 不只是结论
3. 技术债归档不删除, 保留认知过程作为资产