> **最后更新**: Day 11 收工 (2026-04-23)


# Enterprise-RAG 项目地图

> **作用**: 项目真实现状的单一事实来源. Day 9 建立, 以后每日收工更新.
> **权威规则**: 本文件 > /mnt/project/ 快照 > Claude 推测 (按 D8 元规则).
> **最后更新**: Day 11 收工 (2026-04-23)

---

## 📂 1. 源码地图

> 每行格式: 路径 — 做什么 (Day X). 接口 / 关键产出.

### 1.1 loaders (PDF → Chunk)

| 路径 | 功能 | 接口 |
|---|---|---|
| `src/loaders/base.py` | `Chunk` dataclass 定义 (Day 2) | `Chunk(content, chunk_type, metadata)` |
| `src/loaders/pdf_loader.py` | PDF 文字与表格分流 + 清洗 (Day 2) | `load_pdf(path) -> list[Document]` |

**关键约定**: 文件名 `{ticker}_{company}_{year}年年度报告.pdf` → parse_filename() 切出 `ticker/company/year` 放进 metadata

### 1.2 chunking

| 路径 | 功能 | 接口 |
|---|---|---|
| `src/chunking/recursive.py` | 递归字符切分 (参考 LangChain, Day 2) | `chunk_documents(docs, size=400, overlap=50) -> list[Chunk]` |

### 1.3 embeddings

| 路径 | 功能 | 接口 |
|---|---|---|
| `src/embeddings/bge.py` | BGE-small-zh 嵌入器 (单例, Day 3) | `BGEEmbedder().encode_query(q) / encode_chunks(chunks)` |

**关键约定**: 维度 512, CPU 运行, `encode_query()` 加 BGE 查询前缀, `encode()` 不加 (用于 HyDE)

### 1.4 retrievers

| 路径 | 功能 | 接口 |
|---|---|---|
| `src/retrievers/qdrant_store.py` | Qdrant 建库/入库/检索 (Day 4) | `get_client() / create_collection() / upsert_points() / search_similar()` |
| `src/retrievers/bm25_store.py` | BM25 + jieba (Day 5) | `BM25Store().build(chunks) / search(query, limit, filters)` |
| `src/retrievers/financial_dict.py` | 金融词典 (175 词, Day 5) | 被 bm25_store 自动加载到 jieba |
| `src/retrievers/hybrid.py` | Dense + BM25 + RRF 融合 (Day 5) | `HybridRetriever(k, recall_multiplier).search(query, top_k, filters)` |

**关键约定** (D7 教训):
- `HybridRetriever.search(top_k)` 的 top_k 不是独立参数, 内部
  `recall_k = top_k × recall_multiplier` 决定两路候选池大小
- **参数扫描时永远扫 recall_multiplier 或 recall_k, 不扫 top_k** (TD-9-2)

### 1.5 query_transformers (查询改写层, Day 6)

| 路径 | 功能 | 接口 |
|---|---|---|
| `src/query_transformers/multi_query.py` | LLM 改写查询为 N 个独立角度子查询 (Day 6) | `MultiQueryRewriter(num_queries, include_original).rewrite(q) -> list[str]` |
| `src/query_transformers/hyde.py` | LLM 生成假设文档做 embedding (Day 6) | `HyDEGenerator().generate(q) -> str` |
| `src/query_transformers/router.py` | LLM 判断查询类型自动选模式 (Day 6) | `QueryRouter().route(q) -> {"use_multi_query", "use_hyde", ...}` |

### 1.6 rerankers (精排层, Day 7)

| 路径 | 功能 | 接口 |
|---|---|---|
| `src/rerankers/bge_reranker.py` | BGE-reranker-base Cross-Encoder (Day 7) | `BGEReranker().rerank(query, candidates, top_k) -> list[dict]` |

**关键约定**: 懒加载 (2 秒冷启动), 用于 AdvancedRAG 的 `use_reranker=True` 场景

### 1.7 generators

| 路径 | 功能 | 接口 |
|---|---|---|
| `src/generators/llm.py` | LLM 调用 + format_context (Day 4) | `generate_answer(question, search_results) -> str`. Prompt header = `"[i] 来源: {company} {year}年报 第{page}页 ({chunk_type})"` |

**关键事实** (Day 9 Task 0 审计确认): **Prompt 已注入 company/year/page/chunk_type metadata**, TD-8-3 闭合.
**Day 10 更新**: `RAG_SYSTEM_PROMPT` 从 5 条规则 (v1) 升级到 7 条 (v3, 472 字符), 新增规则 5/6 多口径处理. v3 是 v2/v3/v4 三轮迭代中数据最优的版本. 详见 `day10-summary.md` 第 4 节.
**Day 11 RAGAS 验证**: v3 Prompt 在 15 U × 3 mode × 45 条 RAGAS 下 Faithfulness 平均 0.90+, 证明 v3 稳健. 专业级拒答样本 (U3 国电南自 / U4 招行) 被 RAGAS AR 判 0.00 但实际是高质量答案, 详见 TD-11-5 和 day11-summary §6.

### 1.8 pipelines

| 路径 | 功能 | 接口 |
|---|---|---|
| `src/pipelines/naive_rag.py` | Naive RAG baseline (Day 4) | `NaiveRAGPipeline(top_k).query(question, filters, top_k)` |
| `src/pipelines/hybrid_rag.py` | Hybrid (Dense + BM25 + RRF) (Day 5) | `HybridRAGPipeline(rrf_k, recall_multiplier).query(...)` |
| `src/pipelines/advanced_rag.py` | **完整** pipeline, 6 mode + Reranker (Day 6 集成, Day 7 加 Reranker) | `AdvancedRAGPipeline().query(question, mode, filters, top_k, use_reranker, rerank_input_n)` |
**Day 11 补丁** (3 个 pipeline 都改): sources 字段加 `full_content` 字段保留完整 chunk 内容, 给 RAGAS `retrieved_contexts` 使用. 原 `preview[:100]` 保留不删.

**Day 12 P0 变更预告**: `advanced_rag.py` 的 full mode 默认 `use_reranker=False` (Day 11 数据确认 Reranker 净负面, TD-7-2 升 P1).



### 1.9 evaluation (评估工具层, Day 10-11)

| 路径 | 功能 | 接口 |
|---|---|---|
| `src/evaluation/answer_check.py` | LLM 答案拒答态检测 (Day 10) | `is_refusal(answer) -> tuple[bool, str \| None]` |
| `src/evaluation/judge_config.py` | RAGAS judge 集中配置 (Day 11) | `build_judge_llm() / describe_judge()`, 读 `OPENAI_JUDGE_*` 3 环境变量 |
| `src/evaluation/pipeline_adapter.py` | pipeline → RAGAS 格式桥接 (Day 11) | `eval_run(question, mode, use_reranker, filters, top_k, rerank_input_n)` |

**关键约定**:
- 8 条 REFUSAL_PATTERNS 来自 Day 9 U5 10 条拒答样本归纳
- 已知局限 (FN-2): 不识别 "拒答句 + 半作答" 拼接场景 (TD-10-3)
- **Day 11 新发现**: 专业级拒答在 RAGAS `answer_relevancy` 下系统砸 0.00 (TD-11-5)
- judge 当前绑定 OpenAI 官方 gpt-4o-mini + text-embedding-3-small, 中转站 embedding 768 维异常 (TD-11-2 已解决)
---

## 🚀 2. Pipeline 状态速查

### NaiveRAGPipeline (Day 4)
- **功能**: Dense only, Top-K 直接进 LLM
- **用途**: baseline, 对比基准
- **当前状态**: 稳定, 未再改动

### HybridRAGPipeline (Day 5)
- **功能**: Dense + BM25 → RRF 融合 → Top-K
- **用途**: 参数审计基线 (Day 9 脚本 27 用这个)
- **当前状态**: 稳定

### AdvancedRAGPipeline (Day 6-7, **当前主力**)
- **功能**: 6 种 mode × 有无 Reranker
- **支持的 mode**:
  - `naive`: 纯 Dense
  - `hybrid`: Dense + BM25 + RRF
  - `multi_query`: Hybrid + Multi-Query (4 子查询 + 原查询, 二次 RRF)
  - `hyde`: Hybrid + HyDE (1 个假答案探针)
  - `full`: Hybrid + Multi-Query + HyDE (5+1 个探针, 二次 RRF)
  - `auto`: LLM Router 判断用哪组
- **Reranker**: `use_reranker=True` 时, 召回 `rerank_input_n=20` 条 → Cross-Encoder 精排 → 取 top_k=5
- **输出字段**: `question / answer / mode_used / routing_decision / probes / rerank_info / sources`
- **当前状态**: Day 11 RAGAS 45 条验证完成. 关键发现:
  - **hybrid** 是最稳 mode (独赢 3 条, 不独砸)
  - **hyde** 对"话题分散" 样本独有价值 (紫金 U5/U7 独赢)
  - **reranker** 对金融 RAG 净负面 (独赢 0 独砸 2). Day 12 P0 禁默认
- **Day 12 计划变更**: full mode `use_reranker=False` 默认, hyde 加 router 规则 (短 query + 有公司 filter → hyde)

---

## 🗓️ 3. 按天关键产出 (Day 1-9)

### Day 1: 环境 + 心智模型
- 搭 uv 环境 + Docker Qdrant + DeepSeek API
- 产出: `docs/01-mental-model.md` (RAG 整体心智模型)

### Day 2: PDF → Chunk
- 实现 pdfplumber 分流 (文字/表格) + RecursiveCharacterTextSplitter
- **关键决策**: chunk_size=400 / overlap=50 (中文语义单元实验选定)
- 产出: `src/loaders/pdf_loader.py` + `src/chunking/recursive.py`

### Day 3: Embedding
- BGE-small-zh 单例加载器, 维度 512
- 产出: `src/embeddings/bge.py` + 相似度实验脚本

### Day 4: Naive RAG 跑通
- Qdrant 入库 (6881 chunks / 8 份年报)
- NaiveRAGPipeline 端到端跑通
- **关键发现**: 数字类查询频繁失败 → Day 5 动机
- 产出: `src/pipelines/naive_rag.py` + `llm.py`

### Day 5: Hybrid (Dense + BM25 + RRF)
- 金融词典 175 词 + jieba 精确切词
- RRF 融合 (k=60 默认)
- 产出: `src/retrievers/bm25_store.py / hybrid.py` + `src/pipelines/hybrid_rag.py`

### Day 6: Advanced 查询改写
- MultiQuery + HyDE + Router
- AdvancedRAGPipeline 集成全部 (6 mode)
- **关键 TD**: TD-6-5 宽召回稀释 (紫金 p.59 案例)

### Day 7: Reranker 精排
- BGE-reranker-base (Cross-Encoder)
- AdvancedRAG 加 `use_reranker` + `rerank_input_n` 参数
- 18 组 A/B (3 查询 × 3 mode × 有无 Reranker)
- **关键发现**: Reranker 不换 Top-1, Top-5 大洗牌不等于答案提升, 延迟代价 4.5-15 秒

### Day 8: 结论审计 1 (诊断方法论升级)
- 3 步法 (PDF grep + Qdrant scroll + Top-20 召回) 重审 Day 4-7 结论
- **推翻 3/4 先验**: U2 L4→L2, U3 L4→L2, U4 L0→L2a
- TD-7-1 (chunking 丢年份) 归因错误 → 归档
- TD-7-3 升 TD-8-2 (多口径错位, P1 Day 10 主战场)
- 新增 TD-8-1 (表格切断 P3)、TD-8-3 (Prompt metadata 待审)、TD-8-4 (chunker 5 种切断形态待量化)

### Day 9: 结论审计 2 (Task 0 + Task 1 + Task 2)
- **Task 0 (闭合)**: TD-8-3 闭合, Prompt 已拼 metadata, 0 项 Day 8 结论需撤回
- **Task 1 (闭合)**: 参数审计. **recall_mult=4 是拐点最优** ✅. top_k / rrf_k 扫描无效 或 不敏感, 维持默认
- **Task 1 意外产出 D7 元教训**: 实验设计错误 > 代码 bug. 自我攻击清单加第 6 步 (参数耦合检查)
- **Task 2 (闭合)**: L1 诊断 + AdvancedRAG 30 次验证. 关键发现:
  - BM25 >> Dense 在金融查询上, RRF 等权融合稀释 BM25 强信号
  - U3 国电净利润真死穴 (子公司/集团口径错位, L2a 扩展)
  - U5 紫金海外真死穴 (Day 6-7 武器库救不回)
  - U6 宁德营收真救回 (LLM 从审计报告复述页答对)
  - check_answer 函数有拒答态假阳性 bug, 手动修正 U5 10 条
- **Task 2 产出 D9 元教训**: 评估函数设计陷阱, 先检测拒答态再判关键词
- **D8 元教训** (Task 2 设计翻车后): `/mnt/project/` 不是权威源, 关键模块必须交叉验证
- **Task 3/5/4 未完成**, 推迟到 Day 9.5 或 Day 10



### Day 10: TD-8-2 Prompt 工程 (三轮迭代)
- **T0 (闭合)**: 新建 `src/evaluation/answer_check.py`, 含 `is_refusal()` + 8 条 REFUSAL_PATTERNS (Day 9 D9 元教训落地)
- **T1/T2 (部分闭合)**: TD-8-2 Prompt v2 → v3 → v4 三轮迭代. 最终定格 v3 (数据支撑 v3 > v4 > v2 > v1)
  - U4 招行: 从 v1 全拒答 → v3 列 3 口径 (集团余额 68.2B / 本行余额 64.0B / 消费信贷 1.52%)
  - U3 国电南自: 三轮 Prompt 开头都带拒答句, **DeepSeek 否定指令遵循弱** (TD-10-1)
  - v4 比 v3 多 19 字但退化到 1 口径, **规则复杂度有 U 型拐点**
- **3 个元洞见**:
  - Prompt 工程不是线性优化, 有过度约束退化拐点
  - DeepSeek 否定指令对 "拒答" 先验的压制力弱
  - `is_refusal` 评估指标 FN-2 盲区 (半作答识别)
- **D10 元教训** (新): 引述性文档也不是权威源, 原始数据 JSON/CSV 才是
- 新增 TD-10-1/2/3/4 四条技术债, 其中 TD-10-3 (is_refusal FN-2) 是 Day 11 RAGAS 前序依赖
- TD-8-2 从 P1 降级到 P2 (Prompt 层已尽力, 继续深挖需 RAGAS 大样本)



### Day 11: RAGAS 评估上线
- **T0-T1 基础设施**: `judge_config.py` + `pipeline_adapter.py` + 3 个 pipeline 的 `full_content` 补丁
- **T1 调试 6 次 API 坑**: ragas 0.4.3 Collections API 类名/参数和老 cookbook 差异大. 锁版本 (TD-11-3)
- **中转站 → OpenAI 官方**: 用户在美国, embedding 从中转 768 维 → 官方 1536 维 (TD-11-2 解决)
- **T2 全量**: 15 U × 3 mode = 45 条, 耗时 28.6 min
- **T3 3 大发现**:
  - Reranker 独赢 0 独砸 2 → **TD-7-2 P3 升 P1** (Day 12 P0 禁默认)
  - HyDE 对紫金类话题分散样本独赢 (U5/U7) → **新认知**, Day 12 加 router 规则
  - 开放题 U13/U15 3 mode 分数完全一致 → "召回收敛" 现象
- **元规则 × 3**:
  - **D11**: 每份 summary 最末节必须有术语对照 (4 子表)
  - **D12**: 评估集 < 20 条时 mode 排名不稳定, 需 2-3 轮不同 U 组合验证
  - **D13**: 评估指标是变量不是真理 (RAGAS AR 对金融 RAG 拒答失效)
- **新 TD**: TD-11-2 (已解决) / TD-11-3 (锁版本) / TD-11-4 (generator × mode 未验证) / TD-11-5 (AR 失效)

---






## 📊 4. 技术债实时图谱

### P1 (当下必修)
| ID | 内容 | 来源 | 状态 |
|---|---|---|---|
| **TD-7-2** ⬆️ | Reranker 对金融 RAG 净负面 (独赢 0 独砸 2). 升级自 P3 | Day 7 登记 → Day 11 数据升级 | **Day 12 P0**: `advanced_rag.py` full mode 默认禁 Reranker |
| **TD-10-3** | `is_refusal` FN-2 盲区: 不识别"拒答句 + 半作答" 拼接 | Day 10 v3/v4 U4 | Day 11 数据显示 FN-2 候选 8 条, Day 12 P3 人工抽检后决定是否改多分类 |
| **TD-9-1** | 检索层评估方法局限 (6 样本 / 页级金页) | Day 9 Task 1+2 | Day 11 RAGAS 大样本部分解决 (45 条), 但"金页 vs chunk 精度"未解, Day 14 扩库再议 |

### P2 (Day 12-14 排)
| ID | 内容 | 来源 | 状态 |
|---|---|---|---|
| **TD-11-4** 🆕 | generator (DeepSeek V3.2) 对 mode 差异的影响未验证 | Day 11 T3 用户追问 | Day 12 专项: 12 U × hybrid × {DeepSeek, gpt-4o-mini} |
| **TD-11-5** 🆕 | RAGAS `answer_relevancy` 对金融 RAG 专业级拒答系统失效 (答 0.00) | Day 11 T3 | Day 12+ 自定义金融 RAG metric (`AspectCritic` prompt 或 `is_professional_refusal`) |
| **TD-10-1** | DeepSeek 否定指令对 "拒答" 先验压制力弱 | Day 10 v3/v4 | Day 11 RAGAS 数据间接印证 (U3 三 mode 都带拒答句开头), 但非主要瓶颈 |
| **TD-10-2** | hyde mode 召回对金融数字类 query 有负面偏置 | Day 10 U3 | **Day 11 修正**: hyde 对数字题和 hybrid 一致, 对话题分散题反而救回. 考虑降级 P3 |
| **TD-8-2** ⬇️ | 多口径错位. Day 10 Prompt v3 部分闭合 (U4 救一半) | Day 7 升 Day 10 部分闭合 | Day 11 U4 hybrid 给出专业多口径, TD-8-2 事实上已基本闭合 (hybrid 层), Day 14 再审 |
| **TD-8-4** | chunker 5 种切断形态未全量量化 | Day 8 | Day 14 扩库时处理 |

### P3 (观察 / 不专项修)
| ID | 内容 | 来源 | 状态 |
|---|---|---|---|
| **TD-11-3** 🆕 | RAGAS 0.4.3 Collections API 兼容性 6 次坑 | Day 11 T1 | 锁 ragas==0.4.3, Day 18+ FastAPI 时再评估升级 |
| **TD-10-4** | 多规则打架优先级 (规则 2 vs 6) | Day 10 v2 → v3 | Day 11 RAGAS 大样本未再现, 观察 |
| **TD-6-5** | 宽召回稀释 (紫金 p.59) | Day 6 | Day 11 发现 hyde 能救回 (TD-6-5 部分缓解), 但召回机制未变 |
| **TD-8-1** | 表格切断个案 (4.5% 财务关键词) | Day 8 | Day 11 U9 表格题 3 mode 都拒答印证, 但非主要瓶颈 |
| **TD-9-2** | top_k 和 recall_mult 架构耦合 | Day 9 D7 | 文档约束, 不改架构 |
| **TD-9-3 候选** | RRF 等权融合对金融场景不友好 | Day 9 Task 2 Phase 1 | Day 11 ContextPrecision 未在融合参数上显著差异, 维持观察 |

### 已解决 (Day 11)
| ID | 原内容 | 解决方式 |
|---|---|---|
| **TD-11-2** | 中转站 embedding 返 768 维疑似替换 | 切 OpenAI 官方 (用户在美国), 直接 1536 维解决 |

### 已归档 (历史认知过程, 保留不删)
| ID | 原内容 | 归档原因 |
|---|---|---|
| **TD-4-1** | 招行表格数据缺失 | Day 5 推翻 (Hybrid 能答出) |
| **TD-7-1** | chunking 丢年份锚点 (U4 招行) | Day 8 推翻 (真实根因是 L2a 多口径错位, TD-8-2) |
| **TD-8-3** | Prompt 是否注入 metadata 待审 | Day 9 Task 0 闭合 (已注入) |







## 🎯 5. 死穴样本登记 (Day 9 Task 1 发现)

| ID | 查询 | 金页数 | Day 9 现状 | 推测修法 |
|---|---|---|---|---|
| U3 | 国电南自 2024 净利润 | 9 | **top_k=15 仅 1/9 命中**, 所有参数都救不了 | Day 10 优先跑 AdvancedRAG auto/full 模式 |
| U6 | 宁德时代 2025 营业收入 | 6 | top_k=10 全 0 命中, top_k=15 仅 2/6 | 同 U3, 且因 query 过泛可能需 Multi-Query |
| U5 | 紫金海外业务营收和风险 | 9 | 所有参数 1/9, recall_mult≥6 掉到 0 (TD-6-5) | Day 10 验证 Reranker + HyDE 能否救回 p.59 |

---

## 📁 6. 文档地图

### 项目元文档
- `docs/project_map.md` (本文件) — 项目地图
- `docs/01-mental-model.md` (Day 1) — RAG 心智模型
- `docs/02-loader-design.md` (Day 2) — PDF loader 设计

### 日报 (day-summary 系列)
- Day 2-8 每天一份 `docs/day0X-summary.md`
- Day 9 尚未写 (Task 4 待做)

### 项目概述文档
- `docs/project-overview_1-6.md` — Day 1-6 总览
- `docs/project-overview_7.md` — Day 7 Reranker 总览
- `docs/project-overview_1-8.md` — Day 1-8 总览

### Day 9 专项产出
- `scripts/27_audit_retrieval_params.py` — Task 1 参数审计主脚本
- `scripts/debug_topk.py` — Task 1 bug 诊断脚本 (D7 产出)
- `docs/day09-td83-result.md` — Task 0 结论
- `docs/day09-gold-pages.final.json` — Task 1 人工复核金页集
- `docs/day09-param-audit.txt` — Task 1 原始报告 (保留作过程资产)
- `docs/day09-param-audit-patch.md` — Task 1 **修正后权威结论**
- `docs/day09-3way-cache.json` — Task 1 三路缓存 (Task 2 复用)
- `docs/day09-decisions.md` — Day 9 决策追溯 (D1-D9), **Day 10 追加 D10 段**


### Day 10 专项产出
- `scripts/29_test_prompt_v2.py` — T2 U3/U4 Prompt 验证脚本 (含 is_refusal 集成)
- `docs/day10-summary.md` — Day 10 完整笔记 (8 段结构, Prompt 三轮迭代 + 3 元洞见)
- `docs/day09-decisions.md` D10 段追加 — "引述文档不是权威源" 元教训
- `src/evaluation/answer_check.py` 新模块 (见第 1.9 段)
- ### Day 11 专项产出
- `src/evaluation/judge_config.py` — RAGAS judge 集中配置 (新模块)
- `src/evaluation/pipeline_adapter.py` — pipeline → RAGAS 格式桥接 (新模块)
- `src/pipelines/{naive_rag,hybrid_rag,advanced_rag}.py` — 补丁 3 个, 加 `full_content` 字段
- `scripts/30_ragas_smoke.py` — T0 smoke test (U1 茅台硬编码样本)
- `scripts/31_ragas_eval.py` — T2 主评估脚本 (15 U × 3 mode)
- `docs/day11-eval-queries.json` — 15 条 U 定义 (含 reference_answer + confidence 标注)
- `docs/day11-ragas-results.jsonl` — 45 条完整 trace
- `docs/day11-ragas-summary.md` — RAGAS 脚本自动生成聚合表
- `docs/day11-summary.md` — Day 11 正式总结 (10 段结构, 含 §9 术语对照首次落地)
- `docs/day09-decisions.md` D11/D12/D13 段追加 + Day 11 TD 溯源 + Day 11 产出清单

---





## 🔧 7. 入口与工具现状

- `main.py`: **仅含 `print("Hello from enterprise-rag!")`**, 未实现 CLI / 服务入口
- `tests/`: 未建立
- FastAPI 服务: 未建 (Day 18 计划)
- Streamlit 前端: 未建 (Day 19 计划)
- RAGAS 评估: **已建立** (Day 11, `scripts/31_ragas_eval.py`), 45 条数据入库 `docs/day11-ragas-results.jsonl`
- Docker Qdrant: 运行中 (6881 chunks / 8 年报已入库)

---

## 🔒 8. 权威源使用规则 (D8 元规则)

### 本文件是项目现状的权威
- Claude 做下游判断 (设计任务 / 写代码 / 评估技术债) 前, 先读本文件
- `/mnt/project/` 是 Claude 能读的项目快照, **不一定等于用户本地代码**
- 发现 `/mnt/project/` 和 `project_knowledge_search` 不一致, 用户拍板后更新本文件
### D10 延伸 (Day 10 新增): 引述文档也不是权威源
- 引述性 markdown (day0N-summary / day0N-decisions / day0N-l1-diagnosis) 是对原始数据的**解读**, 解读会选择性截取
- 原始数据 (JSON / CSV / pkl / 脚本输出 txt) 才是事实源
- 下游判断前: 原始数据 > 引述文档 > Claude 推测
### 更新时机
- **每日收工** (或任务完成时): 更新第 3 段"按天关键产出"
- **技术债变化时**: 更新第 4 段
- **新模块加入时**: 更新第 1 段
- **Pipeline 迭代时**: 更新第 2 段

### 更新原则
- **只写"当前真相"**, 不写历史过程 (过程记录在 day-summary / decisions)
- **有变动就加 ⚠️ 让 Claude / 用户注意**
- **保持单一视图**: 不和 day-summary 重复内容, 不和 decisions 重复决策

---

**文档状态**: Day 9 初稿, Day 10/11 连续维护. Day 12+ 每日收工继续更新.