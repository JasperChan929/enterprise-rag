# Day 11: RAGAS 评估上线

## 目标与产出

Day 11 主题: 用 RAGAS + gpt-4o-mini judge 对现有 RAG 系统做**首次业界标准量化评估**, 覆盖 15 条金融查询 × 3 种检索 mode = 45 条端到端数据.

交付物:
- `src/evaluation/judge_config.py` - judge 集中配置
- `src/evaluation/pipeline_adapter.py` - pipeline → RAGAS 格式适配
- `src/pipelines/{naive_rag,hybrid_rag,advanced_rag}.py` 补丁 3 个 (加 full_content 字段)
- `scripts/30_ragas_smoke.py` - T0 smoke test
- `scripts/31_ragas_eval.py` - T2 主评估脚本
- `docs/day11-eval-queries.json` - 15 条 U 定义
- `docs/day11-ragas-results.jsonl` - 45 条完整 trace
- `docs/day11-ragas-summary.md` - 自动生成聚合表

---

## 1. 评估集设计

### 1.1 15 条 U 构成

按 query_type 分布:

| 类型 | U | 目的 |
|---|---|---|
| refusal_expected | U1, U11 | 真无答案, 测拒答基线 |
| numeric_simple | U6, U12 | 精确数字, baseline |
| qualitative | U2, U5, U7 | 定性开放 |
| multi_perspective | U3, U4 | 多口径错位 (TD-8-2) |
| multi_chunk_reasoning | U8 | 跨 chunk 聚合 |
| table_extraction | U9 | 表格保留 (TD-8-1) |
| synonym_recall | U10 | 金融词典 BM25 覆盖 |
| hyde_friendly | U13 | 给 hyde 发光机会 |
| reranker_friendly | U14 | 给 reranker 发光机会 |
| full_friendly | U15 | 给 full 发光机会 |

### 1.2 评估集演化 (评估方法学的元学习)

U 集合不是一次定稿, 跑了 2 轮:
- **第 1 轮 12 条**: 偏向覆盖已知 TD, 结果 50-83% 拒答率, 数据信号弱
- **第 2 轮 +3 条 (U13/U14/U15)**: 针对 3 个 mode 各加 1 条"理论擅长" 样本
- **核心发现**: **加了 3 条 U 后, hyde 从聚合"垫底" 变"第一"**. 小样本 mode 排名不稳定

---

## 2. 评估基础设施

### 2.1 judge 与 embedding 配置

- **judge**: `gpt-4o-mini` @ OpenAI 官方
- **embedding**: `text-embedding-3-small` @ OpenAI 官方 (1536 维)
- **环境**: 从中转站切到 OpenAI 官方 (用户在美国), TD-11-2 解决

### 2.2 RAGAS 3 metric

| metric | 类别 | 测什么 |
|---|---|---|
| Faithfulness | LLM only | 答案事实是否有 context 支持 |
| AnswerRelevancy | LLM + embedding | 答案是否回应问题 (反推问题 cosine) |
| ContextPrecisionWithoutReference | LLM only | 检索 chunk 对问题是否相关 |

### 2.3 T1 阶段 6 次 API 兼容性坑

从报错到修通, 一一记录 (Day 12+ 如升级 RAGAS 版本参考):

1. `ResponseRelevancy` 类不存在 → 改 `AnswerRelevancy`
2. `LLMContextPrecisionWithoutReference` 类不存在 → 改 `ContextPrecisionWithoutReference`
3. `AnswerRelevancy` 缺必填 `embeddings`
4. 中转 embedding 返 768 维, OpenAI 官方 1536 维 (TD-11-2)
5. `AnswerRelevancy.ascore()` 不接受 `retrieved_contexts`, 按 metric 分别传参
6. embedding client 必须 `AsyncOpenAI`, 非 `OpenAI` (0.4.3 内部调 `aembed_text`)

---

## 3. 45 条数据核心结果

### 3.1 聚合平均表 (仅作参考, 不代表真实差异)

| mode | faithfulness | answer_relevancy | context_precision | refusal_rate |
|---|---|---|---|---|
| hybrid | 0.908 | 0.216 | 0.611 | 33% |
| hyde | 0.934 | 0.365 | 0.692 | 27% |
| full+reranker | 0.930 | 0.256 | 0.484 | 60% |

### 3.2 mode 胜率表 (真实诊断信号)

15 条 U 里每条哪个 mode 独赢:

| mode | 独赢 | 并列赢 | 平 | 独砸 |
|---|---|---|---|---|
| hybrid | 3 (U8/U14 + 并列) | 1 (U4) | 7 | **0** |
| hyde | **2 (U5/U7)** | 1 (U4) | 7 | 0 |
| reranker | **0** | 0 | 7 | **2 (U4/U8)** |

**定义**: 独赢 = 该 mode 答对, 其他 mode 都拒或明显砸. 独砸 = hybrid 答对但该 mode 变拒答.

---

## 4. 核心发现

### 4.1 Reranker 对金融 RAG 净负面 (TD-7-2 P3 升 P1)

**证据**: Reranker 独赢 0 条, 独砸 2 条 (U4/U8).

**U4 例证**:
- hybrid: 专业多口径答案 (本公司 0.93% / 本集团 0.94%)
- Reranker: 366 字拒答 "未提供...直接数据"

**机制假设**: BGE-Reranker 训练数据偏通用知识问答, 在金融场景把"关键词密度高但信息少" 的分析页顶上来, 挤掉真正给答案的表格行.

**代价**: 23 秒延迟 × 每次查询, 换 0 救回样本 + 2 次破坏.

**Day 12 行动**: full mode 默认禁 Reranker, 保留 API 可选.

### 4.2 HyDE 对"话题分散" 样本有独特救回 (新发现)

**证据**: U5 (紫金海外业务) / U7 (紫金 ESG) 两条死穴, **只有 hyde 答对**.
- U5 hyde: fait=0.86 answ=0.67 cont=1.00
- U7 hyde: fait=1.00 answ=0.64 cont=1.00

**机制假设**: 紫金年报相关内容散落多章节 (经营讨论 / 投资分析 / 风险提示). hybrid 的 BM25+向量集中召回第一相关章节, hyde 生成的"假设答案"包含答案应有的关键词 (如"地缘政治" / "碳排放"), 向量更接近真实答案段落, 召回更分散的相关章节.

**适配场景**:
- ❌ 精确数字查询 (U6/U12 hyde 和 hybrid 完全一致)
- ✅ 话题分散 + 开放语义 + 多章节聚合

**Day 12 行动**: router 加规则 "短 query (<15 字) + 开放词汇 + 有公司 filter → 路由 hyde".

### 4.3 开放题 3 mode 分数精确相同

U13 (紫金战略转型) 和 U15 (比亚迪竞争优势) 3 mode 的 fait/answ/cont **三个值完全一致**:
- U13: 全部 fait=1.00 answ=0.65 cont=1.00
- U15: 全部 fait=1.00 answ=0.47 cont=1.00

**假设**: 对开放题, 3 种 mode 召回的 Top-5 高度重合, 差异被 LLM 抹平.

**对金融 RAG 的启示**: 开放题上 mode 选择不重要, 只要有一个能答, 3 mode 往往都能. mode 价值差异**只在难题上显现**.

### 4.4 RAGAS 指标对金融 RAG 拒答场景不适配

**AnswerRelevancy 的金融 RAG 盲区**: 45 条里 answ=0.00 有 26 条, 其中多数是**专业级诊断性拒答** (U3/U4 等). 例如 U3:

> "根据已有资料无法回答. 参考资料提供审计范围[4], 但未提供合并/母公司口径数据. 仅含子公司数据[2]."

这是金融分析师最想看的答案质量, 但 AnswerRelevancy 判 0.00. 原因: judge 反推问题 "资料里没有什么" 和原查询 cosine 低.

**结论**: answer_relevancy 在金融 RAG 拒答样本上**几乎无诊断价值**. Day 12+ 考虑自定义金融 RAG metric.

**Faithfulness 边缘崩溃**: U14 hyde/reranker 2 条 `fait=ERR`, 45 条中占 4.4%. judge 对超长/结构化答案解析偶发失败. Day 12 不修 judge, 但锁 RAGAS 版本.

**ContextPrecision 最可靠**: 3 mode 间差异最明显, 是唯一能信的 mode 对比指标.

---

## 5. Day 11 元学习 (比单一数据发现更重要)

### 5.1 小样本评估不稳定

- 12 条版结论: hyde 垫底, Reranker 偶尔有用
- 15 条版结论: hyde 有独特价值, Reranker 完全无效

加 3 条 U 颠覆上一轮结论. **原则**: 评估集 < 20 条时, 单次评估不足以下 mode 排名. Day 14 扩库要扩评估集到 30+ 条, 并跑 2-3 次不同 U 组合验证稳定性.

### 5.2 评估指标也是变量

Day 1-10 默认 is_refusal 二分即可, Day 11 发现"专业诊断性拒答" 被所有 RAGAS metric 误判. Metric 选择本身是评估设计决策, 不是自动成立的客观真理.

### 5.3 生成器未作对照变量

Day 11 固定 DeepSeek V3.2 作生成器 (不是老模型, 是当前主力, 水平约 gpt-4o-mini). 但未做 generator × mode 二维实验, 无法排除"mode 差异被生成器能力上限限制" 可能. Day 12 P2 实验补齐.

---

## 6. TD 登记

### 新增

| TD | 描述 | 优先级 |
|---|---|---|
| TD-11-2 | 中转站 embedding 768 维疑似替换 | **已解决** (切 OpenAI 官方) |
| TD-11-3 | RAGAS 0.4.3 Collections API 兼容性坑 6 次 | P3 (锁版本即可) |
| TD-11-4 | generator 对 mode 差异的影响未验证 | P2 (Day 12 专项) |
| TD-11-5 | RAGAS answer_relevancy 对金融 RAG 拒答失效 | P2 (Day 12+ 自定义 metric) |

### 升级

| TD | 变化 | 原因 |
|---|---|---|
| TD-7-2 Reranker 边际价值存疑 | **P3 → P1** | Day 11 确认净负面, 独赢 0 独砸 2 |

---

## 7. Day 12 工作建议

**P0 必做**:
- `advanced_rag.py` 里 full mode 默认 `use_reranker=False`
- 更新 `project_map.md` 记录 Day 11 交付

**P1 必做**:
- router 加 hyde 规则: query < 15 字 + 有公司 filter + 无数字 → 路由 hyde
- 保持现有 hybrid / full 为 fallback

**P2 该做**:
- generator 对比实验: 12 条 U × hybrid mode × {DeepSeek V3.2, gpt-4o-mini} = 24 条, 看 generator 能力对答案质量影响

**P3 可做**:
- FN-2 候选 8 条人工抽检, 估 TD-10-3 污染度, 决定 is_refusal 是否改多分类

**不做**:
- 扩 U 集合 (Day 14 统一扩)
- 换 RAGAS 版本 (锁 0.4.3)
- 重写 metric (Day 14+ 再考虑)

---

## 8. 时间账 / 成本账

- **总耗时**: 约 5.5 小时 (14:30 - 20:00)
- **评估全量跑**: 28.6 分钟 (45 条)
- **OpenAI judge 费用**: 约 $0.3-0.5 (未精确账单)
- **T1 调试 RAGAS API 耗时**: 约 2.5 小时 (6 次兼容性坑)
- **数据解读 + 评估集迭代**: 约 2 小时

---

## 9. 术语对照 (D11 元规则落地)

### 9.1 缩写 / 代号

| 缩写 | 全称 | 含义 |
|---|---|---|
| RAGAS | Retrieval-Augmented Generation Assessment | RAG 系统业界标准评估框架, 开源 |
| FN-2 | False Negative type 2 | Day 10 TD-10-3 子类: is_refusal=True 但答案含有效信息的"半拒答+半作答" |
| T0/T1/T2/T3 | Test 分阶段 | Day 11 内部分: T0 smoke, T1 开发, T2 全量, T3 解读 |
| U / U1-U15 | User query | 评估集中的单条查询样本 |
| TD | Technical Debt | 技术债编号, 从 Day 1 起维护 |

### 9.2 当天新造概念

| 概念 | 定义 |
|---|---|
| 专业级拒答 | 答案形式是拒答但内容含"找到什么/缺什么/部分信息在哪" 的诊断性分析, 金融分析师视角的高质量答案 |
| 独赢 / 独砸 / 并列赢 | Day 11 mode 胜率表的单位. 独赢=该 mode 答对, 其他都拒; 独砸=hybrid 答对但该 mode 变拒答; 并列=多个 mode 都答对 |
| 召回收敛 | 45 条数据中 U13/U15 的 3 mode 分数完全一致现象. 假设: 开放题上 3 mode 召回的 Top-5 高度重合 |
| hyde_friendly / reranker_friendly / full_friendly | 评估集设计时给 3 种 mode 各加 1 条"理论擅长" 样本, 测 mode 差异 |
| 评估集不稳定性 | 15 条 U 下, 加 3 条可以颠覆"mode 排名" 结论. 小样本评估方法学缺陷 |

### 9.3 首次外部技术术语

| 术语 | 本次引入场景 |
|---|---|
| Faithfulness | RAGAS 3 metric 之一: 答案事实是否有 context 支持 |
| AnswerRelevancy | RAGAS 3 metric 之一: 答案反推问题与原 query 相似度 |
| ContextPrecisionWithoutReference | RAGAS 3 metric 之一: 检索 chunk 对 query 相关性 |
| Collections API | RAGAS 0.4.x 引入的 metric 组织方式, v0.3 的 `ragas.metrics` 将被弃用 |
| LLM-only metric / Embedding-dependent metric | RAGAS 指标分类: 前者 (Faithfulness/ContextPrecision) 只用 LLM, 后者 (AnswerRelevancy) 需要 embedding |
| embedding_factory / interface="modern" | RAGAS 0.4.3 构造 embedding client 的工厂函数 |
| BGE-Reranker-base | BAAI 开源重排序模型, Day 7 部署时使用, Day 11 确认对金融 RAG 净负面 |

### 9.4 项目内部专属术语

| 术语 | 含义 |
|---|---|
| 金页 (day09-gold-pages_final.json) | Day 9 人工标注的 U1-U6 "最权威答案来源页", PDF 级真值 |
| pipeline_adapter | Day 11 T1b 产出的桥接层, 包装 AdvancedRAGPipeline.query() 输出成 RAGAS `{user_input, response, retrieved_contexts}` 格式 |
| full_content 补丁 | Day 11 T1 对 3 个 pipeline 的 sources 字段添加完整 chunk 内容, 避免 RAGAS 读 preview[:100] 截断 |
| 紫金死穴 | Day 6 起 U5 紫金海外业务查询始终召回不到 p.59 的长期问题, Day 11 发现 hyde 能救 |
| D11 元规则 | Day 11 定的"每份 day-summary 必须有术语对照段" 规则. 本节即落地 |

---

## 10. 元反思

Day 11 最有价值的不是 45 条数据本身, 而是**多次评估结论翻转的过程**:

- 跑 12 条说 "Reranker 偶尔有效 / hyde 垫底"
- 跑 15 条说 "Reranker 完全无效 / hyde 有独特价值"
- 解读 36 条聚合 mean 说 "3 mode 差别不大"
- 换胜率表说 "hyde 独赢 2 条, Reranker 独砸 2 条"

**同一批系统, 不同评估设计得出不同结论**. 评估不是客观的"测量", 是有主观决策的"设计". Day 11 完成了从 "用 RAGAS 评估" 到 "评估 RAGAS 评估是否评估了真正想测的东西" 的跃迁.

这份元认知是项目从 Day 1 到 Day 11 最重要的积累, 也是 Day 12+ 做任何"改进" 前必须先确认"改进会不会只是优化评估指标而非系统本身" 的前提.