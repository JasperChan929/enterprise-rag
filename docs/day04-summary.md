# Day 4 完整笔记: Naive RAG 端到端跑通
# 代码模块
src/retrievers/
├── __init__.py              ← 空文件
└── qdrant_store.py          ← Qdrant 操作封装
                                - get_client(): 获取连接
                                - create_collection(): 建库
                                - chunks_to_points(): Chunk → PointStruct
                                - upsert_points(): 批量写入
                                - search_similar(): 语义检索(支持 payload 过滤)

src/generators/
├── __init__.py              ← 空文件
└── llm.py                   ← LLM 生成模块
                                - get_llm_client(): OpenAI 兼容客户端
                                - format_context(): 检索结果 → 参考资料文本
                                - generate_answer(): 组装 Prompt + 调用 LLM
                                - RAG_SYSTEM_PROMPT: 系统 Prompt(5 条规则)
                                - RAG_USER_TEMPLATE: 用户 Prompt 模板

src/pipelines/
├── __init__.py              ← 空文件
└── naive_rag.py             ← Naive RAG Pipeline
                                - NaiveRAGPipeline 类
                                - query(): 问题 → 向量化 → 检索 → 生成 → 答案

# 实验脚本
scripts/
├── 11_test_qdrant.py        ← Qdrant 连通性测试(建库/插入/检索/清理)
├── 12_ingest_all.py         ← 4 份年报全量入库(加载→切分→向量化→写入)
└── 13_test_naive_rag.py     ← 端到端 Naive RAG 测试(4 个问题)

# 基础设施
Docker: qdrant/qdrant 容器运行中
  - HTTP API: localhost:6333
  - gRPC: localhost:6334
  - 数据持久化: qdrant_storage/
  - Web UI: localhost:6333/dashboard

Qdrant Collection: financial_reports
  - 6881 points
  - 512 维, Cosine 距离
## 0. 本日目标

> 把 Day 2 的 Chunks + Day 3 的 Embedding 接入向量数据库和 LLM,
> 组装成一个能"问问题拿到答案"的完整 Naive RAG 系统。
> 这是整个项目的 baseline,后续所有优化都和它对比。

---

## 1. 当日产出清单

### 代码模块
- `src/retrievers/qdrant_store.py` — Qdrant 操作封装(建库/入库/检索/过滤)
- `src/generators/llm.py` — LLM 调用 + RAG Prompt 模板
- `src/pipelines/naive_rag.py` — 端到端 Naive RAG Pipeline

### 实验脚本
- `scripts/11_test_qdrant.py` — Qdrant 连通性测试
- `scripts/12_ingest_all.py` — 4 份年报全量入库
- `scripts/13_test_naive_rag.py` — 端到端 RAG 测试(4 个问题)

### 关键数据
- 4 份年报: 茅台 2022/2023 + 宁德时代 2025 + 招商银行 2025
- 总 points 数: 6881
- 向量维度: 512, 距离度量: Cosine
- LLM: DeepSeek-Chat (temperature=0.1)

---

## 2. 完整数据流

```
离线建库:
  4 份 PDF
    → Loader (分流 + 去重 + 清洗)
    → Chunker (递归切分, 400 tokens)
    → BGE Embedder (批量向量化)
    → Qdrant (6881 points, 含 payload metadata)

在线问答:
  用户问题
    → BGE encode_query (加前缀)
    → Qdrant query_points (Top-5 + payload 过滤)
    → Prompt 组装 (参考资料 + 规则 + 问题)
    → DeepSeek-Chat (temperature=0.1)
    → 带引用的答案
```

---

## 3. 核心知识点

### 3.1 向量数据库解决的 4 个问题

Day 3 用 numpy 数组做检索虽然能跑,但有 4 个根本缺陷:

| 问题 | numpy 数组 | 向量数据库 |
|---|---|---|
| 持久化 | 重启就没 | 磁盘持久化 |
| 扩展性 | O(N) 暴力遍历 | O(log N) ANN 索引 |
| 过滤 | 自己写 if-else | payload 原生过滤 |
| 并发 | 不支持 | 生产级并发 |

### 3.2 ANN 算法(近似最近邻)

核心思想: 牺牲一点精度(1-2%),换百倍速度提升。

HNSW (Hierarchical Navigable Small World):
- 分层图索引,高层稀疏粗筛,低层密集精查
- 类比: 在中国地图上找离上海最近的咖啡店,
  先锁定长三角(粗筛) → 上海市(细查) → 具体街道(精确)
- 从 O(N) 降到 O(log N)

面试一句话版: "HNSW 是分层图索引,通过多级粗筛到精查,
把向量检索从 O(N) 降到 O(log N),精度损失 1-2%。"

### 3.3 Qdrant 核心概念

```
Qdrant 实例
└── Collection (集合 ≈ 数据库的"表")
    ├── 配置: vector_size=512, distance=COSINE
    └── Point (一条记录 ≈ 数据库的"行")
        ├── id:      UUID
        ├── vector:  [512 维浮点数]
        └── payload: JSON 元数据
```

payload 设计(金融年报场景):
```json
{
  "content":    "chunk 原文",
  "chunk_type": "text | table",
  "company":    "贵州茅台",
  "stock_code": "600519",
  "year":       2023,
  "page":       6
}
```

payload 支撑的能力:
- 按公司过滤: `company == "贵州茅台"`
- 按年份过滤: `year == 2023`
- 权限隔离: A 部门只能查 A 的数据

### 3.4 Qdrant 选型理由(面试答辩)

1. Rust 写的,性能强,内存省(比 Milvus 省 30-50%)
2. API 干净,Python 客户端好用,社区活跃
3. 天然支持 Hybrid Search(Dense + Sparse),Day 5 无缝衔接

不选 Milvus: 本地部署要 4 个 Docker 容器,配置复杂。
不选 Chroma: 太简单,无生产级特性,面试说不出选型理由。

### 3.5 RAG Prompt 工程(每一句都有目的)

```python
RAG_SYSTEM_PROMPT = """你是一个专业的金融文档问答助手。
请严格基于下面提供的【参考资料】回答用户问题。

规则：
1. 只使用参考资料中的信息回答，不要使用你自己的知识
2. 如果参考资料中没有足够信息回答问题，请明确说"根据已有资料无法回答此问题"
3. 回答中用 [1][2] 等标注引用了哪条参考资料
4. 涉及数字时必须精确引用原文数据，不要四舍五入或估算
5. 保持专业、简洁"""
```

| 指令 | 解决的问题 |
|---|---|
| "只使用参考资料" | 抑制 LLM 调用内部记忆,减少幻觉 |
| "没有信息就说无法回答" | 兜底出口,避免强行编造 |
| "用 [1][2] 标注引用" | 答案可追溯,用户能验证 |
| "数字必须精确,不要估算" | 金融场景核心要求 |
| temperature=0.1 | 低温 = 确定性高,RAG 不需要"创造力" |

### 3.6 Naive RAG Pipeline 架构

```python
class NaiveRAGPipeline:
    def query(self, question, filters=None, top_k=5):
        # Step 1: 查询向量化
        q_vec = self.embedder.encode_query(question)

        # Step 2: Qdrant 检索 Top-K
        results = search_similar(client, q_vec, filters=filters, limit=top_k)

        # Step 3: 组装 Prompt + LLM 生成
        answer = generate_answer(question, results)

        return {"question": question, "answer": answer, "sources": results}
```

这就是 Naive RAG 的全部——查询 → 向量化 → 检索 → 生成。
没有查询改写、没有 Hybrid Search、没有 Reranker。

---

## 4. Naive RAG 测试结果与诊断

### 4.1 测试成绩单

| 问题 | 类型 | 检索质量 | 答案质量 | 根因分析 |
|---|---|---|---|---|
| 茅台 2023 营收 | 精确数字 | 中(Rank 2 命中) | 对(LLM 兜底) | 表格语义信号弱 |
| 宁德核心竞争力 | 定性 | 优(5/5 命中) | 优(结构化回答) | 查询和文档用词接近 |
| 招行不良贷款率 | 精确数字 | 高(页命中) | 拒答 | 表格截断,数值丢失 |
| 茅台未来规划 | 开放性 | 优(多来源) | 优(4 点+引用) | 叙述性段落丰富 |

### 4.2 最有价值的发现

**发现 1: 招行"智能拒答"**

LLM 回答: "根据已有资料无法回答...表格标题包含了'不良贷款率%'的列,
但具体数值在提供的文本片段中缺失。"

这个回答非常优秀:
- 没有编造数字(prompt 兜底策略生效)
- 精准诊断了数据缺失原因(LLM 真的在读 chunks)
- 暴露了 Day 2 的技术债(超长表格被 BGE 截断)

面试金句: "RAG 的故障点不一定在检索排序上,也可能在上游的数据加工环节——
如果 chunk 本身就残缺,检索再准也没用。"

**发现 2: 茅台营收"侥幸答对"**

Top-5 里 Rank 1 不是营收数据(是经营计划),
Rank 2 才是审计表格里的数字。LLM 从 5 条里挑出了正确的那条。

这说明: LLM 的阅读理解能力在"兜底",但这不是可靠方案——
如果 Top-5 里一条相关的都没有,LLM 就无能为力了。
检索质量才是天花板。

**发现 3: metadata 过滤的实际效果**

"茅台 2023 营收"不带过滤 vs 带 `company=贵州茅台` 过滤:
这次碰巧结果一样(Top-3 都是茅台),但如果查"2023 年营收"
(不提公司名),不带过滤会混入宁德/招行内容。

### 4.3 Naive RAG 的规律总结

| 问题类型 | Naive RAG 表现 | 为什么 |
|---|---|---|
| 定性/开放性 | 优 | 大段叙述文本,语义检索擅长 |
| 精确数字(文字中) | 中 | 能命中但排名不稳定 |
| 精确数字(表格中) | 差 | 表格文字密度低 + 可能被截断 |
| 专有名词精确匹配 | 差 | 稠密检索对细微术语区分弱 |

---

## 5. 关键代码片段

### 5.1 Qdrant 入库

```python
from qdrant_client.models import PointStruct
import uuid

def chunks_to_points(chunks, vectors):
    points = []
    for i, chunk in enumerate(chunks):
        payload = {
            "content": chunk.content,
            "chunk_type": chunk.chunk_type,
            **chunk.metadata,
        }
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vectors[i].tolist(),
            payload=payload,
        ))
    return points
```

设计要点:
- UUID 保证全局唯一(多次入库不冲突)
- content 存进 payload(检索后直接拿到原文)
- metadata 展开存(支持 payload 过滤)

### 5.2 带过滤的检索(新版 API)

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue

response = client.query_points(
    collection_name="financial_reports",
    query=query_vector,       # 注意: 参数名是 query 不是 query_vector
    query_filter=Filter(must=[
        FieldCondition(key="company", match=MatchValue(value="贵州茅台")),
        FieldCondition(key="year", match=MatchValue(value=2023)),
    ]),
    limit=5,
    with_payload=True,
)
results = response.points    # 注意: 结果在 .points 属性里
```

注意事项(踩过的坑):
- qdrant-client 1.12+ 用 `query_points` 替代了 `search`
- 参数名从 `query_vector` 改成了 `query`
- 返回值要取 `.points` 才是结果列表

### 5.3 参考资料格式化

```python
def format_context(search_results):
    parts = []
    for i, result in enumerate(search_results, 1):
        meta = result["metadata"]
        header = f"[{i}] 来源: {meta['company']} {meta['year']}年报 第{meta['page']}页"
        parts.append(f"{header}\n{result['content']}")
    return "\n\n".join(parts)
```

每条参考资料都带编号和来源信息,让 LLM 能标注引用。

---

## 6. 踩过的坑

### 坑 1: qdrant-client API 变更

新版 `qdrant-client`(1.12+)把 `search` 改为 `query_points`,
旧 API 直接报错。

教训: 第三方库 API 变更是常态,用之前先看版本号对应的文档。

### 坑 2: DeepSeek API key 缓存

`.env` 改了新 key,但代码里 `load_dotenv()` 没加 `override=True`,
环境变量被旧值缓存,一直 401。

修复: `load_dotenv(override=True)`

教训: Python 的 `os.environ` 进程级缓存,
改 `.env` 后如果进程没重启,必须 `override=True` 才能刷新。

### 坑 3: 年报文件命名不一致

招行文件名 "2025年度报告" 少了一个 "年" 字,
导致 `parse_filename` 的正则匹配失败,metadata 缺失。

修复: 正则改为 `年{1,2}度报告`,兼容两种命名。

教训: 文件命名约定一定要容错,不能假设 100% 规范。

---

## 7. 工程决策记录

### 决策 1: Point ID 用 UUID 而非自增整数

理由:
- 多次入库不冲突(自增需要查询当前最大值)
- 分布式场景下不需要中心化 ID 分配
- Qdrant 对 UUID 有原生优化

### 决策 2: content 存进 payload

理由:
- 检索后直接拿到原文,不需要额外查询
- 避免维护 "id → 文本" 的映射表
- 这是 RAG 的标准做法(trade-off: 占更多存储,但简化了架构)

### 决策 3: temperature=0.1

理由:
- RAG 场景需要确定性(每次问同样问题应该答案稳定)
- 金融数字不能有"创造力"(不能让 LLM 随机变换表述方式导致数字出错)
- 0.1 而不是 0: 留一点灵活度让 LLM 组织语言,完全 0 容易重复

### 决策 4: 开发阶段 recreate=True

入库脚本每次重建 collection,虽然慢但保证数据干净。
生产环境会改为增量 upsert。

---

## 8. 面试话术

### Q: 你为什么选 Qdrant?

> "三个理由。一是 Rust 写的,性能强内存省,本地一个 Docker 容器就能跑。
> 二是 API 设计干净,Python 客户端同步异步双支持。三是天然支持 Hybrid Search,
> 同一个 Collection 可以存 Dense + Sparse 两种向量,
> 后面做 BM25 融合时无缝衔接——这是关键决策点,
> 因为 Chroma 和 FAISS 做不到这一点。"

### Q: 你的 RAG Prompt 怎么设计的?为什么?

> "5 条规则,每条解决一个具体问题。
> '只用参考资料'抑制幻觉,'没信息就说不知道'提供兜底,
> '[1][2] 标注引用'保证可追溯,'数字精确'满足金融场景刚需,
> temperature=0.1 保证确定性。
> 实测招行不良贷款率问题时,LLM 确实拒绝了编造,
> 而且精准指出了'表格标题有但数值缺失'——说明 prompt 设计生效了。"

### Q: Naive RAG 的 baseline 表现怎么样?

> "四类问题测完,定性和开放性问题表现优秀,Top-5 几乎全命中;
> 但精确数字查询是短板,营收数据在 Rank 2 才命中,
> 不良贷款率甚至因为表格截断导致数值缺失,LLM 选择拒答。
> 这个 baseline 直接驱动了后续三个优化:
> Day 5 Hybrid Search 补充关键词精确匹配,
> Day 6 Multi-Query 解决语义鸿沟,
> Day 7 Reranker 过滤 Top-K 里的噪声。"

### Q: 你做这个项目遇到的最有意义的 bug 是什么?

> "招行不良贷款率的'智能拒答'——检索命中了正确的页码(p.32-35),
> 但 LLM 说'表格标题有但数值缺失'。排查发现是超长表格被 BGE 截断,
> 只有表头进了 embedding,数据行全丢了。
> 这让我意识到 RAG 的故障点不一定在检索排序上,
> 也可能在上游的数据加工环节——chunk 本身残缺,检索再准也没用。
> 后来我把这个列为技术债,等评估阶段用数据决定修不修。"

---

## 9. 已知问题与 Day 5-7 优化方向

| 问题 | 影响 | 对应优化 | 哪天做 |
|---|---|---|---|
| 稠密检索对术语区分弱 | "利息收入"被当作"营业收入" | Hybrid Search (BM25) | Day 5 |
| 查询和文档表述差异大 | 用户口语 vs 文档专业语 | Multi-Query / HyDE | Day 6 |
| Top-K 有噪声(不相关但"沾边") | 浪费 LLM 上下文 | Reranker (BGE-Reranker) | Day 7 |
| 超长表格被截断 | 数值丢失,精确查询失败 | 行级切分 / 换大模型 | Day 9 评估后 |
| 重复 chunks 占 Top-K 名额 | 挤掉真正相关的内容 | 去重 / Reranker | Day 7 |

---

## 10. 项目截至 Day 4 的完整模块地图

```
enterprise-rag/
├── src/
│   ├── loaders/
│   │   ├── base.py          ← Chunk 数据结构
│   │   └── pdf_loader.py    ← PDF 加载(分流+去重+清洗)
│   ├── chunking/
│   │   └── recursive.py     ← 递归切分器
│   ├── embeddings/
│   │   └── bge.py           ← BGE 封装(单例+batch+前缀)
│   ├── retrievers/
│   │   └── qdrant_store.py  ← Qdrant 操作(建库/入库/检索)
│   ├── generators/
│   │   └── llm.py           ← LLM 调用 + Prompt 模板
│   └── pipelines/
│       └── naive_rag.py     ← Naive RAG 端到端流程
├── data/
│   ├── raw/                 ← 4 份年报 PDF
│   └── processed/           ← 向量化结果 .pkl
├── scripts/
│   ├── 01-05               ← Day 2 实验脚本
│   ├── 06-10               ← Day 3 实验脚本
│   └── 11-13               ← Day 4 实验脚本
└── docs/
    ├── 01-mental-model.md
    ├── day02-summary.md
    ├── day03-summary.md
    └── day04-summary.md     ← 本文件
```

---

## 11. 自我检验

1. 向量数据库比 numpy 数组多解决了哪 4 个问题?
2. HNSW 的核心思想用一句话怎么说?
3. Qdrant 的 Collection/Point/Payload 分别对应关系数据库的什么?
4. 为什么把 content 存进 payload 而不是只存 id?
5. RAG Prompt 里"没有信息就说无法回答"解决什么问题?
6. temperature=0.1 为什么是 RAG 的标配?
7. Naive RAG 对哪类问题表现好?哪类差?为什么?
8. 招行"智能拒答"的根因是什么?说明 RAG 系统什么规律?
9. qdrant-client 新版 API 和旧版有什么区别?
10. 你为什么选 Qdrant 不选 Milvus 或 Chroma?
