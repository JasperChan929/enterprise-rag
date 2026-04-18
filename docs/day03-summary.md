# Day 3 完整笔记: Embedding — RAG 的数学心脏

#完成的模块
src/embeddings/
├── __init__.py              ← 空文件(包标识)
└── bge.py                   ← BGE Embedding 封装
                                - BGEEmbedder 类(单例模式)
                                - encode(): 单条/批量编码
                                - encode_query(): 查询编码(带 BGE 前缀)
                                - encode_chunks(): 直接编码 Chunk 列表
# 实验脚本
scripts/
├── 06_download_bge.py       ← 下载 BGE 模型 + 验证维度/归一化
├── 07_similarity_experiment.py  ← 5 句话相似度矩阵(验证"语义魔法")
├── 08_test_embedder.py      ← BGEEmbedder 封装测试(单条/批量/前缀/单例)
├── 09_embed_all_chunks.py   ← 493 chunks 批量向量化 + sanity check
└── 10_verify_naive_limit.py ← 多查询验证 Naive 检索局限

# 数据产出
data/processed/
└── maotai_2023_embeddings.pkl  ← 493 chunks + (493,512) 向量矩阵, 1.32MB
# 配置变更
.env  ← 新增 HF_ENDPOINT=https://hf-mirror.com
## 0. 本日目标

> 把 Day 2 产出的 493 个 Chunks 转成 512 维浮点数向量,
> 让计算机能"计算"文本之间的语义相似度,为 Day 4 入库做准备。

---

## 1. 当日产出清单

### 代码模块
- `src/embeddings/bge.py` — BGE Embedding 封装(单例 + 批量 + 查询前缀)

### 实验脚本
- `scripts/06_download_bge.py` — 下载/加载 BGE 模型
- `scripts/07_similarity_experiment.py` — 5 句话语义相似度实验
- `scripts/08_test_embedder.py` — BGEEmbedder 封装测试
- `scripts/09_embed_all_chunks.py` — 批量向量化 493 chunks + sanity check
- `scripts/10_verify_naive_limit.py` — 多查询验证 Naive 检索的局限

### 数据产出
- `data/processed/maotai_2023_embeddings.pkl` — 493 chunks + 向量矩阵(1.32 MB)

### 关键数据
- 模型: BGE-small-zh-v1.5, 512 维, 95MB
- 493 chunks 批量编码: 15.9 秒(32.2 ms/chunk, CPU)
- 向量矩阵: shape (493, 512), L2 归一化

---

## 2. 核心知识点

### 2.1 Embedding 的本质

Embedding 是一个函数,把任意文本映射成固定长度的浮点数向量,
使得语义相似的文本在高维空间里距离也近。

```
"茅台 2023 年营收" → [0.02, -0.18, 0.04, ...] (512 个数字)
"贵州茅台去年挣了多少钱" → [0.03, -0.16, 0.05, ...] ← 方向接近
"今天天气真好" → [0.62, -0.91, 0.11, ...] ← 方向差很多
```

核心性质: 语义相近 → 向量方向一致 → 余弦相似度高。

### 2.2 余弦相似度(RAG 事实标准)

不看距离,只看两个向量的方向夹角:

```
cosine(A, B) = (A · B) / (|A| × |B|)

取值范围: [-1, 1]
  1.0  → 方向完全一致 → 极度相似
  0.0  → 正交 → 无关
 -1.0  → 反向 → 极度相反
```

为什么不用欧氏距离: 文本向量的"长度"没有明确语义,
我们只关心"方向一致性"(语义相似度),余弦只考虑方向,正好合适。

BGE 输出已经 L2 归一化(向量长度恒为 1),所以余弦 = 点积:
```python
cosine(A, B) = A · B  # 因为 |A| = |B| = 1
```
这让计算快了一倍(省了一次开方)。

### 2.3 为什么用 512 维

人类语义太丰富。2 维只能区分"几大类话题",
512 维才能区分细微差别:
- "营业收入" vs "营业利润"(都是财务,但一个是收入一个是利润)
- "净利润" vs "扣非净利润"(差在"扣非"二字)

维度越高表达力越强,但计算/存储代价也越大。
512 维是 BGE-small 的平衡点,够用。

### 2.4 模型是怎么"学会"语义的

核心思想: 对比学习(Contrastive Learning)。

训练时给模型大量正负样本对:
- 正样本: ("如何续签证件", "签证延期办理流程") → 应该靠近
- 负样本: ("如何续签证件", "今天天气真好") → 应该远离

通过几亿样本对训练,模型学会了"什么样的句子应该有相近的向量"。

BGE 还用了 hard negative mining(精心挑选"看着像但不相关"的负样本),
这是它中文效果强的原因之一。

### 2.5 BGE-small-zh-v1.5 选型理由

| 维度 | 选择理由 |
|---|---|
| 中文效果 | 中文 RAG 事实标准,智源团队专门优化 |
| 轻量 | 95MB,CPU 跑 50ms/chunk,不依赖 GPU |
| 零成本 | 开源免费,不像 OpenAI 按调用收费 |
| 配套生态 | 有 BGE-Reranker(Day 7 用),技术栈统一 |
| 生产级 | HuggingFace 下载量千万级,社区活跃 |

不选 OpenAI text-embedding: 中文效果差一截 + 要梯子 + 要花钱。
不选 BGE-large: 大 14 倍换 5-10% 提升,不划算,等评估再决定。

### 2.6 BGE 查询前缀

BGE 官方建议对查询加前缀:
```python
query = "为这个句子生成表示以用于检索相关文章:" + "茅台营收"
```

但实验发现 v1.5 上前缀效果不显著(差异 -0.007,在噪声范围内)。
保留前缀但列为待评估项,Day 9 用统计方法决定。

教训: 单次实验差异 < 1% 不能作为决策依据。

### 2.7 常见误区

1. Embedding 不是 word2vec: BGE 是句子级别的,理解上下文
2. 维度不是越高越好: 够用就行,边际效应递减
3. 中英文不能混用模型: 必须用中文训练的模型
4. 大模型不一定必要: small 先跑通,瓶颈再升级

---

## 3. 关键代码片段

### 3.1 BGEEmbedder 核心设计

```python
class BGEEmbedder:
    _instance = None  # 单例

    def encode(self, texts, batch_size=32):
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,  # L2 归一化
            batch_size=batch_size,
        )
        return vectors

    def encode_query(self, query):
        # 查询加前缀(BGE 官方建议)
        prefixed = "为这个句子生成表示以用于检索相关文章:" + query
        return self.encode(prefixed)
```

设计要点:
- 单例模式: 全局共用一个模型实例,省内存
- batch 推理: 32 个一批,比逐条快 10 倍
- 自动归一化: 后续余弦计算直接用点积
- 查询/文档区分: 查询加前缀,文档不加

### 3.2 相似度计算

```python
# 因为都归一化了,余弦 = 点积
vectors = model.encode(texts, normalize_embeddings=True)
q_vec = model.encode_query(query)
similarities = vectors @ q_vec  # 矩阵乘法,一次算所有
top_k = np.argsort(-similarities)[:5]
```

---

## 4. 关键实验发现

### 实验 1: 5 句话相似度矩阵

验证了 BGE 确实"理解语义":
- "茅台营收" vs "茅台挣了多少钱": ~0.85(高,词完全不同但语义一致)
- "茅台营收" vs "申请退税": ~0.35(低,不同话题)
- "茅台营收" vs "天气真好": ~0.15(极低,完全无关)

### 实验 2: Sanity check(493 chunks 检索)

查询"茅台 2023 年营业收入是多少",Top-5 结果:
- Rank 1 (0.66): p.20 经营计划 ← 不是营收数据
- Rank 2 (0.65): p.56 审计事项 ← 提到营收但非数据
- Rank 3 (0.64): p.135 利息收入 ← 错!混淆了"营业收入"和"利息收入"
- Rank 4 (0.64): p.47 利息收入 ← 重复内容
- Rank 5 (0.64): p.29 董事会公告 ← 不相关

真正的营收表(第 6 页)根本没进 Top-5。

### 诊断

这暴露了 Naive 稠密检索的 3 个致命问题:

| 问题 | 原因 | 解决方案(哪天做) |
|---|---|---|
| "利息收入"被当作"营业收入" | 稠密检索对专业术语区分弱 | Day 5 Hybrid Search(BM25 精确匹配) |
| Top-5 有重复低质量内容 | 无精排机制 | Day 7 Reranker |
| 表格 chunk 排名低 | 文字密度低,语义信号弱 | Day 6 HyDE/Multi-Query |

---

## 5. 面试话术

### Q: 你为什么选 BGE-small 不选 OpenAI?

> "三个理由。一是中文效果:BGE 用对比学习+hard negative mining 专门训练中文,
> 在中文 RAG 评测里 SOTA。二是部署:95MB 本地跑,CPU 50ms 延迟,
> 不依赖外部 API,无合规风险。三是配套:有 BGE-Reranker 做精排,技术栈统一。"

### Q: BGE 输出有什么特点?

> "512 维向量,经过 L2 归一化,长度恒为 1。
> 这意味着余弦相似度可以直接用点积计算,省一次开方,推理更快。"

### Q: 你做 embedding 时踩过什么坑?

> "跑完 Naive 检索发现 Top-5 完全没命中真正的营收表——
> 前两名是讲战略和审计的,Rank 3-4 是'关联方利息收入'(混淆了术语)。
> 而且两个 chunks 相似度只差 0.0001,说明 Top-K 边界很脆弱。
> 这两个发现分别驱动了后续的 Hybrid Search 和 Reranker 优化。"

### Q: 单次实验能作为决策依据吗?

> "不能。BGE 前缀实验差异 -0.007,在统计噪声范围内。
> 我没有立即去掉前缀,而是列为待评估项,等评估集建好后用统计方法决定。
> 这是做这个项目学到的重要经验:单次实验结果不能作为工程决策依据。"

---

## 6. 已知问题

| 问题 | 影响 | 计划 |
|---|---|---|
| Naive 检索对精确数字问题表现差 | Top-5 命中率低 | Day 5-7 Advanced RAG |
| BGE 前缀效果待评估 | 可能 +/- 1% | Day 9 评估集验证 |
| 49 个超长表格 embedding 被截断 | 后段内容检索不到 | Day 9 评估后决策 |

---

## 7. 自我检验

1. 一句话说出 embedding 的本质
2. 余弦相似度的取值范围和含义
3. 为什么 BGE 输出归一化后余弦等于点积?
4. BGE-small 的维度/模型大小/推理速度
5. 对比学习的训练思路(正负样本对)
6. Sanity check 暴露了哪 3 个问题?分别在哪天解决?
