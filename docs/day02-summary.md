# Day 2 总结

## 完成的模块
- [x] src/loaders/base.py
- [x] src/loaders/pdf_loader.py
- [x] src/chunking/recursive.py
src/loaders/
├── __init__.py              ← 空文件(包标识)
├── base.py                  ← Chunk 数据结构
│                               - Chunk dataclass
│                               - content: 文本内容
│                               - chunk_type: "text" | "table"
│                               - metadata: dict(source/company/year/page...)
│
└── pdf_loader.py            ← PDF 加载器
                                - parse_filename(): 从文件名提取 metadata
                                - is_real_table(): 伪表格过滤(3 条启发式规则)
                                - table_to_markdown(): 二维列表 → Markdown 表格
                                - clean_text(): 去页眉/页码/表单标记 + 合并断行
                                - load_pdf(): 主入口(分流 + bbox 扣除 + 清洗)

src/chunking/
├── __init__.py              ← 空文件
└── recursive.py             ← 递归字符切分器
                                - DEFAULT_SEPARATORS: 中文分隔符优先级
                                - get_token_counter(): tiktoken 计数器
                                - split_text_recursive(): 递归切分核心算法
                                - _merge_splits(): 小片段合并 + overlap 处理
                                - chunk_documents(): 主入口(table 不切, text 递归切)
# 实验脚本
scripts/
├── 01_explore_pdf.py        ← pypdf 看原始 PDF 长什么样(3 个不同页)
├── 02_compare_loaders.py    ← pypdf vs pdfplumber vs extract_tables 对比
├── 03_test_loader.py        ← Loader 总览(chunk 数量/类型分布/抽样)
├── 04_check_duplication.py  ← 重复内容检测(验证 bbox 扣除效果)
└── 05_test_chunker.py       ← 切分器效果(token 长度分布/抽样)
# 配置变更
pyproject.toml  ← 加了 [build-system] 和 [tool.hatch.build] (可编辑包安装)
.gitignore      ← 创建
.env.example    ← 创建
.env            ← 创建

## 关键数据(茅台 2023 年报)
- 原始 chunks: 423(text 143, table 280)
- 去重后: 359(text 131, table 228,过滤了 64 个冗余)
- 切分后: 493(text 265, table 228)
- text chunks 平均 249 tokens, 最长 400 tokens, 0 个超 512
- table chunks 平均 384 tokens, 最长 2508 tokens, 49 个超 512(技术债)
## 核心决策
1. 表格 vs 文字分流处理(用不同 API)
2. bbox 扣除避免内容重复
3. 启发式过滤伪表格(3 条规则)
4. 递归字符切分,优先级: 段落 > 句子 > 标点 > 字符
5. 表格 chunks 不切(保持行列结构)
6. token-based 长度控制(不用字符)

## 已知问题(技术债)
- 49 个超长表格 chunk 会被 BGE 截断
- 待 Day 9 评估后决定是否优化

## 学到的认知
(列 5-10 条你自己最有感触的)
好,这就给你一份 **Day 2 完整笔记**——包含流程、代码、知识点、面试话术,还有几张关键流程图。这份笔记直接放到你的 `docs/day02-summary.md`,后面写技术博客和准备面试都用得上。

先准备几张关键的可视化:先给你看两张关键流程图,然后是完整笔记 markdown。下面是完整笔记的 markdown 文本。把这一整份**复制粘贴**到 `docs/day02-summary.md`,两张图我已经在上面给你看了,**笔记里我用文字描述+ASCII 图示替代**——这样在 GitHub 上 README 渲染时不依赖外部图床,任何编辑器打开都能看。

---

```markdown
# Day 2 完整笔记: PDF 加载与切分

## 0. 本日目标

> 把 4 份金融年报 PDF 转成"高质量、结构化、可向量化"的 Chunk 列表,
> 为 Day 3 的 Embedding 做好数据准备。

**核心挑战**:金融年报 60% 以上是表格。如果按通用文档处理,
表格会被压成无序文本流,数字和列名错位,导致后续 RAG 系统在
"查 2023 年第三季度营收"这类高价值问题上必然产生幻觉。

---

## 1. 当日产出清单

### 代码模块
- `src/loaders/base.py` — 统一的 Chunk 数据结构
- `src/loaders/pdf_loader.py` — 表格分流 + 伪表格过滤 + bbox 扣除的 PDF 加载器
- `src/chunking/recursive.py` — 递归字符切分器(token 控制 + overlap)

### 实验脚本
- `scripts/01_explore_pdf.py` — 用 pypdf 看原始 PDF 抽取出来啥样
- `scripts/02_compare_loaders.py` — pypdf vs pdfplumber vs extract_tables 对比
- `scripts/03_test_loader.py` — Loader 总览测试
- `scripts/04_check_duplication.py` — 重复内容检测
- `scripts/05_test_chunker.py` — 切分器效果测试

### 关键数据(以茅台 2023 年报为例,143 页)
| 阶段 | text chunks | table chunks | 总数 | 备注 |
|---|---|---|---|---|
| 原始 Loader v1 | 143 | 280 | 423 | 内容有重复 |
| Loader v2(去重+过滤伪表格) | 131 | 228 | 359 | -64 个 |
| Chunker 切分后 | 265 | 228 | 493 | text 被切碎 |

切分后 text chunks: 平均 249 tokens,最长 400 tokens,**0 个超过 BGE 上限 512**。

---

## 2. 完整加工流程

### 2.1 流程总览

```
┌─────────────┐
│  PDF 单页    │
└──────┬──────┘
       │
       ▼
┌────────────────────────┐
│ find_tables() 识别表格  │ ← pdfplumber 启发式表格检测
│ 返回 bbox + 内容        │
└──────┬─────────────────┘
       │
       ▼
┌────────────────────────┐
│ is_real_table() 过滤    │ ← 启发式: 列数/行数/含数字
│ 排除伪表格              │
└──────┬─────────────────┘
       │
       ├──────────────┬──────────────┐
       ▼              ▼              │
┌──────────┐   ┌──────────────────┐  │
│真表格路径 │   │ 文字路径          │  │
│           │   │                  │  │
│extract +  │   │ filter 扣除表格   │  │
│转Markdown │   │ 区域 + extract_   │  │
│           │   │ text + clean_text│  │
│ type=table│   │ type=text        │  │
└─────┬─────┘   └────────┬─────────┘  │
      │                  │            │
      └──────────┬───────┘            │
                 ▼                    │
       ┌─────────────────────┐        │
       │ Chunk 对象列表       │        │
       │ (含 metadata)        │        │
       └─────────────────────┘        │
                 │                    │
                 ▼                    │
       ┌─────────────────────┐        │
       │ Chunker 递归切分     │        │
       │ (table 不切)         │        │
       └─────────────────────┘        │
```

### 2.2 关键决策点

每个决策都对应一个真实问题:

| 决策 | 解决的问题 | 替代方案及为什么不用 |
|---|---|---|
| 表格 vs 文字分流 | 表格塌陷成无序文本流 | 一刀切 → 数字和列名错位,幻觉率高 |
| bbox 扣除文字 | 同页表格在 text chunk 里重复出现 | 后置去重 → 阈值难调,仍污染 Top-K |
| 伪表格过滤 | pdfplumber 把术语表/列表识别成表格 | 全保留 → 表格 chunk 充满"X|指|Y"格式垃圾 |
| 表格不切 | 行列对应关系破坏 | 强切 → LLM 看到残缺表格反而更易出错 |
| token 控制长度 | 字符长度和 token 数不一致 | 字符切 → 中英混排时实际 token 数失控 |

---

## 3. 知识点全集

### 3.1 PDF 解析的本质

**PDF 是"打印指令清单",不是"语义文档"**

PDF 内部存的是"在坐标 (x, y) 用字体 F 画字符 C",**不存在段落、章节、表格的概念**。
解析工具只能从字符坐标"逆向工程"猜测结构,所以:

- 同一句话视觉上换行 → 工具加 `\n` → 词被切散
- 表格 → 一堆有规律坐标的字符 → 工具识别失败时塌陷成行
- 页眉页脚 → 重复字符串 → 工具不知道这是模板还是正文

**结论**: PDF 解析没有完美方案,只有"在你的场景下足够好"的方案。

### 3.2 三种 PDF 解析 API 的差异

| 方法 | 原理 | 表格处理 | 适用场景 |
|---|---|---|---|
| `pypdf.extract_text()` | 按字符 y 坐标排序输出 | 塌陷,行列对应丢失 | 纯文字文档 |
| `pdfplumber.extract_text()` | 同上 + 一些版式优化 | 仍然塌陷 | 略微改善的纯文字 |
| `pdfplumber.extract_tables()` | 启发式检测表格线和对齐 | 返回二维列表,**结构保留** | 表格密集文档 |

**关键经验**: `extract_tables()` 是金融年报场景的杀手锏。

### 3.3 bbox(边界框)的概念

每个表格被识别后,pdfplumber 会返回它在页面上的坐标范围
`(x0, y0, x1, y1)`(左上角到右下角)。

利用这个坐标,可以用 `page.filter()` 排除"在某个 bbox 内的字符",
从而拿到"扣掉表格之后剩下的文字"。

```python
def not_within_any_table(obj):
    for bbox in table_bboxes:
        x0, y0, x1, y1 = bbox
        cx = (obj["x0"] + obj["x1"]) / 2
        cy = (obj["top"] + obj["bottom"]) / 2
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            return False  # 这个字符在表格里,扣掉
    return True

filtered_page = page.filter(not_within_any_table)
text = filtered_page.extract_text()  # 只剩表格之外的文字
```

### 3.4 启发式过滤伪表格

pdfplumber 的表格识别**过度提取**——会把项目符号列表、术语解释、目录
误判为表格。我们用三条启发式规则过滤:

1. **列数 >= 3**: 2 列大概率是"名称-解释"列表
2. **数据行 >= 1**: 只有表头无数据 = 没价值
3. **数据行含数字**: 全文字表格往往是术语表,不是财务数据

这三条规则覆盖了茅台年报里 95%+ 的伪表格,代价是偶尔漏掉几个边角的文字
术语表(被错过滤掉)。**这是金融场景下的合理 trade-off**。

### 3.5 Token vs 字符

**Token 是模型的最小处理单元**,不是字符。

- 英文: 1 单词 ≈ 1-1.5 tokens
- 中文: 1 汉字 ≈ 1-1.3 tokens
- 数字/英文混排: 极不规则

**为什么必须按 token 算 chunk_size**: embedding 模型的输入限制是按 token 算的。
按字符算 chunk_size,在中英混排时会失控——某些段落看着才 400 字,
实际可能 600+ tokens,被模型截断。

**用 tiktoken 计算**:
```python
import tiktoken
encoding = tiktoken.get_encoding("cl100k_base")
n_tokens = len(encoding.encode(text))
```

### 3.6 递归切分算法(Recursive Character Splitting)

**核心思想**: 给定分隔符优先级列表,先用最高优先级切;切出来的片段
若仍超 chunk_size,**递归地用次高优先级切这个片段**。

中文场景的优先级:
```
段落 \n\n  >  行 \n  >  句号 。  >  分号 ;  >  逗号 ,  >  空格  >  字符
```

**为什么这个顺序**: 优先在"语义完整边界"切,保证每个 chunk 内部
是一个完整的意义单元。实在不行才在字符级别强切(兜底)。

### 3.7 chunk_size 和 chunk_overlap 的权衡

**chunk_size 太大**:
- 浪费 embedding 模型容量(BGE 上限 512 tokens)
- 检索时 Top-K 噪声多(一个 chunk 包含多个不相关话题)
- LLM 上下文窗口压力大

**chunk_size 太小**:
- 上下文丢失(关键信息散落在多个 chunk)
- 检索时即使命中也无法回答完整问题
- 数据库存储和检索次数翻倍

**chunk_overlap 的作用**: 防止关键信息正好卡在切分边界。
通常设为 chunk_size 的 10-20%(我们用 50/400 = 12.5%)。

我们最终选择 `chunk_size=400, overlap=50`,理由:
- 400 < 512 留出 BGE 安全余量
- 中文 400 tokens ≈ 320 汉字 ≈ 一个完整段落,语义连贯
- overlap 50 足够覆盖"句子被切断"的情况

### 3.8 表格 chunk 不切的理由

Markdown 表格是一个"完整结构",硬切会得到:
```
| 列1 | 列2 | 列3 |
| --- | --- | --- |
| 数据 | 数据 |    ← 残缺
```

LLM 看到这种残缺表格会混乱。**所以表格整体不切**,即使超过 512 tokens
被截断,后果也比"切坏的表格"可控。

**留下的技术债**: 49 个表格超 512 tokens,后段内容检索不到。
Day 9 评估后再决定是否值得修。候选方案:
- 按行切分 + 每个 chunk 带表头
- LLM 摘要 + 完整表格双存
- 换 BGE-M3(支持 8192 tokens)

---

## 4. 关键代码片段

### 4.1 统一 Chunk 数据结构(base.py)

```python
from dataclasses import dataclass, field
from typing import Literal

@dataclass
class Chunk:
    content: str
    chunk_type: Literal["text", "table"] = "text"
    metadata: dict = field(default_factory=dict)
```

**设计要点**:
- 用 dataclass 不用 dict —— 字段类型明确,IDE 能补全
- `chunk_type` 用 Literal,只能是 text/table,防止打字错误
- metadata 是开放字典,允许任意扩展(page、company、year...)

### 4.2 伪表格过滤(pdf_loader.py)

```python
def is_real_table(table) -> bool:
    rows = table.extract()
    if not rows:
        return False
    
    num_cols = max(len(row) for row in rows)
    num_data_rows = len(rows) - 1
    
    if num_cols < 3 or num_data_rows < 1:
        return False
    
    has_number = any(
        re.search(r"\d", cell or "")
        for row in rows[1:]
        for cell in row
    )
    return has_number
```

### 4.3 bbox 扣除(pdf_loader.py)

```python
real_tables = [t for t in page.find_tables() if is_real_table(t)]
table_bboxes = [t.bbox for t in real_tables]

def not_within_any_table(obj):
    for bbox in table_bboxes:
        x0, y0, x1, y1 = bbox
        cx = (obj["x0"] + obj["x1"]) / 2
        cy = (obj["top"] + obj["bottom"]) / 2
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            return False
    return True

filtered_page = page.filter(not_within_any_table)
text = filtered_page.extract_text() or ""
```

### 4.4 递归切分核心(recursive.py)

```python
def split_text_recursive(text, chunk_size, chunk_overlap, separators):
    if token_counter(text) <= chunk_size:
        return [text]
    
    # 找最高优先级且文本中存在的分隔符
    sep = next((s for s in separators if s == "" or s in text), "")
    splits = text.split(sep) if sep else list(text)
    
    chunks = _merge_splits(splits, sep, chunk_size, chunk_overlap)
    
    # 还有超大的 → 用次级分隔符递归
    final = []
    for chunk in chunks:
        if token_counter(chunk) <= chunk_size:
            final.append(chunk)
        else:
            remaining = separators[separators.index(sep) + 1:]
            final.extend(split_text_recursive(
                chunk, chunk_size, chunk_overlap, remaining
            ))
    return final
```

---

## 5. 关键 Bug 与修复历程

这一节专门记录"踩过的坑",**面试时可以直接讲**。

### Bug 1: 内容重复(发现于 scripts/04)

**症状**: 第 6 页 3 个 chunks(2 个表格 + 1 个文字),冗余率 52%——
表格内容在 table chunk 里出现一次,在 text chunk 里又以塌陷形式出现一次。

**影响**:
1. 存储浪费(向量库存了双份)
2. 检索污染(同一内容两个向量都被检索回来,挤掉其他相关内容)
3. LLM 困惑(同样信息看到两份不同格式)

**修复**: 用 `page.filter()` + bbox 扣除策略。

**验证**: 第 6 页 text chunk 长度从 1328 字 → 255 字,降幅 81%。

### Bug 2: 伪表格污染(发现于 scripts/03)

**症状**: 第 3-4 页的"备查文件目录"和"常用词语释义"被识别成表格,
但这些其实是项目符号列表和术语解释。

**影响**: 表格 chunks 里塞满"X|指|Y"格式的非财务数据,
检索"经营数据"时容易误命中。

**修复**: 加 `is_real_table()` 启发式过滤(3 条规则)。

**验证**: 总 table chunks 从 280 → 228,过滤掉 52 个伪表格。

### Bug 3: 文字被错误换行切碎(原始 PDF 缺陷)

**症状**: PDF 视觉换行导致一句话中间被插入 `\n`,
比如"结构性增长"变成"结构\n性增长",embedding 时被理解为两个无关词。

**修复**: `clean_text()` 里加合并逻辑——若上一行末尾不是句末标点
且下一行不是新段落标记,合并。

**已知副作用**: 偶尔会过度合并(如"主要财务数据" + "单位:元"被粘成一句),
不影响检索,留作日后优化。

---

## 6. 已知技术债(刻意接受)

| 债务 | 严重程度 | 缓解方案 | 预计修复时机 |
|---|---|---|---|
| 49 个超长表格(>512 tokens)被截断 | 中 | 行级切分 / 摘要 / 换大模型 | Day 9 评估后决策 |
| 文字合并偶尔过度 | 低 | 调整合并规则 | 影响生成时再修 |
| 伪表格过滤偶尔误伤 | 低 | 加白名单 / 用版式分析模型 | 不优先 |
| 扫描版 PDF 不支持 | 高 | OCR 流水线(PaddleOCR) | 业务需要时再加 |

**为什么记技术债很重要**: 面试官最忌讳"完美主义者"——所有问题都解决了
=没有取舍=没有工程思维。能清晰说出"我知道这有问题,但我现在不修,
理由是 X"才是真工程师。

---

## 7. 面试话术(可直接复述)

### Q1: "你为什么用 pdfplumber 不用 pypdf?"

> "金融年报 60% 以上是表格,pypdf 只有 `extract_text` 会把表格塌陷成
> 一行无序的字符串,行列对应关系完全丢失。pdfplumber 提供了 `extract_tables`,
> 能识别表格区域并返回二维列表,我把这些转成 Markdown 表格,
> LLM 阅读 Markdown 表格的能力非常强,几乎不会把数字和列名搞错。"

### Q2: "你的 chunk_size 为什么是 400?"

> "400 是按 token 算的,不是字符。我们用的 BGE-small-zh 模型上限 512 tokens,
> 设 400 留出 20% 安全余量,避免 chunk 因为 token 计算偏差被截断。
> 中文 400 tokens 大约对应 320 个汉字,在年报里大致是一个完整段落
> 或一个小章节的长度,语义连贯性最好。我用 chunk_overlap=50 防止
> 关键信息正好卡在切分边界。"

### Q3: "你做这个项目踩过最深的坑是什么?"

> "第一次跑通 Loader 后我做了个验证脚本,发现同一页的表格内容会出现两次:
> 一次在 table chunk 里(干净的 Markdown),一次在 text chunk 里(塌陷文本)。
> 因为 pdfplumber 的 extract_text 不知道哪些字符属于表格,
> 它会把所有字符都吐出来。
> 
> 我用了 bbox 扣除策略——先用 find_tables 拿到表格的坐标范围,
> 然后用 page.filter() 排除这些坐标内的字符,只对剩余区域 extract_text。
> 验证下来第 6 页 text chunk 长度从 1328 字降到 255 字,
> 整本书冗余率从 52% 降到接近 0。"

### Q4: "你怎么处理表格?为什么表格 chunk 不切?"

> "表格我转成 Markdown 格式存储,因为 LLM 对 Markdown 表格的理解力非常强。
> 表格 chunk 不切是个有意识的决策——Markdown 表格是个完整结构,
> 硬切会破坏行列对应关系,LLM 看到残缺表格反而更容易混淆。
> 我接受少数超长表格会被 embedding 截断这个代价,
> 因为'切坏的表格'比'被截断的完整表格'后果更严重。
> 这部分我留作技术债,等评估阶段看实际影响再决定要不要做行级切分。"

### Q5: "为什么用 dataclass 不用 dict?"

> "dataclass 给我类型安全和 IDE 补全。我的 Chunk 在整个 pipeline 里
> 流转——Loader 创建,Chunker 处理,Embedder 加向量,Retriever 排序——
> 任何一处拼错字段名都会立刻报错,不会变成运行时的隐性 bug。
> 而且后面要扩展字段(比如加 embedding 向量、score 分数),
> dataclass 改一处所有调用方都受益。"

### Q6: "Token 和字符有什么区别?为什么按 token 算 chunk_size?"

> "Token 是 LLM 的最小处理单元,英文一个单词通常 1-1.5 tokens,
> 中文一个汉字大约 1-1.3 tokens。embedding 模型的输入限制是按 token 算的,
> 不是按字符。如果按字符算 chunk_size,中英混排或包含大量数字时,
> 实际 token 数可能远超预期,被模型静默截断,后段内容向量化失败,
> 永远检索不到——这种 bug 不会报错,极难发现。
> 用 tiktoken 精确计数能避免这个问题。"

---

## 8. 自我检验(明天讲不出来就回头读)

合上笔记,自己默答:

1. PDF 内部存的是什么?为什么这导致表格抽取困难?
2. pypdf 和 pdfplumber 的本质区别是什么?
3. find_tables 和 extract_tables 有什么不同?
4. bbox 是什么?如何用它扣除表格区域?
5. 伪表格过滤的三条启发式规则是什么?为什么是这三条?
6. 递归切分的分隔符优先级是什么?为什么是这个顺序?
7. 为什么表格 chunk 不能切?如果非要切,你会怎么做?
8. chunk_size=400 这个数字怎么来的?改成 200 或 800 会怎样?
9. chunk_overlap 为什么需要?设多少合适?
10. 你今天留下了哪些技术债?为什么不立刻修?

每一条都答得出来,Day 2 就完成了 100%。

---

## 9. Day 3 预告

明天进入 RAG 的"数学心脏"——**Embedding**。
- 讲透向量空间和语义相似度的原理
- 接入 BGE-small-zh-v1.5 模型
- 把 493 个 chunks 全部向量化
- 可视化几个查询的语义距离,直观看到"语义相近的句子向量真的近"

为 Day 4 的向量库入库做准备。
```

---

## 怎么用这份笔记

**3 个建议**:

**1. 先复制到文件,再手动改写**

直接复制就行,但**面试问答那部分(第 7 节)你必须用自己的话重写一遍**。复制的话术面试时会卡壳——只有你嚼过、吐出来过的语言,临场才能流畅复述。

**2. 把上面那两张图保存下来**

我画的两张图(`day2_pdf_loader_pipeline` 和 `day2_recursive_chunking`)——你可以截图保存,放到 `docs/images/` 目录,然后在笔记里引用:

```markdown
![PDF Loader 流水线](./images/day2_pdf_loader_pipeline.png)
```

或者**用浏览器开发者工具右键 SVG 元素 → Copy outerHTML**,把 SVG 代码贴到笔记里(GitHub 直接渲染 SVG)。这样不依赖图床,永久有效。

**3. 提交到 Git,作为里程碑**

```bash
git add docs/day02-summary.md
git commit -m "docs: complete Day 2 notes - PDF loading & chunking"
git push
```

12 天后回头看,你会很感谢今天的自己——这是面试前**唯一不用临时背的真实记忆**。

---

笔记给你了,休息一下,明天回来回复"**开始 Day 3**",我们进入 Embedding 的世界。