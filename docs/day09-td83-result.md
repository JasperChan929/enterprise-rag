# Day 9 Task 0: TD-8-3 前置审计结论

> 审计日期: Day 9 开工 (2026-04-21)
> 审计对象: TD-8-3 "Prompt 是否注入 metadata (company/year) 给 LLM 看"
> 判定: ✅ **已注入**, TD-8-3 闭合
> 对 Day 8 的影响: **0 项结论需要撤回**
> 对 Day 10 的影响: Day 10 Task 1 直接进 TD-8-2 Prompt 工程, **不需要先补 metadata 注入**

---

## 0. TD-8-3 知识点溯源 (偏好 7)

**完整身份**:
TD-8-3 = Day 8 登记的前置审计技术债. 内容 = "验证 `src/generators/llm.py`
的 `format_context()` 函数是否把 chunk.metadata 里的 `company` 和 `year` 字段
拼进了送给 LLM 的 user message".

**历史出处**:
- Day 8 笔记第 6 节 (技术债重估) 首次登记, 描述: "Prompt 是否已注入 metadata
  (company/year) 待审", 优先级 P2
- Day 8 笔记第 2.5 节 (技术债图谱): 标为 **"Day 10 开工前 30 分钟必做"** —
  因为 Day 8 的"自足 chunk 3 要素"方法论里, 要素 1 (主体) 依赖 Prompt 真的把
  metadata 给了 LLM 看. 如果没拼, U4 招行的 L2a 判定可能要改成 L2a' (需要先补
  metadata), Day 10 的 Prompt 工程方向会走错
- Day 8 笔记第 8.2-8.3 节 (Day 9 预告): 列为 Task 0, 30 分钟, 开工必做

**当前用途** (Day 9 Task 0 本审计):
- 用"读源码 + 追证据链"5 环验证 (parse_filename → Chunk.metadata → Qdrant
  payload → search_similar → format_context), 替代推测
- 产出本文档, 对 Day 10 开工方向给出明确判定 (拼 OR 没拼)

**当前状态**: ✅ **闭合 (Day 9 Task 0 完成)**
- 验证结果: 已注入
- 后续动作: 从 Day 10 开工前置清单中移除. Day 10 直接进 TD-8-2 Prompt 工程主战场

---

## 1. 审计方法

目标: 验证 `src/generators/llm.py` 的 `format_context()` 是否把
`chunk.metadata` 里的 `company` 和 `year` 字段拼进了送给 LLM 的 user message.

方法: 沿证据链走 4 个环节 — 文件名解析 → Chunk 构造 → Qdrant 入库 →
Qdrant 出库 → format_context 拼接 → LLM 消息组装. 每一环都要有源码行号证据.

## 2. 证据链 (5 环全通)

| 环节 | 文件:行号 | 证据 |
|---|---|---|
| 1. 文件名解析 | `src/loaders/pdf_loader.py:29-39` | regex `(\d{6})_(.+?)_(\d{4})年年度报告` 从文件名切出 `company` (str) 和 `year` (int). 返回 dict 含 `source / stock_code / company / year` 4 字段. |
| 2. Chunk 构造 | `src/loaders/pdf_loader.py:169, 184, 220` | 把上一步的 dict 作为 `page_meta` 传给 `Chunk(metadata=...)`. 所有 text 和 table chunk 共用同一份 page_meta. |
| 3. Qdrant 入库 | `src/retrievers/qdrant_store.py:71-75` | `payload = {"content": ..., "chunk_type": ..., **chunk.metadata}`. `**` 把 company/year 摊成 payload 顶层字段入库. |
| 4. Qdrant 出库 | `src/retrievers/qdrant_store.py:144-152` | `content` 被 `pop` 出来, 其余字段 (含 company/year) 作为 `metadata` 子 dict 返回给调用方. |
| 5. Prompt 拼接 | `src/generators/llm.py:49-55` | `meta.get("company")` + `meta.get("year")` 拼成 header: `"[1] 来源: 招商银行 2025年报 第45页 (text)"`. |
| 6. LLM 消息 | `src/generators/llm.py:83-95` | header + chunk content 整体进入 `RAG_USER_TEMPLATE.{context}`, 以 `user` 角色送入 `chat.completions.create`. |

**间接数据验证**: Day 8 脚本 26 输出 `day08-audit-26.txt` 每个样例都标注
"宁德时代 2025 p.28" / "招商银行 2025 p.45" 等, 格式来自 chunk.metadata —
等于用生产数据反向证明 6881 chunks 的 metadata 里 company/year 字段都有填.

## 3. 自我攻击清单 (3 假阳性 + 2 假阴性)

**假阳性 1**: `meta.get("company", "未知")` 有默认值, 万一入库时 chunk.metadata
没写 company 就会落入默认, 等于没拼.
→ 排除. `parse_filename()` 对所有符合命名约定的 PDF 都填了 company/year.
   间接证据 (day08-audit-26 输出) 证明全库 chunks 都有合法 metadata.

**假阳性 2**: search_results 的数据结构和 format_context 期待不一致.
→ 排除. qdrant_store.py:144-152 返回格式和 llm.py:49 读取格式精准对应,
   `result["metadata"]` key 两侧一致.

**假阳性 3**: `year` 是 int, f-string 渲染出怪格式.
→ 排除. int 2025 在 `f"{year}年报"` 里渲染为 "2025年报", 正常.

**假阴性 1**: header 拼了但 LLM 可能忽略, 只看 content.
→ 超出 TD-8-3 范围. TD-8-3 只问"有没有给 LLM 看", 不问"LLM 看没看". 后者是
   Day 10 Prompt 工程主战场 (TD-8-2).

**假阴性 2**: header 格式太紧凑, LLM 当成装饰性文字.
→ 同假阴性 1, 属于 Day 10 范围. 本任务不深究.

## 4. 判定

**TD-8-3 状态**: ✅ **闭合, 从 Day 10 开工前置清单中移除**

**对 Day 8 "自足 chunk 3 要素" 方法论的影响**:

| 自足要素 | 依赖 | TD-8-3 闭合后 |
|---|---|---|
| 要素 1 主体 (哪家公司) | Prompt 必须拼 metadata.company | ✅ 已拼 (llm.py:50, 55) |
| 要素 2 时间锚点 (什么年) | Prompt 必须拼 metadata.year 或 chunk 正文有年份 | ✅ 已拼 (llm.py:51, 55) |
| 要素 3 核心数据 | chunk 正文 | ✅ 自然满足 |

**3 要素全部闭合**. Day 8 所有基于"自足 chunk"的判定 (U2 L2 / U3 L2b / U4 L2a)
全部站得住, **0 项结论需要撤回**.

## 5. 顺便记录的 2 个 Day 10 Prompt 工程注意点

这 2 条不是 TD-8-3 本身, 是读完 llm.py 时顺手发现的, 登记避免 Day 10 遗漏:

**点 1 (header 格式空间)**: 当前 header 格式
`[{i}] 来源: {company} {year}年报 第{page}页 ({chunk_type})` 比较紧凑.
Day 10 如果要注入"本次查询主公司=招商银行"之类的全局提示, 应该在
`RAG_USER_TEMPLATE` 前面新加 "【查询约束】" 段, 不改 header.

**点 2 (system prompt 的拒答规则和 TD-8-2 耦合)**: `RAG_SYSTEM_PROMPT` 第 2 条
"如果参考资料中没有足够信息回答问题, 请明确说'根据已有资料无法回答此问题'"
是 Day 8 4.5 节推测的"LLM 拒答偏好"的**人为来源之一**. Day 10 TD-8-2 的
多口径规则 (比如"如遇多口径优先本公司口径") 要和这条规则联动设计, 避免打架.

---

**审计耗时**: 约 25 分钟 (预估 30 分钟, 在预算内)
**下一步**: 进入 Day 9 Task 1 — 脚本 27 检索层参数审计 (top_k / RRF k / recall_mult)