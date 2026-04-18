# Enterprise RAG - 项目全景 (Day 1-6 完成)

## 12 天路线图
- ✅ Day 1: 环境 + 心智模型
- ✅ Day 2: PDF 加载 + 切分
- ✅ Day 3: BGE Embedding
- ✅ Day 4: Qdrant + LLM + Naive RAG
- ✅ Day 5: Hybrid Search (BM25 + RRF)
- ✅ Day 6: Advanced RAG (Multi-Query + HyDE + Router)
- ⏳ Day 7: Cross-Encoder Reranker (本日)
- ⏳ Day 8: 评估集构造 (30-50 QA 对)
- ⏳ Day 9: RAGAS 评估 + 消融对比
- ⏳ Day 10: 幻觉抑制
- ⏳ Day 11: FastAPI 服务化 + Streamlit
- ⏳ Day 12: README + 架构图 + 技术博客

## 准确率进步追踪
| Day | Pipeline | 拒答率 |
|---|---|---|
| 4 | Naive | 75% |
| 5 | Hybrid | 50% |
| 6 | Auto | 25% |
| 7+ | 预期 | 15% |

## 核心技术栈
- DeepSeek-Chat (LLM)
- BGE-small-zh-v1.5 (Embedding, 512 维)
- BGE-Reranker-base (Day 7 要引入)
- Qdrant (向量库, Docker)
- jieba + rank-bm25 (稀疏检索)
- tiktoken (token 计数)
- pdfplumber (PDF 加载)
- FastAPI + Streamlit (Day 11 规划)

## 关键技术债优先级
**P1 待修** (影响主线):
- (无, Day 6 技术债全是 P2-P4)

**P2 下阶段修**:
- TD-5-1: 招行 PDF 表格抽取率 1% (Day 9 评估后定)
- TD-6-3: Router 规则需实证校准 (Day 9 评估后定)
- TD-6-4: 紫金 PDF header 清理不彻底
- TD-6-5: Full 模式宽召回稀释 (**Day 7 Reranker 直接解**)

**P3 计划内**:
- TD-6-1: Filter 盲区 (跨年查询)
- TD-6-2: Router 对"公司+术语"误判

## 我的面试目标
用这个项目展示:
1. 完整 RAG 工程能力 (Naive → Advanced → Production)
2. 诊断驱动的优化思维 (不是堆栈技术)
3. 区分 bug/特性/数据边界/方法边界的工程成熟度