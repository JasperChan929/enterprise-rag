# Day 11 RAGAS 聚合结果 (自动生成)

- 总记录数: 45
- 生成时间: Day 11 T2 自动生成

## mode × metric 平均分

| mode | faithfulness | answer_relevancy | context_precision | refusal_rate |
|---|---|---|---|---|
| hybrid | 0.908 (n=15) | 0.216 (n=15) | 0.611 (n=15) | 33% (5/15) |
| hyde | 0.934 (n=14) | 0.365 (n=15) | 0.692 (n=15) | 27% (4/15) |
| full+reranker | 0.930 (n=14) | 0.256 (n=15) | 0.484 (n=15) | 60% (9/15) |

## 按 query_type 拆分 (faithfulness 平均)

| query_type | hybrid | hyde | full+reranker |
|---|---|---|---|
| full_friendly | 1.000 | 1.000 | 1.000 |
| hyde_friendly | 1.000 | 1.000 | 1.000 |
| multi_chunk_reasoning | 1.000 | 0.500 | 1.000 |
| multi_perspective | 0.733 | 1.000 | 0.929 |
| numeric_simple | 1.000 | 1.000 | 1.000 |
| qualitative | 0.875 | 0.952 | 0.803 |
| refusal_expected | 1.000 | 1.000 | 1.000 |
| reranker_friendly | 1.000 | - | - |
| synonym_recall | 0.750 | 1.000 | 1.000 |
| table_extraction | 0.778 | 0.714 | 0.750 |

## TD-10-3 FN-2 候选 (is_refusal=True 但答案较长)

定义: is_refusal 命中但 answer_length > 200 字符 — 疑似'拒答句+半作答'

| qid | mode | answer_length | faithfulness | refusal_pattern |
|---|---|---|---|---|
| U1 | hyde | 227 | 1.000 | 未提供 |
| U4 | full+reranker | 366 | 0.857 | 未提供 |
| U5 | hybrid | 222 | 0.875 | 参考资料中未 |
| U5 | full+reranker | 285 | 0.909 | 无法回答 |
| U8 | full+reranker | 215 | 1.000 | 参考资料中未 |
| U9 | hybrid | 305 | 0.778 | 未提供 |
| U9 | hyde | 249 | 0.714 | 参考资料未 |
| U9 | full+reranker | 209 | 0.750 | 未提供 |

*共 8 条疑似 FN-2. T3 人工抽检核对, Day 12 决策修 is_refusal 用*

## 明细参见

- `docs/day11-ragas-results.jsonl` - 每行 1 条 (U, mode) 完整 trace
- T3 人工抽检重点看: is_refusal=True 的条目 + faithfulness < 0.5 的条目