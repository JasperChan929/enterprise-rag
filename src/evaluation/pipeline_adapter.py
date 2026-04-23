"""Pipeline adapter: 把 AdvancedRAG 输出适配成 RAGAS 需要的格式.

===============================================================================
出处与目的
===============================================================================
- Day 11 T1b 最终版. 背景: naive_rag/hybrid_rag/advanced_rag 三个 pipeline
  的 sources 字段原本只给 `preview[:100]` 截断, RAGAS Faithfulness 的 NLI
  阶段需要完整 chunk content 才能判 claim 是否被支持. 100 字截断会导致
  judge 误判 "claim 无支持", 是 Day 10 §3.3 "评估指标和答案质量脱钩" 的
  评估层镜像.
- Day 11 T1 决策: 修 pipeline 源码加 `full_content` 字段 (不删 preview,
  加字段不破坏现有调用). 本 adapter 取 full_content, 不取 preview.
- 3 个 pipeline 补丁在 day11-summary §T1 有完整记录.

===============================================================================
关键设计决策
===============================================================================
- **不走 pipeline 内部原始 results**, 而是通过 .query() 标准接口 + 新加的
  full_content 字段. 这样评估的就是真实 pipeline 行为, 不绕 Day 7 定稿代码.
- AdvancedRAGPipeline 单例: 避免每次 eval_run 都重建 BGE embedder / BM25
  索引 / Reranker. 首次 use_reranker=True 时有 2 秒冷启动.
- 返回字段对齐 RAGAS SingleTurnSample 格式: user_input / response /
  retrieved_contexts.

===============================================================================
接口
===============================================================================
    from src.evaluation.pipeline_adapter import eval_run

    result = eval_run(
        question="比亚迪汽车业务2025年的毛利率是多少",
        mode="hybrid",                   # naive/hybrid/multi_query/hyde/full/auto
        use_reranker=False,
        filters={"company": "比亚迪"},
        top_k=5,
        rerank_input_n=20,
    )
    # result = {
    #   "question": str,                   -> RAGAS user_input
    #   "answer": str,                     -> RAGAS response
    #   "retrieved_contexts": list[str],   -> RAGAS retrieved_contexts, 带来源 header + 完整 chunk
    #   "mode_used": str,
    #   "top_pages": list[tuple[str, int]], -> 给人工抽检
    #   "rerank_info": dict | None,
    #   "routing_decision": dict | None,
    # }

===============================================================================
已知局限
===============================================================================
- 单例 pipeline 在多线程/多进程场景不安全 (Day 11 评估脚本串行跑, 不受影响).
- full_content 字段依赖 pipeline 补丁已应用. 如果本地 pipeline 没打补丁,
  fallback 到 preview 并打印 RuntimeWarning (只警告一次).
"""
from __future__ import annotations

import warnings
from src.pipelines.advanced_rag import AdvancedRAGPipeline
from src.pipelines.advanced_rag import AdvancedRAGPipeline, Mode  # Mode 是 Literal 类型别名

_pipeline: AdvancedRAGPipeline | None = None
_warned_no_full_content = False


def _get_pipeline() -> AdvancedRAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = AdvancedRAGPipeline()
    return _pipeline


def _extract_context(src: dict) -> str:
    """从 sources 里一条 src 字典抽出 context 文本.

    优先用 full_content (Day 11 T1 pipeline 补丁后有).
    如果没有 full_content, fallback 到 preview 并警告一次.
    """
    global _warned_no_full_content

    if "full_content" in src and src["full_content"]:
        return src["full_content"]

    if not _warned_no_full_content:
        warnings.warn(
            "pipeline sources 里没有 full_content 字段. "
            "需要先应用 Day 11 T1 的 3 个 pipeline 补丁. "
            "本次 fallback 到 preview[:100], 评估结果会系统性偏低.",
            RuntimeWarning,
            stacklevel=3,
        )
        _warned_no_full_content = True
    return src.get("preview", "")


def eval_run(
    question: str,
    mode: str = "hybrid",
    use_reranker: bool = False,
    filters: dict | None = None,
    top_k: int = 5,
    rerank_input_n: int = 20,
) -> dict:
    """跑一次完整 pipeline, 返回 RAGAS 需要的字段."""
    p = _get_pipeline()

    from typing import cast
    rag_result = p.query(
        question=question,
        mode=cast(Mode, mode),  # adapter 入参是 str, pipeline 要 Literal, 这里信任
        filters=filters,
        top_k=top_k,
        use_reranker=use_reranker,
        rerank_input_n=rerank_input_n,
    )

    retrieved_contexts = []
    top_pages = []
    for i, src in enumerate(rag_result.get("sources", []), 1):
        header = (f"[{i}] 来源: {src.get('company', '?')} "
                  f"{src.get('year', '?')}年报 第{src.get('page', '?')}页 "
                  f"({src.get('type', '?')})")
        content = _extract_context(src)
        retrieved_contexts.append(f"{header}\n{content}")
        top_pages.append((src.get("company", "?"), src.get("page", "?")))

    return {
        "question": question,
        "answer": rag_result.get("answer", ""),
        "retrieved_contexts": retrieved_contexts,
        "mode_used": rag_result.get("mode_used", mode),
        "top_pages": top_pages,
        "rerank_info": rag_result.get("rerank_info"),
        "routing_decision": rag_result.get("routing_decision"),
    }


if __name__ == "__main__":
    print("自测 1: U1 hybrid mode (无 reranker)")
    r = eval_run(
        "比亚迪汽车业务2025年的毛利率是多少",
        mode="hybrid",
        filters={"company": "比亚迪"},
    )
    print(f"  question:           {r['question']}")
    print(f"  mode_used:          {r['mode_used']}")
    print(f"  answer:             {r['answer'][:150]}...")
    print(f"  retrieved_contexts: {len(r['retrieved_contexts'])} 条")
    if r['retrieved_contexts']:
        print(f"  第 1 条 context len: {len(r['retrieved_contexts'][0])}  "
              f"(应远大于 100, 否则补丁未生效)")
    print(f"  top_pages:          {r['top_pages']}")
    print()

    print("自测 2: U1 full + reranker")
    r2 = eval_run(
        "比亚迪汽车业务2025年的毛利率是多少",
        mode="full",
        use_reranker=True,
        filters={"company": "比亚迪"},
    )
    print(f"  mode_used:          {r2['mode_used']}")
    print(f"  rerank_info:        {r2['rerank_info']}")
    print(f"  top_pages:          {r2['top_pages']}")