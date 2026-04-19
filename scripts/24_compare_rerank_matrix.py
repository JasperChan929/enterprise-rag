"""Day 7: Reranker A/B 矩阵对比.

对 3 个代表性查询 × 3 个检索模式 × 有无 Reranker = 18 次查询,
量化延迟代价和排序变化, 为 Day 7 笔记准备数据.

查询设计原则:
  - 避开数据边界类 (Day 6 宁德 / Day 7 招行已证明 RAG 救不了这类)
  - 覆盖不同类型: 精确数字 / 多维度 / 跨公司

输出:
  控制台打印对比矩阵 (CSV 友好格式, 方便粘到笔记)
"""
from dotenv import load_dotenv
load_dotenv(override=True)

import time
from src.pipelines.advanced_rag import AdvancedRAGPipeline


QUERIES = [
    {
        "name": "茅台营收",
        "query": "贵州茅台2023年的营业收入是多少?",
        "filters": {"company": "贵州茅台"},
    },
    {
        "name": "比亚迪毛利率",
        "query": "比亚迪2025年汽车业务毛利率",
        "filters": {"company": "比亚迪"},
    },
    {
        "name": "长江电力发电量",
        "query": "长江电力2024年总发电量是多少?",
        "filters": {"company": "长江电力"},
    },
]

MODES = ["hybrid", "hyde", "full"]


def run_one(rag, query_cfg: dict, mode: str, use_reranker: bool) -> dict:
    """跑一次查询, 返回延迟和 Top-5 信息.
    
    Args:
        rag: AdvancedRAGPipeline 实例
        query_cfg: 查询配置, 含 query / filters / name
        mode: 检索模式 (hybrid / hyde / full)
        use_reranker: 是否启用 Reranker 精排
    
    Returns:
        {
            "answer": 完整答案 (用于人工判定),
            "pages": Top-5 页码列表,
            "elapsed": 端到端耗时(秒),
            "rerank_ms": Reranker 精排耗时(ms), 未开启时为 None,
        }
    """
    t0 = time.time()
    result = rag.query(
        question=query_cfg["query"],
        mode=mode,
        filters=query_cfg["filters"],
        top_k=5,
        use_reranker=use_reranker,
        rerank_input_n=20,
    )
    elapsed = time.time() - t0
    
    rerank_ms = None
    if result.get("rerank_info"):
        rerank_ms = result["rerank_info"]["elapsed_ms"]
    
    return {
        "answer": result["answer"],
        "pages": [s["page"] for s in result["sources"]],
        "elapsed": elapsed,
        "rerank_ms": rerank_ms,
    }


def print_query_block(query_cfg: dict, results: dict):
    """打印单个查询的对比块.
    
    results 结构: {(mode, use_rr): run_one 返回值}
    """
    print("\n" + "=" * 78)
    print(f"查询: {query_cfg['name']}  |  {query_cfg['query']}")
    print("=" * 78)
    
    # 矩阵头
    print(f"\n{'Mode':<10} {'RR':<5} {'耗时':<8} {'精排':<8} {'Top-5 页面':<30}")
    print("-" * 78)
    
    for mode in MODES:
        for use_rr in [False, True]:
            r = results[(mode, use_rr)]
            rr_label = "ON" if use_rr else "OFF"
            rerank_str = f"{r['rerank_ms']:.0f}ms" if r['rerank_ms'] else "-"
            pages_str = str(r['pages'])
            print(f"{mode:<10} {rr_label:<5} {r['elapsed']:<6.2f}s  "
                  f"{rerank_str:<8} {pages_str:<30}")
    
    # 打印每个模式的 A/B 答案对比 (只展开答案, 便于人工判读)
    for mode in MODES:
        print(f"\n--- {mode} 模式答案对比 ---")
        a = results[(mode, False)]
        b = results[(mode, True)]
        print(f"  [RR=OFF] {a['answer'][:200]}")
        print(f"  [RR=ON ] {b['answer'][:200]}")


def print_summary(all_results: dict):
    """打印跨查询的汇总统计.
    
    all_results 结构: {query_name: {(mode, use_rr): run_one 返回值}}
    """
    print("\n\n" + "=" * 78)
    print("汇总: 延迟代价量化")
    print("=" * 78)
    print(f"\n{'查询':<15} {'Mode':<10} {'OFF 耗时':<10} {'ON 耗时':<10} "
          f"{'精排耗时':<10} {'延迟代价':<10}")
    print("-" * 78)
    
    for q_name, results in all_results.items():
        for mode in MODES:
            a = results[(mode, False)]
            b = results[(mode, True)]
            cost = b['elapsed'] - a['elapsed']
            rerank_str = f"{b['rerank_ms']:.0f}ms" if b['rerank_ms'] else "-"
            print(f"{q_name:<15} {mode:<10} {a['elapsed']:<8.2f}s  "
                  f"{b['elapsed']:<8.2f}s  {rerank_str:<10} "
                  f"+{cost*1000:.0f}ms")


def main():
    print("初始化 AdvancedRAGPipeline...")
    rag = AdvancedRAGPipeline()
    print("预热 Reranker...")
    _ = rag.reranker
    
    all_results = {}
    
    for query_cfg in QUERIES:
        print(f"\n\n{'#'*78}")
        print(f"# 测试查询: {query_cfg['name']}")
        print(f"{'#'*78}")
        
        results = {}
        for mode in MODES:
            for use_rr in [False, True]:
                print(f"  跑 mode={mode}, use_reranker={use_rr}...")
                results[(mode, use_rr)] = run_one(rag, query_cfg, mode, use_rr)
        
        all_results[query_cfg['name']] = results
        print_query_block(query_cfg, results)
    
    print_summary(all_results)
    print("\n完成.")


if __name__ == "__main__":
    main()