"""6 种模式的终极对比实验。

在精心选择的测试查询上, 对比以下模式:
  - naive:       纯 Dense (Day 4 baseline)
  - hybrid:      Day 5 的 Hybrid
  - multi_query: Hybrid + Multi-Query
  - hyde:        Hybrid + HyDE
  - full:        全开
  - auto:        LLM Router 决策

目的:
  1. 量化每种模式在不同查询类型上的效果
  2. 验证 auto 模式是否真的"聪明"
  3. 为 Day 6 笔记和 Day 9 RAGAS 评估提供直观依据
"""
import time

from dotenv import load_dotenv
load_dotenv(override=True)

from rich.console import Console
from rich.panel import Panel

from src.pipelines.advanced_rag import AdvancedRAGPipeline

console = Console()

# ============================================================
# 初始化
# ============================================================

console.print("\n[bold]初始化 Advanced RAG Pipeline (这需要 10-15 秒)...[/bold]")
t0 = time.time()
rag = AdvancedRAGPipeline()
console.print(f"[green]✅ 就绪 ({time.time()-t0:.1f}s)[/green]\n")


# ============================================================
# 测试查询 (选择有代表性的 4 个, 避免总时间太长)
# ============================================================

test_cases = [
    # 1. 事实问句 (HyDE 应该最强)
    ("茅台 2023 年的营业收入是多少?", {"company": "贵州茅台"}),

    # 2. 抽象开放 (Multi-Query 应该最强)
    ("宁德时代磷酸铁锂电池的产品情况", {"company": "宁德时代"}),

    # 3. 专业术语 (Hybrid 就够了, 不需要额外)
    ("招商银行不良贷款率", {"company": "招商银行"}),

    # 4. 多维度复杂 (Full 应该最强)
    ("紫金矿业海外业务的营收和风险", {"company": "紫金矿业"}),
]

from src.pipelines.advanced_rag import Mode
MODES:list[Mode] = ["naive", "hybrid", "multi_query", "hyde", "full", "auto"]


# ============================================================
# 跑实验
# ============================================================

for q_idx, (question, filters) in enumerate(test_cases, 1):
    console.print(f"\n{'=' * 85}")
    console.print(f"[bold cyan]Query {q_idx}: {question}[/bold cyan]")
    console.print(f"[dim]Filter: {filters}[/dim]")
    console.print(f"{'=' * 85}\n")

    for mode in MODES:
        console.print(f"\n[bold yellow]--- Mode: {mode} ---[/bold yellow]")

        t0 = time.time()
        try:
            result = rag.query(question, mode=mode, filters=filters)
            elapsed = time.time() - t0
        except Exception as e:
            console.print(f"[red]❌ 失败: {e}[/red]")
            continue

        # 答案
        console.print(Panel(
            result["answer"],
            title=f"[green]答案 (mode_used={result['mode_used']}, {elapsed:.1f}s)[/green]",
            width=85,
        ))

        # 如果是 auto 模式, 展示 Router 决策
        if mode == "auto" and result["routing_decision"]:
            rd = result["routing_decision"]
            console.print(
                f"[dim]  Router 决策: multi_query={rd['use_multi_query']}, "
                f"hyde={rd['use_hyde']} → {rd['reason']}[/dim]"
            )

        # 探针信息
        probe_types = [p["type"] for p in result["probes"]]
        console.print(f"[dim]  探针: {probe_types}[/dim]")

        # Top 3 来源
        console.print("[dim]  Top-3 来源:[/dim]")
        for i, s in enumerate(result["sources"][:3], 1):
            preview = s["preview"].replace("\n", " ")[:60]
            console.print(
                f"[dim]    {i}. [{s['score']:.4f}] p.{s['page']} ({s['type']}) | {preview}...[/dim]"
            )


# ============================================================
# 收尾
# ============================================================

console.print(Panel.fit(
    "[bold]观察要点:[/bold]\n"
    "1. 茅台营收: naive 答对但 hybrid 拒答? hyde/full 能不能救回?\n"
    "2. 宁德产品情况: naive/hybrid 都拒答? multi_query/full 有没有突破?\n"
    "3. 招行不良贷款率: hybrid 已经够好, 其他模式是不是多余的?\n"
    "4. 紫金海外营收+风险: full 模式是不是给了最全面的答案?\n"
    "5. auto 模式选的策略是不是和手动指定的效果一致?\n"
    "6. 延迟对比: naive/hybrid 最快, full 最慢, auto 居中吗?",
    title="[cyan]Day 6 Step 4 完成检查[/cyan]",
))