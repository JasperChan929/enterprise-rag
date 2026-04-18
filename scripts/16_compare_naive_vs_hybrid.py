"""Naive RAG vs Hybrid RAG 对比实验。

核心目的:
  1. 在同一组查询上, 对比 Naive (仅 Dense) 和 Hybrid (Dense+BM25+RRF) 的表现
  2. 可解释: 每条 Hybrid 结果标注 dense_rank / bm25_rank / rrf_score
  3. 为 Day 9 RAGAS 评估积累直觉
"""
from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.pipelines.hybrid_rag import HybridRAGPipeline
from src.pipelines.naive_rag import NaiveRAGPipeline

console = Console()

# ============================================================
# 初始化两个 Pipeline
# ============================================================

console.print("\n[bold]初始化 Naive RAG...[/bold]")
naive = NaiveRAGPipeline(top_k=5)
console.print("[green]✅ Naive 就绪[/green]\n")

console.print("[bold]初始化 Hybrid RAG (含 BM25 构建)...[/bold]")
hybrid = HybridRAGPipeline(top_k=5)
console.print("[green]✅ Hybrid 就绪[/green]\n")


# ============================================================
# 测试问题集
# ============================================================

test_cases = [
    ("茅台 2023 年的营业收入是多少?", {"company": "贵州茅台"}),
    ("招商银行 2025 年的不良贷款率是多少?", {"company": "招商银行"}),
    ("宁德时代磷酸铁锂电池的产品情况", {"company": "宁德时代"}),
    ("比亚迪 2025 年研发投入是多少?", {"company": "比亚迪"}),
    ("长江电力的装机容量", {"company": "长江电力"}),
    ("紫金矿业 2025 年矿产金产量", {"company": "紫金矿业"}),
    ("国电南自 2024 年净利润是多少?", {"company": "国电南自", "year": 2024}),
]


# ============================================================
# 跑对比
# ============================================================

for question, filters in test_cases:
    console.print(f"\n{'=' * 80}")
    console.print(f"[bold cyan]Q: {question}[/bold cyan]")
    console.print(f"[dim]Filter: {filters}[/dim]")
    console.print(f"{'=' * 80}\n")

    # ---- Naive ----
    naive_result = naive.query(question, filters=filters)
    console.print(Panel(
        naive_result["answer"],
        title="[bold magenta]📊 Naive RAG 答案[/bold magenta]",
        width=80,
    ))
    console.print("[dim]Naive 检索来源:[/dim]")
    for i, s in enumerate(naive_result["sources"], 1):
        preview = s['preview'].replace('\n', ' ')
        console.print(
            f"  [dim]{i}. [{s['score']:.4f}] "
            f"p.{s['page']} ({s['type']}) | {preview[:60]}...[/dim]"
        )

    # ---- Hybrid ----
    hybrid_result = hybrid.query(question, filters=filters)
    console.print()
    console.print(Panel(
        hybrid_result["answer"],
        title="[bold green]🔀 Hybrid RAG 答案[/bold green]",
        width=80,
    ))

    # Hybrid 的来源表格(带融合诊断)
    table = Table(
        title="Hybrid 检索来源(含融合诊断)",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("#", width=3)
    table.add_column("RRF", justify="right", width=7)
    table.add_column("Dense", justify="center", width=10)
    table.add_column("BM25", justify="center", width=10)
    table.add_column("Page", width=5)
    table.add_column("Type", width=6)
    table.add_column("Preview", max_width=35)

    for i, s in enumerate(hybrid_result["sources"], 1):
        dense_info = (
            f"#{s['dense_rank']} ({s['dense_score']:.3f})"
            if s["dense_rank"] else "-"
        )
        bm25_info = (
            f"#{s['bm25_rank']} ({s['bm25_score']:.1f})"
            if s["bm25_rank"] else "-"
        )
        preview = s["preview"].replace("\n", " ")[:35]
        table.add_row(
            str(i),
            f"{s['rrf_score']:.4f}",
            dense_info,
            bm25_info,
            str(s["page"]),
            str(s["type"]),
            preview,
        )
    console.print(table)


# ============================================================
# 收尾
# ============================================================

console.print(Panel.fit(
    "[bold]观察要点:[/bold]\n"
    "1. 哪些查询 Hybrid 答案明显更好? 看'融合诊断'里两路的排名分布\n"
    "2. RRF Top-5 里是否出现了'Dense 单路进不去 Top-5'的文档? 说明 BM25 救场\n"
    "3. 对于两路 Top-1 一致的共识题, RRF 会不会过度集中? 多样性如何?\n"
    "4. 招行不良贷款率: 对比 Day 4 的拒答, 现在能否答出具体数字(0.93% / 0.94%)?",
    title="[cyan]Day 5 完成检查[/cyan]",
))