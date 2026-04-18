"""端到端测试 Naive RAG Pipeline。
这是你的系统第一次"开口说话"!
"""
import os
from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from src.pipelines.naive_rag import NaiveRAGPipeline

console = Console()

# 初始化 pipeline
console.print("\n[bold]初始化 Naive RAG Pipeline...[/bold]")
rag = NaiveRAGPipeline(top_k=5)
console.print("[green]✅ 就绪[/green]\n")

# 测试问题集(覆盖不同类型)
test_questions = [
    # (问题, 过滤条件, 预期难度)
    ("茅台 2023 年的营业收入是多少?", {"company": "贵州茅台"}, "精确数字(难)"),
    ("宁德时代的核心竞争力是什么?", {"company": "宁德时代"}, "定性问题(易)"),
    ("招商银行 2025 年的不良贷款率是多少?", {"company": "招商银行"}, "精确数字(难)"),
    ("茅台对未来发展有什么规划?", {"company": "贵州茅台"}, "开放性(中)"),
]

for question, filters, difficulty in test_questions:
    console.print(f"\n{'='*70}")
    console.print(f"[bold cyan]Q: {question}[/bold cyan]")
    console.print(f"[dim]难度: {difficulty} | 过滤: {filters}[/dim]")
    console.print(f"{'='*70}")

    result = rag.query(question, filters=filters)

    # 答案
    console.print(Panel(result["answer"], title="[bold green]答案[/bold green]", width=70))

    # 来源
    console.print("[dim]检索来源:[/dim]")
    for i, s in enumerate(result["sources"], 1):
        console.print(
            f"  [dim]{i}. [{s['score']:.4f}] {s['company']} {s['year']}年报"
            f" p.{s['page']} ({s['type']})[/dim]"
        )