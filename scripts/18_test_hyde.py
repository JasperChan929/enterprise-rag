"""HyDE 假答案生成 + 检索效果对比实验。

目的:
  1. 看 LLM 生成的假答案是什么样子 (是否像真的年报)
  2. 对比用'原查询 vs HyDE 假答案'做检索的 Top-3 差异
  3. 直观看到 HyDE 在事实性查询上的威力
"""
from dotenv import load_dotenv
load_dotenv(override=True)

from rich.console import Console
from rich.panel import Panel

from src.embeddings.bge import BGEEmbedder
from src.query_transformers.hyde import HyDEGenerator
from src.retrievers.qdrant_store import get_client, search_similar

console = Console()

# ============================================================
# 初始化
# ============================================================

console.print("\n[bold]初始化组件...[/bold]")
hyde = HyDEGenerator()
embedder = BGEEmbedder()
client = get_client()
collection = "financial_reports"
console.print("[green]✅ 就绪[/green]\n")

# ============================================================
# 测试查询集
# ============================================================

test_cases = [
    # 精心选: Day 5 翻车 or 边缘的查询, 看 HyDE 能不能救
    ("茅台 2023 年的营业收入是多少", {"company": "贵州茅台"}),
    ("招商银行 2025 年的不良贷款率", {"company": "招商银行"}),
    ("比亚迪 2025 年研发投入", {"company": "比亚迪"}),
    ("紫金矿业 2025 年矿产金产量", {"company": "紫金矿业"}),
    ("宁德时代磷酸铁锂电池的产品情况", {"company": "宁德时代"}),
    ("国电南自 2024 年净利润", {"company": "国电南自", "year": 2024}),
]

# ============================================================
# 逐条: 生成 HyDE + 对比检索
# ============================================================

for i, (query, filters) in enumerate(test_cases, 1):
    console.print(f"\n{'=' * 80}")
    console.print(f"[bold cyan]Query {i}: {query}[/bold cyan]")
    console.print(f"[dim]Filter: {filters}[/dim]")
    console.print(f"{'=' * 80}\n")

    # ---- Step 1: 生成 HyDE 假答案 ----
    try:
        fake_answer = hyde.generate(query)
    except Exception as e:
        console.print(f"[red]❌ HyDE 生成失败: {e}[/red]")
        continue

    console.print(Panel(
        fake_answer,
        title="[yellow]📝 HyDE 假答案 (陈述句风格)[/yellow]",
        width=80,
    ))

    # ---- Step 2: 分别用原查询 vs 假答案做检索 ----

    # 路径 A: 原查询 (用 encode_query, 带 BGE 前缀)
    q_vec_original = embedder.encode_query(query)
    results_original = search_similar(
        client, q_vec_original.tolist(), collection, limit=3, filters=filters
    )

    # 路径 B: HyDE 假答案 (用 encode, 因为这是陈述句不是查询)
    q_vec_hyde = embedder.encode(fake_answer)
    results_hyde = search_similar(
        client, q_vec_hyde.tolist(), collection, limit=3, filters=filters
    )

    # ---- Step 3: 并排展示 ----
    console.print("\n[bold magenta]📊 原查询 Top-3:[/bold magenta]")
    for j, r in enumerate(results_original, 1):
        meta = r["metadata"]
        preview = r["content"][:70].replace("\n", " ")
        console.print(
            f"  {j}. [{r['score']:.4f}] p.{meta.get('page')} "
            f"({meta.get('chunk_type')}) | {preview}..."
        )

    console.print("\n[bold green]🎯 HyDE 假答案 Top-3:[/bold green]")
    for j, r in enumerate(results_hyde, 1):
        meta = r["metadata"]
        preview = r["content"][:70].replace("\n", " ")
        console.print(
            f"  {j}. [{r['score']:.4f}] p.{meta.get('page')} "
            f"({meta.get('chunk_type')}) | {preview}..."
        )

    # ---- Step 4: 诊断 Top-3 重合度 ----
    pages_orig = {r["metadata"].get("page") for r in results_original}
    pages_hyde = {r["metadata"].get("page") for r in results_hyde}
    overlap = pages_orig & pages_hyde
    only_hyde = pages_hyde - pages_orig

    console.print(
        f"\n[dim]重合页码: {sorted(overlap) if overlap else '无'} | "
        f"仅 HyDE 召回: {sorted(only_hyde) if only_hyde else '无'}[/dim]"
    )


# ============================================================
# 收尾
# ============================================================

console.print(Panel.fit(
    "[bold]观察要点:[/bold]\n"
    "1. HyDE 假答案像不像真的年报? (专业术语、句式、维度)\n"
    "2. 假答案里的具体数字是不是瞎编的? (对比真实答案)\n"
    "3. 原查询 vs HyDE 的 Top-3 页码差多少? 差得越多越说明 HyDE 在探索不同的向量区域\n"
    "4. HyDE 独家召回的页面 (only HyDE) 有没有真相关的内容?\n"
    "5. 对于事实性问题 (营收/产量), HyDE 的 Top-1 是不是比原查询更精准?",
    title="[cyan]Step 2 完成检查[/cyan]",
))