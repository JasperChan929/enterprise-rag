"""Multi-Query 改写质量验证实验。

目的: 亲眼看到 LLM 把原查询改写成了什么, 
     评估子查询是否真的"从不同角度探索"。
"""
from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.panel import Panel

from src.query_transformers.multi_query import MultiQueryRewriter

console = Console()

# ============================================================
# 初始化
# ============================================================

console.print("\n[bold]初始化 Multi-Query 改写器...[/bold]")
rewriter = MultiQueryRewriter(num_queries=4, include_original=True)
console.print("[green]✅ 就绪[/green]\n")

# ============================================================
# 测试查询集 (精心挑选, 覆盖不同难度)
# ============================================================

test_queries = [
    # 1. 极度泛化的查询 (Day 5 翻车案例)
    "宁德时代磷酸铁锂电池的产品情况",

    # 2. 精确数字查询 (看 LLM 会不会把它改坏)
    "茅台 2023 年的营业收入是多少",

    # 3. 专业术语查询
    "招商银行不良贷款率",

    # 4. 开放性查询
    "比亚迪的核心竞争力",

    # 5. 多维度查询
    "紫金矿业的海外业务布局",

    # 6. 时间跨度查询
    "国电南自 2024 年相比 2023 年有什么变化",
]

# ============================================================
# 逐条改写 + 展示
# ============================================================

for i, query in enumerate(test_queries, 1):
    console.print(f"\n{'=' * 75}")
    console.print(f"[bold cyan]Query {i}: {query}[/bold cyan]")
    console.print(f"{'=' * 75}")

    try:
        sub_queries = rewriter.rewrite(query)
    except Exception as e:
        console.print(f"[red]❌ 改写失败: {e}[/red]")
        continue

    console.print(f"\n[bold]生成了 {len(sub_queries)} 个查询:[/bold]")
    for j, sq in enumerate(sub_queries):
        tag = "[dim][原始][/dim]" if j == 0 else f"[green][子 {j}][/green]"
        console.print(f"  {tag} {sq}")

# ============================================================
# 观察要点
# ============================================================

console.print(Panel.fit(
    "[bold]观察要点:[/bold]\n"
    "1. 子查询是'不同角度'还是'表面同义'? (好: 产品列表/技术/应用/市场; 差: 产品情况/产品详情/产品说明)\n"
    "2. 抽象查询 (产品情况) 被拆得多具体? 有没有冒出具体产品名或指标?\n"
    "3. 精确查询 (营业收入是多少) 有没有被改坏? 数字应被保留\n"
    "4. LLM 会不会跑偏到无关维度 (比如查'营业收入'扯到'战略规划')?\n"
    "5. 对比你自己改写: 如果让你改写同样查询, 你会不会这样想?",
    title="[cyan]Step 1 完成检查[/cyan]",
))