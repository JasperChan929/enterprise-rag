"""Query Router 决策质量验证。

目的:
  1. 看 LLM 对各种查询是否判断合理
  2. 检查边界 case (极短术语 / 复杂多维度)
  3. 为 Step 4 完整 Pipeline 打基础
"""
from dotenv import load_dotenv
load_dotenv(override=True)

from rich.console import Console
from rich.table import Table

from src.query_transformers.router import QueryRouter

console = Console()

console.print("\n[bold]初始化 Query Router...[/bold]")
router = QueryRouter()
console.print("[green]✅ 就绪[/green]\n")

# ============================================================
# 测试查询集 (覆盖 4 种组合 + 边界 case)
# ============================================================

test_queries = [
    # === 精准术语 (预期: 两者都 false) ===
    "招商银行不良贷款率",
    "比亚迪毛利率",
    "净资产收益率",

    # === 事实性问句 (预期: HyDE only) ===
    "茅台 2023 年营业收入是多少",
    "国电南自 2024 年净利润",
    "紫金矿业矿产金产量是多少吨",

    # === 抽象开放 (预期: Multi-Query only) ===
    "比亚迪的核心竞争力",
    "宁德时代磷酸铁锂电池的产品情况",
    "招商银行的战略方向",

    # === 复杂多维度 (预期: 两者都 true) ===
    "紫金矿业海外业务的营收和风险",
    "长江电力的装机容量、发电量和上网电价",

    # === 边界 case (看 Router 怎么判) ===
    "茅台",                              # 极短, 只有公司名
    "研发",                              # 极短, 只有关键词
    "你好",                              # 完全无关
]


# ============================================================
# 跑路由 + 表格展示
# ============================================================

table = Table(
    title="Query Router 决策结果",
    show_header=True,
    header_style="bold cyan",
)
table.add_column("Query", max_width=35)
table.add_column("Multi-Q", justify="center", width=8)
table.add_column("HyDE", justify="center", width=6)
table.add_column("策略", justify="center", width=18)
table.add_column("Reason", max_width=30)

for query in test_queries:
    decision = router.route(query)
    mq = decision["use_multi_query"]
    hd = decision["use_hyde"]

    # 策略总结
    if mq and hd:
        strategy = "🔥 Full"
    elif mq:
        strategy = "🔀 +Multi-Query"
    elif hd:
        strategy = "📝 +HyDE"
    else:
        strategy = "⚡ Hybrid only"

    table.add_row(
        query,
        "✓" if mq else "✗",
        "✓" if hd else "✗",
        strategy,
        decision["reason"],
    )

console.print(table)

# ============================================================
# 观察要点
# ============================================================

from rich.panel import Panel

console.print(Panel.fit(
    "[bold]观察要点:[/bold]\n"
    "1. 精准术语类 (不良贷款率/毛利率) 是否被判为 'Hybrid only'?\n"
    "2. 事实性问句 (...是多少) 是否被判为 '+HyDE'?\n"
    "3. 抽象查询 ('核心竞争力''产品情况') 是否被判为 '+Multi-Query'?\n"
    "4. 复杂多维度 ('营收和风险') 是否两者都开?\n"
    "5. 边界 case ('茅台' '研发' '你好') Router 怎么兜底?\n"
    "6. reason 字段有没有给出合理的判断依据?",
    title="[cyan]Step 3 完成检查[/cyan]",
))