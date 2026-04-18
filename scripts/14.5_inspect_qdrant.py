"""摸清 Qdrant 里当前有哪些公司、哪些年份、每家多少 chunks。
在建 BM25 前必须先确认数据和词典对齐。
"""
from collections import Counter
from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.table import Table

from src.retrievers.qdrant_store import get_client

console = Console()

client = get_client()
collection = "financial_reports"

# 全量 scroll 出 payload(不要向量, 省流量)
all_payloads = []
offset = None
while True:
    response, offset = client.scroll(
        collection_name=collection,
        limit=500,
        offset=offset,
        with_payload=True,
        with_vectors=False,
    )
    for point in response:
        all_payloads.append(point.payload or {})
    if offset is None:
        break

console.print(f"\n[bold]Qdrant 总 points: {len(all_payloads)}[/bold]\n")

# 1) 公司 × 年份 分布
combos = Counter(
    (p.get("company", "未知"), p.get("year", "未知"))
    for p in all_payloads
)

table = Table(title="公司 × 年份 × chunks 分布")
table.add_column("股票代码", style="cyan")
table.add_column("公司", style="green")
table.add_column("年份", style="yellow")
table.add_column("chunks 数", justify="right")

# 顺便拉一下 stock_code
code_map = {}
for p in all_payloads:
    company = p.get("company", "未知")
    code = p.get("stock_code", "??")
    if company not in code_map:
        code_map[company] = code

for (company, year), count in sorted(combos.items()):
    table.add_row(code_map.get(company, "??"), str(company), str(year), str(count))

console.print(table)

# 2) chunk_type 分布
type_counter = Counter(p.get("chunk_type", "未知") for p in all_payloads)
console.print(f"\n[bold]chunk_type 分布:[/bold] {dict(type_counter)}")

# 3) 每家公司的 chunk_type 分布(看表格/文字比例是否合理)
console.print("\n[bold]每家公司的 text / table 分布:[/bold]")
for company in sorted(set(p.get("company", "未知") for p in all_payloads)):
    company_chunks = [p for p in all_payloads if p.get("company") == company]
    sub_types = Counter(p.get("chunk_type", "未知") for p in company_chunks)
    text_n = sub_types.get("text", 0)
    table_n = sub_types.get("table", 0)
    total = text_n + table_n
    if total > 0:
        ratio = table_n / total * 100
        console.print(
            f"  {company:8s} : text={text_n:4d}  table={table_n:4d}  "
            f"(表格占比 {ratio:.1f}%)"
        )