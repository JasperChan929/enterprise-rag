"""验证 BM25 索引 + Dense vs BM25 对比实验。

核心目的:
  1. 从 Qdrant 加载全部 6881 chunks, 重建 BM25 索引
  2. 跑 6 个行业代表性查询, 对比 Dense 和 BM25 的 Top-3 差异
  3. 直观看到各自的优势场景
"""
from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.panel import Panel

from src.embeddings.bge import BGEEmbedder
from src.loaders.base import Chunk
from src.retrievers.bm25_store import BM25Store
from src.retrievers.qdrant_store import get_client, search_similar

console = Console()

# ============================================================
# Step 1: 从 Qdrant 加载全部 chunks
# ============================================================

console.print("\n[bold]Step 1: 从 Qdrant 加载 chunks[/bold]")

client = get_client()
collection = "financial_reports"

all_points = []
offset = None
while True:
    response, offset = client.scroll(
        collection_name=collection,
        limit=500,
        offset=offset,
        with_payload=True,
        with_vectors=False,
    )
    all_points.extend(response)
    if offset is None:
        break

console.print(f"   加载了 {len(all_points)} 个 points")

# 转回 Chunk 对象(content + metadata 分离)
chunks = []
for p in all_points:
    payload = dict(p.payload or {})
    content = payload.pop("content", "")
    chunk_type = payload.pop("chunk_type", "text")
    chunks.append(Chunk(
        content=content,
        chunk_type=chunk_type,
        metadata=payload,
    ))

# ============================================================
# Step 2: 构建 BM25 索引
# ============================================================

console.print("\n[bold]Step 2: 构建 BM25 索引[/bold]")
bm25 = BM25Store()
bm25.build(chunks)

# ============================================================
# Step 3: Dense vs BM25 对比实验
# ============================================================

console.print("\n[bold]Step 3: Dense vs BM25 检索对比[/bold]\n")

embedder = BGEEmbedder()

# 6 类代表性查询(每家公司/行业一个)
test_queries = [
    # (查询, 过滤, 难度描述)
    ("茅台 2023 年营业收入", {"company": "贵州茅台"}, "精确数字·表格密集"),
    ("招商银行不良贷款率", {"company": "招商银行"}, "专业术语·但已知招行表格丢失"),
    ("宁德时代磷酸铁锂电池出货量", {"company": "宁德时代"}, "行业术语+数字"),
    ("比亚迪 2025 年研发投入", {"company": "比亚迪"}, "通用财务指标"),
    ("长江电力装机容量", {"company": "长江电力"}, "行业术语"),
    ("紫金矿业矿产金产量", {"company": "紫金矿业"}, "行业专有术语"),
    ("国电南自 2024 年净利润", {"company": "国电南自", "year": 2024}, "多条件过滤"),
]

for query, filters, difficulty in test_queries:
    console.print(f"\n{'=' * 75}")
    console.print(f"[bold cyan]Query: {query}[/bold cyan]")
    console.print(f"[dim]难度: {difficulty} | Filter: {filters}[/dim]")
    console.print(f"{'=' * 75}")

    # 查看分词(诊断用)
    tokens = bm25.debug_tokenize(query)
    console.print(f"[yellow]分词: {tokens}[/yellow]")

    # ---- Dense ----
    q_vec = embedder.encode_query(query)
    dense_results = search_similar(
        client, q_vec.tolist(), collection, limit=3, filters=filters
    )

    console.print("\n[bold magenta]📊 Dense Top-3:[/bold magenta]")
    for i, r in enumerate(dense_results, 1):
        meta = r["metadata"]
        preview = r["content"][:75].replace("\n", " ")
        console.print(
            f"  {i}. [{r['score']:.4f}] p.{meta.get('page')} "
            f"({meta.get('chunk_type')}) | {preview}..."
        )

    # ---- BM25 ----
    bm25_results = bm25.search(query, limit=3, filters=filters)

    console.print("\n[bold green]🔤 BM25 Top-3:[/bold green]")
    if not bm25_results:
        console.print("  [red](无匹配, 可能是分词后查询为空)[/red]")
    else:
        for i, r in enumerate(bm25_results, 1):
            meta = r["metadata"]
            preview = r["content"][:75].replace("\n", " ")
            console.print(
                f"  {i}. [{r['score']:.2f}] p.{meta.get('page')} "
                f"({meta.get('chunk_type')}) | {preview}..."
            )

# ============================================================
# 小结
# ============================================================

console.print(Panel.fit(
    "[bold]观察要点:[/bold]\n"
    "1. 哪些查询 BM25 大幅优于 Dense?(专有名词/行业术语类)\n"
    "2. 哪些查询 Dense 反超?(语义改写类)\n"
    "3. 招行查询: 两者都翻车 → 印证表格数据丢失的诊断\n"
    "4. 有没有查询两边 Top-3 完全不重合?(这种场景 RRF 融合价值最大)",
    title="[cyan]下一步: Step 3 RRF 融合[/cyan]",
))