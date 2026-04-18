"""把 4 份年报全部加载、切分、向量化、写入 Qdrant。

这是 Day 4 的核心脚本,跑完后 Qdrant 里就有完整的知识库了。
"""
import os
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from src.chunking.recursive import chunk_documents
from src.embeddings.bge import BGEEmbedder
from src.loaders.pdf_loader import load_pdf
from src.retrievers.qdrant_store import (
    chunks_to_points,
    create_collection,
    get_client,
    upsert_points,
)

# 配置
DATA_DIR = Path("data/raw")
COLLECTION = "financial_reports"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

# 找到所有年报 PDF
pdf_files = sorted(DATA_DIR.glob("*.pdf"))
print(f"找到 {len(pdf_files)} 份年报:")
for f in pdf_files:
    print(f"  {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)")
print()

# Step 1: 初始化
print("=" * 60)
print("Step 1: 初始化 Qdrant + Embedding 模型")
print("=" * 60)
client = get_client()
create_collection(client, COLLECTION, recreate=True)  # 开发阶段每次重建
embedder = BGEEmbedder()
print()

# Step 2: 逐份处理
all_chunks = []
all_vectors = []

for pdf_path in pdf_files:
    print("=" * 60)
    print(f"处理: {pdf_path.name}")
    print("=" * 60)

    # 加载
    t0 = time.time()
    raw_chunks = load_pdf(pdf_path)
    chunks = chunk_documents(raw_chunks, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    t1 = time.time()
    print(f"  加载+切分: {len(chunks)} chunks ({t1-t0:.1f}s)")

    # 向量化
    t0 = time.time()
    vectors = embedder.encode_chunks(chunks, batch_size=32, show_progress=False)
    t1 = time.time()
    print(f"  向量化: {vectors.shape} ({t1-t0:.1f}s)")

    all_chunks.extend(chunks)
    all_vectors.append(vectors)

    # 入库
    import numpy as np
    points = chunks_to_points(chunks, vectors)
    upsert_points(client, points, COLLECTION, batch_size=100)
    print()

# Step 3: 验证
print("=" * 60)
print("Step 3: 验证入库结果")
print("=" * 60)
collection_info = client.get_collection(COLLECTION)
print(f"  Collection: {COLLECTION}")
print(f"  总 points 数: {collection_info.points_count}")
print(f"  向量维度: {collection_info.config.params.vectors.size}") # type: ignore
print(f"  距离度量: {collection_info.config.params.vectors.distance}") # type: ignore

# Step 4: 快速检索测试
print("\n" + "=" * 60)
print("Step 4: 快速检索测试")
print("=" * 60)

test_queries = [
    ("茅台 2023 年营业收入", None),
    ("宁德时代的核心竞争力", None),
    ("招商银行不良贷款率", None),
    ("茅台 2023 年营业收入", {"company": "贵州茅台"}),  # 带过滤
]

from src.retrievers.qdrant_store import search_similar

for query, filters in test_queries:
    q_vec = embedder.encode_query(query)
    results = search_similar(client, q_vec.tolist(), COLLECTION, limit=3, filters=filters)

    filter_str = f" [filter: {filters}]" if filters else ""
    print(f"\n🔍 {query}{filter_str}")
    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        company = meta.get("company", "?")
        page = meta.get("page", "?")
        score = r["score"]
        preview = r["content"][:80].replace("\n", " ")
        print(f"   {i}. [{score:.4f}] {company} p.{page} | {preview}...")