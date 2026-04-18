"""把茅台年报的 493 个 chunks 全部向量化,保存到磁盘。
为 Day 4 入向量库做准备。
"""
import os
import pickle
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from src.chunking.recursive import chunk_documents
from src.embeddings.bge import BGEEmbedder
from src.loaders.pdf_loader import load_pdf

PDF_PATH = Path("data/raw/600519_贵州茅台_2023年年度报告.pdf")
OUTPUT_PATH = Path("data/processed/maotai_2023_embeddings.pkl")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Step 1: 加载 + 切分
print("=" * 60)
print("Step 1: 加载 + 切分 PDF")
print("=" * 60)
t0 = time.time()
raw_chunks = load_pdf(PDF_PATH)
chunks = chunk_documents(raw_chunks, chunk_size=400, chunk_overlap=50)
t1 = time.time()
print(f"完成: {len(chunks)} 个 chunks,耗时 {t1-t0:.1f} 秒\n")

# Step 2: 加载 embedder
print("=" * 60)
print("Step 2: 加载 BGE 模型")
print("=" * 60)
embedder = BGEEmbedder()
print()

# Step 3: 批量编码
print("=" * 60)
print(f"Step 3: 批量向量化 {len(chunks)} 个 chunks")
print("=" * 60)
t0 = time.time()
vectors = embedder.encode_chunks(chunks, batch_size=32, show_progress=True)
t1 = time.time()
print(f"\n完成: {vectors.shape} 矩阵,耗时 {t1-t0:.1f} 秒")
print(f"平均每个 chunk: {(t1-t0)/len(chunks)*1000:.1f} ms\n")

# Step 4: 保存(chunks + vectors 一起存,Day 4 入库时直接用)
print("=" * 60)
print("Step 4: 保存到磁盘")
print("=" * 60)
data = {"chunks": chunks, "vectors": vectors}
with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(data, f)

file_size_mb = OUTPUT_PATH.stat().st_size / 1024 / 1024
print(f"✅ 保存到 {OUTPUT_PATH}")
print(f"   文件大小: {file_size_mb:.2f} MB")
print(f"   chunks 数量: {len(chunks)}")
print(f"   向量矩阵: {vectors.shape}")

# Step 5: 简单的"sanity check"
print("\n" + "=" * 60)
print("Step 5: Sanity check(用茅台问题去匹配 chunks)")
print("=" * 60)

import numpy as np

test_query = "茅台 2023 年的营业收入是多少"
print(f"测试查询: {test_query}")

q_vec = embedder.encode_query(test_query)
similarities = vectors @ q_vec  # 因为都归一化了,点积=余弦相似度
top5_idx = np.argsort(-similarities)[:5]

print(f"\nTop-5 最相似的 chunks:")
for rank, idx in enumerate(top5_idx, 1):
    chunk = chunks[idx]
    print(f"\n  Rank {rank}: 相似度 {similarities[idx]:.4f}")
    print(f"    类型: {chunk.chunk_type}, 页码: {chunk.metadata.get('page')}")
    print(f"    内容预览: {chunk.content[:150].replace(chr(10), ' ')}...")