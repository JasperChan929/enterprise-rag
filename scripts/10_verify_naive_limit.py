"""验证 Naive RAG 的局限是普遍现象,不是单个查询的偶然。
跑 5 个不同类型的查询,看 Top-3 命中情况。
"""
import os
import pickle
from dotenv import load_dotenv
load_dotenv()

import numpy as np
from src.embeddings.bge import BGEEmbedder

# 加载 Day 3 保存的数据
with open("data/processed/maotai_2023_embeddings.pkl", "rb") as f:
    data = pickle.load(f)

chunks = data["chunks"]
vectors = data["vectors"]
embedder = BGEEmbedder()

# 5 个不同类型的查询
test_queries = [
    "茅台 2023 年营业收入是多少",       # 财务数字 → 难
    "茅台的董事长是谁",                # 实体查询 → 中等
    "茅台 2023 年研发投入了多少",       # 财务数字 → 难
    "茅台的核心竞争力是什么",          # 开放性问题 → 容易
    "茅台对 2024 年的展望",            # 时间相关 → 中等
]

for query in test_queries:
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")
    
    q_vec = embedder.encode_query(query)
    sims = vectors @ q_vec
    top3 = np.argsort(-sims)[:3]
    
    for rank, idx in enumerate(top3, 1):
        chunk = chunks[idx]
        print(f"  Rank {rank} ({sims[idx]:.4f}, {chunk.chunk_type}, p.{chunk.metadata.get('page')})")
        print(f"    {chunk.content[:120].replace(chr(10), ' ')}...")