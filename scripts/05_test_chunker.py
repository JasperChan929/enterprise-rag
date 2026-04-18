"""测试切分器:量化效果,看长度分布是否健康。"""
from pathlib import Path

from src.chunking.recursive import chunk_documents, get_token_counter
from src.loaders.pdf_loader import load_pdf

PDF_PATH = Path("data/raw/600519_贵州茅台_2023年年度报告.pdf")
CHUNK_SIZE = 400  # tokens
CHUNK_OVERLAP = 50

# 1. 加载
print("=" * 60)
print("Step 1: 加载 PDF")
print("=" * 60)
raw_chunks = load_pdf(PDF_PATH)
print(f"原始 chunks: {len(raw_chunks)}")
print(f"  text: {sum(1 for c in raw_chunks if c.chunk_type == 'text')}")
print(f"  table: {sum(1 for c in raw_chunks if c.chunk_type == 'table')}")

# 2. 切分
print("\n" + "=" * 60)
print(f"Step 2: 切分 (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
print("=" * 60)
chunks = chunk_documents(raw_chunks, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
print(f"切分后 chunks: {len(chunks)}")
print(f"  text: {sum(1 for c in chunks if c.chunk_type == 'text')}")
print(f"  table: {sum(1 for c in chunks if c.chunk_type == 'table')}")

# 3. 长度分析
print("\n" + "=" * 60)
print("Step 3: token 长度分布")
print("=" * 60)
token_counter = get_token_counter()
text_lengths = [token_counter(c.content) for c in chunks if c.chunk_type == "text"]
table_lengths = [token_counter(c.content) for c in chunks if c.chunk_type == "table"]

if text_lengths:
    print(f"\nText chunks:")
    print(f"  数量: {len(text_lengths)}")
    print(f"  平均: {sum(text_lengths)/len(text_lengths):.0f} tokens")
    print(f"  最短: {min(text_lengths)} tokens")
    print(f"  最长: {max(text_lengths)} tokens")
    print(f"  > 512 的: {sum(1 for l in text_lengths if l > 512)} 个 (会被 BGE 截断)")

if table_lengths:
    print(f"\nTable chunks (未切分):")
    print(f"  数量: {len(table_lengths)}")
    print(f"  平均: {sum(table_lengths)/len(table_lengths):.0f} tokens")
    print(f"  最长: {max(table_lengths)} tokens")
    print(f"  > 512 的: {sum(1 for l in table_lengths if l > 512)} 个 (会被截断,后续优化)")

# 4. 抽样几个切分后的 chunk
print("\n" + "=" * 60)
print("Step 4: 抽样切分后的 text chunks")
print("=" * 60)
text_chunks = [c for c in chunks if c.chunk_type == "text"]
for c in text_chunks[5:8]:  # 看几个中间的
    print(f"\n--- 第 {c.metadata['page']} 页, split_index={c.metadata.get('split_index')} ---")
    print(f"长度: {token_counter(c.content)} tokens")
    print(f"内容: {c.content[:200]}{'...' if len(c.content) > 200 else ''}")