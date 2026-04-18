"""测试 PDF 加载器。看看 Chunk 数量、类型分布、抽样几个看看质量。"""
from collections import Counter
from pathlib import Path
import src
from src.loaders.pdf_loader import load_pdf

PDF_PATH = Path("data/raw/600519_贵州茅台_2023年年度报告.pdf")

chunks = src.loaders.pdf_loader.load_pdf(PDF_PATH)

# 总览
print(f"总 chunk 数: {len(chunks)}")
type_counter = Counter(c.chunk_type for c in chunks)
print(f"类型分布: {dict(type_counter)}")
print(f"前 3 个 chunk 的 metadata 长这样:")
for c in chunks[:3]:
    print(f"  {c}")

# 看几个表格 chunk 的实际内容
print("\n" + "=" * 60)
print("抽样表格 chunks:")
print("=" * 60)
table_chunks = [c for c in chunks if c.chunk_type == "table"]
for c in table_chunks[:2]:  # 看前 2 个表格
    print(f"\n--- 第 {c.metadata['page']} 页, 表格 #{c.metadata['table_index']} ---")
    print(c.content)