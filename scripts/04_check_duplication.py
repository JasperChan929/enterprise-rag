"""验证一个怀疑:同一段表格内容,是不是既出现在 table chunk 里,
   又出现在同一页的 text chunk 里?"""
from pathlib import Path
from src.loaders.pdf_loader import load_pdf

PDF_PATH = Path("data/raw/600519_贵州茅台_2023年年度报告.pdf")
chunks = load_pdf(PDF_PATH)

# 找第 6 页(我们之前重点分析的季度数据页)
page_6_chunks = [c for c in chunks if c.metadata.get("page") == 6]

print(f"第 6 页一共 {len(page_6_chunks)} 个 chunks:\n")

for i, c in enumerate(page_6_chunks):
    print(f"--- Chunk {i}: type={c.chunk_type} ---")
    print(c.content[:500])
    print(f"...(总长 {len(c.content)} 字)\n")