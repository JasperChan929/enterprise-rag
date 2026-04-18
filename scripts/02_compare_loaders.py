"""
Day 2 实验 2: 对比 pypdf 和 pdfplumber 在同一页上的输出差异
重点看带表格的第 6 页
"""
from pypdf import PdfReader
import pdfplumber
from pathlib import Path

PDF_PATH = Path("data/raw/600519_贵州茅台_2023年年度报告.pdf")
TARGET_PAGE = 5  # 0-indexed, 即第 6 页

# === 方案 A: pypdf ===
print("=" * 60)
print("方案 A: pypdf")
print("=" * 60)
reader = PdfReader(PDF_PATH)
text_pypdf = reader.pages[TARGET_PAGE].extract_text()
print(text_pypdf)
print(f"\n字符数: {len(text_pypdf)}")

# === 方案 B: pdfplumber 默认抽取 ===
print("\n" + "=" * 60)
print("方案 B: pdfplumber (默认 extract_text)")
print("=" * 60)
with pdfplumber.open(PDF_PATH) as pdf:
    page = pdf.pages[TARGET_PAGE]
    text_plumber = page.extract_text()
    print(text_plumber)
    print(f"\n字符数: {len(text_plumber)}")

# === 方案 C: pdfplumber 表格识别 ===
print("\n" + "=" * 60)
print("方案 C: pdfplumber (extract_tables - 专门提取表格)")
print("=" * 60)
with pdfplumber.open(PDF_PATH) as pdf:
    page = pdf.pages[TARGET_PAGE]
    tables = page.extract_tables()
    print(f"识别出 {len(tables)} 个表格\n")
    for i, table in enumerate(tables):
        print(f"--- 表格 {i+1} ---")
        for row in table:
            print(row)
        print()