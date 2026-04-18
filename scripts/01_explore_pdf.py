"""
Day 2 实验 1: 看看 pypdf 把茅台年报抽成什么样
目的: 直观感受 PDF 解析的"脏"
"""
from pypdf import PdfReader
from pathlib import Path

PDF_PATH = Path("data/raw/600519_贵州茅台_2023年年度报告.pdf")

reader = PdfReader(PDF_PATH)
print(f"总页数: {len(reader.pages)}")
print(f"文件大小: {PDF_PATH.stat().st_size / 1024 / 1024:.2f} MB")
print()

# 试 3 个不同页,看看不同位置长什么样
for page_num in [5, 15, 50]:
    page = reader.pages[page_num]
    text = page.extract_text()
    print(f"{'='*60}")
    print(f"第 {page_num + 1} 页 (字符数: {len(text)})")
    print(f"{'='*60}")
    print(text[:800])  # 只看前 800 字,避免刷屏
    print()