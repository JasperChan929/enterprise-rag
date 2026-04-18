"""验证 embedding 的"语义魔法":
看看模型能不能正确理解金融场景的同义/近义/无关。
"""
import os
from dotenv import load_dotenv
load_dotenv()

import numpy as np
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.table import Table

console = Console()

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
model = SentenceTransformer(MODEL_NAME)

# 5 个精心挑选的句子,设计成 3 组关系
sentences = [
    "茅台 2023 年的营业收入是多少",        # 0  ← 财务问题(标准说法)
    "贵州茅台去年挣了多少钱",              # 1  ← 财务问题(口语说法)
    "请问茅台公司 2023 年的营收情况",      # 2  ← 财务问题(另一种问法)
    "如何申请个人所得税退税",              # 3  ← 不相关(税务)
    "今天天气真好,适合出去散步",          # 4  ← 完全不相关(闲聊)
]

# 一次性向量化所有句子(batch 推理,快)
console.print(f"\n[bold]Step 1: 向量化 {len(sentences)} 个句子[/bold]")
vectors = model.encode(sentences, normalize_embeddings=True)
console.print(f"   ✅ 完成,得到 {vectors.shape} 的矩阵\n")

# 因为已经归一化了,余弦相似度 = 点积
# similarity_matrix[i][j] = sentences[i] 和 sentences[j] 的相似度
similarity_matrix = vectors @ vectors.T

# 打印相似度矩阵
console.print("[bold]Step 2: 相似度矩阵[/bold]\n")
table = Table(show_header=True, header_style="bold")
table.add_column("ID", justify="right")
for i in range(len(sentences)):
    table.add_column(f"S{i}", justify="right")
table.add_column("内容", style="dim")

for i, sent in enumerate(sentences):
    row = [f"S{i}"]
    for j in range(len(sentences)):
        sim = similarity_matrix[i][j]
        # 颜色编码:高相似度绿色,低相似度红色
        if i == j:
            color = "white"  # 自己和自己
        elif sim > 0.7:
            color = "green"
        elif sim > 0.5:
            color = "yellow"
        else:
            color = "red"
        row.append(f"[{color}]{sim:.3f}[/{color}]")
    row.append(sent[:30])
    table.add_row(*row)

console.print(table)

# 打印关键观察
console.print("\n[bold]Step 3: 关键观察[/bold]\n")
console.print(f"  S0 ↔ S1 (营收·标准 vs 口语):  {similarity_matrix[0][1]:.3f}  ← 应该高")
console.print(f"  S0 ↔ S2 (营收·两种问法):       {similarity_matrix[0][2]:.3f}  ← 应该最高")
console.print(f"  S1 ↔ S2 (口语 vs 另一问法):    {similarity_matrix[1][2]:.3f}  ← 应该高")
console.print(f"  S0 ↔ S3 (营收 vs 退税):        {similarity_matrix[0][3]:.3f}  ← 应该低")
console.print(f"  S0 ↔ S4 (营收 vs 天气):        {similarity_matrix[0][4]:.3f}  ← 应该最低")
console.print(f"  S3 ↔ S4 (退税 vs 天气):        {similarity_matrix[3][4]:.3f}  ← 应该低(都和金融无关)")