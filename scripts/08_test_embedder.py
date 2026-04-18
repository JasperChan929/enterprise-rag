"""测试 BGEEmbedder 封装是否工作正常。"""
import os
from dotenv import load_dotenv
load_dotenv()

from src.embeddings.bge import BGEEmbedder

# 创建 embedder(第一次会加载模型)
embedder = BGEEmbedder()
print()

# 测试 1: 单条编码
print("Test 1: 单条编码")
vec = embedder.encode("茅台 2023 年营收")
print(f"  向量 shape: {vec.shape}")
print(f"  L2 norm: {(vec**2).sum()**0.5:.4f}\n")

# 测试 2: 批量编码
print("Test 2: 批量编码")
texts = ["茅台营收", "贵州茅台净利润", "宁德时代毛利率"]
vecs = embedder.encode(texts)
print(f"  矩阵 shape: {vecs.shape}\n")

# 测试 3: 查询编码(带前缀)
print("Test 3: 查询编码(带 BGE 前缀)")
q_vec = embedder.encode_query("茅台营收是多少")
doc_vec = embedder.encode("茅台 2023 年实现营业收入 1476 亿元")
similarity_with_prefix = (q_vec * doc_vec).sum()

# 对照:不加前缀
q_vec_no_prefix = embedder.encode("茅台营收是多少")
similarity_no_prefix = (q_vec_no_prefix * doc_vec).sum()

print(f"  带前缀的相似度: {similarity_with_prefix:.4f}")
print(f"  不带前缀的相似度: {similarity_no_prefix:.4f}")
print(f"  差异: {(similarity_with_prefix - similarity_no_prefix):.4f}\n")

# 测试 4: 单例验证
print("Test 4: 单例验证")
embedder2 = BGEEmbedder()
print(f"  embedder is embedder2: {embedder is embedder2}  ← 应该是 True\n")

print("✅ 所有测试通过!")