"""第一次下载 BGE 模型。
首次运行会下载 ~95MB,以后从本地缓存秒加载。
"""
import os
from dotenv import load_dotenv

# 必须在 import sentence_transformers 之前加载环境变量
load_dotenv()

from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-zh-v1.5"

print(f"正在下载/加载模型: {MODEL_NAME}")
print(f"使用镜像: {os.environ.get('HF_ENDPOINT', 'huggingface.co (默认)')}")
print("第一次会下载 ~95MB,请耐心等待...")
print()

model = SentenceTransformer(MODEL_NAME)

print("✅ 模型加载成功!")
print(f"   嵌入维度: {model.get_sentence_embedding_dimension()}")
print(f"   最大输入 tokens: {model.max_seq_length}")
print(f"   模型设备: {model.device}")

# 测试一下能不能跑
test_text = "贵州茅台 2023 年实现营业收入"
vector = model.encode(test_text)
print(f"\n测试编码: '{test_text}'")
print(f"   向量维度: {vector.shape}")
print(f"   前 5 个数字: {vector[:5]}")
print(f"   向量长度(L2 norm): {(vector**2).sum() ** 0.5:.4f}  ← 应该接近 1.0(已归一化)")