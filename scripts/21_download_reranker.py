# scripts/21_download_reranker.py
"""预下载 BGE-Reranker-base 到 HuggingFace 缓存目录。

只需跑一次。后续 BGEReranker() 初始化会直接读缓存,不再联网。
"""
from sentence_transformers import CrossEncoder
import time

MODEL_NAME = "BAAI/bge-reranker-base"

print(f"开始下载 {MODEL_NAME} (~278MB)...")
t0 = time.time()

model = CrossEncoder(MODEL_NAME)

print(f"✅ 下载并加载完成, 耗时 {time.time()-t0:.1f}s")
print(f"模型类型: {type(model).__name__}")
print(f"max_seq_length: {model.max_seq_length}")

# 试打个分,确认能跑
test_pairs = [
    ("招商银行不良贷款率", "2024年集团不良贷款率0.93%"),
    ("招商银行不良贷款率", "今天天气不错"),
]
scores = model.predict(test_pairs)
print(f"\n测试打分:")
for (q, d), s in zip(test_pairs, scores):
    print(f"  score={s:+.4f}  |  q={q}  |  d={d}")