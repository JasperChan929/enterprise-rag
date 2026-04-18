"""测试连上 Qdrant,做最基础的 hello world。"""
from qdrant_client import QdrantClient
import qdrant_client
from qdrant_client.models import Distance, VectorParams, PointStruct
import numpy as np

# 连接(默认 localhost:6333)
client = QdrantClient(host="localhost", port=6333)

# 测试 1: 列出现有 collections
collections = client.get_collections()
print(f"现有 collections: {[c.name for c in collections.collections]}")

# 测试 2: 创建一个测试 collection
COLLECTION = "test_hello"
client.recreate_collection(  # recreate = 如果存在先删除再建
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=4, distance=Distance.COSINE),
)
print(f"\n✅ 创建 collection: {COLLECTION}")

# 测试 3: 插入 3 个 4 维向量(玩具数据)
points = [
    PointStruct(id=1, vector=[0.1, 0.2, 0.3, 0.4], payload={"text": "苹果"}),
    PointStruct(id=2, vector=[0.2, 0.1, 0.4, 0.3], payload={"text": "香蕉"}),
    PointStruct(id=3, vector=[0.9, 0.8, 0.1, 0.2], payload={"text": "汽车"}),
]
client.upsert(collection_name=COLLECTION, points=points)
print(f"✅ 插入 {len(points)} 个 points")

# 测试 4: 检索
query_vec = [0.15, 0.15, 0.35, 0.35]  # 接近"苹果"和"香蕉"
print(f"\n🔎 正在使用 query_points 检索...")
# query_points 是目前 QdrantClient 最明确的入口
response = client.query_points(
    collection_name=COLLECTION,
    query=query_vec,    # 这里的参数名就叫 query
    limit=2,
)

# 注意：query_points 返回的是一个对象，结果在 .points 属性里
results = response.points
print(f"\n检索结果(查询向量 {query_vec}):")
for r in results:
    print(f"  id={r.id}, score={r.score:.4f}, payload={r.payload}")

# 清理
client.delete_collection(COLLECTION)
print(f"\n🧹 清理 collection: {COLLECTION}")
