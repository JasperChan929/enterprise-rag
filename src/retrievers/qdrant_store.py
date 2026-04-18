"""Qdrant 向量库管理:建库、入库、检索。

统一使用 qdrant-client 新版 API (query_points)。
"""
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from src.loaders.base import Chunk

# 默认配置
DEFAULT_COLLECTION = "financial_reports"
VECTOR_DIM = 512  # BGE-small-zh


def get_client(host: str = "localhost", port: int = 6333) -> QdrantClient:
    """获取 Qdrant 客户端。"""
    return QdrantClient(host=host, port=port)


def create_collection(
    client: QdrantClient,
    collection_name: str = DEFAULT_COLLECTION,
    vector_dim: int = VECTOR_DIM,
    recreate: bool = False,
):
    """创建 collection。

    Args:
        recreate: 如果 True,先删后建(开发时用);False 则已存在时跳过
    """
    # 检查是否已存在
    existing = [c.name for c in client.get_collections().collections]

    if collection_name in existing:
        if recreate:
            client.delete_collection(collection_name)
            print(f"🗑️  删除旧 collection: {collection_name}")
        else:
            print(f"ℹ️  collection 已存在: {collection_name},跳过创建")
            return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_dim,
            distance=Distance.COSINE,
        ),
    )
    print(f"✅ 创建 collection: {collection_name} (dim={vector_dim}, cosine)")


def chunks_to_points(chunks: list[Chunk], vectors) -> list[PointStruct]:
    """把 Chunk 列表 + 向量矩阵转成 Qdrant 的 PointStruct 列表。

    每个 Point 的 payload 里存了原始文本和所有 metadata,
    这样检索后可以直接拿到文本,不需要额外查询。
    """
    points = []
    for i, chunk in enumerate(chunks):
        # 构造 payload: metadata 展开 + content
        payload = {
            "content": chunk.content,
            "chunk_type": chunk.chunk_type,
            **chunk.metadata,  # source, company, year, page, etc.
        }

        point = PointStruct(
            id=str(uuid.uuid4()),  # UUID 保证唯一
            vector=vectors[i].tolist(),  # numpy → list
            payload=payload,
        )
        points.append(point)

    return points


def upsert_points(
    client: QdrantClient,
    points: list[PointStruct],
    collection_name: str = DEFAULT_COLLECTION,
    batch_size: int = 100,
):
    """批量写入 Points。

    分批 upsert,避免单次请求太大导致超时。
    """
    total = len(points)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = points[start:end]
        client.upsert(collection_name=collection_name, points=batch)
        print(f"  写入 {end}/{total} ...")

    print(f"✅ 共写入 {total} 个 points 到 {collection_name}")


def search_similar(
    client: QdrantClient,
    query_vector: list[float],
    collection_name: str = DEFAULT_COLLECTION,
    limit: int = 5,
    filters: dict | None= None,
) -> list[dict]:
    """语义检索:找最相似的 Top-K chunks。

    Args:
        query_vector: 查询向量
        limit: 返回几个
        filters: 可选过滤条件,如 {"company": "贵州茅台", "year": 2023}

    Returns:
        [{"content": ..., "score": ..., "metadata": {...}}, ...]
    """
    # 构造过滤条件
    qdrant_filter = None
    if filters:
        conditions = []
        for key, value in filters.items():
            conditions.append(
                FieldCondition(key=key, match=MatchValue(value=value))
            )
        qdrant_filter = Filter(must=conditions)

    # 新版 API: query_points
    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=qdrant_filter,
        limit=limit,
        with_payload=True,
    )

    # 整理输出格式
    results = []
    for point in response.points:
        payload = point.payload or {}
        content = payload.pop("content", "")
        results.append({
            "content": content,
            "score": point.score,
            "metadata": payload,
        })

    return results