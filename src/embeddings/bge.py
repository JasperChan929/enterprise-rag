"""BGE Embedding 模块。

提供文本到向量的转换,封装了:
  - 模型懒加载(第一次用才加载)
  - 单例(全局共用一个模型实例)
  - 自动归一化(便于后续余弦计算)
  - 批量推理(显著快于循环单条)
"""
import os
from typing import Union

import numpy as np
from sentence_transformers import SentenceTransformer

from src.loaders.base import Chunk


# ============================================================
# 配置
# ============================================================

DEFAULT_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
DEFAULT_BATCH_SIZE = 32  # batch 推理大小,32 是速度和内存的平衡

# BGE 系列推荐:对查询加前缀,提升检索效果
# 这是 BGE 官方建议,它训练时见过这种格式
BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章:"


# ============================================================
# Embedder 类
# ============================================================

class BGEEmbedder:
    """BGE embedding 封装类。
    
    用法:
        embedder = BGEEmbedder()
        # 单条
        vec = embedder.encode("茅台营收")
        # 批量
        vecs = embedder.encode(["句子1", "句子2", "句子3"])
        # 编码 Chunks(直接处理 Day 2 的产出)
        vecs = embedder.encode_chunks(chunks)
        # 编码查询(自动加 BGE 推荐前缀)
        q_vec = embedder.encode_query("茅台营收是多少")
    """
    
    _instance = None  # 单例存储
    
    def __new__(cls, model_name: str = DEFAULT_MODEL):
        """单例模式:同一个 model_name 只创建一个实例。"""
        if cls._instance is None or cls._instance.model_name != model_name:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, model_name: str = DEFAULT_MODEL):
        # 防止重复初始化(单例)
        if self._initialized:
            return
        
        print(f"📦 加载 Embedding 模型: {model_name}")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_embedding_dimension()
        print(f"   维度: {self.dim}, 设备: {self.model.device}")
        
        self._initialized = True
    
    def encode(
        self, 
        texts: Union[str, list[str]], 
        batch_size: int = DEFAULT_BATCH_SIZE,
        show_progress: bool = False,
    ) -> np.ndarray:
        """编码文本(单条或批量)。
        
        Args:
            texts: 单个字符串 或 字符串列表
            batch_size: 批大小
            show_progress: 是否显示进度条(批量大时建议开)
        
        Returns:
            shape=(N, dim) 的向量矩阵
            如果输入是单个字符串,返回 shape=(dim,) 的一维向量
        """
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]
        
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,  # ⭐ L2 归一化
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        
        return vectors[0] if is_single else vectors
    
    def encode_query(self, query: str) -> np.ndarray:
        """编码用户查询。

        BGE 官方建议:对查询(不是文档)加一个前缀,
        这是它训练时见过的格式,能提升检索精度 1-3%。
        文档不加前缀。
        """
        prefixed = BGE_QUERY_PREFIX + query
        return self.encode(prefixed)
    
    def encode_chunks(
        self, 
        chunks: list[Chunk],
        batch_size: int = DEFAULT_BATCH_SIZE,
        show_progress: bool = True,
    ) -> np.ndarray:
        """直接编码 Chunk 列表,从中抽 content。"""
        texts = [c.content for c in chunks]
        return self.encode(texts, batch_size=batch_size, show_progress=show_progress)