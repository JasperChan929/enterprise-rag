"""数据结构定义。整个项目里"被加载的内容"统一用这个 Chunk 表示。"""
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Chunk:
    """一个最小的检索单元。

    后面所有模块(Chunker、Embedder、Retriever)都围绕这个数据结构流转。
    """
    content: str                                   # 实际文本内容
    chunk_type: Literal["text", "table"] = "text"  # 类型标记
    metadata: dict = field(default_factory=dict)   # 元数据字典

    def __repr__(self) -> str:
        preview = self.content[:60].replace("\n", " ")
        return f"Chunk(type={self.chunk_type}, page={self.metadata.get('page')}, preview='{preview}...')"