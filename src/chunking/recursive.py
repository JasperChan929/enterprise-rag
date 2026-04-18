"""递归字符切分器(参考 LangChain RecursiveCharacterTextSplitter)。

核心思想:
  按一组分隔符优先级,递归地把过长片段切小,直到都满足 chunk_size 上限。
  优先在"语义边界"(段落、句号)切,实在不行才在字符级别切。
"""
from typing import Callable

import tiktoken

from src.loaders.base import Chunk


# 中文场景的分隔符优先级,从高到低
DEFAULT_SEPARATORS = [
    "\n\n",   # 段落
    "\n",     # 行
    "。",     # 句号
    "；",     # 分号
    "！",     # 感叹号
    "？",     # 问号
    "，",     # 逗号
    " ",      # 空格
    "",       # 字符级兜底
]


def get_token_counter() -> Callable[[str], int]:
    """返回一个 token 计数函数。

    用 OpenAI 的 cl100k_base tokenizer 作为近似——它和 BGE 的实际 tokenizer
    不完全一致,但数量级接近,够用作切分控制。
    
    生产环境如果想精确,可以加载 BGE 的 tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-zh-v1.5")
    但 transformers 包很大,先用 tiktoken 替代。
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    return lambda text: len(encoding.encode(text))


def split_text_recursive(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str] = None,
    token_counter: Callable[[str], int] = None,
) -> list[str]:
    """递归切分纯文本,返回字符串列表。

    Args:
        text: 待切分的文本
        chunk_size: 每个 chunk 的最大 token 数
        chunk_overlap: 相邻 chunk 之间的重叠 token 数
        separators: 分隔符优先级列表
        token_counter: token 计数函数

    Returns:
        切分后的字符串列表(每个长度 ≤ chunk_size tokens)
    """
    separators = separators or DEFAULT_SEPARATORS
    token_counter = token_counter or get_token_counter()

    # 文本本身就够小 → 直接返回
    if token_counter(text) <= chunk_size:
        return [text] if text.strip() else []

    # 找到第一个能用的分隔符(在文本中实际存在的最高优先级分隔符)
    sep = ""
    for s in separators:
        if s == "" or s in text:
            sep = s
            break

    # 用这个分隔符切
    if sep == "":
        # 兜底:按字符切
        splits = list(text)
    else:
        splits = text.split(sep)

    # 尝试合并相邻的小片段,凑成接近 chunk_size 的 chunks
    chunks = _merge_splits(splits, sep, chunk_size, chunk_overlap, token_counter)

    # 检查是否有还超大的 chunk → 用次级分隔符递归切
    final_chunks = []
    for chunk in chunks:
        if token_counter(chunk) <= chunk_size:
            final_chunks.append(chunk)
        else:
            # 递归:用剩下的分隔符再切
            remaining_seps = separators[separators.index(sep) + 1:] if sep else [""]
            sub_chunks = split_text_recursive(
                chunk, chunk_size, chunk_overlap, remaining_seps, token_counter
            )
            final_chunks.extend(sub_chunks)

    return [c for c in final_chunks if c.strip()]


def _merge_splits(
    splits: list[str],
    separator: str,
    chunk_size: int,
    chunk_overlap: int,
    token_counter: Callable[[str], int],
) -> list[str]:
    """把切出来的小片段合并成接近 chunk_size 的 chunks,带 overlap。

    例: splits = ["句1", "句2", "句3", "句4"], separator = "。"
        如果每句 ~100 tokens, chunk_size=250
        → 输出: ["句1。句2", "句2。句3", "句3。句4"]  (overlap 1 句)
    """
    chunks = []
    current_chunk: list[str] = []
    current_size = 0

    for split in splits:
        split_size = token_counter(split)

        # 如果加上新片段会超 chunk_size,且当前 chunk 非空 → 收口当前 chunk
        if current_size + split_size > chunk_size and current_chunk:
            chunks.append(separator.join(current_chunk))

            # 处理 overlap:从尾部回溯,留出一些片段作为下一个 chunk 的开头
            while current_size > chunk_overlap and current_chunk:
                removed = current_chunk.pop(0)
                current_size -= token_counter(removed)

        current_chunk.append(split)
        current_size += split_size

    # 收口最后一个
    if current_chunk:
        chunks.append(separator.join(current_chunk))

    return chunks


def chunk_documents(
    chunks: list[Chunk],
    chunk_size: int = 400,
    chunk_overlap: int = 50,
    min_chunk_size: int = 30,
) -> list[Chunk]:
    """对一组 Chunk 应用切分策略。

    策略:
      - table chunks: pass through(保留完整表格)
      - text chunks: 递归切分到 chunk_size 以内
      - 切出来的小片段如果 < min_chunk_size,丢弃(过滤噪声)

    Args:
        chunks: load_pdf() 返回的原始 chunks
        chunk_size: 文字 chunk 的最大 token 数
        chunk_overlap: overlap token 数
        min_chunk_size: 最小 token 数,小于此值的丢弃

    Returns:
        切分后的 chunks 列表
    """
    token_counter = get_token_counter()
    result: list[Chunk] = []

    for chunk in chunks:
        if chunk.chunk_type == "table":
            # 表格保持完整
            result.append(chunk)
            continue

        # 文字递归切分
        sub_texts = split_text_recursive(
            chunk.content, chunk_size, chunk_overlap, token_counter=token_counter
        )

        for i, sub_text in enumerate(sub_texts):
            # 过滤太短的
            if token_counter(sub_text) < min_chunk_size:
                continue

            # 创建新 chunk,继承父 chunk 的 metadata + 加上切片索引
            new_metadata = {**chunk.metadata, "split_index": i}
            result.append(Chunk(
                content=sub_text,
                chunk_type="text",
                metadata=new_metadata,
            ))

    return result