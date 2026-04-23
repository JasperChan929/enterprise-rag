"""BM25 稀疏检索模块。

职责:
  - 对 Chunks 的 content 分词 → 构建 BM25Okapi 索引
  - 提供查询接口, 返回 Top-K 命中(支持 metadata 过滤)
  - 配合 Dense 检索做 Hybrid

设计要点:
  - BM25Okapi 是 Elasticsearch/Lucene 同款算法, 工业界事实标准
  - 索引在内存中, 不落盘(Day 5 决策: YAGNI, Day 11 服务化时再评估)
  - 分词结果缓存在 self.tokenized_corpus, 方便调试
"""
import re
from typing import Optional

import jieba
from rank_bm25 import BM25Okapi

from src.loaders.base import Chunk
from src.retrievers.financial_dict import load_to_jieba


# ============================================================
# 预处理: 分词 + 去噪
# ============================================================

# 过滤的噪声字符: 中英文标点、空格、特殊符号
# 保留: 中文字、英文字母、数字(金融数字是关键 信号)
NOISE_PATTERN = re.compile(
    r"[\s\u3000,。;:!?、\"\"''()《》【】\[\]\.\,\;\:\-\—\/\\|\*\#\$\%\^\&\+\=\~\`]+"
)

# 单字停用词(保留数字和英文字母, 因为 5G/3年/A股 都是有信息量的)
STOPWORDS = set("""的 了 在 是 有 和 就 不 也 都 又 或 这 那 其 以 及 与 之 被 将 为 对""".split())


def tokenize(text: str) -> list[str]:
    """中文分词 + 清洗。

    步骤:
      1. jieba 精确模式切分
      2. 过滤空串/纯标点 token
      3. 过滤单字停用词
      4. 英文转小写(字母一致性)

    为什么不过滤单字数字: "3年"的"3"是有信息量的, 不能丢。
    """
    tokens = jieba.lcut(text, cut_all=False)
    cleaned = []
    for t in tokens:
        t = t.strip().lower()
        if not t:
            continue
        # 整体是标点 → 丢
        if NOISE_PATTERN.fullmatch(t):
            continue
        # 单字停用词 → 丢
        if t in STOPWORDS:
            continue
        cleaned.append(t)
    return cleaned


# ============================================================
# BM25 存储
# ============================================================

class BM25Store:
    """BM25 索引的封装。

    用法:
        store = BM25Store()
        store.build(chunks)
        results = store.search("招商银行不良贷款率", limit=5)
    """

    def __init__(self):
        # 加载金融词典到 jieba(进程级, 多次调用无副作用)
        n = load_to_jieba()
        print(f"金融词典已加载: {n} 个术语")

        self.chunks: list[Chunk] = []
        self.tokenized_corpus: list[list[str]] = []
        self.bm25: Optional[BM25Okapi] = None

    def build(self, chunks: list[Chunk]) -> None:
        """从 Chunks 构建 BM25 索引。"""
        self.chunks = chunks

        print(f"🔨 对 {len(chunks)} 个 chunks 分词...")
        self.tokenized_corpus = [tokenize(c.content) for c in chunks]

        # 诊断统计
        lengths = [len(t) for t in self.tokenized_corpus]
        empty_count = sum(1 for l in lengths if l == 0)
        avg_len = sum(lengths) / len(lengths) if lengths else 0
        print(f"   平均词数/chunk: {avg_len:.1f}")
        print(f"   最少/最多: {min(lengths)}/{max(lengths)}")
        if empty_count > 0:
            print(f"   ⚠️  {empty_count} 个 chunk 分词后为空(可能是纯标点/数字)")

        print(f"🔨 构建 BM25Okapi 索引...")
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print(f"✅ BM25 索引就绪")

    def search(
        self,
        query: str,
        limit: int = 5,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """BM25 检索, 返回 Top-K。

        Args:
            query: 原始查询字符串(内部会分词)
            limit: 返回数量
            filters: metadata 等值过滤, 如 {"company": "贵州茅台", "year": 2023}

        Returns:
            [{"content", "score", "metadata", "chunk_index"}, ...]

        注意:
            BM25 score 是**无界正数**(5.2 / 28.7 都可能), 数量级和 Dense 完全不同。
            Hybrid 融合必须用 RRF, 不能用分数加权。
        """
        if self.bm25 is None:
            raise RuntimeError("索引未构建, 请先调用 build()")

        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        # 计算所有文档的 BM25 分数
        scores = self.bm25.get_scores(q_tokens)

        # 先按 filters 筛
        if filters:
            valid_indices = [
                i for i, c in enumerate(self.chunks)
                if self._match_filter(c, filters)
            ]
            if not valid_indices:
                return []
        else:
            valid_indices = list(range(len(self.chunks)))

        # 在候选集中取 Top-K
        ranked = sorted(valid_indices, key=lambda i: -scores[i])[:limit]

        results = []
        for idx in ranked:
            if scores[idx] <= 0:
                continue  # 0 分的无匹配, 不返回
            chunk = self.chunks[idx]
            results.append({
                "content": chunk.content,
                "score": float(scores[idx]),
                "metadata": {**chunk.metadata, "chunk_type": chunk.chunk_type},
                "chunk_index": idx,
            })

        return results

    @staticmethod
    def _match_filter(chunk: Chunk, filters: dict) -> bool:
        """等值过滤(和 Qdrant FieldCondition + MatchValue 对齐)。"""
        for key, value in filters.items():
            if chunk.metadata.get(key) != value:
                return False
        return True

    def debug_tokenize(self, text: str) -> list[str]:
        """暴露分词结果, 方便调试。"""
        return tokenize(text)