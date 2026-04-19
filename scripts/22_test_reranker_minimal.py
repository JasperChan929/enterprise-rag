# scripts/22_test_reranker_minimal.py
"""Reranker 最小 demo: 用 4 条人造数据验证打分行为符合直觉.

目的不是跑完整 RAG, 而是**建立对 Cross-Encoder 打分的直觉**:
  - 高相关 (精确匹配数字) 应该打高分
  - 字面相关但语义偏离 应该打中等
  - 完全无关 应该打0
"""
from dotenv import load_dotenv
load_dotenv(override=True)

from src.rerankers.bge_reranker import BGEReranker


def main():
    print("=" * 60)
    print("BGE-Reranker 最小 demo")
    print("=" * 60)

    reranker = BGEReranker(verbose=True)

    query = "招商银行 2024 年不良贷款率是多少?"

    # 构造 4 条差异明显的候选,覆盖不同相关度层级
    candidates = [
        {
            "id": "A",
            "desc": "🎯 精确答案 - 应该打最高分",
            "content": "2024年末,本集团不良贷款率为0.93%,较上年末下降0.02个百分点。",
        },
        {
            "id": "B",
            "desc": "🟡 业务线细分 - 字面相关但不是整体口径",
            "content": "2024年末,房地产业不良贷款率为4.64%,主要受行业下行影响。",
        },
        {
            "id": "C",
            "desc": "🟠 沾边 - 提到信贷但是风险管控策略描述",
            "content": "本行持续加强消费信贷类业务风险管控,坚持审慎授信原则。",
        },
        {
            "id": "D",
            "desc": "❌ 完全无关 - 应该打最低分",
            "content": "贵州茅台 2023 年营业收入 14,769,360.50 万元,同比增长 18.04%。",
        },
    ]

    print(f"\nQuery: {query}\n")
    print("原始顺序 (模拟 Hybrid 随机返回的顺序):")
    for c in candidates:
        print(f"  [{c['id']}] {c['desc']}")
        print(f"      {c['content']}")

    # 精排
    reranked = reranker.rerank(query, candidates, top_k=None)  # 不截断, 全看

    print("\n" + "=" * 60)
    print("精排后:")
    print("=" * 60)
    for c in reranked:
        print(f"  Rank {c['rerank_rank']}  score={c['rerank_score']:+7.3f}  [{c['id']}]"
              f"  {c['desc']}")

    # ==== 断言: 验证排序符合直觉 ====
    print("\n验证:")
    id_order = [c["id"] for c in reranked]
    assert id_order[0] == "A", f"❌ 精确答案应排第一, 实际是 {id_order[0]}"
    print(f"  ✅ 精确答案 A 排第一")
    assert id_order[-1] == "D", f"❌ 无关文档应排最后, 实际是 {id_order[-1]}"
    print(f"  ✅ 无关文档 D 排最后")
    print(f"  ✅ 完整排序: {' > '.join(id_order)}")


if __name__ == "__main__":
    main()