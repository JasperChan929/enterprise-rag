"""Day 11 T0 smoke test: 验证 RAGAS + 中转站 + judge 模型能否跑通.

===============================================================================
目的 (唯一目的)
===============================================================================
    验证 3 件事:
    1. ragas 0.4.3 能装上, Collections API 能 import
    2. 中转站的 OpenAI 兼容接口 + 我们选的 judge 模型能被 llm_factory 接受
    3. 一条硬编码的 (query, answer, context) 能跑出合法 Faithfulness 分数 [0.0, 1.0]

故意不做的事 (避免多因素耦合 - Day 9 D7 教训):
    - 不调 RAG pipeline (pipeline 的 context 字段残缺是另一个问题, 留到 T1)
    - 不跑多样本 / 多 metric / 多 mode (smoke test = 点火, 不是压力测试)
    - 不测 judge 质量 (质量问题放到 T3 用大样本 + 人工抽检)

===============================================================================
成功判据
===============================================================================
    - 脚本结束不抛异常
    - Faithfulness 分数落在 [0.0, 1.0] 闭区间
    - judge 调用次数 ≤ 5 (Faithfulness 是 2-stage: claim extract + NLI)

失败兜底:
    - KeyError: 环境变量缺失 → .env 改一行变量名再跑
    - AuthError / 404: 中转站 base_url 或 key 不对 → 改 .env 再跑
    - ImportError: ragas 版本不对 → 重装 ragas==0.4.3
    - 分数 NaN / > 1.0 / < 0.0: RAGAS 和 judge 不兼容 → 停, 报告给 Claude 诊断

===============================================================================
如何跑
===============================================================================
    # 1. 确保 .env 里有 3 条 (变量名可改, 但记得同步改本脚本顶部):
    #    OPENAI_JUDGE_API_KEY=sk-xxx       (中转 key)
    #    OPENAI_JUDGE_BASE_URL=https://... (中转 base_url, 结尾 /v1)
    #    OPENAI_JUDGE_MODEL=gpt-4o-mini    (中转支持的 judge 模型名)

    # 2. 安装依赖 (若已装跳过):
    uv pip install ragas==0.4.3

    # 3. 跑:
    uv run python scripts/30_ragas_smoke.py
"""
from __future__ import annotations

import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv(override=True)


# =============================================================================
# 阶段 0: 环境自检 (脚本开头就失败, 不让隐式错配进到 RAGAS 调用)
# =============================================================================
REQUIRED_ENV_VARS = [
    "OPENAI_JUDGE_API_KEY",
    "OPENAI_JUDGE_BASE_URL",
    "OPENAI_JUDGE_MODEL",
]

def check_env() -> dict[str, str]:
    """检查 .env 是否配齐 3 个变量, 返回读到的值."""
    missing = [v for v in REQUIRED_ENV_VARS if not os.environ.get(v)]
    if missing:
        print(f"❌ .env 缺少变量: {missing}")
        print()
        print("请在 .env 加入以下 3 行 (不要覆盖原有 OPENAI_API_KEY):")
        print("  OPENAI_JUDGE_API_KEY=sk-中转站key")
        print("  OPENAI_JUDGE_BASE_URL=https://api.中转站.com/v1")
        print("  OPENAI_JUDGE_MODEL=gpt-4o-mini")
        print()
        print("如果你 .env 里已经用了别的变量名, 请修改本脚本顶部的")
        print("REQUIRED_ENV_VARS 列表来匹配.")
        sys.exit(1)

    values = {v: os.environ[v] for v in REQUIRED_ENV_VARS}
    print("✅ 环境变量自检通过:")
    print(f"   OPENAI_JUDGE_BASE_URL = {values['OPENAI_JUDGE_BASE_URL']}")
    print(f"   OPENAI_JUDGE_MODEL    = {values['OPENAI_JUDGE_MODEL']}")
    # 不打印 API_KEY 明文, 只验证存在
    key = values["OPENAI_JUDGE_API_KEY"]
    print(f"   OPENAI_JUDGE_API_KEY  = {key[:7]}...{key[-4:]} (长度 {len(key)})")
    print()
    return values


# =============================================================================
# 阶段 1: RAGAS 版本断言 (版本不对立刻挂)
# =============================================================================
def check_ragas_version() -> None:
    """验证 ragas 装的是 0.4.3. 避免老教程代码混入."""
    try:
        import ragas
    except ImportError:
        print("❌ ragas 未安装. 请先跑: uv pip install ragas==0.4.3")
        sys.exit(1)

    version = getattr(ragas, "__version__", "unknown")
    print(f"✅ ragas 已安装, 版本 = {version}")
    if not version.startswith("0.4"):
        print(f"⚠️  期望 0.4.x, 实际 {version}. Collections API 可能不可用")
        print("    建议: uv pip install --force-reinstall ragas==0.4.3")
        # 不直接退出 — 允许你知情地继续, 但后续 API 调用可能报错
    print()


# =============================================================================
# 阶段 2: 硬编码 1 条测试样本 (绕开 pipeline, 隔离变量)
# =============================================================================
# 用 U1 茅台 2023 营业总收入的"已知答对"场景.
# - query / answer / context 全部硬编码, 不依赖 Qdrant / BGE / DeepSeek
# - 这样 smoke test 失败时, 100% 是 RAGAS / 中转 / judge 的问题, 不可能是 pipeline 问题
SMOKE_SAMPLE = {
    "user_input": "贵州茅台 2023 年营业总收入是多少?",
    "response": "贵州茅台 2023 年营业总收入为 1,505.60 亿元 [1]。",
    "retrieved_contexts": [
        "[1] 来源: 贵州茅台 2023年报 第2页 (text)\n"
        "2023 年, 公司实现营业总收入 1,505.60 亿元, 同比增长 18.04%; "
        "实现归属于上市公司股东的净利润 747.34 亿元, 同比增长 19.16%. "
        "营业收入主要由茅台酒和系列酒两大产品线构成."
    ],
}


# =============================================================================
# 阶段 3: 构造 judge client + RAGAS Faithfulness metric
# =============================================================================
async def run_smoke(env: dict[str, str]) -> float:
    """跑 1 条 Faithfulness 评估, 返回分数.

    使用 Collections API (ragas 0.4 推荐, 避免 deprecated evaluate()).
    """
    # 用 AsyncOpenAI 显式指向中转站. 不用默认 OPENAI_API_KEY, 避免和 DeepSeek 串线
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.metrics.collections import Faithfulness

    print("→ 构造 AsyncOpenAI client (指向中转站)...")
    client = AsyncOpenAI(
        api_key=env["OPENAI_JUDGE_API_KEY"],
        base_url=env["OPENAI_JUDGE_BASE_URL"],
    )

    print(f"→ 构造 llm_factory (model={env['OPENAI_JUDGE_MODEL']})...")
    llm = llm_factory(env["OPENAI_JUDGE_MODEL"], client=client)

    print("→ 构造 Faithfulness scorer...")
    scorer = Faithfulness(llm=llm)

    print("→ 调用 scorer.ascore() — 这一步会向中转站发 2-4 次请求...")
    print(f"   query:    {SMOKE_SAMPLE['user_input']}")
    print(f"   response: {SMOKE_SAMPLE['response']}")
    print(f"   contexts: {len(SMOKE_SAMPLE['retrieved_contexts'])} 条")
    print()

    result = await scorer.ascore(
        user_input=SMOKE_SAMPLE["user_input"],
        response=SMOKE_SAMPLE["response"],
        retrieved_contexts=SMOKE_SAMPLE["retrieved_contexts"],
    )

    # Collections API 返回的是 MetricResult 对象, 分数在 .value
    score = result.value if hasattr(result, "value") else float(result)
    return float(score)


# =============================================================================
# 阶段 4: 成功判据校验
# =============================================================================
def validate_score(score: float) -> bool:
    """校验分数是否合法. 返回 True 表示 T0 通过."""
    import math

    print()
    print("=" * 60)
    print(f"📊 Faithfulness 分数: {score}")
    print("=" * 60)

    if math.isnan(score):
        print("❌ 分数是 NaN — RAGAS 无法解析 judge 响应")
        print("   可能原因:")
        print("   - judge 模型返回格式不对 (非 JSON)")
        print("   - judge 模型对英文 prompt 指令遵循差")
        print("   → 把完整终端输出贴给 Claude 诊断")
        return False

    if not (0.0 <= score <= 1.0):
        print(f"❌ 分数越界 [{score}] — RAGAS 逻辑有问题或版本不兼容")
        return False

    print("✅ 分数在合法区间 [0.0, 1.0]")
    print()
    print("T0 通过. 进入 T1: 评估集设计 + pipeline context 完整化")
    print()
    print("本例是 U1 已知答对场景, 理论 Faithfulness 应该接近 1.0.")
    print("如果分数 < 0.5, judge 可能对中文年报场景有偏差, 记一笔")
    print("到 day11 笔记的 judge 观察段, T3 人工抽检时重点看.")
    return True


# =============================================================================
# 主流程
# =============================================================================
def main() -> None:
    print("=" * 60)
    print("Day 11 T0: RAGAS + 中转站 + judge 模型 smoke test")
    print("=" * 60)
    print()

    env = check_env()
    check_ragas_version()

    try:
        score = asyncio.run(run_smoke(env))
    except ImportError as e:
        print(f"❌ Import 失败: {e}")
        print("   可能原因: ragas 版本不对 (Collections API 在 0.4 才有)")
        print("   修法: uv pip install --force-reinstall ragas==0.4.3")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 调用失败: {type(e).__name__}: {e}")
        print()
        print("   分类:")
        print("   - 401/403 AuthenticationError: 中转 key 无效 / 过期")
        print("   - 404: base_url 或 model 名错误")
        print("   - Timeout: 中转站连不上")
        print("   - JSON/Parse 错误: judge 模型返回格式不符 RAGAS 期望")
        print()
        print("   → 把完整 traceback 贴给 Claude")
        sys.exit(1)

    ok = validate_score(score)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()