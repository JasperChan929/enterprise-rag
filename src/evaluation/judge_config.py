"""Judge 模型集中配置.

===============================================================================
出处与目的
===============================================================================
- Day 11 T1a 新建. 背景: T0 验证了中转站 + gpt-4o-mini judge 能跑通,
  但 judge 配置当时硬编码在 scripts/30_ragas_smoke.py. 如果 Day 12+
  要换 judge (比如 qwen-max / claude-3.5-haiku 做对比), 必须改脚本.
- 本模块把 "判官是谁" 集中到一个文件. Day 12 换 judge 只改这一个文件,
  所有下游脚本自动跟上.
- 和 src/generators/llm.py 的关系: llm.py 管 "生成器" (答题的 LLM),
  judge_config.py 管 "评估器" (判答的 LLM). 两者 client 隔离, key 隔离,
  不共享命名空间. 见 Day 9 D8 "交叉验证前置" 元规则.

===============================================================================
使用方式
===============================================================================
    from src.evaluation.judge_config import build_judge_llm
    llm = build_judge_llm()  # 返回 ragas llm_factory 产出对象
    # 传给 ragas.metrics.collections.Faithfulness(llm=llm) 等 scorer
"""
from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv(override=True)


# =============================================================================
# 环境变量名 (T0 已确定. Day 12 换 judge 只改 .env 不改代码)
# =============================================================================
ENV_API_KEY = "OPENAI_JUDGE_API_KEY"
ENV_BASE_URL = "OPENAI_JUDGE_BASE_URL"
ENV_MODEL = "OPENAI_JUDGE_MODEL"


def check_env() -> dict[str, str]:
    """确认 3 个环境变量齐备, 返回值. 缺一抛 RuntimeError."""
    missing = [v for v in (ENV_API_KEY, ENV_BASE_URL, ENV_MODEL)
               if not os.environ.get(v)]
    if missing:
        raise RuntimeError(
            f".env 缺少 judge 配置: {missing}. "
            f"参见 scripts/30_ragas_smoke.py 文档注释或 docs/day11-summary.md §配置"
        )
    return {
        "api_key": os.environ[ENV_API_KEY],
        "base_url": os.environ[ENV_BASE_URL],
        "model": os.environ[ENV_MODEL],
    }


def build_judge_llm():
    """构造 RAGAS 0.4 Collections API 用的 judge llm.

    返回: ragas.llms.base.BaseRagasLLM 实例, 可直接传给 scorer(llm=...)
    """
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory

    cfg = check_env()
    client = AsyncOpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
    return llm_factory(cfg["model"], client=client)


def describe_judge() -> str:
    """产出一行描述, 用于日志 / 报告首页的配置声明 (不打印 api_key 明文)."""
    cfg = check_env()
    key = cfg["api_key"]
    key_masked = f"{key[:7]}...{key[-4:]}"
    return (f"judge = {cfg['model']} @ {cfg['base_url']} "
            f"(key={key_masked}, len={len(key)})")


if __name__ == "__main__":
    # 自测: 能不能 build, describe 打印啥
    print(describe_judge())
    llm = build_judge_llm()
    print(f"✅ build_judge_llm() 成功: {type(llm).__name__}")