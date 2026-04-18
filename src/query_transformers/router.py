"""Query Router: 用 LLM 判断查询类型, 自动选择最合适的技术组合。

核心设计:
  - LLM 输出结构化 JSON (use_multi_query, use_hyde)
  - 给 LLM 明确的判断标准 + 示例, 不让它自由发挥
  - 有明确的降级策略: LLM 失败就保守地关掉所有技术

架构位置:
  query → [Router] → 决策 → Pipeline 按决策执行对应的技术组合
"""
import json
import os
from typing import Optional

from openai import OpenAI


# ============================================================
# Prompt 模板
# ============================================================

ROUTER_SYSTEM_PROMPT = """你是一个专业的查询分类器。用户会给你
一个关于上市公司年报的查询, 你的任务是判断该查询需要使用哪些
高级检索技术。

## 可选技术

1. **Multi-Query (横向扩展)**:
   - 适用: 查询抽象、泛化、开放性, 需要从多个角度探索
   - 特征词: "情况" "特点" "优势" "竞争力" "业务布局" "战略"
   - 不适用: 精准具体的问题

2. **HyDE (纵向对齐)**:
   - 适用: 查询是问句形式, 需要的答案是陈述性数字/指标
   - 特征词: "是多少" "多少" "率" "比例" "金额" 等具体指标词
   - 不适用: 极短的专业术语(如"不良贷款率", 无问句结构)

## 判断标准

按以下逻辑判断 (逐条检查, 独立决策):

- 查询**抽象、泛化、需要多角度覆盖** → use_multi_query = true
- 查询**问具体数字/指标, 且是问句形式** → use_hyde = true
- 查询**仅是短专业术语, 无明显问法** → 两者都 false (Hybrid 已足够)
- 查询**既抽象又需要具体数据** → 两者都 true

## 示例

查询: "比亚迪的核心竞争力"
{"use_multi_query": true, "use_hyde": false, "reason": "开放性问题, 需多角度"}

查询: "茅台 2023 年营业收入是多少"
{"use_multi_query": false, "use_hyde": true, "reason": "具体数字问句"}

查询: "招商银行不良贷款率"
{"use_multi_query": false, "use_hyde": false, "reason": "精准术语, Hybrid 足够"}

查询: "宁德时代磷酸铁锂电池的产品情况"
{"use_multi_query": true, "use_hyde": false, "reason": "抽象的'情况'需多角度"}

查询: "紫金矿业海外业务的营收和风险"
{"use_multi_query": true, "use_hyde": true, "reason": "多维度 + 具体指标"}

查询: "国电南自 2024 年净利润"
{"use_multi_query": false, "use_hyde": true, "reason": "具体财务指标"}

## 输出格式

只输出 JSON, 不要任何其他文字:

{
  "use_multi_query": true/false,
  "use_hyde": true/false,
  "reason": "一句话说明判断依据"
}
"""


USER_TEMPLATE = """查询: {query}

请输出判断 JSON:"""


# ============================================================
# Router 类
# ============================================================

class QueryRouter:
    """Query Router: 判断查询应该用哪些高级技术。

    用法:
        router = QueryRouter()
        decision = router.route("茅台营业收入是多少")
        # decision = {
        #     "use_multi_query": False,
        #     "use_hyde": True,
        #     "reason": "具体数字问句"
        # }
    """

    def __init__(self, model: Optional[str] = None):
        self.model = model or os.environ.get("LLM_MODEL", "deepseek-chat")
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com"),
        )

    def route(self, query: str) -> dict:
        """判断查询类型, 返回路由决策。

        Returns:
            {
                "use_multi_query": bool,
                "use_hyde": bool,
                "reason": str,
            }

        降级策略:
            如果 LLM 调用失败或 JSON 解析失败, 返回保守的"全关"决策,
            即退化到纯 Hybrid。这样至少不会比 Day 5 差。
        """
        fallback = {
            "use_multi_query": False,
            "use_hyde": False,
            "reason": "router fallback (LLM 调用失败)",
        }

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                    {"role": "user", "content": USER_TEMPLATE.format(query=query)},
                ],
                temperature=0.1,  # 低温度, 决策要稳定
                max_tokens=150,
                response_format={"type": "json_object"},
            )

            raw = response.choices[0].message.content or "{}"
            decision = json.loads(raw)

            # 字段校验 + 类型强制
            return {
                "use_multi_query": bool(decision.get("use_multi_query", False)),
                "use_hyde": bool(decision.get("use_hyde", False)),
                "reason": str(decision.get("reason", "no reason")),
            }

        except Exception as e:
            print(f"⚠️ Router 失败, 降级到 Hybrid: {e}")
            return fallback