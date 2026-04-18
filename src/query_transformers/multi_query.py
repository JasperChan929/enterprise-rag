"""Multi-Query 查询改写器。

核心思路:
  用 LLM 把一个用户查询改写成 N 个从不同角度探索的子查询,
  每个子查询独立跑检索, 最后用 RRF 合并。

设计要点:
  - Prompt 明确要求"独立角度", 不要表面同义改写
  - 针对金融年报场景, 提示 LLM 常见的探索维度
  - 保留原查询作为第 0 个子查询(保底, 防止 LLM 改写跑偏)
  - 用 JSON 输出, 稳定可解析
"""
import json
import os
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv
load_dotenv(override=True)  # ← 必须有 override=True

# ============================================================
# Prompt 模板
# ============================================================

MULTI_QUERY_SYSTEM_PROMPT = """你是一个专业的金融文档检索助手。
用户会给你一个关于上市公司年度报告的查询, 你的任务是把这个查询
**拆解成 4 个从不同角度探索的子查询**, 帮助更全面地检索信息。

## 拆解原则

1. **每个子查询关注一个独立的方面**, 而不是同义改写。
   比如用户问"产品情况", 不要生成"产品概况/产品详情/产品说明", 
   而要从"产品系列""技术参数""应用场景""市场表现"等**不同维度**展开。

2. **紧贴年报常见的披露维度**:
   - 财务数据: 营收/利润/毛利率/现金流
   - 业务构成: 产品线/业务板块/地区分布/客户结构
   - 技术研发: 研发投入/专利/核心技术
   - 风险披露: 经营风险/市场风险/合规风险
   - 公司治理: 股东结构/董事会/管理层
   - 未来规划: 战略方向/业务目标

3. **保留关键实体**: 公司名、年份、产品名等关键词应出现在子查询中。

4. **子查询要具体、陈述化**: 避免"...的情况"这种抽象词, 
   用"XX 具体包括哪些"、"XX 的数据是多少"等更贴近文档语言的表述。
5. **⚠️ 严格约束**: 
   - **不要添加用户未提及的时间信息**。如果用户没说年份, 就不要在子查询里自己加 "2023 年"、"2024 年" 等。
   - **不要添加用户未提及的具体产品/业务名**, 除非是明显的同义展开。
   - **不要跳出原查询的核心意图**, 所有子查询必须围绕用户真正想问的事。
## 输出格式

只输出 JSON, 不要任何额外文字:

```json
{
  "sub_queries": [
    "子查询 1",
    "子查询 2",
    "子查询 3",
    "子查询 4"
  ]
}
```
"""


USER_TEMPLATE = """用户查询: {query}

请生成 4 个不同角度的子查询。"""


# ============================================================
# 改写器
# ============================================================

class MultiQueryRewriter:
    """Multi-Query 查询改写器。

    用法:
        rewriter = MultiQueryRewriter()
        sub_queries = rewriter.rewrite("宁德时代磷酸铁锂电池的产品情况")
        # → [
        #     "宁德时代有哪些磷酸铁锂电池产品系列和具体型号",
        #     "宁德时代磷酸铁锂电池的技术参数和能量密度",
        #     ...
        # ]
    """

    def __init__(
        self,
        model: Optional[str] = None,
        num_queries: int = 4,
        include_original: bool = True,
    ):
        """
        Args:
            model: LLM 模型名, 默认从 .env 读
            num_queries: 生成几个子查询
            include_original: 是否把原查询也作为一路(保底)
        """
        self.model = model or os.environ.get("LLM_MODEL", "deepseek-chat")
        self.num_queries = num_queries
        self.include_original = include_original

        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com"),
        )

    def rewrite(self, query: str) -> list[str]:
        """改写一个查询成多个子查询。

        Returns:
            子查询列表。如果 include_original=True, 原查询会作为第 0 项。
        """
        # 动态替换 prompt 里的数字(num_queries 可配)
        system_prompt = MULTI_QUERY_SYSTEM_PROMPT.replace(
            "4 个", f"{self.num_queries} 个"
        ).replace("4", str(self.num_queries))

        user_message = USER_TEMPLATE.format(query=query)

        # 调用 LLM
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,  # 稍高一点鼓励多样性, 但别太高避免跑偏
            max_tokens=600,
            response_format={"type": "json_object"},  # DeepSeek 支持 JSON mode
        )

        # 解析 JSON
        raw = response.choices[0].message.content or "{}"
        try:
            data = json.loads(raw)
            sub_queries = data.get("sub_queries", [])
        except json.JSONDecodeError:
            print(f"⚠️ JSON 解析失败, 原始输出: {raw}")
            sub_queries = []

        # 清洗: 去空、去重(以防 LLM 重复生成)
        sub_queries = [q.strip() for q in sub_queries if q and q.strip()]
        sub_queries = list(dict.fromkeys(sub_queries))  # 保序去重

        # 保底: LLM 解析失败或生成为空
        if not sub_queries:
            print(f"⚠️ LLM 未生成有效子查询, 仅使用原查询")
            return [query]

        # 拼接原查询
        if self.include_original:
            return [query] + sub_queries
        else:
            return sub_queries