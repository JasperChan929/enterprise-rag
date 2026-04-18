"""HyDE (Hypothetical Document Embeddings) 查询改写器。

核心思路:
  让 LLM 为用户查询生成一段"假设性答案", 用这个假答案去做向量检索,
  因为假答案的语言风格更接近真实文档(陈述句 vs 问句)。

⚠️ 关键安全约束:
  HyDE 的假答案仅用于检索, 绝不能作为最终答案返回给用户。
  最终答案必须由真实文档生成。本模块只负责"生成假答案",
  调用方(Pipeline)必须确保假答案不进入生成环节。

设计要点:
  - Prompt 引导 LLM 生成"像年报一样的陈述句"
  - temperature=0.3 鼓励生成多样但不跑偏
  - 返回的是一段文本, 由调用方自行 embed 和检索
"""
import os
from typing import Optional

from openai import OpenAI


# ============================================================
# Prompt 模板
# ============================================================

HYDE_SYSTEM_PROMPT = """你是一个专业的金融分析师。用户会给你一个
关于上市公司年度报告的问题, 你的任务是**生成一段假设性的答案**,
这段答案将用于向量检索 (不会直接返回给用户)。

## 生成原则

1. **陈述句, 不要问句**: 答案应该像年报/研报里的原文那样陈述,
   而不是解释或反问。
   
   ✅ 好: "贵州茅台 2023 年度实现营业收入 1,500 亿元, 同比增长 18%"
   ❌ 差: "这个问题的答案是需要查看年报的营业收入栏目"

2. **具体、专业**: 使用金融年报常见的表达方式、关键指标和维度。
   
   ✅ 好: "主营业务收入""同比增长""归属于上市公司股东的净利润"
   ❌ 差: "赚了很多钱""发展很好""表现不错"

3. **可以编造具体数字**: 数字是假的没关系, 重要的是语言风格和
   结构对齐真实文档。不要写 "[XX 亿元]" 这种占位符。

4. **长度适中**: 100-200 字, 涵盖 2-3 个相关维度。太短信号不足,
   太长引入噪声。

5. **紧扣问题**: 答案应直接回应问题, 不要跑偏到无关话题。

## 示例

问题: "招商银行 2025 年的不良贷款率是多少?"

答案: 招商银行 2025 年末不良贷款率为 0.95%, 较上年末下降 0.03 个
百分点。不良贷款余额为 680.52 亿元, 同比微增。从业务板块看, 公司
贷款不良率 1.18%, 零售贷款不良率 0.78%. 分地区看, 长三角和珠三角
地区不良率较低, 分别为 0.68% 和 0.72%. 公司持续加强风险管控, 
拨备覆盖率维持在 430% 以上的审慎水平。

---

现在, 请为用户的问题生成假设性答案。"""


USER_TEMPLATE = """问题: {query}

请生成假设性答案:"""


# ============================================================
# HyDE 生成器
# ============================================================

class HyDEGenerator:
    """HyDE 假答案生成器。

    用法:
        hyde = HyDEGenerator()
        fake_answer = hyde.generate("茅台 2023 年的营业收入是多少?")
        # fake_answer 长这样:
        # "贵州茅台 2023 年度实现营业收入 1,500 亿元人民币..."
        
        # 然后调用方用 fake_answer 做向量检索
        vec = embedder.encode(fake_answer)  
        # 注意: 不是 encode_query, 因为假答案是陈述句不是查询
    """

    def __init__(
        self,
        model: Optional[str] = None,
        max_tokens: int = 300,
    ):
        """
        Args:
            model: LLM 模型名, 默认从 .env 读
            max_tokens: 生成长度上限 (200 字中文 ≈ 300 tokens)
        """
        self.model = model or os.environ.get("LLM_MODEL", "deepseek-chat")
        self.max_tokens = max_tokens

        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com"),
        )

    def generate(self, query: str) -> str:
        """为查询生成一段假设性答案。

        Returns:
            假答案文本 (陈述句风格, 类似年报原文)
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": HYDE_SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(query=query)},
            ],
            temperature=0.3,  # 鼓励一些多样性, 但别跑偏
            max_tokens=self.max_tokens,
        )

        fake_answer = response.choices[0].message.content or ""
        return fake_answer.strip()