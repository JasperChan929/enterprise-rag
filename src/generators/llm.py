"""LLM 生成模块:调用 DeepSeek/OpenAI 兼容 API 生成答案。

核心职责:
  - 管理 API 客户端(单例)
  - 组装 RAG Prompt(把检索到的 chunks 塞进去）
  - 调用 LLM 生成带引用的答案
"""
import os
from dotenv import load_dotenv
load_dotenv(override=True)

from openai import OpenAI


# RAG 专用 Prompt 模板
RAG_SYSTEM_PROMPT = RAG_SYSTEM_PROMPT = """你是一个专业的金融文档问答助手。请严格基于下面提供的【参考资料】回答用户问题。

规则:
1. 只使用参考资料中的信息回答, 不要使用你自己的知识
2. 如果参考资料中没有足够信息回答问题, 请明确说"根据已有资料无法回答此问题"
3. 回答中用 [1][2] 等标注引用了哪条参考资料
4. 涉及数字时必须精确引用原文数据, 不要四舍五入或估算
5. 当参考资料包含多个口径的同一指标 (如本公司 / 本集团 / 合并 / 母公司 / 业务分部 / 子公司), 优先给出与问题字面最匹配的口径, 并简要列出其他口径供参考。多口径并列不是"资料不足", 不要因此拒答。
6. 如果参考资料提供的是相关但口径不匹配的数据 (如问集团但只有子公司, 问本期但只有上期), 应明确告知用户: 已找到 X 口径的数据 (附数值和引用), 但未找到问题所问的 Y 口径数据。不要混用口径作答。
7. 保持专业、简洁"""

RAG_USER_TEMPLATE = """【参考资料】
{context}

【用户问题】
{question}

请基于参考资料回答："""


def get_llm_client() -> OpenAI:
    """获取 LLM 客户端。配置从 .env 读取。"""
    return OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com"),
    )


def format_context(search_results: list[dict]) -> str:
    """把检索结果格式化成 Prompt 里的【参考资料】部分。

    每条结果编号,带上来源信息,方便 LLM 标注引用。
    """
    context_parts = []
    for i, result in enumerate(search_results, 1):
        meta = result["metadata"]
        source = meta.get("company", "未知")
        year = meta.get("year", "")
        page = meta.get("page", "?")
        chunk_type = meta.get("chunk_type", "text")

        header = f"[{i}] 来源: {source} {year}年报 第{page}页 ({chunk_type})"
        content = result["content"]
        context_parts.append(f"{header}\n{content}")

    return "\n\n".join(context_parts)


def generate_answer(
    question: str,
    search_results: list[dict],
    model: str| None = None,
    temperature: float = 0.1,
) -> str:
    """基于检索结果生成答案。

    Args:
        question: 用户问题
        search_results: search_similar() 返回的结果列表
        model: LLM 模型名,默认从 .env 读
        temperature: 生成温度,RAG 场景建议低温(更精确)

    Returns:
        LLM 生成的答案文本
    """
    model = model or os.environ.get("LLM_MODEL", "deepseek-chat")
    client = get_llm_client()

    # 组装 prompt
    context = format_context(search_results)
    user_message = RAG_USER_TEMPLATE.format(
        context=context,
        question=question,
    )

    # 调用 LLM
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
        max_tokens=1024,
    )

    return response.choices[0].message.content or ""