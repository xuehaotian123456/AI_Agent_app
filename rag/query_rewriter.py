"""
查询改写模块
基于历史对话进行查询增强，解决指代消解问题
"""
from typing import List, Optional
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model, robust_llm_caller
from utils.logger_handler import logger


class QueryRewriter:
    """查询改写器：将依赖上下文的查询改写为独立完整的问题"""

    def __init__(self):
        self.rewrite_prompt = PromptTemplate.from_template(
            """你是一个查询改写助手。基于历史对话，将用户的最新查询改写为独立、完整的问题。

历史对话：
{history}

最新查询：{current_query}

要求：
1. 如果最新查询包含代词（如"它"、"这个"、"那款"），替换为具体指代的对象
2. 保持原意不变，只补充缺失的上下文信息
3. 如果最新查询已经是完整问题，直接返回原查询
4. 只输出改写后的查询，不要添加任何解释

改写后的查询："""
        )

    def rewrite(self, current_query: str, messages: List[dict], request_id: str = "") -> str:
        """
        基于历史对话改写查询

        Args:
            current_query: 当前用户查询
            messages: 历史消息列表 [{"role": "user/assistant", "content": "..."}]
            request_id: 请求ID（用于日志追踪）

        Returns:
            改写后的查询（如果不需要改写则返回原查询）
        """
        # 如果历史消息少于2条，不需要改写
        if len(messages) < 2:
            return current_query

        try:
            # 获取最近的2轮对话（避免上下文过长）
            recent_messages = messages[-4:] if len(messages) >= 4 else messages

            # 构建对话历史字符串
            history_text = "\n".join([
                f"{'用户' if msg['role'] == 'user' else '助手'}: {msg['content']}"
                for msg in recent_messages
            ])

            # 格式化 prompt
            formatted_prompt = self.rewrite_prompt.format(
                history=history_text,
                current_query=current_query
            )

            # 调用 LLM 进行改写（使用带重试的调用器）
            response = robust_llm_caller(formatted_prompt)
            rewritten_query = response.content.strip()

            # 验证改写结果
            if rewritten_query and len(rewritten_query) >= 5:
                log_prefix = f"[QueryRewriter-{request_id}]" if request_id else "[QueryRewriter]"
                logger.info(f"{log_prefix} 查询改写: '{current_query}' -> '{rewritten_query}'")
                return rewritten_query
            else:
                logger.warning(f"[QueryRewriter] 改写结果为空或过短，使用原始查询")
                return current_query

        except Exception as e:
            log_prefix = f"[QueryRewriter-{request_id}]" if request_id else "[QueryRewriter]"
            logger.error(f"{log_prefix} 查询改写失败: {e}，使用原始查询")
            return current_query

    def is_low_quality_answer(self, answer: str) -> bool:
        """
        检测答案是否为低质量回答

        Args:
            answer: 生成的答案

        Returns:
            True 如果是低质量回答
        """
        low_quality_indicators = [
            "资料不足",
            "没有找到相关信息",
            "无法回答",
            "抱歉，我",
            "我不知道",
        ]

        answer_lower = answer.lower()
        return any(indicator in answer_lower for indicator in low_quality_indicators)

    def generate_fallback_answer(self, original_query: str, context: str) -> str:
        """
        当主回答质量不佳时，生成降级回答

        Args:
            original_query: 原始查询
            context: 检索到的上下文

        Returns:
            降级回答
        """
        if context and len(context) > 50:
            # 如果有检索到的内容，尝试简单总结
            return f"根据检索到的资料，关于'{original_query}'的相关信息如下：\n\n{context[:500]}..."
        else:
            return f"抱歉，我暂时没有找到关于'{original_query}'的详细信息。您可以尝试：\n1. 换一种提问方式\n2. 提供更具体的关键词\n3. 咨询人工客服"


# 全局单例
query_rewriter = QueryRewriter()
