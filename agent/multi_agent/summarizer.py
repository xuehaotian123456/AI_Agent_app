"""
Summarizer Agent — 总结智能体

职责：
1. 综合所有Agent的输出（Plan + 多轮Contexts + Reflection + 工具结果）
2. 生成结构化的最终答案（含引用）
3. 格式化输出（报告类使用Markdown）

设计要点：
- 输入是跨轮累积的 contexts（reducer），拿到完整证据链
- 使用 PydanticOutputParser + 花括号转义（已验证模式）
- 解析失败降级为纯文本，保证可用性
- 返回 AnswerWithCitations 结构化答案
"""
import time
from typing import Dict, Any, Optional
import json

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from model.factory import robust_llm_caller, chat_model, get_last_model_used
from utils.prompt_loader import load_prompt_from_file
from utils.path_tool import get_abs_path
from utils.logger_handler import logger
from utils.observability import observe_if_available, update_observation_safe

from rag.output_models import AnswerWithCitations
from agent.multi_agent.models import ReflectionResult
from agent.multi_agent.state import MultiAgentState


class SummarizerNode:
    """
    总结智能体节点

    作为 LangGraph 节点，输入 MultiAgentState，返回部分状态更新
    """

    def __init__(self):
        # 加载总结提示词
        prompt_path = get_abs_path("prompts/summarizer.txt")
        try:
            template = load_prompt_from_file(prompt_path)
        except Exception:
            logger.warning("[Summarizer] 提示词文件加载失败，使用内置模板")
            template = self._get_builtin_prompt()

        self.prompt_template = PromptTemplate.from_template(template)

        # PydanticOutputParser（复用现有模式）
        self.parser = PydanticOutputParser(pydantic_object=AnswerWithCitations)
        format_instructions = self.parser.get_format_instructions()
        self.escaped_instructions = format_instructions.replace("{", "{{").replace("}", "}}")

    @observe_if_available(name="summarizer")
    def __call__(self, state: MultiAgentState) -> Dict[str, Any]:
        """执行总结"""
        start_time = time.time()

        query = state.get("query", state.get("original_query", ""))
        original_query = state.get("original_query", query)
        plan = state.get("plan", {})
        contexts = state.get("contexts", [])
        reflection = state.get("reflection", {})
        tool_outputs = state.get("tool_outputs", {})
        messages = state.get("messages", [])
        draft_answer = state.get("draft_answer", "")

        # 合并上下文
        merged_context = "\n---\n".join(contexts) if contexts else "（无检索结果）"

        # 如果有草稿答案（Reflection阶段生成的），也纳入
        if draft_answer and draft_answer not in merged_context:
            merged_context = merged_context + f"\n\n[草稿参考]\n{draft_answer[:500]}"

        logger.info(
            f"[Summarizer] 开始总结: query='{original_query[:60]}...', "
            f"contexts={len(contexts)}, intent={plan.get('intent')}"
        )

        try:
            # 格式化 Prompt
            formatted = self.prompt_template.format(
                query=original_query,
                plan=str(plan.get("summary", "")) + f" | intent: {plan.get('intent', 'knowledge')}",
                context=merged_context[:4000],
                history=self._format_history(messages),
                tool_outputs=str(tool_outputs)[:1000],
                reflection=str(reflection.get("reason", "")) + f" | confidence: {reflection.get('confidence', '?')}",
                format_instructions=self.escaped_instructions,
            )

            # 调用 LLM
            response = robust_llm_caller(formatted)
            model_used = get_last_model_used()
            content = response.content if hasattr(response, 'content') else str(response)

            # 解析结构化输出
            answer, structured_result, citations = self._parse_output(content)

        except Exception as e:
            logger.error(f"[Summarizer] LLM调用失败: {e}", exc_info=True)
            answer = draft_answer or "抱歉，生成回答时出现错误，请稍后重试。"
            structured_result = None
            citations = []
            model_used = "unknown"

        elapsed = round((time.time() - start_time) * 1000, 2)
        confidence = structured_result.get("confidence") if structured_result else reflection.get("confidence", 0.5)

        logger.info(
            f"[Summarizer] 完成: answer_len={len(answer)}, "
            f"citations={len(citations)}, confidence={confidence}, "
            f"elapsed={elapsed}ms, model={model_used}"
        )

        update_observation_safe(
            output={"answer_preview": answer[:200], "citations_count": len(citations)},
            metadata={"model_used": model_used, "elapsed_ms": elapsed, "confidence": confidence},
        )

        return {
            "answer": answer,
            "structured_result": structured_result,
            "node_times": [{"node": "summarizer", "elapsed_ms": elapsed}],
            "models_used": [model_used],
            "llm_usage": [{"node": "summarizer", "model": model_used}],
        }

    # ==================== 内部方法 ====================

    def _parse_output(self, content: str) -> tuple:
        """
        解析LLM输出为 (answer, structured_result, citations)

        三层fallback：
        1. PydanticParser直接解析
        2. 提取JSON块解析
        3. 纯文本降级
        """
        try:
            parsed = self.parser.parse(content)
            structured = parsed.model_dump()
            answer = parsed.answer
            citations = [cit.model_dump() for cit in parsed.citations] if parsed.citations else []
            return answer, structured, citations
        except Exception as e1:
            logger.warning(f"[Summarizer] 直接解析失败: {e1}")

            try:
                # 提取JSON
                start = content.find("{")
                end = content.rfind("}")
                if start != -1 and end != -1:
                    json_str = content[start:end + 1]
                    parsed = AnswerWithCitations.model_validate_json(json_str)
                    structured = parsed.model_dump()
                    answer = parsed.answer
                    citations = [cit.model_dump() for cit in parsed.citations] if parsed.citations else []
                    return answer, structured, citations
            except Exception as e2:
                logger.warning(f"[Summarizer] JSON提取也失败: {e2}")

        # 最终降级：纯文本
        clean_content = content.strip()
        if len(clean_content) > 10:
            return clean_content, None, []
        return "抱歉，我无法根据现有资料回答您的问题。请尝试换个方式提问。", None, []

    @staticmethod
    def _format_history(messages: list) -> str:
        """格式化对话历史"""
        if not messages:
            return "（新对话）"
        recent = messages[-6:]
        lines = []
        for msg in recent:
            role = "用户" if msg.get("role") == "user" else "助手"
            content = str(msg.get("content", ""))[:150]
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    @staticmethod
    def _get_builtin_prompt() -> str:
        return """你是专业的知识总结专家。

用户问题: {query}
规划: {plan}
参考资料: {context}
对话历史: {history}
工具结果: {tool_outputs}
反思: {reflection}

请基于参考资料生成准确的回答。{format_instructions}"""
