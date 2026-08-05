"""
Reflection Agent — 反思智能体（置信度自检 + 失败重检索决策）

职责：
1. 对累积上下文生成草稿答案（用于质量检查）
2. 复用 SelfCorrector.check_answer_quality() 做质量检查
3. 结构化输出 ReflectionResult（通过LLM）
4. 融合启发式信号（文档数=0 / 低质关键词）
5. 判定是否需要重试 → 改写查询 → 准备下一轮检索

设计要点：
- Fail-open：解析失败时默认通过，避免流水线死锁
- 有界循环：retry_count < max_retries 保证终止
- 复用现有 SelfCorrector：已验证的check_answer_quality + rewrite_query
- 飞轮日志：每次retry都记录到 FlywheelService
"""
import time
from typing import Dict, Any

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from model.factory import robust_llm_caller, get_last_model_used
from utils.prompt_loader import load_prompt_from_file, load_rag_prompts
from utils.path_tool import get_abs_path
from utils.logger_handler import logger
from utils.observability import observe_if_available, update_observation_safe
from utils.config_handler import multi_agent_conf

from agent.multi_agent.models import ReflectionResult
from agent.multi_agent.state import MultiAgentState


class ReflectionNode:
    """
    反思智能体节点

    作为 LangGraph 节点，输入 MultiAgentState，返回部分状态更新
    """

    def __init__(self, self_corrector=None, flywheel=None):
        """
        Args:
            self_corrector: SelfCorrector 实例（复用现有质量检查逻辑）
            flywheel: FlywheelService 实例（可选）
        """
        self.self_corrector = self_corrector
        self.flywheel = flywheel

        reflection_conf = multi_agent_conf.get("reflection", {})
        self.max_retries = reflection_conf.get("max_retries", 2)
        self.confidence_threshold = reflection_conf.get("confidence_threshold", 0.6)

        # 加载反思提示词
        prompt_path = get_abs_path("prompts/reflector.txt")
        try:
            template = load_prompt_from_file(prompt_path)
        except Exception:
            logger.warning("[Reflection] 提示词文件加载失败，使用内置模板")
            template = self._get_builtin_prompt()

        self.prompt_template = PromptTemplate.from_template(template)

        # PydanticOutputParser
        self.parser = PydanticOutputParser(pydantic_object=ReflectionResult)
        format_instructions = self.parser.get_format_instructions()
        self.escaped_instructions = format_instructions.replace("{", "{{").replace("}", "}}")

    @observe_if_available(name="reflector")
    def __call__(self, state: MultiAgentState) -> Dict[str, Any]:
        """执行反思"""
        start_time = time.time()

        query = state.get("query", state.get("original_query", ""))
        plan = state.get("plan", {})
        contexts = state.get("contexts", [])
        retrieval_rounds = state.get("retrieval_rounds", [])
        tool_calls = state.get("tool_calls", [])
        retry_count = state.get("retry_count", 0)

        # 合并累积的上下文
        merged_context = "\n---\n".join(contexts) if contexts else "（无检索结果）"

        # 检查检索结果是否为空
        latest_round = retrieval_rounds[-1] if retrieval_rounds else {}
        docs_count = latest_round.get("docs_count", 0)
        all_tools_failed = all(not tc.get("success", False) for tc in tool_calls) if tool_calls else False

        logger.info(
            f"[Reflection] 第{retry_count + 1}轮反思: "
            f"contexts={len(contexts)}, docs={docs_count}, "
            f"tools_ok={not all_tools_failed}"
        )

        # 1. 生成草稿答案（用于质量检查）
        draft_answer = self._generate_draft(query, merged_context)

        # 2. 质量检查（复用 SelfCorrector）
        check_result = self._run_quality_check(query, merged_context, draft_answer)

        # 3. 结构化反思（通过LLM）
        reflection = self._run_structured_reflection(
            query=query,
            plan=plan,
            context=merged_context,
            draft_answer=draft_answer,
            check_result=check_result,
        )

        # 4. 融合启发式信号
        if docs_count == 0 or all_tools_failed:
            reflection["confidence"] = min(reflection.get("confidence", 0.5), 0.1)  # type: ignore
        if self._is_low_quality(draft_answer):
            reflection["confidence"] = min(reflection.get("confidence", 0.5), 0.3)  # type: ignore

        # 5. 判定是否需要重试
        needs_retry = (
            reflection.get("verdict") == "retry"
            and retry_count < self.max_retries
            and not all_tools_failed
        )

        # 强制重试：有文档但全体工具失败且未达上限
        if all_tools_failed and retry_count < self.max_retries:
            needs_retry = False  # 工具全失败，重试也无意义
            reflection["verdict"] = "pass"
            reflection["reason"] = "所有工具调用失败，直接生成最佳可用答案"

        elapsed = round((time.time() - start_time) * 1000, 2)
        logger.info(
            f"[Reflection] 判定: verdict={reflection.get('verdict')}, "
            f"confidence={reflection.get('confidence', '?')}, "
            f"needs_retry={needs_retry}, elapsed={elapsed}ms"
        )

        update_observation_safe(
            output={
                "verdict": reflection.get("verdict"),
                "confidence": reflection.get("confidence"),
                "needs_retry": needs_retry,
            },
            metadata={"elapsed_ms": elapsed},
        )

        # 构建返回
        result = {
            "reflection": reflection,
            "draft_answer": draft_answer,
            "quality_checks": [check_result],
            "node_times": [{"node": "reflection", "elapsed_ms": elapsed}],
            "models_used": [get_last_model_used()],
        }

        if needs_retry:
            # 改写查询
            refined_query = self._rewrite_query(query, merged_context, draft_answer, check_result)
            result["query"] = refined_query
            result["retry_count"] = retry_count + 1

            # 记录飞轮事件
            if self.flywheel:
                try:
                    self.flywheel.log_event("reflection_fail", {
                        "original_query": query,
                        "refined_query": refined_query,
                        "reason": reflection.get("reason", ""),
                        "confidence": reflection.get("confidence", 0),
                        "docs_count": docs_count,
                        "retry_count": retry_count,
                    })
                except Exception:
                    pass
        else:
            result["retry_count"] = retry_count

        return result

    # ==================== 内部方法 ====================

    def _generate_draft(self, query: str, context: str) -> str:
        """快速生成草稿答案（用于质量检查）"""
        try:
            rag_prompt = load_rag_prompts()
            from langchain_core.prompts import PromptTemplate
            pt = PromptTemplate.from_template(rag_prompt)
            formatted = pt.format(input=query, context=context[:3000])
            response = robust_llm_caller(formatted)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.warning(f"[Reflection] 草稿生成失败: {e}")
            return "（草稿生成失败）"

    def _run_quality_check(self, query: str, context: str, draft: str) -> dict:
        """运行质量检查（复用 SelfCorrector）"""
        if self.self_corrector:
            try:
                return self.self_corrector.check_answer_quality(query, context[:2000], draft)
            except Exception as e:
                logger.warning(f"[Reflection] 质量检查失败: {e}")

        # 降级：内置简单检查
        return {"verdict": "通过", "reason": "降级质量检查", "check_time": 0}

    def _run_structured_reflection(
        self, query: str, plan: dict, context: str, draft_answer: str, check_result: dict
    ) -> dict:
        """通过LLM生成结构化的反思结果"""
        try:
            formatted = self.prompt_template.format(
                query=query,
                plan=str(plan.get("summary", "")) + " | sub_tasks: " + str(len(plan.get("sub_tasks", []))),
                context=context[:2000],
                draft_answer=draft_answer[:1000],
                format_instructions=self.escaped_instructions,
            )
            response = robust_llm_caller(formatted)
            content = response.content if hasattr(response, 'content') else str(response)

            try:
                parsed = self.parser.parse(content)
                return parsed.model_dump()
            except Exception:
                # 提取JSON再试
                import json
                try:
                    start = content.find("{")
                    end = content.rfind("}")
                    if start != -1 and end != -1:
                        json_str = content[start:end + 1]
                        parsed = ReflectionResult.model_validate_json(json_str)
                        return parsed.model_dump()
                except Exception:
                    pass

                # 使用check_result的判定
                verdict = "pass" if check_result.get("verdict") in ("通过", "pass") else "retry"
                return ReflectionResult(
                    verdict=verdict,
                    confidence=0.5,
                    reason=check_result.get("reason", "降级判定")[:100],
                ).model_dump()

        except Exception as e:
            logger.warning(f"[Reflection] 结构化反思失败: {e}")
            return ReflectionResult(
                verdict="pass",
                confidence=0.5,
                reason=f"反思异常，默认通过: {str(e)[:100]}",
            ).model_dump()

    def _rewrite_query(self, original_query: str, context: str, draft: str, check: dict) -> str:
        """改写查询（复用 SelfCorrector）"""
        if self.self_corrector:
            try:
                return self.self_corrector.rewrite_query(
                    original_query=original_query,
                    context=context[:1000],
                    initial_answer=draft[:500],
                    check_reason=check.get("reason", "需要补充信息"),
                )
            except Exception as e:
                logger.warning(f"[Reflection] 查询改写失败: {e}")

        # 降级：直接返回原查询
        return original_query

    @staticmethod
    def _is_low_quality(answer: str) -> bool:
        """检测低质量回答"""
        indicators = ["资料不足", "没有找到", "无法回答", "抱歉", "我不知道", "暂无相关信息"]
        return any(ind in answer for ind in indicators)

    @staticmethod
    def _get_builtin_prompt() -> str:
        return """你是答案质量评估专家。
用户问题: {query}
规划: {plan}
上下文: {context}
草稿答案: {draft_answer}
请输出JSON格式的评估结果: {format_instructions}"""
