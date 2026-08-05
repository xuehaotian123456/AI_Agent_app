"""
Planner Agent — 规划智能体

职责：
1. 分析用户查询意图（知识问答/报告生成/闲聊/故障诊断/对比选择）
2. 从 ToolRegistry 获取可用工具集
3. 将复杂问题拆解为1-N个子任务
4. 为每个子任务指定需要调用的工具
5. 输出结构化的 ExecutionPlan

设计要点：
- 使用 PydanticOutputParser + 花括号转义（已验证模式）
- 解析失败时优雅降级为单任务fallback plan
- 工具选择由 ToolRegistry.get_available_tools() 按意图预筛选
"""
import time
from typing import Dict, Any
import json

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from model.factory import robust_llm_caller, get_last_model_used
from utils.prompt_loader import load_prompt_from_file
from utils.path_tool import get_abs_path
from utils.logger_handler import logger
from utils.observability import observe_if_available, update_observation_safe
from utils.config_handler import multi_agent_conf

from agent.multi_agent.models import ExecutionPlan, SubTask
from agent.multi_agent.state import MultiAgentState


class PlannerNode:
    """
    规划智能体节点

    作为 LangGraph 的节点函数，输入 MultiAgentState，返回部分状态更新
    """

    def __init__(self):
        self.max_subtasks = multi_agent_conf.get("planner", {}).get("max_subtasks", 3)

        # 加载提示词
        prompt_path = get_abs_path("prompts/planner.txt")
        try:
            template = load_prompt_from_file(prompt_path)
        except Exception:
            logger.warning("[Planner] 提示词文件加载失败，使用内置模板")
            template = self._get_builtin_prompt()

        self.prompt_template = PromptTemplate.from_template(template)

        # PydanticOutputParser + 花括号转义
        self.parser = PydanticOutputParser(pydantic_object=ExecutionPlan)
        format_instructions = self.parser.get_format_instructions()
        escaped_instructions = format_instructions.replace("{", "{{").replace("}", "}}")

        # 将格式指令拼接到prompt后
        combined = template + "\n\n{format_instructions}"
        self.full_prompt = PromptTemplate.from_template(combined)
        self.format_instructions = escaped_instructions

    @observe_if_available(name="planner")
    def __call__(self, state: MultiAgentState) -> Dict[str, Any]:
        """执行规划"""
        start_time = time.time()
        query = state.get("query", state.get("original_query", ""))
        messages = state.get("messages", [])

        logger.info(f"[Planner] 开始规划: query='{query[:80]}...'")

        try:
            # 1. 获取可用工具列表
            from agent.tools.tool_registry import tool_registry
            available_tools_text = tool_registry.get_schema_text()

            # 2. 构建对话历史文本
            history_text = self._format_history(messages)

            # 3. 格式化 Prompt
            formatted = self.full_prompt.format(
                query=query,
                history=history_text if history_text else "（无历史对话）",
                max_subtasks=self.max_subtasks,
                available_tools=available_tools_text,
                format_instructions=self.format_instructions,
            )

            # 4. 调用 LLM
            response = robust_llm_caller(formatted)
            model_used = get_last_model_used()
            elapsed = time.time() - start_time

            # 5. 解析结构化输出
            plan = self._parse_with_fallback(response.content if hasattr(response, 'content') else str(response), query)
            plan_dict = plan.model_dump()

            logger.info(
                f"[Planner] 规划完成: intent={plan.intent}, "
                f"sub_tasks={len(plan.sub_tasks)}, "
                f"tools={[t for st in plan.sub_tasks for t in st.required_tools]}, "
                f"elapsed={elapsed:.2f}s"
            )

            update_observation_safe(
                output={"intent": plan.intent, "sub_tasks_count": len(plan.sub_tasks)},
                metadata={"model_used": model_used, "elapsed_ms": round(elapsed * 1000, 2)},
            )

            return {
                "plan": plan_dict,
                "original_query": state.get("original_query", query),
                "node_times": [{"node": "planner", "elapsed_ms": round(elapsed * 1000, 2)}],
                "models_used": [model_used],
                "llm_usage": [{"node": "planner", "model": model_used}],
            }

        except Exception as e:
            logger.error(f"[Planner] 规划失败: {e}", exc_info=True)
            elapsed = time.time() - start_time
            # 降级：单任务fallback
            fallback_plan = ExecutionPlan(
                intent="knowledge",
                summary="（降级）直接检索回答",
                sub_tasks=[SubTask(
                    task_id="t1",
                    description=query,
                    search_query=query,
                    required_tools=["hybrid_search"],
                )],
                requires_retrieval=True,
                reasoning="规划器异常，降级为单任务检索模式"
            )
            return {
                "plan": fallback_plan.model_dump(),
                "original_query": state.get("original_query", query),
                "node_times": [{"node": "planner", "elapsed_ms": round(elapsed * 1000, 2)}],
                "models_used": ["fallback"],
            }

    def _parse_with_fallback(self, llm_output: str, original_query: str) -> ExecutionPlan:
        """解析LLM输出，失败时返回降级plan"""
        try:
            # 尝试直接解析
            return self.parser.parse(llm_output)
        except Exception as e:
            logger.warning(f"[Planner] 直接解析失败，尝试提取JSON: {e}")

            # 尝试从文本中提取JSON块
            try:
                json_str = self._extract_json(llm_output)
                return ExecutionPlan.model_validate_json(json_str)
            except Exception as e2:
                logger.warning(f"[Planner] JSON提取也失败: {e2}，使用fallback plan")

                # 最终降级
                return ExecutionPlan(
                    intent="knowledge",
                    summary="（降级）直接检索回答",
                    sub_tasks=[SubTask(
                        task_id="t1",
                        description=original_query,
                        search_query=original_query,
                        required_tools=["hybrid_search"],
                    )],
                    requires_retrieval=True,
                    reasoning=f"解析失败:{type(e).__name__}"
                )

    @staticmethod
    def _extract_json(text: str) -> str:
        """从LLM输出中提取JSON块"""
        # 找 { ... } 块
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start:end + 1]
        raise ValueError("未找到JSON块")

    @staticmethod
    def _format_history(messages: list) -> str:
        """格式化对话历史"""
        if not messages:
            return ""
        recent = messages[-6:]  # 最近3轮
        lines = []
        for msg in recent:
            role = "用户" if msg.get("role") == "user" else "助手"
            content = str(msg.get("content", ""))[:200]
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    @staticmethod
    def _get_builtin_prompt() -> str:
        """内置fallback提示词"""
        return """你是智能任务规划专家。

可用工具: {available_tools}
对话历史: {history}
用户查询: {query}

请输出JSON格式的执行计划，不要添加任何前缀或解释。"""
