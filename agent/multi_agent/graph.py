"""
Multi-Agent LangGraph 状态图组装

四智能体闭环：
    START → Planner → [条件路由] → Retriever → Reflector → [条件路由] → Summarizer → END
                              ↓                                        ↑
                         (skip retrieval)                    (retry: 回到 Retriever)

关键设计：
1. 图原生的重试循环——Reflector→Retriever 条件边，非外部while循环
2. 有界循环——retry_count < max_retries 保证终止
3. Checkpoint——MemorySaver 支持多轮对话和断点续传
4. Annotated reducer——跨轮累积检索历史和工具调用轨迹
"""
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agent.multi_agent.state import MultiAgentState
from utils.logger_handler import logger


def build_multi_agent_graph(
    planner,
    retriever,
    reflector,
    summarizer,
    max_retries: int = 2
) -> StateGraph:
    """
    构建四智能体闭环 LangGraph

    Args:
        planner: PlannerNode 实例
        retriever: RetrieverNode 实例
        reflector: ReflectionNode 实例
        summarizer: SummarizerNode 实例
        max_retries: 最大重试次数（有界循环保证）

    Returns:
        编译后的 StateGraph（含 MemorySaver checkpointer）
    """

    # === 构建图 ===
    builder = StateGraph(MultiAgentState)

    # 添加四个节点
    builder.add_node("planner", planner)
    builder.add_node("retriever", retriever)
    builder.add_node("reflector", reflector)
    builder.add_node("summarizer", summarizer)

    # === 边和条件路由 ===

    # 入口
    builder.add_edge(START, "planner")

    # Planner → Retriever 或 Summarizer（条件路由）
    builder.add_conditional_edges(
        "planner",
        _route_after_plan,
        {
            "retriever": "retriever",
            "summarizer": "summarizer",
        }
    )

    # Retriever → Reflector（固定边）
    builder.add_edge("retriever", "reflector")

    # Reflector → Retriever（重试） 或 Summarizer（通过）（条件路由）
    builder.add_conditional_edges(
        "reflector",
        lambda state: _route_after_reflection(state, max_retries),
        {
            "retriever": "retriever",
            "summarizer": "summarizer",
        }
    )

    # Summarizer → END（固定边）
    builder.add_edge("summarizer", END)

    # === 编译 ===
    graph = builder.compile(checkpointer=MemorySaver())

    logger.info(
        f"[Multi-Agent Graph] 图编译完成: "
        f"4 nodes (planner/retriever/reflector/summarizer), "
        f"2 conditional edges, max_retries={max_retries}"
    )

    return graph


def _route_after_plan(state: MultiAgentState) -> str:
    """
    Planner 之后的条件路由

    如果 plan.requires_retrieval=False（纯对话/闲聊），跳过检索直接到总结
    否则进入 Retriever 执行工具调用
    """
    plan = state.get("plan", {})
    requires_retrieval = plan.get("requires_retrieval", True)

    if not requires_retrieval:
        logger.info("[Router] Planner → Summarizer (skip retrieval)")
        return "summarizer"
    else:
        logger.info("[Router] Planner → Retriever")
        return "retriever"


def _route_after_reflection(state: MultiAgentState, max_retries: int = 2) -> str:
    """
    Reflector 之后的条件路由（核心重试逻辑）

    回到 Retriever 的条件：
    1. reflection.verdict == "retry"（反思判定需要重试）
    2. retry_count < max_retries（有界循环，保证终止）

    否则进入 Summarizer 生成最终答案
    """
    reflection = state.get("reflection", {})
    retry_count = state.get("retry_count", 0)

    verdict = reflection.get("verdict", "pass")

    should_retry = (
        verdict == "retry"
        and retry_count < max_retries
    )

    if should_retry:
        logger.info(
            f"[Router] Reflector → Retriever (RETRY {retry_count + 1}/{max_retries}): "
            f"reason={reflection.get('reason', 'N/A')[:80]}"
        )
        return "retriever"
    else:
        if verdict == "retry":
            logger.warning(
                f"[Router] Reflector → Summarizer (MAX_RETRIES reached: {retry_count})"
            )
        else:
            logger.info(
                f"[Router] Reflector → Summarizer (PASS: confidence={reflection.get('confidence', '?')})"
            )
        return "summarizer"
