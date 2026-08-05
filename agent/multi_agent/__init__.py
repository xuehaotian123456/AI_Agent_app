"""
Multi-Agent 知识问答系统 — 四智能体闭环

模块结构：
    models.py    — Pydantic数据模型（ExecutionPlan, ReflectionResult, MultiAgentResponse等）
    state.py     — LangGraph TypedDict 状态定义（含 Annotated reducer）
    planner.py   — PlannerNode（规划智能体：意图识别 + 任务拆解 + 工具选择）
    retriever.py — RetrieverNode（检索智能体：工具调用编排 + 三路多路召回）
    reflection.py — ReflectionNode（反思智能体：置信度自检 + 失败重检索决策）
    summarizer.py — SummarizerNode（总结智能体：证据链综合 + 结构化输出）
    graph.py     — build_multi_agent_graph()（四节点条件图组装）

使用方式：
    from agent.multi_agent.graph import build_multi_agent_graph
    from agent.multi_agent.planner import PlannerNode
    from agent.multi_agent.retriever import RetrieverNode
    from agent.multi_agent.reflection import ReflectionNode
    from agent.multi_agent.summarizer import SummarizerNode

注意：节点类（PlannerNode等）需要项目完整依赖（langchain_chroma等），
     纯数据模型和状态定义可独立使用。
"""
from agent.multi_agent.graph import build_multi_agent_graph
from agent.multi_agent.state import MultiAgentState
from agent.multi_agent.models import (
    ExecutionPlan,
    SubTask,
    ReflectionResult,
    RetrievalRound,
    MultiAgentResponse,
    ToolCallTrace,
)

# 节点类使用延迟导入（避免未安装 langchain_chroma 等依赖时导入失败）


def _lazy_import(name):
    """延迟导入节点类"""
    _imports = {
        "PlannerNode": "agent.multi_agent.planner",
        "RetrieverNode": "agent.multi_agent.retriever",
        "ReflectionNode": "agent.multi_agent.reflection",
        "SummarizerNode": "agent.multi_agent.summarizer",
    }
    if name not in _imports:
        raise AttributeError(f"module has no attribute '{name}'")
    import importlib
    module = importlib.import_module(_imports[name])
    return getattr(module, name)


def __getattr__(name):
    if name in ("PlannerNode", "RetrieverNode", "ReflectionNode", "SummarizerNode"):
        return _lazy_import(name)
    raise AttributeError(f"module 'agent.multi_agent' has no attribute '{name}'")


__all__ = [
    "build_multi_agent_graph",
    "MultiAgentState",
    "ExecutionPlan",
    "SubTask",
    "ReflectionResult",
    "RetrievalRound",
    "MultiAgentResponse",
    "ToolCallTrace",
    "PlannerNode",
    "RetrieverNode",
    "ReflectionNode",
    "SummarizerNode",
]
