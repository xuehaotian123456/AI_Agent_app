"""
Agent 模块

包含：
- ReactAgent: 原有单Agent（兼容Streamlit UI）
- Multi-Agent: 四智能体闭环系统（API v2）
- Tools: 企业级工具调用体系（ToolRegistry + 增强工具集）
- Middleware: 工具监控 + 提示词动态切换

注意：为避免循环导入和缺失依赖，所有子模块使用延迟导入。
"""


def get_react_agent():
    """延迟获取 ReactAgent（避免导入时触发 langchain_chroma 等依赖）"""
    from agent.react_agent import ReactAgent
    return ReactAgent


def get_multi_agent_orchestrator(*args, **kwargs):
    """延迟获取 MultiAgentOrchestrator"""
    from agent.multi_agent_orchestrator import MultiAgentOrchestrator
    return MultiAgentOrchestrator(*args, **kwargs)


__all__ = [
    "get_react_agent",
    "get_multi_agent_orchestrator",
]
