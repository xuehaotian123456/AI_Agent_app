"""
Agent 工具模块

提供企业级 Function Calling 工具管理体系：

- ToolRegistry: 工具注册中心（统一管理、Schema导出、调用追踪）
- ToolDefinition: 工具标准化定义（OpenAI Function Calling 兼容）
- enhanced_tools: 15+增强工具实现（5类：检索/数据查询/外部API/分析/系统）
- middleware: 工具监控中间件

使用方式：
    from agent.tools import tool_registry, register_all_tools
    register_all_tools()  # 启动时调用一次
    result = tool_registry.invoke("hybrid_search", {"query": "扫地机器人选购"})
"""
from agent.tools.tool_registry import (
    ToolRegistry,
    ToolDefinition,
    ToolCategory,
    ToolCallRecord,
    tool_registry,
)
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch
from utils.logger_handler import logger


def register_all_tools(rag_service=None):
    """
    注册所有增强工具到全局 tool_registry

    在应用启动时调用一次，将所有可用工具注册到工具注册中心。
    之后 Planner/Retriever 可通过 tool_registry 发现和调用工具。

    Args:
        rag_service: 可选的 RAG 服务实例（用于需要向量检索的工具）
    """
    from agent.tools.enhanced_tools import (
        hybrid_search_tool,
        entity_lookup_tool,
        keyword_search_tool,
        similar_doc_search_tool,
        query_user_usage_tool,
        get_device_specs_tool,
        compare_models_tool,
        get_weather_tool,
        get_maintenance_schedule_tool,
        check_firmware_version_tool,
        analyze_usage_pattern_tool,
        calculate_consumable_life_tool,
        troubleshoot_issue_tool,
        get_user_context_tool,
        log_user_feedback_tool,
        build_tool_definitions,
    )

    definitions = build_tool_definitions()
    handlers = {
        "hybrid_search": hybrid_search_tool,
        "entity_lookup": entity_lookup_tool,
        "keyword_search": keyword_search_tool,
        "similar_doc_search": similar_doc_search_tool,
        "query_user_usage": query_user_usage_tool,
        "get_device_specs": get_device_specs_tool,
        "compare_models": compare_models_tool,
        "get_weather": get_weather_tool,
        "get_maintenance_schedule": get_maintenance_schedule_tool,
        "check_firmware_version": check_firmware_version_tool,
        "analyze_usage_pattern": analyze_usage_pattern_tool,
        "calculate_consumable_life": calculate_consumable_life_tool,
        "troubleshoot_issue": troubleshoot_issue_tool,
        "get_user_context": get_user_context_tool,
        "log_user_feedback": log_user_feedback_tool,
    }

    registered_count = 0
    for name, definition in definitions.items():
        if name in handlers:
            try:
                tool_registry.register(definition, handlers[name])
                registered_count += 1
            except Exception as e:
                logger.error(f"[工具注册] 注册 '{name}' 失败: {e}")

    logger.info(f"[工具注册] 已注册 {registered_count}/{len(definitions)} 个工具到 ToolRegistry")

    # 同时注册到原有的 LangChain 工具列表（向后兼容）
    try:
        _register_langchain_tools()
    except Exception as e:
        logger.debug(f"[工具注册] LangChain工具兼容注册跳过: {e}")

    return registered_count


def _register_langchain_tools():
    """将增强工具也注册为 LangChain 兼容的 @tool 装饰器函数（向后兼容）"""
    from langchain_core.tools import tool as lc_tool
    from agent.tools.enhanced_tools import hybrid_search_tool

    @lc_tool(description="从向量存储中检索参考资料（混合检索）")
    def rag_summarize(query: str) -> str:
        result = hybrid_search_tool(query=query, top_k=5)
        if result["success"]:
            data = result["data"]
            if data.get("results"):
                contents = [r["content"][:300] for r in data["results"]]
                return "\n---\n".join(contents)
        return "未找到相关资料"

    # 挂载到模块
    globals()["rag_summarize"] = rag_summarize


# 导出列表
__all__ = [
    "ToolRegistry",
    "ToolDefinition",
    "ToolCategory",
    "ToolCallRecord",
    "tool_registry",
    "register_all_tools",
    "monitor_tool",
    "log_before_model",
    "report_prompt_switch",
]
