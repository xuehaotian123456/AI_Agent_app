from typing import Callable
from utils.prompt_loader import load_system_prompts,load_report_prompts
from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from utils.logger_handler import logger


@wrap_tool_call
def monitor_tool(              #工具执行的监控
        request: ToolCallRequest,               #请求的数据封装
        handler: Callable[[ToolCallRequest],ToolMessage | Command],               #执行的函数本身
) -> ToolMessage | Command:
    logger.info(f"[monitor_tool]工具执行开始：{request.tool_call['name']}")
    logger.info(f"[monitor_tool]工具执行参数：{request.tool_call['args']}")
    try:
        result = handler(request)
        logger.info(f"[monitor_tool]工具执行结果：{result}")
        if request.tool_call["name"] == "fill_context_for_report":
            request.runtime.context["report"] = True
        return result
    except Exception as e:
        logger.error(f"[monitor_tool]| 工具执行异常：{request.tool_call['name']}:{e}")
        raise e #抛出异常
@before_model
def log_before_model(
        state : AgentState, # 状态记录
        runtime : Runtime, # 记录执行过程的上下文信息
):  #模型执行前的日志
    logger.info(f"[log_before_model]即将调用模型，带有：{len(state['messages'])}条信息。")

    logger.debug(f"[log_before_model] {type(state['messages'][-1]).__name__} |当前信息：{state['messages'][-1].content.strip()}]")
    return None
@dynamic_prompt    #每一次生成提示词之前调用此函数
def report_prompt_switch(request : ModelRequest):
    is_report = request.runtime.context.get("report", False)#提示词动态切换的日志
    if is_report:
        logger.info(f"[report_prompt_switch]当前提示词切换为报告模式")
        return load_report_prompts()
    else:
        logger.info(f"[report_prompt_switch]当前提示词切换为正常模式")
        return load_system_prompts()