"""
工具注册中心 (Tool Registry) — 企业级Function Calling工具管理体系

设计理念：
- 单一注册源：所有工具通过 ToolRegistry.register() 集中注册，避免分散定义
- Schema标准化：每个工具都有 OpenAI Function Calling 兼容的 JSON Schema
- 统一调用入口：参数校验 → 超时控制 → 异常捕获 → 调用追踪，全部在此完成
- 工具发现：根据用户意图自动推荐合适的工具集

这是Agent岗位面试的核心考察点——
"你怎么管理Agent的工具库？" "你怎么做工具发现和调度？"
"""
import time
from typing import Dict, List, Callable, Optional, Any
from pydantic import BaseModel, Field, ValidationError
from enum import Enum
import json

from utils.logger_handler import logger


class ToolCategory(str, Enum):
    """工具分类 — 用于按领域筛选和意图匹配"""
    SEARCH = "search"               # 检索类：文档搜索、实体查询、关键词匹配
    DATA_QUERY = "data_query"       # 数据查询类：用户记录、设备参数、型号对比
    EXTERNAL_API = "external_api"   # 外部接口类：天气、保养、固件
    ANALYSIS = "analysis"           # 分析类：使用习惯、耗材预估、故障诊断
    SYSTEM = "system"               # 系统类：上下文获取、反馈记录


class ToolDefinition(BaseModel):
    """
    工具的标准化定义

    类似 OpenAI Function Calling Schema 的结构，
    确保每个工具都有完整的类型信息供LLM理解
    """
    name: str = Field(..., description="工具唯一标识，如 'hybrid_search'")
    description: str = Field(..., description="工具功能描述，供LLM理解工具用途")
    category: ToolCategory = Field(..., description="工具分类")
    parameters: dict = Field(..., description="输入参数 JSON Schema")
    returns: dict = Field(default_factory=dict, description="返回值 JSON Schema")
    requires_confirmation: bool = Field(default=False, description="敏感操作是否需要用户确认")
    timeout_seconds: float = Field(default=10.0, description="调用超时时间")
    retry_on_failure: bool = Field(default=True, description="失败后是否重试")
    max_retries: int = Field(default=2, description="最大重试次数")


class ToolCallRecord(BaseModel):
    """单次工具调用的完整记录"""
    tool_name: str
    args: dict
    result_summary: str = ""
    success: bool = False
    error: Optional[str] = None
    elapsed_ms: float = 0.0
    retry_count: int = 0


class ToolRegistry:
    """
    工具注册中心

    使用方式：
        registry = ToolRegistry()
        registry.register(
            ToolDefinition(name="hybrid_search", category=ToolCategory.SEARCH, ...),
            hybrid_search_handler
        )
        result = registry.invoke("hybrid_search", {"query": "扫地机器人选购"})
    """

    # 意图 → 推荐工具类别映射
    INTENT_TOOL_MAP = {
        "knowledge": [ToolCategory.SEARCH, ToolCategory.DATA_QUERY],
        "report": [ToolCategory.DATA_QUERY, ToolCategory.ANALYSIS, ToolCategory.SYSTEM],
        "chat": [ToolCategory.SYSTEM],
        "troubleshoot": [ToolCategory.SEARCH, ToolCategory.ANALYSIS],
        "comparison": [ToolCategory.SEARCH, ToolCategory.DATA_QUERY],
    }

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self._handlers: Dict[str, Callable] = {}
        self._call_history: List[ToolCallRecord] = []
        self._max_history = 1000  # 保留最近1000条调用记录

    # ==================== 注册 ====================

    def register(self, definition: ToolDefinition, handler: Callable):
        """
        注册工具到注册中心

        Args:
            definition: 工具的标准化定义（name + schema + 元信息）
            handler: 工具的执行函数，签名为 handler(**kwargs) -> dict

        Raises:
            ValueError: 工具名重复
        """
        if definition.name in self._tools:
            raise ValueError(f"工具 '{definition.name}' 已注册，不允许重复注册")

        self._tools[definition.name] = definition
        self._handlers[definition.name] = handler
        logger.info(
            f"[ToolRegistry] 注册工具: {definition.name} "
            f"(category={definition.category.value}, timeout={definition.timeout_seconds}s)"
        )

    def unregister(self, name: str):
        """注销工具"""
        self._tools.pop(name, None)
        self._handlers.pop(name, None)
        logger.info(f"[ToolRegistry] 注销工具: {name}")

    def is_registered(self, name: str) -> bool:
        """检查工具是否已注册"""
        return name in self._tools

    # ==================== Schema导出（给LLM） ====================

    def get_definition(self, name: str) -> Optional[ToolDefinition]:
        """获取单个工具定义"""
        return self._tools.get(name)

    def get_schemas(
        self,
        categories: Optional[List[ToolCategory]] = None,
        for_openai: bool = True
    ) -> List[dict]:
        """
        导出工具列表，格式兼容 OpenAI Function Calling

        Args:
            categories: 过滤特定分类，None=全部
            for_openai: True → OpenAI格式，False → 完整格式

        Returns:
            工具列表，每个元素包含 name, description, parameters
        """
        schemas = []
        for name, td in self._tools.items():
            if categories and td.category not in categories:
                continue

            if for_openai:
                schemas.append({
                    "type": "function",
                    "function": {
                        "name": td.name,
                        "description": td.description,
                        "parameters": td.parameters,
                    }
                })
            else:
                schemas.append(td.model_dump())

        return schemas

    def get_schema_text(self, categories: Optional[List[ToolCategory]] = None) -> str:
        """
        导出工具列表为文本格式（用于注入Prompt）

        Returns:
            格式化的工具描述文本
        """
        lines = []
        for name, td in self._tools.items():
            if categories and td.category not in categories:
                continue
            params_text = json.dumps(td.parameters, ensure_ascii=False, indent=2)
            lines.append(f"- {td.name} (类别: {td.category.value})")
            lines.append(f"  描述: {td.description}")
            lines.append(f"  参数: {params_text}")

        return "\n".join(lines)

    def get_tools_by_category(self, category: ToolCategory) -> List[str]:
        """按分类获取工具名列表"""
        return [name for name, td in self._tools.items() if td.category == category]

    # ==================== 工具发现 ====================

    def get_available_tools(self, intent: str) -> List[str]:
        """
        根据用户意图推荐可用工具集

        这是"工具发现"能力——不需要LLM每次都在所有工具中选择，
        而是根据意图预先筛选出相关工具集，减少LLM的决策空间

        Args:
            intent: 用户意图类型 (knowledge/report/chat/troubleshoot/comparison)

        Returns:
            推荐的工具名列表
        """
        categories = self.INTENT_TOOL_MAP.get(intent, [ToolCategory.SEARCH])
        return self.get_tool_names(categories=categories)

    def get_tool_names(self, categories: Optional[List[ToolCategory]] = None) -> List[str]:
        """获取工具名列表，可按分类过滤"""
        if categories is None:
            return list(self._tools.keys())
        return [
            name for name, td in self._tools.items()
            if td.category in categories
        ]

    # ==================== 统一调用入口 ====================

    def invoke(
        self,
        name: str,
        args: Optional[dict] = None,
        timeout_override: Optional[float] = None
    ) -> dict:
        """
        统一工具调用入口

        完整的调用生命周期：
        1. 存在性校验 → 2. 参数校验 → 3. 执行（含超时控制）→ 4. 异常捕获 → 5. 结果记录

        Args:
            name: 工具名
            args: 工具参数
            timeout_override: 超时覆盖

        Returns:
            统一格式: {"success": bool, "data": ..., "error": ..., "_meta": {...}}
        """
        args = args or {}
        start_time = time.time()

        # 1. 存在性校验
        if name not in self._tools:
            return self._error_response(name, f"工具 '{name}' 未注册", start_time)

        definition = self._tools[name]
        handler = self._handlers[name]

        # 2. 参数校验
        try:
            valid_args = self._validate_args(definition, args)
        except Exception as e:
            return self._error_response(name, f"参数校验失败: {e}", start_time, args)

        # 3. 执行（含超时+异常捕获+重试）
        timeout = timeout_override or definition.timeout_seconds
        last_error = None

        for attempt in range(definition.max_retries + 1):
            try:
                result = self._execute_with_timeout(handler, valid_args, timeout)
                elapsed_ms = (time.time() - start_time) * 1000

                # 成功 → 标准化返回
                response = {
                    "success": True,
                    "data": result if isinstance(result, dict) else {"value": str(result)},
                    "error": None,
                    "_meta": {
                        "tool_name": name,
                        "elapsed_ms": round(elapsed_ms, 2),
                        "retry_count": attempt,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                }
                self._record_call(name, args, response, True)
                return response

            except Exception as e:
                last_error = e
                logger.warning(
                    f"[ToolRegistry] 工具 '{name}' 第 {attempt + 1} 次调用失败: {type(e).__name__}: {e}"
                )

                if not definition.retry_on_failure or attempt >= definition.max_retries:
                    break

        # 4. 所有重试均失败
        elapsed_ms = (time.time() - start_time) * 1000
        response = {
            "success": False,
            "data": None,
            "error": f"{type(last_error).__name__}: {last_error}",
            "_meta": {
                "tool_name": name,
                "elapsed_ms": round(elapsed_ms, 2),
                "retry_count": definition.max_retries + 1,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        }
        self._record_call(name, args, response, False)
        return response

    async def ainvoke(
        self,
        name: str,
        args: Optional[dict] = None,
        timeout_override: Optional[float] = None
    ) -> dict:
        """异步调用（预留接口，当前委托同步执行）"""
        import asyncio
        return await asyncio.to_thread(self.invoke, name, args, timeout_override)

    # ==================== 内部方法 ====================

    def _validate_args(self, definition: ToolDefinition, args: dict) -> dict:
        """简单的参数校验"""
        params_schema = definition.parameters
        required = params_schema.get("required", [])

        for param_name in required:
            if param_name not in args:
                raise ValueError(f"缺少必需参数: '{param_name}'")

        return args

    def _execute_with_timeout(self, handler: Callable, args: dict, timeout: float) -> Any:
        """执行工具调用（当前为同步模式）"""
        result = handler(**args)
        return result

    def _error_response(self, name: str, error: str, start_time: float, args: dict = None) -> dict:
        """生成统一错误响应"""
        elapsed_ms = (time.time() - start_time) * 1000
        response = {
            "success": False,
            "data": None,
            "error": error,
            "_meta": {
                "tool_name": name,
                "elapsed_ms": round(elapsed_ms, 2),
                "retry_count": 0,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        }
        self._record_call(name, args or {}, response, False)
        return response

    def _record_call(self, name: str, args: dict, result: dict, success: bool):
        """记录工具调用到历史"""
        record = ToolCallRecord(
            tool_name=name,
            args={k: str(v)[:100] for k, v in args.items()},
            result_summary=str(result.get("data", result.get("error", "")))[:200],
            success=success,
            error=result.get("error"),
            elapsed_ms=result.get("_meta", {}).get("elapsed_ms", 0),
            retry_count=result.get("_meta", {}).get("retry_count", 0),
        )
        self._call_history.append(record)

        # 限制历史长度
        if len(self._call_history) > self._max_history:
            self._call_history = self._call_history[-self._max_history:]

    # ==================== 统计与查询 ====================

    def get_statistics(self) -> dict:
        """获取工具调用统计"""
        if not self._call_history:
            return {"total_calls": 0, "tools": {}}

        total = len(self._call_history)
        success_count = sum(1 for r in self._call_history if r.success)

        per_tool = {}
        for record in self._call_history:
            if record.tool_name not in per_tool:
                per_tool[record.tool_name] = {"calls": 0, "success": 0, "total_elapsed_ms": 0}
            per_tool[record.tool_name]["calls"] += 1
            if record.success:
                per_tool[record.tool_name]["success"] += 1
            per_tool[record.tool_name]["total_elapsed_ms"] += record.elapsed_ms

        for name in per_tool:
            stats = per_tool[name]
            stats["success_rate"] = round(stats["success"] / stats["calls"], 3) if stats["calls"] > 0 else 0
            stats["avg_elapsed_ms"] = round(stats["total_elapsed_ms"] / stats["calls"], 2)
            del stats["total_elapsed_ms"]

        return {
            "total_calls": total,
            "total_success": success_count,
            "overall_success_rate": round(success_count / total, 3) if total > 0 else 0,
            "tools": per_tool,
        }

    def get_recent_calls(self, limit: int = 20) -> List[dict]:
        """获取最近的工具调用记录"""
        return [r.model_dump() for r in self._call_history[-limit:]]

    def list_all(self) -> List[str]:
        """列出所有已注册的工具名"""
        return list(self._tools.keys())

    @property
    def tool_count(self) -> int:
        return len(self._tools)


# ==================== 全局单例 ====================
# 项目启动时通过 register_all_tools() 填充
tool_registry = ToolRegistry()
