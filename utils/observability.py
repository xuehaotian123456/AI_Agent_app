"""
LangFuse 可观测性工具。
默认通过环境变量启用：
- LANGFUSE_PUBLIC_KEY
- LANGFUSE_SECRET_KEY
- LANGFUSE_HOST（可选，默认云端）
"""

import os
from typing import Any, Dict, Optional, Callable

from utils.logger_handler import logger

try:
    from langfuse import Langfuse
    from langfuse.decorators import observe as _observe
    from langfuse.decorators import langfuse_context
    _LANGFUSE_IMPORT_OK = True
except Exception:
    Langfuse = None  # type: ignore
    _observe = None  # type: ignore
    langfuse_context = None  # type: ignore
    _LANGFUSE_IMPORT_OK = False


_langfuse_client: Optional["Langfuse"] = None
_langfuse_enabled = False


def initialize_langfuse() -> bool:
    """初始化 LangFuse 客户端。失败时降级为无观测，不影响主流程。"""
    global _langfuse_client, _langfuse_enabled

    if not _LANGFUSE_IMPORT_OK:
        logger.warning("[Observability] langfuse 未安装，跳过可观测性上报")
        _langfuse_enabled = False
        return False

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
    host = os.getenv("LANGFUSE_BASE_URL", os.getenv("LANGFUSE_HOST", "")).strip()

    if not public_key or not secret_key:
        logger.info("[Observability] 未配置 LANGFUSE_PUBLIC_KEY/SECRET_KEY，可观测性已关闭")
        _langfuse_enabled = False
        return False

    try:
        kwargs: Dict[str, Any] = {
            "public_key": public_key,
            "secret_key": secret_key,
        }
        if host:
            kwargs["host"] = host
        _langfuse_client = Langfuse(**kwargs)
        _langfuse_enabled = True
        logger.info("[Observability] LangFuse 已启用")
        return True
    except Exception as e:
        logger.warning(f"[Observability] LangFuse 初始化失败，已降级关闭: {e}")
        _langfuse_client = None
        _langfuse_enabled = False
        return False


def shutdown_langfuse() -> None:
    """优雅刷新并关闭 LangFuse 客户端。"""
    global _langfuse_client
    if _langfuse_client is None:
        return
    try:
        _langfuse_client.flush()
    except Exception as e:
        logger.warning(f"[Observability] LangFuse flush 失败: {e}")
    finally:
        _langfuse_client = None


def is_langfuse_enabled() -> bool:
    return _langfuse_enabled and _LANGFUSE_IMPORT_OK


def observe_if_available(name: Optional[str] = None) -> Callable:
    """LangFuse 可用时返回 observe 装饰器，否则返回 no-op 装饰器。"""
    def _noop_decorator(func):
        return func

    if is_langfuse_enabled() and _observe is not None:
        if name:
            return _observe(name=name)
        return _observe()
    return _noop_decorator


def update_trace_safe(*, user_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
    """安全更新当前 trace，不抛出异常。"""
    if not is_langfuse_enabled() or langfuse_context is None:
        return
    try:
        kwargs: Dict[str, Any] = {}
        if user_id:
            kwargs["user_id"] = user_id
        if metadata is not None:
            kwargs["metadata"] = metadata
        if kwargs:
            langfuse_context.update_current_trace(**kwargs)
    except Exception as e:
        logger.warning(f"[Observability] 更新 trace 失败: {e}")


def update_observation_safe(*, input: Any = None, output: Any = None, metadata: Optional[Dict[str, Any]] = None) -> None:
    """安全更新当前 observation，不抛出异常。"""
    if not is_langfuse_enabled() or langfuse_context is None:
        return
    try:
        kwargs: Dict[str, Any] = {}
        if input is not None:
            kwargs["input"] = input
        if output is not None:
            kwargs["output"] = output
        if metadata is not None:
            kwargs["metadata"] = metadata
        if kwargs:
            langfuse_context.update_current_observation(**kwargs)
    except Exception as e:
        logger.warning(f"[Observability] 更新 observation 失败: {e}")
