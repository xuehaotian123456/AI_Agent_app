"""
FastAPI 封装的智能 RAG Agent 服务
提供 RESTful API 接口，支持多用户会话隔离、自我纠错、结构化输出
"""
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional, List, Any, Dict

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from rag.self_corrector import CorrectiveRAGService




from rag.rag_service import RagSummarizeService
from rag.query_rewriter import query_rewriter  # 新增导入
from utils.logger_handler import logger
from utils.observability import (
    initialize_langfuse,
    shutdown_langfuse,
    observe_if_available,
    update_trace_safe,
    update_observation_safe,
)



# ==================== 全局变量 ====================
rag_service: Optional[CorrectiveRAGService] = None
_LANGFUSE_ENABLED = initialize_langfuse()


# ==================== 生命周期管理 ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动和关闭时的资源管理"""
    global rag_service
    logger.info("[API] 正在初始化 RAG 服务...")
    start_time = time.time()

    try:
        # 步骤 1：创建基础 RAG 服务
        base_rag_service = RagSummarizeService(
            use_langgraph=True,
            enable_structured_output=True
        )

        # 步骤 2：用 CorrectiveRAGService 包装，启用自我纠错
        rag_service = CorrectiveRAGService(
            rag_service=base_rag_service,  # 传入基础服务实例
            enable_correction=True,  # 启用自我纠错
            max_retries=2,
            confidence_threshold=0.6
        )

        elapsed = time.time() - start_time
        logger.info(f"[API] Corrective RAG Service 初始化成功，耗时: {elapsed:.2f}s")
    except Exception as e:
        logger.error(f"[API] RAG 服务初始化失败: {e}", exc_info=True)
        raise

    yield

    logger.info("[API] 正在关闭 RAG 服务...")
    shutdown_langfuse()
    rag_service = None
    logger.info("[API] 服务已关闭")


# ==================== FastAPI 应用 ====================
app = FastAPI(
    title="智扫通 RAG Agent API",
    description="带有记忆、自我纠错和混合检索的智能问答系统",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 中间件（允许前端跨域访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应指定具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== 数据模型 ====================
class QueryRequest(BaseModel):
    """聊天请求模型"""
    query: str = Field(..., min_length=1, max_length=2000, description="用户查询内容")
    user_id: str = Field(default="default_user", description="用户ID，用于会话隔离")
    session_id: Optional[str] = Field(default=None, description="会话ID，不传则自动生成")
    use_history: bool = Field(default=True, description="是否使用历史对话上下文")
    enable_correction: bool = Field(default=True, description="是否启用自我纠错机制")


class Citation(BaseModel):
    """引用来源模型"""
    source: str = Field(..., description="资料来源")
    content: str = Field(..., description="引用内容片段")
    relevance_score: Optional[float] = Field(default=None, description="相关性分数")


class QueryResponse(BaseModel):
    """聊天响应模型"""
    request_id: str = Field(..., description="请求唯一ID")
    answer: str = Field(..., description="生成的答案")
    citations: Optional[List[Citation]] = Field(default=None, description="引用来源列表")
    corrected: bool = Field(default=False, description="是否触发了自我纠错")
    retry_count: int = Field(default=0, description="纠错重试次数")
    final_query: Optional[str] = Field(default=None, description="最终用于检索的查询")
    quality_checks: Optional[List[Dict[str, Any]]] = Field(default=None, description="质量检查详情")
    retrieval_metadata: Optional[dict] = Field(default=None, description="检索元数据")
    llm_usage: Optional[dict] = Field(default=None, description="LLM token 用量信息")
    response_time_ms: float = Field(..., description="响应时间（毫秒）")
    model_used: str = Field(default="unknown", description="生成答案实际使用的模型（primary/backup）")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    rag_service_ready: bool
    vector_store_ready: bool
    hybrid_search_enabled: bool
    timestamp: str


class ErrorResponse(BaseModel):
    """错误响应模型"""
    request_id: str
    error_code: str
    error_message: str
    details: Optional[str] = None


# ==================== API 路由 ====================

@app.post("/api/v1/chat", response_model=QueryResponse, tags=["聊天"])
@observe_if_available(name="api.chat")
async def chat(request: QueryRequest):
    """
    智能问答接口

    - 支持多用户会话隔离
    - 自动启用混合检索（向量 + BM25 + 重排序）
    - 可选的自我纠错机制
    - 返回结构化答案和引用来源
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    logger.info(f"[API-{request_id}] 收到请求: user_id={request.user_id}, query={request.query[:50]}...")
    update_trace_safe(
        user_id=request.user_id,
        metadata={
            "request_id": request_id,
            "session_id": request.session_id,
            "use_history": request.use_history,
            "enable_correction": request.enable_correction,
            "langfuse_enabled": _LANGFUSE_ENABLED,
            "api_version": "v1.1-observability",
        },
    )

    try:
        # 验证服务状态
        if rag_service is None:
            raise HTTPException(status_code=503, detail="RAG 服务未就绪")

        session_id = request.session_id or f"{request.user_id}_session"

        # 【新增】如果启用历史，先获取历史消息用于查询改写
        enhanced_query = request.query
        if request.use_history:
            try:
                # 从基础 RAG 服务的 LangGraph 中获取历史消息
                config = {"configurable": {"thread_id": session_id}}
                state_snapshot = rag_service.rag_service.graph.get_state(config)

                if state_snapshot and state_snapshot.values:
                    messages = state_snapshot.values.get("messages", [])

                    # 如果有历史对话，进行查询改写
                    if len(messages) >= 2:  # 至少有一轮完整对话
                        enhanced_query = query_rewriter.rewrite(
                            current_query=request.query,
                            messages=messages,
                            request_id=request_id
                        )
            except Exception as e:
                logger.warning(f"[API-{request_id}] 获取历史失败，使用原始查询: {e}")
                enhanced_query = request.query

        # 调用 RAG 服务（使用增强后的查询）
        result = rag_service.rag_summarize_with_correction(
            query=enhanced_query,
            thread_id=session_id,
            enable_correction=request.enable_correction
        )

        # 提取答案（CorrectiveRAGService 的返回结构）
        answer = result.get("answer", "")
        corrected = result.get("corrected", False)  # 是否触发了自我纠错
        retry_count = result.get("retry_count", 0)

        retrieval_metadata = result.get("retrieval_metadata") or {}
        if not retrieval_metadata:
            try:
                docs = rag_service.rag_service.retriever_docs(result.get("final_query", enhanced_query))
                retrieval_metadata = {
                    "method": "hybrid" if (
                        getattr(rag_service.rag_service.vector_store, "hybrid_enabled", False)
                        and hasattr(rag_service.rag_service.retriever, "hybrid_search")
                    ) else "vector",
                    "docs_count": len(docs),
                    "elapsed_time": None
                }
            except Exception as e:
                logger.warning(f"[API-{request_id}] 检索元数据兜底失败: {e}")
                retrieval_metadata = {"method": "unknown", "docs_count": 0, "elapsed_time": None}

        # 提取结构化结果（如果存在）
        structured_result = result.get("structured_result")
        citations = []

        if structured_result and structured_result.get("citations"):
            citations = [
                Citation(
                    source=cit.get("source", "未知来源"),
                    content=cit.get("content", ""),
                    relevance_score=cit.get("relevance_score")
                )
                for cit in structured_result["citations"]
            ]

        # 计算响应时间
        response_time_ms = (time.time() - start_time) * 1000

        # 构建响应
        response = QueryResponse(
            request_id=request_id,
            answer=answer,
            citations=citations,
            corrected=corrected,  # 使用实际的纠错状态
            retry_count=retry_count,
            final_query=result.get("final_query", enhanced_query),
            quality_checks=result.get("quality_checks", []),
            retrieval_metadata=retrieval_metadata,
            llm_usage=result.get("llm_usage", {}),
            response_time_ms=round(response_time_ms, 2),
            model_used=result.get("model_used", "unknown")
        )

        logger.info(
            f"[API-{request_id}] 响应成功: "
            f"time={response_time_ms:.0f}ms, "
            f"corrected={corrected}, "
            f"retries={retry_count}, "
            f"docs={retrieval_metadata.get('docs_count', 0)}"
        )
        update_observation_safe(
            input={
                "query": request.query,
                "enhanced_query": enhanced_query,
                "session_id": session_id,
            },
            output={
                "answer_preview": answer[:200],
                "citations_count": len(citations),
                "corrected": corrected,
                "retry_count": retry_count,
            },
            metadata={
                "request_id": request_id,
                "response_time_ms": round(response_time_ms, 2),
                "retrieval_metadata": retrieval_metadata,
                "llm_usage": result.get("llm_usage", {}),
                "model_used": result.get("model_used", "unknown"),
            },
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        response_time_ms = (time.time() - start_time) * 1000
        logger.error(f"[API-{request_id}] 处理失败: {e}", exc_info=True)
        update_observation_safe(
            input={"query": request.query, "session_id": request.session_id},
            output={"error": str(e)},
            metadata={"request_id": request_id, "response_time_ms": round(response_time_ms, 2)},
        )

        raise HTTPException(
            status_code=500,
            detail=f"内部服务器错误: {str(e)}"
        )


@app.post("/api/v1/chat/stream", tags=["聊天"])
async def chat_stream(request: QueryRequest):
    """
    流式问答接口（SSE）

    注意：当前版本暂不支持流式输出，后续可扩展
    """
    raise HTTPException(
        status_code=501,
        detail="流式接口暂未实现，请使用 /api/v1/chat"
    )


@app.get("/health", response_model=HealthResponse, tags=["监控"])
async def health_check():
    """
    健康检查接口

    返回服务状态、向量库状态等信息
    """
    try:
        rag_ready = rag_service is not None
        vector_ready = False
        hybrid_enabled = False

        if rag_service:
            try:
                # 测试向量库连接（通过基础服务访问）
                test_docs = rag_service.rag_service.retriever_docs("测试")
                vector_ready = len(test_docs) >= 0
                vector_store = rag_service.rag_service.vector_store
                hybrid_enabled = (
                    getattr(vector_store, 'hybrid_enabled', False) and
                    getattr(vector_store, 'hybrid_retriever', None) is not None
                )
            except Exception as e:
                logger.warning(f"[健康检查] 向量库测试失败: {e}")

        from datetime import datetime

        return HealthResponse(
            status="healthy" if rag_ready and vector_ready else "degraded",
            rag_service_ready=rag_ready,
            vector_store_ready=vector_ready,
            hybrid_search_enabled=hybrid_enabled,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"[健康检查] 异常: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/sessions/{user_id}", tags=["会话管理"])
async def get_user_sessions(user_id: str):
    """
    获取用户的会话历史摘要

    注意：由于 MemorySaver 的限制，这里仅返回占位符
    生产环境建议使用 Redis 或数据库存储会话历史
    """
    return {
        "user_id": user_id,
        "sessions": [],
        "message": "会话历史功能待实现（需持久化存储支持）"
    }


@app.delete("/api/v1/sessions/{session_id}", tags=["会话管理"])
async def clear_session(session_id: str):
    """
    清除指定会话的历史记录

    注意：当前版本不支持动态清除，需要重启服务
    """
    logger.warning(f"[API] 尝试清除会话 {session_id}，但当前不支持此操作")
    return {
        "session_id": session_id,
        "cleared": False,
        "message": "会话清除功能待实现"
    }


# ==================== 全局异常处理 ====================
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理器"""
    request_id = str(uuid.uuid4())[:8]
    logger.error(f"[API-{request_id}] 未捕获的异常: {exc}", exc_info=True)

    return ErrorResponse(
        request_id=request_id,
        error_code="INTERNAL_ERROR",
        error_message="服务器内部错误",
        details=str(exc) if app.debug else None
    )


# ==================== 启动入口 ====================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8010,
        reload=True,
        log_level="info"
    )
