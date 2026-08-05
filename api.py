"""
FastAPI 封装的智能 RAG Agent 服务
提供 RESTful API 接口，支持多用户会话隔离、自我纠错、结构化输出

v1: /api/v1/chat — CorrectiveRAG 服务（保持兼容）
v2: /api/v2/chat — Multi-Agent 四智能体闭环（新增）
"""
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional, List, Any, Dict
import json
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from rag.self_corrector import CorrectiveRAGService
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

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
from config.redis_config import (
    redis_client,
    async_redis_client,
    setup_exact_cache,
    setup_semantic_cache,
)
from services.RedisSessionManager import RedisSessionManager
from services.RetrievalCache import RetrievalCache
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
import os



# ==================== 全局变量 ====================
rag_service: Optional[CorrectiveRAGService] = None
multi_agent_orchestrator = None  # v2 Multi-Agent编排器
session_manager: Optional[RedisSessionManager] = None
retrieval_cache: Optional[RetrievalCache] = None
_LANGFUSE_ENABLED = initialize_langfuse()


# ==================== 生命周期管理 ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动和关闭时的资源管理"""
    global rag_service, session_manager, retrieval_cache, multi_agent_orchestrator
    logger.info("[API] 正在初始化 RAG 服务...")
    start_time = time.time()

    try:
        # 步骤 1：初始化 Redis 缓存（选择一种策略）
        cache_enabled = setup_exact_cache()  # 精确匹配缓存
        # cache_enabled = setup_semantic_cache()  # 或者使用语义缓存

        if cache_enabled:
            logger.info("[API] Redis LLM 缓存已启用")
        else:
            logger.warning("[API] Redis LLM 缓存启用失败，将继续运行但不使用缓存")

        # 步骤 2：初始化会话管理器
        session_manager = RedisSessionManager(
            redis_client=async_redis_client,
            ttl_hours=2,
            max_history=20
        )
        logger.info("[API] Redis 会话管理器已初始化")

        # 步骤 3：初始化检索缓存
        retrieval_cache = RetrievalCache(
            redis_client=async_redis_client,
            ttl_seconds=3600  # 1 小时
        )
        logger.info("[API] Redis 检索缓存已初始化")

        # 步骤 4：初始化限流器
        try:
            await FastAPILimiter.init(async_redis_client)
            logger.info("[API] Redis 限流器已启用")
        except Exception as e:
            logger.warning(f"[API] 限流器初始化失败: {e}")

        # 步骤 5：创建基础 RAG 服务
        base_rag_service = RagSummarizeService(
            use_langgraph=True,
            enable_structured_output=True
        )

        # 步骤 6：用 CorrectiveRAGService 包装，启用自我纠错
        rag_service = CorrectiveRAGService(
            rag_service=base_rag_service,  # 传入基础服务实例
            enable_correction=True,  # 启用自我纠错
            max_retries=2,
            confidence_threshold=0.6
        )

        elapsed = time.time() - start_time
        logger.info(f"[API] Corrective RAG Service 初始化成功，耗时: {elapsed:.2f}s")

        # 步骤 7：初始化 Multi-Agent 编排器（v2）
        try:
            from agent.multi_agent_orchestrator import MultiAgentOrchestrator
            from services.FlywheelService import get_flywheel
            from rag.knowledge_graph import KnowledgeGraph

            flywheel = get_flywheel(redis_client=redis_client)

            # 初始化KG（尝试加载已有索引）
            kg = KnowledgeGraph()
            if base_rag_service.vector_store and base_rag_service.vector_store.all_text_chunks:
                kg.load_or_build(base_rag_service.vector_store.all_text_chunks)
                logger.info(f"[API] KG索引已初始化: {kg.entity_count} entities")
            else:
                logger.info("[API] KG索引跳过（无可用文本块）")

            multi_agent_orchestrator = MultiAgentOrchestrator(
                rag_service=base_rag_service,
                session_manager=session_manager,
                retrieval_cache=retrieval_cache,
                flywheel=flywheel,
                knowledge_graph=kg,
            )
            logger.info("[API] Multi-Agent 编排器(v2)初始化成功")
        except Exception as e:
            logger.warning(f"[API] Multi-Agent v2初始化失败（v1仍可用）: {e}")
            multi_agent_orchestrator = None

        elapsed = time.time() - start_time
        logger.info(f"[API] 全部服务初始化完成，总耗时: {elapsed:.2f}s")
    except Exception as e:
        logger.error(f"[API] RAG 服务初始化失败: {e}", exc_info=True)
        raise

    yield

    logger.info("[API] 正在关闭 RAG 服务...")
    shutdown_langfuse()

    # 关闭 Redis 连接
    try:
        await async_redis_client.close()
        logger.info("[API] Redis 连接已关闭")
    except Exception as e:
        logger.warning(f"[API] 关闭 Redis 连接失败: {e}")

    rag_service = None
    session_manager = None
    retrieval_cache = None
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
    session_id: str = Field(..., description="会话ID")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    rag_service_ready: bool
    vector_store_ready: bool
    hybrid_search_enabled: bool
    redis_connected: bool
    timestamp: str


class ErrorResponse(BaseModel):
    """错误响应模型"""
    request_id: str
    error_code: str
    error_message: str
    details: Optional[str] = None


# ==================== V2 API: Multi-Agent 四智能体闭环 ====================

class MultiAgentChatRequest(BaseModel):
    """v2 Multi-Agent 聊天请求"""
    query: str = Field(..., min_length=1, max_length=2000, description="用户查询内容")
    user_id: str = Field(default="default_user", description="用户ID")
    session_id: Optional[str] = Field(default=None, description="会话ID")
    use_history: bool = Field(default=True, description="是否使用历史对话")
    enable_correction: bool = Field(default=True, description="是否启用反思纠错")


@app.post("/api/v2/chat", tags=["Multi-Agent v2"])
@observe_if_available(name="api.v2.chat")
async def multi_agent_chat(request: MultiAgentChatRequest):
    """
    Multi-Agent 智能问答（四智能体闭环）

    - Planner: 意图分析 + 任务拆解 + 工具选择
    - Retriever: 工具调用编排 + 三路多路召回
    - Reflector: 置信度自检 + 失败重检索决策
    - Summarizer: 证据链综合 + 结构化输出

    返回完整决策轨迹（Plan/Reflection/检索轮次/工具调用/飞轮指标）
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    if multi_agent_orchestrator is None:
        raise HTTPException(status_code=503, detail="Multi-Agent v2 服务未就绪（v1仍可用）")

    logger.info(f"[API-v2-{request_id}] 收到请求: user={request.user_id}, query={request.query[:50]}...")

    try:
        session_id = request.session_id or f"{request.user_id}_v2_{uuid.uuid4().hex[:8]}"

        # 获取历史消息
        messages = []
        if request.use_history and session_manager:
            try:
                history = await session_manager.get_history(session_id)
                messages = [{"role": m["role"], "content": m["content"]} for m in history]
                await session_manager.add_message(session_id, "user", request.query)
            except Exception as e:
                logger.warning(f"[API-v2-{request_id}] 历史获取失败: {e}")

        # 执行 Multi-Agent 流程
        response = multi_agent_orchestrator.chat(
            query=request.query,
            session_id=session_id,
            user_id=request.user_id,
            messages=messages,
            enable_correction=request.enable_correction,
        )
        response.request_id = request_id
        response.session_id = session_id

        # 保存助手回复
        if session_manager:
            try:
                await session_manager.add_message(session_id, "assistant", response.answer)
            except Exception:
                pass

        logger.info(
            f"[API-v2-{request_id}] 完成: "
            f"time={response.response_time_ms}ms, "
            f"intent={response.plan.get('intent', '?') if response.plan else '?'}, "
            f"retries={response.retry_count}, "
            f"confidence={response.confidence}"
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API-v2-{request_id}] 处理失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")


@app.post("/api/v2/chat/stream", tags=["Multi-Agent v2"])
async def multi_agent_chat_stream(request: MultiAgentChatRequest):
    """
    Multi-Agent 流式问答（SSE）

    返回每个Agent的执行阶段和最终答案的token流
    """
    if multi_agent_orchestrator is None:
        raise HTTPException(status_code=503, detail="Multi-Agent v2 服务未就绪")

    session_id = request.session_id or f"{request.user_id}_v2_stream_{uuid.uuid4().hex[:8]}"
    messages = []

    if request.use_history and session_manager:
        try:
            history = await session_manager.get_history(session_id)
            messages = [{"role": m["role"], "content": m["content"]} for m in history]
        except Exception:
            pass

    return StreamingResponse(
        multi_agent_orchestrator.chat_stream(
            query=request.query,
            session_id=session_id,
            user_id=request.user_id,
            messages=messages,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.get("/api/v2/flywheel/metrics", tags=["Multi-Agent v2"])
async def get_flywheel_metrics(days: int = 7):
    """
    获取飞轮聚合指标

    - 失败率、平均置信度、重试分布
    - 高频失败查询族
    - 当前策略自适应状态
    """
    try:
        from services.FlywheelService import get_flywheel
        fw = get_flywheel()
        return fw.get_metrics(days=days)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/flywheel/failures", tags=["Multi-Agent v2"])
async def get_flywheel_failures(limit: int = 50):
    """
    获取最近的飞轮失败事件
    """
    try:
        from services.FlywheelService import get_flywheel
        fw = get_flywheel()
        failures = fw.get_recent_failures(limit=limit)
        return {"count": len(failures), "failures": failures}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/tools/stats", tags=["Multi-Agent v2"])
async def get_tool_statistics():
    """
    获取工具调用统计

    - 各工具调用次数、成功率、平均耗时
    - 最近调用记录
    """
    try:
        from agent.tools.tool_registry import tool_registry
        return {
            "statistics": tool_registry.get_statistics(),
            "recent_calls": tool_registry.get_recent_calls(limit=10),
            "registered_tools": tool_registry.list_all(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/kg/entities", tags=["Multi-Agent v2"])
async def search_kg_entities(keyword: str = "", limit: int = 20):
    """
    搜索知识图谱实体（调试用）
    """
    try:
        from rag.knowledge_graph import get_knowledge_graph
        kg = get_knowledge_graph()
        if not kg.is_built:
            return {"status": "not_built", "entities": []}
        entities = kg.search_entities(keyword, limit=limit) if keyword else []
        return {"status": "ok", "entity_count": kg.entity_count, "entities": entities}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== V1: 原有端点（保持不变） ====================

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
            "api_version": "v1.2-redis",
        },
    )

    try:
        # 验证服务状态
        if rag_service is None:
            raise HTTPException(status_code=503, detail="RAG 服务未就绪")

        # 生成或使用提供的 session_id
        session_id = request.session_id or f"{request.user_id}_session_{uuid.uuid4().hex[:8]}"

        # 【新增】如果启用历史，先获取历史消息用于查询改写和上下文注入
        enhanced_query = request.query
        chat_history_context = ""

        if request.use_history and session_manager:
            try:
                # 从 Redis 获取历史消息
                history = await session_manager.get_history(session_id)

                # 如果有历史对话，进行查询改写
                if len(history) >= 2:  # 至少有一轮完整对话
                    messages_for_rewrite = []
                    for msg in history:
                        messages_for_rewrite.append({
                            "role": msg["role"],
                            "content": msg["content"]
                        })

                    enhanced_query = query_rewriter.rewrite(
                        current_query=request.query,
                        messages=messages_for_rewrite,
                        request_id=request_id
                    )

                # 获取格式化的对话上下文
                chat_history_context = await session_manager.get_context(
                    session_id=session_id,
                    max_tokens=2048
                )

                # 保存用户消息
                await session_manager.add_message(session_id, "user", request.query)

            except Exception as e:
                logger.warning(f"[API-{request_id}] 获取历史失败，使用原始查询: {e}")
                enhanced_query = request.query
                chat_history_context = ""
        else:
            # 不使用历史，直接保存当前消息
            if session_manager:
                await session_manager.add_message(session_id, "user", request.query)

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

        # 保存助手回复到 Redis
        if session_manager:
            await session_manager.add_message(session_id, "assistant", answer)

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
            model_used=result.get("model_used", "unknown"),
            session_id=session_id
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
    流式问答接口（SSE - Server-Sent Events）

    支持真正的 token 级别流式输出，适合需要实时显示回答的场景
    """
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG 服务未就绪")

    request_id = str(uuid.uuid4())[:8]
    session_id = request.session_id or f"{request.user_id}_session_{uuid.uuid4().hex[:8]}"

    async def generate_stream():
        """生成 SSE 格式的流式数据"""
        try:
            # 获取历史上下文
            history_context = ""
            if request.use_history and session_manager:
                history_context = await session_manager.get_history_context(
                    session_id, max_turns=3
                )

            # 构建完整查询
            full_query = f"{history_context}\n{request.query}" if history_context else request.query

            # 发送开始事件
            start_data = json.dumps({
                "event": "start",
                "request_id": request_id,
                "session_id": session_id
            }, ensure_ascii=False)
            yield f"data: {start_data}\n\n"

            # 执行流式查询
            full_answer = ""
            start_time = time.time()

            for token in rag_service.rag_summarize_stream(full_query, thread_id=request.user_id):
                full_answer += token
                # 转义 JSON 特殊字符
                escaped_token = token.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
                token_data = json.dumps({
                    "event": "token",
                    "content": token
                }, ensure_ascii=False)
                yield f"data: {token_data}\n\n"

            # 计算响应时间
            response_time = round((time.time() - start_time) * 1000, 2)

            # 发送结束事件
            end_data = json.dumps({
                "event": "end",
                "request_id": request_id,
                "session_id": session_id,
                "answer": full_answer,
                "response_time_ms": response_time,
                "model_used": "primary"
            }, ensure_ascii=False)
            yield f"data: {end_data}\n\n"

            # 保存会话历史
            if session_manager and request.use_history:
                await session_manager.add_message(session_id, request.query, full_answer)

        except Exception as e:
            logger.error(f"[API-{request_id}] 流式请求失败: {e}", exc_info=True)
            error_data = json.dumps({
                "event": "error",
                "request_id": request_id,
                "error": str(e)
            }, ensure_ascii=False)
            yield f"data: {error_data}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # 禁用 Nginx 缓冲
        }
    )


@app.get("/health", response_model=HealthResponse, tags=["监控"])
async def health_check():
    """
    健康检查接口

    返回服务状态、向量库状态、Redis 连接状态等信息
    """
    try:
        rag_ready = rag_service is not None
        vector_ready = False
        hybrid_enabled = False
        redis_connected = False

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

        # 测试 Redis 连接
        try:
            redis_client.ping()
            redis_connected = True
        except Exception as e:
            logger.warning(f"[健康检查] Redis 连接失败: {e}")

        from datetime import datetime

        return HealthResponse(
            status="healthy" if (rag_ready and vector_ready and redis_connected) else "degraded",
            rag_service_ready=rag_ready,
            vector_store_ready=vector_ready,
            hybrid_search_enabled=hybrid_enabled,
            redis_connected=redis_connected,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"[健康检查] 异常: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/sessions/{user_id}", tags=["会话管理"])
async def get_user_sessions(user_id: str):
    """
    获取用户的会话历史摘要

    注意：当前实现返回占位符，生产环境需要维护 session_id 索引
    """
    return {
        "user_id": user_id,
        "sessions": [],
        "message": "会话历史功能待实现（需维护 session_id 索引）"
    }


@app.delete("/api/v1/sessions/{session_id}", tags=["会话管理"])
async def clear_session(session_id: str):
    """
    清除指定会话的历史记录
    """
    try:
        if session_manager:
            await session_manager.clear_history(session_id)
            return {
                "session_id": session_id,
                "cleared": True,
                "message": "会话已清除"
            }
        else:
            raise HTTPException(status_code=503, detail="会话管理器未就绪")
    except Exception as e:
        logger.error(f"[API] 清除会话失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/cache/clear", tags=["缓存管理"])
async def clear_cache(
    cache_type: str = Field(default="all", description="缓存类型: retrieval, llm, all")
):
    """
    清除缓存

    Args:
        cache_type: 缓存类型
            - retrieval: 清除检索缓存
            - llm: 清除 LLM 响应缓存
            - all: 清除所有缓存
    """
    try:
        cleared = []

        if cache_type in ["retrieval", "all"]:
            if retrieval_cache:
                await retrieval_cache.clear()
                cleared.append("retrieval")

        if cache_type in ["llm", "all"]:
            # LangChain 的 Redis 缓存没有直接的清除方法
            # 可以通过删除 Redis key 来实现
            pattern = "langchain:cache:*"
            keys = []
            for key in redis_client.scan_iter(match=pattern):
                keys.append(key)

            if keys:
                redis_client.delete(*keys)
                cleared.append("llm")

        return {
            "cleared": cleared,
            "message": f"已清除缓存: {', '.join(cleared) if cleared else '无'}"
        }
    except Exception as e:
        logger.error(f"[API] 清除缓存失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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

