"""
Multi-Agent 编排器

初始化并运行四智能体闭环，提供同步/流式两种接口。
供 API v2 端点调用。

使用方式：
    orchestrator = MultiAgentOrchestrator()
    response = orchestrator.chat("小户型适合哪种扫地机器人？", session_id="u1")
    print(response.answer, response.plan, response.reflection)
"""
import time
import uuid
from typing import Dict, Any, Optional, AsyncIterator

from utils.logger_handler import logger
from utils.config_handler import multi_agent_conf
from utils.observability import observe_if_available, update_trace_safe

from agent.multi_agent.graph import build_multi_agent_graph
from agent.multi_agent.planner import PlannerNode
from agent.multi_agent.retriever import RetrieverNode
from agent.multi_agent.reflection import ReflectionNode
from agent.multi_agent.summarizer import SummarizerNode
from agent.multi_agent.models import MultiAgentResponse


class MultiAgentOrchestrator:
    """
    四智能体编排器

    职责：
    1. 初始化所有Agent节点 + LangGraph状态图
    2. 提供 chat() 同步接口（非流式）
    3. 提供 chat_stream() 流式接口（SSE）
    4. 构建 MultiAgentResponse（含完整决策轨迹）
    """

    def __init__(
        self,
        rag_service=None,
        session_manager=None,
        retrieval_cache=None,
        flywheel=None,
        knowledge_graph=None,
    ):
        """
        Args:
            rag_service: RagSummarizeService 实例（用于检索）
            session_manager: Redis会话管理器（用于多轮对话）
            retrieval_cache: Redis检索缓存
            flywheel: 飞轮服务
            knowledge_graph: 知识图谱实例
        """
        self.rag_service = rag_service
        self.session_manager = session_manager
        self.retrieval_cache = retrieval_cache
        self.flywheel = flywheel
        self.knowledge_graph = knowledge_graph

        # 配置
        self.max_retries = multi_agent_conf.get("reflection", {}).get("max_retries", 2)

        # 初始化 SelfCorrector（复用现有实现）
        self_corrector = None
        if rag_service:
            try:
                from rag.self_corrector import SelfCorrector
                self_corrector = SelfCorrector(
                    max_retries=self.max_retries,
                    confidence_threshold=multi_agent_conf.get("reflection", {}).get("confidence_threshold", 0.6),
                )
            except Exception as e:
                logger.warning(f"[Orchestrator] SelfCorrector 初始化失败: {e}")

        # 初始化向量存储引用
        vector_store = rag_service.vector_store if rag_service else None

        # 初始化四个Agent节点
        self.planner = PlannerNode()
        self.retriever = RetrieverNode(
            vector_store=vector_store,
            retrieval_cache=retrieval_cache,
            knowledge_graph=knowledge_graph,
            flywheel=flywheel,
        )
        self.reflector = ReflectionNode(self_corrector=self_corrector, flywheel=flywheel)
        self.summarizer = SummarizerNode()

        # 构建图
        self.graph = build_multi_agent_graph(
            planner=self.planner,
            retriever=self.retriever,
            reflector=self.reflector,
            summarizer=self.summarizer,
            max_retries=self.max_retries,
        )

        # 注册工具
        try:
            from agent.tools import register_all_tools
            register_all_tools()
            logger.info("[Orchestrator] 工具注册完成")
        except Exception as e:
            logger.warning(f"[Orchestrator] 工具注册失败: {e}")

        logger.info(
            f"[Orchestrator] Multi-Agent编排器初始化完成: "
            f"max_retries={self.max_retries}, "
            f"rag={'yes' if rag_service else 'no'}, "
            f"kg={'yes' if knowledge_graph and knowledge_graph.is_built else 'no'}"
        )

    # ==================== 同步接口 ====================

    def chat(
        self,
        query: str,
        session_id: str = "default",
        user_id: str = "default_user",
        messages: Optional[list] = None,
        enable_correction: bool = True,
    ) -> MultiAgentResponse:
        """
        同步执行四智能体问答

        Args:
            query: 用户查询
            session_id: 会话ID（用于checkpoint持久化）
            user_id: 用户ID
            messages: 多轮对话历史 [{role, content}]
            enable_correction: 是否启用反思纠错

        Returns:
            MultiAgentResponse（含完整决策轨迹）
        """
        request_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        logger.info(
            f"[Orchestrator-{request_id}] 收到请求: "
            f"query='{query[:80]}...', session={session_id}"
        )

        # 构建初始状态
        initial_state = {
            "query": query,
            "original_query": query,
            "messages": messages or [],
            "session_id": session_id,
            "user_id": user_id,
            "retry_count": 0,
        }

        # 执行图
        config = {"configurable": {"thread_id": session_id}}
        result = self.graph.invoke(initial_state, config)

        # 构建响应
        response = self._build_response(request_id, session_id, result, start_time)

        return response

    # ==================== 流式接口 ====================

    async def chat_stream(
        self,
        query: str,
        session_id: str = "default",
        user_id: str = "default_user",
        messages: Optional[list] = None,
    ):
        """
        流式执行四智能体问答（SSE）

        通过 asyncio.Queue 桥接同步 graph.stream 和异步 SSE

        Yields:
            SSE事件字符串
        """
        import asyncio
        import json as _json

        request_id = str(uuid.uuid4())[:8]

        yield f"data: {_json.dumps({'event': 'start', 'request_id': request_id, 'session_id': session_id}, ensure_ascii=False)}\n\n"

        try:
            initial_state = {
                "query": query,
                "original_query": query,
                "messages": messages or [],
                "session_id": session_id,
                "user_id": user_id,
                "retry_count": 0,
            }

            config = {"configurable": {"thread_id": session_id}}

            # 在 executor 中运行同步 graph.stream
            loop = asyncio.get_event_loop()

            def _run_graph():
                events = []
                for event in self.graph.stream(initial_state, config, stream_mode="updates"):
                    events.append(event)
                return events

            events = await loop.run_in_executor(None, _run_graph)

            # 解析事件流
            for event in events:
                for node_name, node_output in event.items():
                    if node_name == "planner":
                        plan = node_output.get("plan", {})
                        yield f"data: {_json.dumps({'event': 'stage', 'stage': 'planner', 'data': {'intent': plan.get('intent'), 'sub_tasks': len(plan.get('sub_tasks', []))}}, ensure_ascii=False)}\n\n"

                    elif node_name == "retriever":
                        rounds = node_output.get("retrieval_rounds", [])
                        latest = rounds[-1] if rounds else {}
                        yield f"data: {_json.dumps({'event': 'stage', 'stage': 'retriever', 'data': {'docs_count': latest.get('docs_count', 0), 'kg_used': latest.get('kg_used', False)}}, ensure_ascii=False)}\n\n"

                    elif node_name == "reflector":
                        reflection = node_output.get("reflection", {})
                        yield f"data: {_json.dumps({'event': 'stage', 'stage': 'reflector', 'data': {'verdict': reflection.get('verdict'), 'confidence': reflection.get('confidence')}}, ensure_ascii=False)}\n\n"

                    elif node_name == "summarizer":
                        answer = node_output.get("answer", "")
                        structured = node_output.get("structured_result", {})
                        yield f"data: {_json.dumps({'event': 'stage', 'stage': 'summarizer', 'data': {'answer_preview': answer[:200]}}, ensure_ascii=False)}\n\n"

                        # Token级别流式输出（按字符拆分模拟）
                        for i in range(0, len(answer), 5):
                            chunk = answer[i:i + 5]
                            yield f"data: {_json.dumps({'event': 'token', 'content': chunk}, ensure_ascii=False)}\n\n"

            yield f"data: {_json.dumps({'event': 'end', 'request_id': request_id}, ensure_ascii=False)}\n\n"

        except Exception as e:
            logger.error(f"[Orchestrator-{request_id}] 流式失败: {e}", exc_info=True)
            yield f"data: {_json.dumps({'event': 'error', 'request_id': request_id, 'error': str(e)}, ensure_ascii=False)}\n\n"

    # ==================== 内部方法 ====================

    def _build_response(
        self,
        request_id: str,
        session_id: str,
        result: dict,
        start_time: float
    ) -> MultiAgentResponse:
        """从图执行结果构建 MultiAgentResponse"""
        response_time = round((time.time() - start_time) * 1000, 2)

        # 提取引用
        citations = []
        structured = result.get("structured_result")
        if structured:
            citations = structured.get("citations", [])

        # 提取使用的模型
        models = result.get("models_used", [])
        model_used = models[-1] if models else "unknown"

        return MultiAgentResponse(
            request_id=request_id,
            session_id=session_id,
            answer=result.get("answer", ""),
            citations=citations,
            plan=result.get("plan"),
            reflection=result.get("reflection"),
            retrieval_rounds=result.get("retrieval_rounds", []),
            tool_calls=result.get("tool_calls", []),
            retry_count=result.get("retry_count", 0),
            confidence=result.get("reflection", {}).get("confidence", 0.0) if result.get("reflection") else 0.0,
            kg_used=result.get("kg_used", False),
            model_used=model_used,
            llm_usage=result.get("llm_usage", []),
            response_time_ms=response_time,
        )
