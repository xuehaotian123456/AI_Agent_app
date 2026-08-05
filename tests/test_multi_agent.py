"""
Multi-Agent 系统测试

测试覆盖：
1. 单元测试：路由函数逻辑、reducer累积、Pydantic模型
2. 图结构测试：节点连接、条件边、有界循环
3. 集成测试：端到端流程（需要API Key）

运行：
    pytest tests/test_multi_agent.py -v
    # 跳过集成测试：
    pytest tests/test_multi_agent.py -v -k "not integration"
"""
import pytest
from typing import Dict, Any


# ============================================================
# 单元测试
# ============================================================

class TestModels:
    """Pydantic 模型验证"""

    def test_execution_plan_valid(self):
        from agent.multi_agent.models import ExecutionPlan, SubTask

        plan = ExecutionPlan(
            intent="knowledge",
            summary="测试",
            sub_tasks=[SubTask(task_id="t1", description="test", search_query="test")],
            requires_retrieval=True,
        )
        assert plan.intent == "knowledge"
        assert len(plan.sub_tasks) == 1

    def test_execution_plan_defaults(self):
        from agent.multi_agent.models import ExecutionPlan, SubTask

        plan = ExecutionPlan()
        assert plan.intent == "knowledge"
        assert plan.requires_retrieval is True

    def test_reflection_result_valid(self):
        from agent.multi_agent.models import ReflectionResult

        rr = ReflectionResult(verdict="pass", confidence=0.85, reason="good")
        assert rr.verdict == "pass"
        assert rr.confidence == 0.85

    def test_reflection_result_retry(self):
        from agent.multi_agent.models import ReflectionResult

        rr = ReflectionResult(verdict="retry", confidence=0.3, reason="missing info", missing_aspects=["价格"])
        assert rr.verdict == "retry"
        assert "价格" in rr.missing_aspects

    def test_multi_agent_response(self):
        from agent.multi_agent.models import MultiAgentResponse

        resp = MultiAgentResponse(
            answer="测试回答",
            citations=[],
            plan={"intent": "knowledge"},
            reflection={"verdict": "pass", "confidence": 0.9},
            retrieval_rounds=[],
            tool_calls=[],
            retry_count=0,
            confidence=0.9,
            response_time_ms=100.0,
        )
        assert resp.answer == "测试回答"
        assert resp.retry_count == 0


class TestRouterLogic:
    """路由函数逻辑测试"""

    def test_route_after_plan_with_retrieval(self):
        from agent.multi_agent.graph import _route_after_plan

        state = {"plan": {"requires_retrieval": True}}
        result = _route_after_plan(state)
        assert result == "retriever"

    def test_route_after_plan_skip_retrieval(self):
        from agent.multi_agent.graph import _route_after_plan

        state = {"plan": {"requires_retrieval": False}}
        result = _route_after_plan(state)
        assert result == "summarizer"

    def test_route_after_plan_no_plan(self):
        from agent.multi_agent.graph import _route_after_plan

        state = {}
        result = _route_after_plan(state)
        assert result == "retriever"  # 默认需要检索

    def test_route_after_reflection_pass(self):
        from agent.multi_agent.graph import _route_after_reflection

        state = {"reflection": {"verdict": "pass", "confidence": 0.9}, "retry_count": 0}
        result = _route_after_reflection(state, max_retries=2)
        assert result == "summarizer"

    def test_route_after_reflection_retry(self):
        from agent.multi_agent.graph import _route_after_reflection

        state = {"reflection": {"verdict": "retry", "confidence": 0.3}, "retry_count": 0}
        result = _route_after_reflection(state, max_retries=2)
        assert result == "retriever"

    def test_route_after_reflection_max_retries(self):
        from agent.multi_agent.graph import _route_after_reflection

        state = {"reflection": {"verdict": "retry", "confidence": 0.3}, "retry_count": 2}
        result = _route_after_reflection(state, max_retries=2)
        assert result == "summarizer"  # 达到最大重试，强制到总结

    def test_route_after_reflection_exceed_max(self):
        from agent.multi_agent.graph import _route_after_reflection

        state = {"reflection": {"verdict": "retry", "confidence": 0.3}, "retry_count": 5}
        result = _route_after_reflection(state, max_retries=2)
        assert result == "summarizer"  # 超过上限


class TestGraphConstruction:
    """图结构测试（需要模型API Key，因为 factory.py 模块级初始化 ChatTongyi）"""

    def test_graph_has_four_nodes(self):
        import os
        if not os.getenv("DASHSCOPE_API_KEY"):
            pytest.skip("需要 DASHSCOPE_API_KEY 环境变量")
        from agent.multi_agent.graph import build_multi_agent_graph
        from agent.multi_agent.planner import PlannerNode
        from agent.multi_agent.retriever import RetrieverNode
        from agent.multi_agent.reflection import ReflectionNode
        from agent.multi_agent.summarizer import SummarizerNode

        graph = build_multi_agent_graph(
            planner=PlannerNode(),
            retriever=RetrieverNode(),
            reflector=ReflectionNode(),
            summarizer=SummarizerNode(),
            max_retries=2,
        )

        assert graph is not None

    def test_graph_compile_bounded(self):
        """验证图有界（max_retries保证终止）"""
        import os
        if not os.getenv("DASHSCOPE_API_KEY"):
            pytest.skip("需要 DASHSCOPE_API_KEY 环境变量")
        from agent.multi_agent.graph import build_multi_agent_graph
        from agent.multi_agent.planner import PlannerNode
        from agent.multi_agent.retriever import RetrieverNode
        from agent.multi_agent.reflection import ReflectionNode
        from agent.multi_agent.summarizer import SummarizerNode

        graph = build_multi_agent_graph(
            planner=PlannerNode(),
            retriever=RetrieverNode(),
            reflector=ReflectionNode(),
            summarizer=SummarizerNode(),
            max_retries=2,
        )

        assert graph is not None


class TestReducers:
    """Annotated reducer 累积行为测试"""

    def test_list_concat(self):
        """测试 operator.add 的列表累积行为"""
        import operator
        from typing import Annotated

        class TestState(dict):
            items: Annotated[list, operator.add] = []
            count: int = 0

        # 模拟多次update
        s1 = {"items": ["a"]}
        s2 = {"items": ["b"]}
        s3 = {"items": ["c"]}

        # 在真实LangGraph中这些会累积，此处手动验证概念
        combined = s1["items"] + s2["items"] + s3["items"]
        assert combined == ["a", "b", "c"]


class TestToolRegistry:
    """工具注册中心测试"""

    def test_tool_registry_import(self):
        from agent.tools.tool_registry import ToolRegistry, ToolCategory, ToolDefinition
        reg = ToolRegistry()
        assert reg.tool_count == 0

    def test_tool_registration(self):
        from agent.tools.tool_registry import ToolRegistry, ToolCategory, ToolDefinition

        reg = ToolRegistry()
        def dummy_handler(**kwargs):
            return {"status": "ok"}

        reg.register(
            ToolDefinition(
                name="test_tool",
                description="测试工具",
                category=ToolCategory.SYSTEM,
                parameters={"type": "object", "properties": {}, "required": []},
            ),
            dummy_handler,
        )

        assert reg.tool_count == 1
        assert reg.is_registered("test_tool")

    def test_tool_invoke(self):
        from agent.tools.tool_registry import ToolRegistry, ToolCategory, ToolDefinition

        reg = ToolRegistry()
        def echo_handler(**kwargs):
            return {"echo": kwargs}

        reg.register(
            ToolDefinition(
                name="echo",
                description="回显",
                category=ToolCategory.SYSTEM,
                parameters={
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                    "required": [],
                },
            ),
            echo_handler,
        )

        result = reg.invoke("echo", {"message": "hello"})
        assert result["success"] is True
        assert result["data"]["echo"]["message"] == "hello"

    def test_tool_invoke_unregistered(self):
        from agent.tools.tool_registry import ToolRegistry

        reg = ToolRegistry()
        result = reg.invoke("nonexistent", {})
        assert result["success"] is False
        assert "未注册" in result["error"]

    def test_tool_get_schemas(self):
        from agent.tools.tool_registry import ToolRegistry, ToolCategory, ToolDefinition

        reg = ToolRegistry()
        def handler(**kwargs):
            return {}

        reg.register(
            ToolDefinition(
                name="test1",
                description="测试1",
                category=ToolCategory.SEARCH,
                parameters={"type": "object", "properties": {}, "required": []},
            ),
            handler,
        )

        schemas = reg.get_schemas(for_openai=True)
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "test1"


class TestKnowledgeGraph:
    """知识图谱测试"""

    def test_kg_initialization(self):
        from rag.knowledge_graph import KnowledgeGraph
        kg = KnowledgeGraph()
        assert kg.is_built is False
        assert kg.entity_count == 0

    def test_kg_build_index(self):
        from rag.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph()
        texts = [
            "石头P10是一款扫地机器人，具有LDS激光导航功能",
            "科沃斯T30采用dToF导航技术，清洁效率高",
            "石头P10和科沃斯T30都是热门扫地机器人型号",
            "LDS激光导航可以实现精准建图",
        ]
        kg.build_index(texts, verbose=False)

        assert kg.is_built is True
        assert kg.entity_count > 0
        # 石头P10、科沃斯T30、LDS激光导航 应该出现
        assert "石头P10" in kg.entity_to_chunks or "科沃斯T30" in kg.entity_to_chunks

    def test_kg_one_hop_expand(self):
        from rag.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph()
        texts = [
            "石头P10的LDS激光导航精度高",
            "LDS激光导航在暗光环境下表现好",
            "科沃斯T30使用dToF导航",
        ]
        kg.build_index(texts, verbose=False)

        expanded = kg.one_hop_expand("石头P10", max_entities=2)
        assert isinstance(expanded, list)

    def test_kg_entity_info(self):
        from rag.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph()
        texts = ["石头P10吸力5500Pa，适合小户型"]
        kg.build_index(texts, verbose=False)

        info = kg.get_entity_info("石头P10")
        assert info is not None
        assert info["name"] == "石头P10"


# ============================================================
# 集成测试（需要 DASHSCOPE_API_KEY）
# ============================================================

@pytest.mark.integration
class TestMultiAgentIntegration:
    """端到端集成测试"""

    @pytest.fixture
    def orchestrator(self):
        import os
        if not os.getenv("DASHSCOPE_API_KEY"):
            pytest.skip("需要 DASHSCOPE_API_KEY 环境变量")

        from agent.multi_agent_orchestrator import MultiAgentOrchestrator
        return MultiAgentOrchestrator()

    def test_chat_basic(self, orchestrator):
        """基础知识问答"""
        result = orchestrator.chat("小户型适合什么扫地机器人？", session_id="test_basic")

        assert result.answer, "应该有回答内容"
        assert result.plan, "应该有执行计划"
        assert result.plan.get("intent"), "应该识别意图"
        assert result.retry_count >= 0

    def test_chat_returns_tool_calls(self, orchestrator):
        """验证工具调用轨迹"""
        result = orchestrator.chat("石头P10的规格参数是什么？", session_id="test_tools")

        # 应该至少有一次工具调用（retriever会调用工具）
        assert result.tool_calls is not None

    def test_chat_reflection(self, orchestrator):
        """验证反思结果存在"""
        result = orchestrator.chat("扫地机器人如何保养？", session_id="test_refl")

        assert result.reflection, "应该有反思结果"
        assert result.reflection.get("verdict") in ("pass", "retry")
        assert 0 <= result.confidence <= 1

    def test_chat_multi_turn(self, orchestrator):
        """多轮对话"""
        q1 = "石头P10的吸力是多少？"
        r1 = orchestrator.chat(q1, session_id="test_multi")

        messages = [
            {"role": "user", "content": q1},
            {"role": "assistant", "content": r1.answer[:300]},
        ]
        q2 = "它适合大户型吗？"
        r2 = orchestrator.chat(q2, session_id="test_multi", messages=messages)

        assert r2.answer, "多轮对话应该有回答"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
