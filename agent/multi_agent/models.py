"""
Multi-Agent 系统数据模型

定义四智能体闭环中所有节点间的数据传递结构。
每个模型都使用 Pydantic 强类型，确保：
1. LLM 结构化输出的格式约束
2. Agent 间数据传递的类型安全
3. 序列化到 JSON 用于 API 响应和飞轮日志

复用 rag.output_models.AnswerWithCitations 作为最终答案结构。
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class SubTask(BaseModel):
    """
    规划子任务

    每个子任务代表一个独立的检索或工具调用目标
    """
    task_id: str = Field(..., description="子任务唯一ID，如't1','t2'")
    description: str = Field(..., description="子任务描述，说明要完成什么")
    search_query: str = Field(default="", description="检索查询词，空字符串表示无需检索")
    required_tools: List[str] = Field(
        default_factory=list,
        description="该子任务需要调用的工具名列表，如['hybrid_search','get_device_specs']"
    )
    expected_knowledge: List[str] = Field(
        default_factory=list,
        description="期望获取的知识点，用于后续质量检查"
    )


class ExecutionPlan(BaseModel):
    """
    规划器输出：执行计划

    这是 Planner Agent 的核心产出，指导后续所有 Agent 的工作
    """
    intent: str = Field(
        default="knowledge",
        description="用户意图: knowledge(知识问答) | report(报告生成) | chat(闲聊) | troubleshoot(故障诊断) | comparison(对比选择)"
    )
    summary: str = Field(default="", description="对用户意图的一句话重述，写入Trace")
    sub_tasks: List[SubTask] = Field(default_factory=list, description="拆解后的子任务列表")
    requires_retrieval: bool = Field(default=True, description="是否需要执行RAG检索")
    reasoning: str = Field(default="", description="规划推理过程说明")


class ReflectionResult(BaseModel):
    """
    反思器输出：质量评估结果

    Reflector Agent 的产出，决定答案是否通过或需要重试
    """
    verdict: str = Field(
        default="pass",
        description="判定结果: pass(通过) | retry(重试)"
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="置信度 0-1"
    )
    reason: str = Field(default="", description="判定原因说明")
    missing_aspects: List[str] = Field(
        default_factory=list,
        description="缺失的知识点，供改写查询时定向补充"
    )


class RetrievalRound(BaseModel):
    """单轮检索记录"""
    round_number: int = Field(default=1)
    query: str = Field(default="")
    method: str = Field(default="hybrid", description="检索方法")
    docs_count: int = Field(default=0)
    kg_used: bool = Field(default=False)
    kg_entities: List[str] = Field(default_factory=list)
    elapsed_ms: float = Field(default=0.0)


class ToolCallTrace(BaseModel):
    """单次工具调用轨迹"""
    tool_name: str
    args_summary: str = Field(default="", description="参数摘要（截断）")
    result_summary: str = Field(default="", description="结果摘要（截断）")
    success: bool = True
    elapsed_ms: float = 0.0


class MultiAgentResponse(BaseModel):
    """
    Multi-Agent 完整响应模型

    API v2 /api/v2/chat 返回此结构，包含所有Agent的决策轨迹
    """
    request_id: str = Field(default="")
    session_id: str = Field(default="")
    answer: str = Field(..., description="最终答案文本")
    citations: List[Dict[str, Any]] = Field(default_factory=list, description="引用来源列表")
    plan: Optional[Dict[str, Any]] = Field(default=None, description="Planner的执行计划")
    reflection: Optional[Dict[str, Any]] = Field(default=None, description="Reflector的反思结果")
    retrieval_rounds: List[Dict[str, Any]] = Field(default_factory=list, description="各轮检索详情")
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list, description="工具调用轨迹")
    retry_count: int = Field(default=0, description="重试次数")
    confidence: float = Field(default=0.0, description="最终置信度")
    kg_used: bool = Field(default=False, description="是否使用了知识图谱")
    model_used: str = Field(default="unknown", description="实际使用的LLM模型")
    llm_usage: List[Dict[str, Any]] = Field(default_factory=list, description="LLM用量追踪")
    response_time_ms: float = Field(default=0.0, description="总响应时间(毫秒)")
