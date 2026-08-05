"""
Multi-Agent StateGraph 状态定义

使用 Annotated reducer 模式实现跨轮次的状态累积——
这是 LangGraph 的核心设计模式之一，确保每次 retry 追加数据而非覆盖，
让 Summarizer 拿到完整的多轮检索证据链。

关键模式：
    contexts: Annotated[List[str], operator.add]
    → 每轮新context追加到列表末尾，而非覆盖
"""
from typing import Annotated, TypedDict, List, Optional, Dict, Any
import operator
from langchain_core.documents import Document


class MultiAgentState(TypedDict, total=False):
    """
    四智能体闭环的完整状态定义

    total=False 表示所有字段都是可选的（TypedDict 不能有默认值，
    由各个 Agent Node 根据职责填充对应字段）
    """

    # ==================== 输入层 ====================
    query: str                              # 当前查询（可能是改写后的）
    original_query: str                     # 用户原始查询（不变）
    messages: List[Dict[str, str]]          # 对话历史 [{role, content}]
    session_id: str                         # 会话ID
    user_id: str                            # 用户ID

    # ==================== Planner 输出 ====================
    plan: Optional[Dict[str, Any]]          # ExecutionPlan.model_dump()
    tool_outputs: Dict[str, str]            # 轻量工具调用结果 {tool_name: result_text}

    # ==================== Retriever 输出（Annotated reducer 累积） ====================
    retrieval_docs: Annotated[List[Document], operator.add]
    """跨轮累积的检索文档列表——每次retry追加"""
    contexts: Annotated[List[str], operator.add]
    """跨轮累积的格式化上下文——每次retry追加"""
    retrieval_rounds: Annotated[List[Dict[str, Any]], operator.add]
    """跨轮累积的检索元数据——每次retry追加"""
    tool_calls: Annotated[List[Dict[str, Any]], operator.add]
    """跨轮累积的工具调用轨迹——每次retry追加"""
    kg_used: bool                           # 是否使用了知识图谱

    # ==================== Reflector 输出 ====================
    reflection: Optional[Dict[str, Any]]    # ReflectionResult.model_dump()
    draft_answer: str                       # 反思前生成的草稿答案（用于质量检查）
    quality_checks: Annotated[List[Dict[str, Any]], operator.add]
    """跨轮累积的质量检查结果——每次retry追加"""
    retry_count: int                        # 当前重试次数

    # ==================== Summarizer 输出 ====================
    answer: str                             # 最终答案
    structured_result: Optional[Dict[str, Any]]  # AnswerWithCitations.model_dump()

    # ==================== 观测层（Annotated reducer 累积） ====================
    node_times: Annotated[List[Dict[str, Any]], operator.add]
    """各节点执行耗时 [{node, elapsed_ms}]"""
    llm_usage: Annotated[List[Dict[str, Any]], operator.add]
    """LLM调用用量 [{node, model, usage}]"""
    models_used: Annotated[List[str], operator.add]
    """使用的模型列表"""
