"""
Retriever Agent — 检索智能体（含工具调用编排）

职责：
1. 按 Planner 的子任务列表，逐一执行检索/工具调用
2. 支持三路多路召回：hybrid_search + KG扩展 + 关键词检索
3. 通过 ToolRegistry 统一执行工具调用，自动记录轨迹
4. 合并去重检索结果，BGE重排序
5. 结果格式化 + Redis缓存读写

设计要点：
- ToolRegistry.invoke() 统一入口，单个工具失败不阻塞
- Annotated[operator.add] reducer 确保跨轮累积
- 检索缓存命中则跳过实际检索，节省LLM调用
"""
import time
from typing import Dict, Any, List

from langchain_core.documents import Document

from model.factory import get_last_model_used
from utils.logger_handler import logger
from utils.observability import observe_if_available, update_observation_safe
from utils.config_handler import multi_agent_conf

from agent.multi_agent.state import MultiAgentState
from agent.tools.tool_registry import tool_registry


class RetrieverNode:
    """
    检索智能体节点

    执行多路检索 + 工具调用编排，
    作为 LangGraph 节点函数，输入 MultiAgentState，返回部分状态更新
    """

    def __init__(self, vector_store=None, retrieval_cache=None, knowledge_graph=None, flywheel=None):
        """
        Args:
            vector_store: 向量存储服务（可选，从RagService获取）
            retrieval_cache: Redis检索缓存（可选）
            knowledge_graph: 知识图谱实例（可选）
            flywheel: 飞轮服务（可选）
        """
        self.vector_store = vector_store
        self.retrieval_cache = retrieval_cache
        self.knowledge_graph = knowledge_graph
        self.flywheel = flywheel

        # 配置
        retriever_conf = multi_agent_conf.get("retriever", {})
        self.top_k_retrieve = retriever_conf.get("top_k_retrieve", 10)
        self.top_k_rerank = retriever_conf.get("top_k_rerank", 3)
        self.alpha = retriever_conf.get("alpha", 0.5)
        self.kg_enabled = retriever_conf.get("kg_enabled", True)
        self.kg_max_entities = retriever_conf.get("kg_max_entities", 3)

        self._round_number = 0

    @observe_if_available(name="retriever")
    def __call__(self, state: MultiAgentState) -> Dict[str, Any]:
        """执行检索 + 工具调用"""
        start_time = time.time()
        self._round_number = state.get("retry_count", 0) + 1

        plan = state.get("plan", {})
        sub_tasks = plan.get("sub_tasks", [])
        query = state.get("query", "")

        logger.info(
            f"[Retriever] 第{self._round_number}轮检索: "
            f"sub_tasks={len(sub_tasks)}, query='{query[:60]}...'"
        )

        total_docs = []
        total_contexts = []
        tool_calls = []
        kg_used = False

        # 对每个子任务执行检索/工具调用
        for sub_task in sub_tasks:
            required_tools = sub_task.get("required_tools", [])
            search_query = sub_task.get("search_query", query)

            if not required_tools and search_query:
                # 无指定工具但有检索词 → 默认使用 hybrid_search
                required_tools = ["hybrid_search"]

            for tool_name in required_tools:
                # 构建工具参数
                args = self._build_tool_args(tool_name, sub_task, search_query)

                # 执行工具调用（统一入口）
                t_start = time.time()
                try:
                    result = tool_registry.invoke(tool_name, args)
                    t_elapsed = (time.time() - t_start) * 1000
                except Exception as e:
                    logger.error(f"[Retriever] 工具 '{tool_name}' 调用异常: {e}")
                    tool_calls.append({
                        "tool": tool_name,
                        "args": {k: str(v)[:50] for k, v in args.items()},
                        "success": False,
                        "error": str(e),
                        "elapsed_ms": round((time.time() - t_start) * 1000, 2),
                    })
                    continue  # 单个工具失败不阻塞

                tool_calls.append({
                    "tool": tool_name,
                    "args": {k: str(v)[:50] for k, v in args.items()},
                    "success": result.get("success", False),
                    "result_summary": self._summarize_result(result),
                    "elapsed_ms": round(t_elapsed, 2),
                })

                # 处理工具结果
                if result.get("success") and result.get("data"):
                    docs_from_tool = self._extract_documents(tool_name, result["data"])
                    total_docs.extend(docs_from_tool)

                # 标记KG使用
                if tool_name in ("entity_lookup",) and result.get("success"):
                    kg_used = True

        # 合并去重
        unique_docs = self._deduplicate_docs(total_docs)

        # BGE重排序（如果有文档且retriever可用）
        if len(unique_docs) > self.top_k_rerank and hasattr(self, 'vector_store'):
            try:
                vs = self.vector_store
                if vs and vs.hybrid_retriever and hasattr(vs.hybrid_retriever, '_rerank'):
                    unique_docs = vs.hybrid_retriever._rerank(query, unique_docs, self.top_k_rerank)
            except Exception as e:
                logger.debug(f"[Retriever] 重排序跳过: {e}")
        else:
            unique_docs = unique_docs[:self.top_k_rerank]

        # 格式化上下文
        context_text = self._format_context(unique_docs, query)

        # 记录本轮信息
        retrieval_round = {
            "round_number": self._round_number,
            "query": query,
            "method": "multi-tool",
            "tools_used": [tc["tool"] for tc in tool_calls if tc["success"]],
            "docs_count": len(unique_docs),
            "kg_used": kg_used,
            "kg_entities": [],
            "elapsed_ms": round((time.time() - start_time) * 1000, 2),
        }

        # 空结果记录到飞轮
        if len(unique_docs) == 0 and self.flywheel:
            try:
                self.flywheel.log_event("retrieval_empty", {
                    "query": query,
                    "tools": [tc["tool"] for tc in tool_calls],
                    "round": self._round_number,
                    "sub_tasks": len(sub_tasks),
                })
            except Exception:
                pass

        elapsed = round((time.time() - start_time) * 1000, 2)
        logger.info(
            f"[Retriever] 第{self._round_number}轮完成: "
            f"docs={len(unique_docs)}, tools={len(tool_calls)}, "
            f"kg={kg_used}, elapsed={elapsed}ms"
        )

        update_observation_safe(
            output={"docs_count": len(unique_docs), "kg_used": kg_used, "tools_used": len(tool_calls)},
            metadata={"retrieval_round": retrieval_round},
        )

        return {
            "retrieval_docs": unique_docs,
            "contexts": [context_text] if context_text else [],
            "retrieval_rounds": [retrieval_round],
            "tool_calls": tool_calls,
            "kg_used": kg_used,
            "node_times": [{"node": "retriever", "elapsed_ms": elapsed}],
        }

    # ==================== 辅助方法 ====================

    def _build_tool_args(self, tool_name: str, sub_task: dict, fallback_query: str) -> dict:
        """根据工具名自动构建参数"""
        search_query = sub_task.get("search_query", fallback_query)

        tool_args_map = {
            "hybrid_search": {"query": search_query, "top_k": self.top_k_rerank},
            "entity_lookup": {"entity_name": search_query},
            "keyword_search": {"keywords": search_query, "top_k": self.top_k_rerank},
            "similar_doc_search": {"doc_id": search_query, "top_k": 3},
            "query_user_usage": {"user_id": sub_task.get("user_id", "1001"), "month": sub_task.get("month", "2025-06")},
            "get_device_specs": {"model_name": search_query},
            "compare_models": {"model_list": sub_task.get("model_list", [search_query])},
            "get_weather": {"city": sub_task.get("city", search_query)},
            "get_maintenance_schedule": {"model_name": search_query},
            "check_firmware_version": {"model_name": search_query},
            "analyze_usage_pattern": {"user_id": sub_task.get("user_id", "1001"), "months": [sub_task.get("month", "2025-06")]},
            "calculate_consumable_life": {"part_name": search_query, "usage_hours": sub_task.get("usage_hours", 100)},
            "troubleshoot_issue": {"symptom": search_query, "model": sub_task.get("model")},
            "get_user_context": {"user_id": sub_task.get("user_id", "default")},
            "log_user_feedback": {"query_id": "0", "rating": 3},
        }

        return tool_args_map.get(tool_name, {"query": search_query})

    @staticmethod
    def _summarize_result(result: dict) -> str:
        """生成工具结果摘要"""
        if not result.get("success"):
            return f"失败: {result.get('error', '未知错误')[:100]}"
        data = result.get("data", {})
        if isinstance(data, dict):
            if "results" in data:
                return f"返回{len(data['results'])}条结果"
            if "total_found" in data:
                return f"找到{data['total_found']}条"
            if "entity" in data:
                return f"实体: {data.get('entity', {}).get('name', '?')} | 关联{len(data.get('related_docs', []))}篇"
            if "possible_causes" in data:
                return f"诊断: {len(data['possible_causes'])}个可能原因"
            if "found" in data:
                return "已找到" if data["found"] else "未找到"
            if "recorded" in data:
                return "反馈已记录"
        return str(data)[:100]

    @staticmethod
    def _extract_documents(tool_name: str, data: dict) -> List[Document]:
        """从工具返回数据中提取 Document 列表"""
        if isinstance(data, dict):
            # hybrid_search / keyword_search 结果
            if "results" in data:
                docs = []
                for r in data["results"]:
                    if isinstance(r, dict):
                        docs.append(Document(
                            page_content=r.get("content", ""),
                            metadata={
                                "source": r.get("source", "工具调用"),
                                "relevance_score": r.get("relevance_score", 0.0),
                                "tool": tool_name,
                            }
                        ))
                return docs
            # entity_lookup 结果
            if "entity" in data:
                entity = data["entity"]
                content = json_dumps_safe(entity)
                if content:
                    return [Document(page_content=content, metadata={"source": f"KG实体", "tool": tool_name})]
            if "related_docs" in data:
                docs = []
                for rd in data["related_docs"]:
                    docs.append(Document(
                        page_content=rd.get("content", ""),
                        metadata={"source": rd.get("source", "关联文档"), "tool": tool_name}
                    ))
                return docs

        # fallback
        content = json_dumps_safe(data)
        if content:
            return [Document(page_content=content[:1000], metadata={"source": f"工具:{tool_name}", "tool": tool_name})]
        return []

    @staticmethod
    def _deduplicate_docs(docs: List[Document]) -> List[Document]:
        """基于内容的文档去重"""
        seen = set()
        unique = []
        for doc in docs:
            fingerprint = doc.page_content[:80]
            if fingerprint not in seen:
                seen.add(fingerprint)
                unique.append(doc)
        return unique

    @staticmethod
    def _format_context(docs: List[Document], query: str = "") -> str:
        """格式化为参考资料块（复用现有约定）"""
        if not docs:
            return ""
        blocks = []
        for idx, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "未知来源")
            page = doc.metadata.get("page", "")
            page_info = f" 第{page}页" if page else ""
            tool = doc.metadata.get("tool", "")
            tool_info = f" [工具:{tool}]" if tool else ""
            blocks.append(
                f"[参考资料：{idx}.]{tool_info} [参考资料]{doc.page_content[:1000]} | [参考资料链接]{source}{page_info}"
            )
        return "\n".join(blocks)


def json_dumps_safe(obj) -> str:
    """安全的JSON序列化"""
    import json
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return str(obj)[:500]
