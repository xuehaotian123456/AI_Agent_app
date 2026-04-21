"""
用户提问，搜索参考资料，将提问和参考资料提交给模型，让模型总结回复
使用 LangGraph 实现精细控制的 RAG Agent
"""
from typing import TypedDict, List, Optional
from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts, load_rag_structured_prompts
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model, robust_llm_caller, get_last_model_used
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from utils.config_handler import chroma_conf, rag_conf
from pydantic import ValidationError
import json
import time

class GraphState(TypedDict):
    """定义 LangGraph 的状态结构"""
    messages: List[dict]       # 存储所有对话消息 [{"role": "user/assistant", "content": "..."}]
    context: str               # 检索到的文档上下文
    answer: str                # 生成的答案
    query: str                 # 当前用户查询
    structured_result: Optional[dict]  # 新增：结构化解析结果
    retrieval_docs: Optional[List[Document]]  # 新增：原始检索文档（用于构建引用）
    generation_model_used: Optional[str]  # 生成答案实际使用的模型（primary/backup）
    llm_usage: Optional[dict]  # LLM token 用量等元数据


def print_prompt(prompt):
    print("="*20)
    print(prompt.to_string())
    print("="*20)
    return prompt


class RagSummarizeService(object):
    def __init__(self, use_langgraph: bool = True, enable_structured_output: bool = True):
        self.vector_store = VectorStoreService()
        # 启动时加载/恢复文档并初始化混合检索器，确保 API 调用链能命中混合检索能力
        try:
            self.vector_store.load_documents()
        except Exception as e:
            from utils.logger_handler import logger
            logger.warning(f"[RAG初始化] 文档加载失败，将继续使用现有向量库: {e}")
        self.retriever = self.vector_store.get_retriever()

        # 初始化结构化输出解析器
        self.enable_structured_output = enable_structured_output

        if enable_structured_output:
            # 结构化模式：加载专用的结构化 prompt
            self.prompt_text = load_rag_structured_prompts()

            from rag.output_models import AnswerWithCitations
            self.parser = PydanticOutputParser(pydantic_object=AnswerWithCitations)
            format_instructions = self.parser.get_format_instructions()

            # 关键修复：转义 format_instructions 中的花括号
            escaped_format_instructions = format_instructions.replace('{', '{{').replace('}', '}}')

            # 将 Pydantic 的格式指令拼接到 prompt 后面
            combined_template = self.prompt_text + "\n\n" + escaped_format_instructions
            self.prompt_template = PromptTemplate.from_template(combined_template)
        else:
            # 传统模式：加载传统的纯文本 prompt
            self.prompt_text = load_rag_prompts()
            self.prompt_template = PromptTemplate.from_template(self.prompt_text)

        self.model = chat_model
        self.use_langgraph = use_langgraph

        # 获取混合检索配置
        hybrid_config = chroma_conf.get("hybrid_search", {})
        self.top_k_retrieve = hybrid_config.get("top_k_retrieve", 10)
        self.top_k_rerank = hybrid_config.get("top_k_rerank", 3)
        self.alpha = hybrid_config.get("alpha", 0.5)

        if use_langgraph:
            # 先初始化 checkpointer
            self.checkpointer = MemorySaver()
            # 再构建图（此时 checkpointer 已存在）
            self.graph = self._build_graph()
        else:
            # 保持原有的简单链式调用
            self.chain = self._init_chain()

    @staticmethod
    def _extract_llm_usage(response) -> dict:
        """从模型响应中提取 token 用量，提取失败则返回空字典。"""
        usage = {}
        try:
            metadata = getattr(response, "response_metadata", None) or {}
            if isinstance(metadata, dict):
                token_usage = metadata.get("token_usage") or metadata.get("usage") or {}
                if isinstance(token_usage, dict):
                    usage = token_usage
        except Exception:
            usage = {}
        return usage

    def _init_chain(self):
        """初始化传统的链式调用（非 LangGraph 模式）"""
        if self.enable_structured_output:
            # 使用结构化输出解析器
            chain = self.prompt_template | self.model | self.parser
        else:
            # 传统模式：使用字符串输出解析器
            chain = self.prompt_template | self.model | StrOutputParser()
        return chain

    def _build_graph(self):
        """构建 LangGraph 图结构"""

        def retrieve(state: GraphState) -> GraphState:
            """检索节点：根据用户查询检索相关文档"""
            query = state["query"]
            start_time = time.time()

            # 判断是否使用混合检索
            if self.vector_store.hybrid_enabled and hasattr(self.retriever, 'hybrid_search'):
                # 使用混合检索
                docs = self.retriever.hybrid_search(
                    query=query,
                    top_k_retrieve=self.top_k_retrieve,
                    top_k_rerank=self.top_k_rerank,
                    alpha=self.alpha
                )
                retrieval_method = "hybrid"
            else:
                # 使用传统向量检索
                docs = self.retriever.invoke(query)
                retrieval_method = "vector"

            # 保存原始文档（用于构建引用）
            state["retrieval_docs"] = docs

            # 格式化检索到的文档
            context = ""
            for idx, doc in enumerate(docs, 1):
                source = doc.metadata.get('source', '未知来源')
                page = doc.metadata.get('page', '')
                page_info = f" 第{page}页" if page else ""
                context += f"[参考资料：{idx}.] [参考资料]{doc.page_content} | [参考资料链接]{source}{page_info}\n"

            state["context"] = context

            # 记录检索元数据
            elapsed_time = time.time() - start_time
            state.setdefault("retrieval_metadata", {})
            state["retrieval_metadata"]["method"] = retrieval_method
            state["retrieval_metadata"]["docs_count"] = len(docs)
            state["retrieval_metadata"]["elapsed_time"] = round(elapsed_time, 3)

            return state

        def generate(state: GraphState) -> GraphState:
            """生成节点：基于检索到的上下文生成答案"""
            query = state["query"]
            context = state["context"]

            try:
                if self.enable_structured_output:
                    # 使用结构化输出
                    formatted_prompt = self.prompt_template.format(
                        input=query,
                        context=context
                    )

                    # 使用带重试的调用器
                    response = robust_llm_caller(formatted_prompt)
                    state["generation_model_used"] = get_last_model_used()
                    state["llm_usage"] = self._extract_llm_usage(response)

                    # 尝试解析为结构化输出
                    try:
                        parsed_result = self.parser.parse(response.content)
                        # 修复：使用 model_dump() 替代 dict()
                        state["structured_result"] = parsed_result.model_dump()
                        state["answer"] = parsed_result.answer

                        # 将结构化结果转换为消息格式
                        answer_text = parsed_result.answer
                        if parsed_result.citations:
                            citation_sources = [f"[{i+1}] {cit.source}" for i, cit in enumerate(parsed_result.citations)]
                            answer_text += "\n\n参考资料：" + "\n".join(citation_sources)

                    except (ValidationError, json.JSONDecodeError) as e:
                        # 解析失败，降级为普通文本
                        from utils.logger_handler import logger
                        logger.warning(f"[结构化输出] 解析失败，降级为普通文本: {e}")
                        state["answer"] = response.content if hasattr(response, 'content') else str(response)
                        state["structured_result"] = None
                else:
                    # 使用传统方式
                    formatted_prompt = self.prompt_template.format(
                        input=query,
                        context=context
                    )
                    # 使用带重试的调用器
                    response = robust_llm_caller(formatted_prompt)
                    state["generation_model_used"] = get_last_model_used()
                    state["llm_usage"] = self._extract_llm_usage(response)
                    state["answer"] = response.content if hasattr(response, 'content') else str(response)
                    state["structured_result"] = None

            except Exception as e:
                from utils.logger_handler import logger
                logger.error(f"[生成节点] 异常: {e}", exc_info=True)
                state["answer"] = "抱歉，生成回答时出现错误，请稍后重试。"
                state["structured_result"] = None
                state["generation_model_used"] = "unknown"
                state["llm_usage"] = {}

            # 确保 messages 列表存在（首次调用新 thread_id 时可能不存在）
            if "messages" not in state:
                state["messages"] = []

            # 将问答添加到消息历史中
            state["messages"].append({"role": "user", "content": query})
            state["messages"].append({"role": "assistant", "content": state["answer"]})

            return state

        # 构建图
        builder = StateGraph(GraphState)

        # 添加节点
        builder.add_node("retrieve", retrieve)
        builder.add_node("generate", generate)

        # 添加边：retrieve -> generate
        builder.add_edge("retrieve", "generate")

        # 设置入口点
        builder.set_entry_point("retrieve")

        # 编译图并传入 checkpointer
        graph = builder.compile(checkpointer=self.checkpointer)
        return graph

    def retriever_docs(self, query: str) -> list[Document]:
        """检索相关文档"""
        # 判断是否使用混合检索
        if self.vector_store.hybrid_enabled and hasattr(self.retriever, 'hybrid_search'):
            return self.retriever.hybrid_search(
                query=query,
                top_k_retrieve=self.top_k_retrieve,
                top_k_rerank=self.top_k_rerank,
                alpha=self.alpha
            )
        return self.retriever.invoke(query)

    def rag_summarize_stream(self, query: str, thread_id: str = "default"):
        """
        流式执行 RAG 总结（生成 token 流）

        Args:
            query: 用户查询
            thread_id: 会话ID

        Yields:
            str: 逐个 token 的输出内容
        """
        if not self.use_langgraph:
            # 非 LangGraph 模式的流式调用
            context_docs = self.retriever_docs(query)
            context = ""
            counter = 0
            for doc in context_docs:
                counter += 1
                source = doc.metadata.get('source', '未知来源')
                page = doc.metadata.get('page', '')
                page_info = f" 第{page}页" if page else ""
                context += f"[参考资料：{counter}.] [参考资料]{doc.page_content} | [参考资料链接]{source}{page_info}\n"

            formatted_prompt = self.prompt_template.format(
                input=query,
                context=context
            )

            # 使用模型的 stream 方法
            for chunk in self.model.stream(formatted_prompt):
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content

        else:
            # LangGraph 模式的流式调用
            config = {"configurable": {"thread_id": thread_id}}

            initial_state = {
                "messages": [],
                "context": "",
                "answer": "",
                "query": query,
                "structured_result": None,
                "retrieval_docs": [],
                "generation_model_used": "unknown",
                "llm_usage": {}
            }

            # 使用 stream 方法执行图
            for event in self.graph.stream(initial_state, config=config, stream_mode="updates"):
                # event 是一个字典，键是节点名，值是该节点的输出
                for node_name, node_output in event.items():
                    if node_name == "generate":
                        # 从 generate 节点提取流式内容
                        answer = node_output.get("answer", "")
                        if answer:
                            yield answer

    def rag_summarize(self, query: str, thread_id: str = "default") -> str:
        """
        执行 RAG 总结

        Args:
            query: 用户查询
            thread_id: 会话ID，用于区分不同用户的对话历史

        Returns:
            模型生成的答案
        """
        if not self.use_langgraph:
            # 使用传统方式
            context_docs = self.retriever_docs(query)
            context = ""
            counter = 0
            for doc in context_docs:
                counter += 1
                source = doc.metadata.get('source', '未知来源')
                page = doc.metadata.get('page', '')
                page_info = f" 第{page}页" if page else ""
                context += f"[参考资料：{counter}.] [参考资料]{doc.page_content} | [参考资料链接]{source}{page_info}\n"

            if self.enable_structured_output:
                try:
                    result = self.chain.invoke({"input": query, "context": context})
                    return result.answer
                except Exception as e:
                    from utils.logger_handler import logger
                    logger.warning(f"[结构化输出] 失败，降级为普通文本: {e}")
                    # 临时切换为非结构化模式
                    old_mode = self.enable_structured_output
                    self.enable_structured_output = False
                    self.chain = self._init_chain()
                    result = self.chain.invoke({"input": query, "context": context})
                    # 恢复
                    self.enable_structured_output = old_mode
                    self.chain = self._init_chain()
                    return result
            else:
                return self.chain.invoke({"input": query, "context": context})
        else:
            # 使用 LangGraph
            config = {"configurable": {"thread_id": thread_id}}

            # 初始化状态
            initial_state = {
                "messages": [],
                "context": "",
                "answer": "",
                "query": query,
                "structured_result": None,
                "retrieval_docs": [],
                "generation_model_used": "unknown",
                "llm_usage": {}
            }

            # 执行图
            result = self.graph.invoke(initial_state, config=config)

            return result["answer"]

    def rag_summarize_structured(self, query: str, thread_id: str = "default") -> dict:
        """
        执行 RAG 总结并返回结构化结果（新增方法）

        Args:
            query: 用户查询
            thread_id: 会话ID

        Returns:
            包含结构化结果的字典
        """
        if not self.enable_structured_output:
            raise ValueError("结构化输出未启用，请在初始化时设置 enable_structured_output=True")

        config = {"configurable": {"thread_id": thread_id}}

        # 仅设置 query，让 checkpointer 自动加载 thread 历史，避免多轮上下文被清空
        initial_state = {
            "query": query
        }

        result = self.graph.invoke(initial_state, config=config)

        return {
            "answer": result["answer"],
            "structured_result": result.get("structured_result"),
            "messages": result["messages"],
            "context": result["context"],
            "retrieval_metadata": result.get("retrieval_metadata", {}),
            "generation_model_used": result.get("generation_model_used", "unknown"),
            "llm_usage": result.get("llm_usage", {})
        }

    def rag_summarize_with_history(self, query: str, thread_id: str = "default") -> dict:
        """
        执行 RAG 总结并返回完整状态（包含对话历史）

        Args:
            query: 用户查询
            thread_id: 会话ID

        Returns:
            包含答案和对话历史的字典
        """
        config = {"configurable": {"thread_id": thread_id}}

        # 不再手动初始化 messages，让 checkpointer 自动加载历史
        initial_state = {
            "query": query
        }

        result = self.graph.invoke(initial_state, config=config)

        return {
            "answer": result["answer"],
            "messages": result["messages"],
            "context": result["context"],
            "structured_result": result.get("structured_result"),
            "generation_model_used": result.get("generation_model_used", "unknown"),
            "llm_usage": result.get("llm_usage", {})
        }


if __name__ == '__main__':
    # 测试结构化输出
    print("=" * 50)
    print("测试：结构化输出")
    print("=" * 50)

    rag = RagSummarizeService(use_langgraph=True, enable_structured_output=True)

    result = rag.rag_summarize_structured("小户型适合哪些扫地机器人？", thread_id="test_user")

    print(f"\n回答：{result['answer']}")
    print(f"\n结构化结果：")
    if result['structured_result']:
        import json
        print(json.dumps(result['structured_result'], ensure_ascii=False, indent=2))

    print(f"\n检索元数据：{result.get('retrieval_metadata', {})}")

