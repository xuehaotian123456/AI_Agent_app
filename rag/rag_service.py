"""
用户提问，搜索参考资料，将提问和参考资料提交给模型，让模型总结回复
使用 LangGraph 实现精细控制的 RAG Agent
"""
from typing import TypedDict, List, Optional
from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from utils.config_handler import chroma_conf

class GraphState(TypedDict):
    """定义 LangGraph 的状态结构"""
    messages: List[dict]       # 存储所有对话消息 [{"role": "user/assistant", "content": "..."}]
    context: str               # 检索到的文档上下文
    answer: str                # 生成的答案
    query: str                 # 当前用户查询


def print_prompt(prompt):
    print("="*20)
    print(prompt.to_string())
    print("="*20)
    return prompt


class RagSummarizeService(object):
    def __init__(self, use_langgraph: bool = True):
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever()
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

    def _init_chain(self):
        """初始化传统的链式调用（非 LangGraph 模式）"""
        chain = self.prompt_template | print_prompt | self.model | StrOutputParser()
        return chain

    def _build_graph(self):
        """构建 LangGraph 图结构"""

        def retrieve(state: GraphState) -> GraphState:
            """检索节点：根据用户查询检索相关文档"""
            query = state["query"]

            # 判断是否使用混合检索
            if self.vector_store.hybrid_enabled and hasattr(self.retriever, 'hybrid_search'):
                # 使用混合检索
                docs = self.retriever.hybrid_search(
                    query=query,
                    top_k_retrieve=self.top_k_retrieve,
                    top_k_rerank=self.top_k_rerank,
                    alpha=self.alpha
                )
            else:
                # 使用传统向量检索
                docs = self.retriever.invoke(query)

            # 格式化检索到的文档
            context = ""
            for idx, doc in enumerate(docs, 1):
                source = doc.metadata.get('source', '未知来源')
                context += f"[参考资料：{idx}.] [参考资料]{doc.page_content} | [参考资料链接]{source}\n"

            state["context"] = context
            return state

        def generate(state: GraphState) -> GraphState:
            """生成节点：基于检索到的上下文生成答案"""
            # 使用 prompt template 格式化输入
            formatted_prompt = self.prompt_template.format(
                input=state["query"],
                context=state["context"]
            )

            # 调用模型生成答案
            response = self.model.invoke(formatted_prompt)
            answer = response.content if hasattr(response, 'content') else str(response)

            state["answer"] = answer

            # 确保 messages 列表存在（首次调用新 thread_id 时可能不存在）
            if "messages" not in state:
                state["messages"] = []

            # 将问答添加到消息历史中
            state["messages"].append({"role": "user", "content": state["query"]})
            state["messages"].append({"role": "assistant", "content": answer})

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
                context += f"[参考资料：{counter}.] [参考资料]{doc.page_content} | [参考资料链接]{doc.metadata['source']}\n"
            return self.chain.invoke({"input": query, "context": context})
        else:
            # 使用 LangGraph
            config = {"configurable": {"thread_id": thread_id}}

            # 初始化状态
            initial_state = {
                "messages": [],
                "context": "",
                "answer": "",
                "query": query
            }

            # 执行图
            result = self.graph.invoke(initial_state, config=config)

            return result["answer"]

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
            "context": result["context"]
        }


# ... existing code ...

if __name__ == '__main__':
    # 测试 LangGraph 版本
    rag = RagSummarizeService(use_langgraph=True)

    # 单轮对话测试
    print("=" * 50)
    print("测试1：单轮对话")
    print("=" * 50)
    answer = rag.rag_summarize("小户型适合哪些扫地机器人？", thread_id="user_123")
    print(f"答案：{answer}")

    # 多轮对话测试
    print("\n" + "=" * 50)
    print("测试2：多轮对话（同一 thread_id）")
    print("=" * 50)

    questions = [
        "小户型适合哪些扫地机器人？",
        "这些机器人的价格大概是多少？",
        "哪款性价比最高？"
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n第{i}轮问题：{question}")
        result = rag.rag_summarize_with_history(question, thread_id="user_123")
        print(f"答案：{result['answer']}")
        print(f"对话历史长度：{len(result['messages'])}")

    # 测试多用户会话隔离
    print("\n" + "=" * 50)
    print("测试3：验证多用户会话是否互不干扰")
    print("=" * 50)

    # 用户A的会话 - 进行多轮对话
    print("\n--- 用户A的会话 ---")
    result_a1 = rag.rag_summarize_with_history("扫地机器人的电池续航一般是多久？", thread_id="user_A")
    print(f"用户A第1轮问题：扫地机器人的电池续航一般是多久？")
    print(f"用户A第1轮答案：{result_a1['answer'][:100]}...")
    print(f"用户A对话历史长度：{len(result_a1['messages'])}")

    result_a2 = rag.rag_summarize_with_history("那充电时间需要多久？", thread_id="user_A")
    print(f"\n用户A第2轮问题：那充电时间需要多久？")
    print(f"用户A第2轮答案：{result_a2['answer'][:100]}...")
    print(f"用户A对话历史长度：{len(result_a2['messages'])}")

    # 用户B的会话 - 完全不同的话题
    print("\n--- 用户B的会话（应该与用户A完全独立）---")
    result_b1 = rag.rag_summarize_with_history("扫地机器人如何避障？", thread_id="user_B")
    print(f"用户B第1轮问题：扫地机器人如何避障？")
    print(f"用户B第1轮答案：{result_b1['answer'][:100]}...")
    print(f"用户B对话历史长度：{len(result_b1['messages'])}")

    # 关键验证点：用户A继续对话，不应该受到用户B的影响
    print("\n--- 验证：用户A继续对话（不应受用户B影响）---")
    result_a3 = rag.rag_summarize_with_history("续航和充电有什么关系吗？", thread_id="user_A")
    print(f"用户A第3轮问题：续航和充电有什么关系吗？")
    print(f"用户A第3轮答案：{result_a3['answer'][:100]}...")
    print(f"用户A对话历史长度：{len(result_a3['messages'])}")

    # 最终验证总结
    print("\n" + "=" * 50)
    print("验证总结：")
    print("=" * 50)
    print(f"✓ 用户A总对话轮数：{len(result_a3['messages']) // 2} 轮（应该是3轮）")
    print(f"✓ 用户B总对话轮数：{len(result_b1['messages']) // 2} 轮（应该是1轮）")
    print(f"✓ 用户A的问题都是关于'续航/充电'的话题")
    print(f"✓ 用户B的问题是关于'避障'的话题")
    print(f"✓ 如果两个用户的对话历史长度正确且话题不混淆，说明会话隔离成功！")

    # 额外验证：查看用户A的完整对话历史
    print("\n--- 用户A的完整对话历史 ---")
    for idx, msg in enumerate(result_a3['messages'], 1):
        role = "用户" if msg['role'] == 'user' else "AI助手"
        content_preview = msg['content'][:80] + "..." if len(msg['content']) > 80 else msg['content']
        print(f"{idx}. [{role}]: {content_preview}")

    print("\n--- 用户B的完整对话历史 ---")
    for idx, msg in enumerate(result_b1['messages'], 1):
        role = "用户" if msg['role'] == 'user' else "AI助手"
        content_preview = msg['content'][:80] + "..." if len(msg['content']) > 80 else msg['content']
        print(f"{idx}. [{role}]: {content_preview}")
