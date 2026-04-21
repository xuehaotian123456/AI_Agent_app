"""
对比结构化输出 vs 传统输出的质量
"""
from rag.rag_service import RagSummarizeService

queries = [
    "小户型适合哪些扫地机器人？",
    "扫地机器人如何避障？",
]

rag_structured = RagSummarizeService(enable_structured_output=True)
rag_traditional = RagSummarizeService(enable_structured_output=False)

for query in queries:
    print("=" * 60)
    print(f"问题：{query}")
    print("=" * 60)

    # 结构化输出
    print("\n【结构化模式】")
    result = rag_structured.rag_summarize_structured(query)
    print(f"回答：{result['answer']}")
    if result['structured_result']:
        print(f"引用数：{len(result['structured_result'].get('citations', []))}")
        print(f"置信度：{result['structured_result'].get('confidence')}")

    # 传统输出
    print("\n【传统模式】")
    answer = rag_traditional.rag_summarize(query)
    print(f"回答：{answer}")

    print("\n")