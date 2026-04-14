"""
测试混合检索效果
对比传统向量检索和混合检索的差异
"""
import os
import json
from typing import List, Dict
from rag.vector_store import VectorStoreService
from rag.rag_service import RagSummarizeService
from utils.logger_handler import logger
from qa_generator import QAGenerator


def evaluate_retrieval_quality(
    test_samples: List[Dict],
    retriever,
    retriever_name: str = "Retriever",
    top_k: int = 5
) -> Dict[str, float]:
    """
    评估检索质量

    Args:
        test_samples: 测试样本列表，每个样本包含 'query' 和 'relevant_ids'
        retriever: 检索器实例
        retriever_name: 检索器名称（用于打印）
        top_k: 检索返回的文档数量

    Returns:
        包含各项指标的字典
    """
    print(f"\n{'='*80}")
    print(f"开始评估 {retriever_name} (top_k={top_k})")
    print(f"{'='*80}")

    hits = 0
    reciprocal_ranks = []
    total_precision = 0
    total_recall = 0

    for idx, sample in enumerate(test_samples, 1):
        query = sample['query']
        relevant_ids = sample['relevant_ids']

        # 执行检索
        retrieved_docs = retriever.invoke(query) if hasattr(retriever, 'invoke') else retriever.hybrid_search(query, top_k_rerank=top_k)

        # 提取检索结果的文件名
        retrieved_basenames = [os.path.basename(doc.metadata.get('source', '')) for doc in retrieved_docs[:top_k]]

        # 计算 Hit Rate@k
        is_hit = any(rid in retrieved_basenames for rid in relevant_ids)
        if is_hit:
            hits += 1

        # 计算 MRR (Mean Reciprocal Rank)
        first_rank = None
        for rank, basename in enumerate(retrieved_basenames, 1):
            if basename in relevant_ids:
                first_rank = rank
                break

        if first_rank:
            reciprocal_ranks.append(1.0 / first_rank)
        else:
            reciprocal_ranks.append(0.0)

        # 计算 Precision@k 和 Recall@k
        hits_in_top_k = sum(1 for rid in relevant_ids if rid in retrieved_basenames)
        precision = hits_in_top_k / len(retrieved_basenames) if retrieved_basenames else 0
        recall = hits_in_top_k / len(relevant_ids) if relevant_ids else 0

        total_precision += precision
        total_recall += recall

        # 打印详细信息（前10个样本）
        if idx <= 10:
            print(f"[{idx}/{len(test_samples)}] 问题: {query}")
            print(f"   检索结果: {retrieved_basenames}")
            print(f"   目标文件: {relevant_ids}")
            print(f"   命中: {'✓' if is_hit else '✗'}, 排名: {first_rank or '未找到'}")
            print(f"   P@{top_k}: {precision:.2f}, R@{top_k}: {recall:.2f}")

    # 计算平均指标
    n = len(test_samples)
    metrics = {
        'hit_rate': hits / n,
        'mrr': sum(reciprocal_ranks) / n,
        'precision': total_precision / n,
        'recall': total_recall / n,
    }

    print(f"\n{'='*80}")
    print(f"{retriever_name} 评估结果 (基于{n}个样本, top_k={top_k}):")
    print(f"  Hit Rate@{top_k}:  {metrics['hit_rate']:.2%}")
    print(f"  MRR@{top_k}:       {metrics['mrr']:.4f}")
    print(f"  Precision@{top_k}: {metrics['precision']:.2%}")
    print(f"  Recall@{top_k}:    {metrics['recall']:.2%}")
    print(f"{'='*80}")

    return metrics


def compare_retrievers(dataset_size=20, top_k=5):
    """
    对比传统向量检索和混合检索的效果

    Args:
        dataset_size: 用于测试的问答对数量
        top_k: 检索返回的文档数量
    """
    print("\n" + "=" * 80)
    print("检索器对比评测")
    print("=" * 80)

    # 加载评测集
    dataset = QAGenerator.load_dataset("data/eval_dataset.json")

    if not dataset:
        print("❌ 未找到评测集，请先运行 test_qa_generation()")
        return

    # 限制测试数量
    test_samples = dataset[:dataset_size]
    print(f"\n使用 {len(test_samples)} 个问题进行评测...\n")

    # 初始化向量存储
    vs = VectorStoreService()
    vs.load_documents()

    results = {}

    # 1. 评估传统向量检索
    traditional_retriever = vs.vector_store.as_retriever(search_kwargs={"k": top_k})
    results['traditional'] = evaluate_retrieval_quality(
        test_samples,
        traditional_retriever,
        "传统向量检索",
        top_k
    )

    # 2. 评估混合检索（如果启用）
    if vs.hybrid_enabled and vs.hybrid_retriever:
        results['hybrid'] = evaluate_retrieval_quality(
            test_samples,
            vs.hybrid_retriever,
            "混合检索 (向量+BM25+重排序)",
            top_k
        )

        # 3. 对比分析
        print(f"\n{'='*80}")
        print("📊 对比分析")
        print(f"{'='*80}")
        print(f"{'指标':<20} {'传统检索':>12} {'混合检索':>12} {'提升':>12}")
        print(f"{'-'*60}")

        for metric_name in ['hit_rate', 'mrr', 'precision', 'recall']:
            trad_val = results['traditional'][metric_name]
            hyb_val = results['hybrid'][metric_name]
            improvement = ((hyb_val - trad_val) / trad_val * 100) if trad_val > 0 else 0

            metric_display = {
                'hit_rate': f'Hit Rate@{top_k}',
                'mrr': f'MRR@{top_k}',
                'precision': f'Precision@{top_k}',
                'recall': f'Recall@{top_k}'
            }

            symbol = "↑" if improvement > 0 else ("↓" if improvement < 0 else "→")
            print(f"{metric_display[metric_name]:<20} {trad_val:>11.2%} {hyb_val:>11.2%} {symbol}{abs(improvement):>9.1f}%")

        print(f"{'='*80}")

        # 给出建议
        if results['hybrid']['hit_rate'] > results['traditional']['hit_rate']:
            print("✅ 混合检索效果更好，建议在生产环境使用！")
        else:
            print("⚠️  混合检索效果不如预期，可能需要调整参数（alpha、top_k等）")
    else:
        print("\n⚠️  混合检索未启用，无法对比")

    return results


def test_hybrid_retrieval():
    """测试混合检索功能"""
    print("=" * 80)
    print("测试混合检索 vs 传统向量检索")
    print("=" * 80)

    # 初始化向量存储服务
    vs = VectorStoreService()
    vs.load_documents()

    # 测试查询
    test_queries = [
        "小户型适合哪些扫地机器人？",
        "扫地机器人如何避障？",
        "电池续航和充电时间",
    ]

    for query in test_queries:
        print(f"\n{'=' * 80}")
        print(f"查询: {query}")
        print(f"{'=' * 80}")

        # 1. 传统向量检索
        print("\n【传统向量检索结果】")
        traditional_retriever = vs.vector_store.as_retriever(search_kwargs={"k": 3})
        traditional_docs = traditional_retriever.invoke(query)

        for i, doc in enumerate(traditional_docs, 1):
            source = doc.metadata.get('source', '未知')
            content_preview = doc.page_content[:100].replace('\n', ' ')
            print(f"{i}. [{source}] {content_preview}...")

        # 2. 混合检索（如果启用）
        if vs.hybrid_enabled and vs.hybrid_retriever:
            print(f"\n【混合检索结果】")
            hybrid_docs = vs.hybrid_retriever.hybrid_search(
                query=query,
                top_k_retrieve=10,
                top_k_rerank=3,
                alpha=0.5
            )

            for i, doc in enumerate(hybrid_docs, 1):
                source = doc.metadata.get('source', '未知')
                content_preview = doc.page_content[:100].replace('\n', ' ')
                print(f"{i}. [{source}] {content_preview}...")
        else:
            print("\n【混合检索未启用】")

    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)


def test_rag_with_hybrid():
    """测试 RAG 服务使用混合检索"""
    print("\n" + "=" * 80)
    print("测试 RAG 服务（混合检索）")
    print("=" * 80)

    rag = RagSummarizeService(use_langgraph=True)

    query = "小户型适合哪些扫地机器人？"
    print(f"\n查询: {query}\n")

    result = rag.rag_summarize_with_history(query, thread_id="test_user")

    print(f"答案:\n{result['answer']}")
    print(f"\n使用的参考资料数量: {len(result['context'].split('[参考资料：')) - 1}")
    print(f"对话历史长度: {len(result['messages'])}")


def test_qa_generation():
    """测试问答对自动生成功能"""
    print("\n" + "=" * 80)
    print("测试问答对自动生成")
    print("=" * 80)

    # 初始化生成器
    generator = QAGenerator()

    # 从文件生成评测集
    print("\n开始从文档生成问答对...")
    dataset = generator.build_eval_dataset_from_files(
        num_questions=3,
        output_path="data/eval_dataset.json"
    )

    print(f"\n✓ 成功生成 {len(dataset)} 个问答对")

    if dataset:
        print("\n" + "-" * 80)
        print("前10个问答对示例:")
        print("-" * 80)
        for i, item in enumerate(dataset[:10], 1):
            print(f"\n{i}. 问题: {item['query']}")
            print(f"   来源: {item['source']}")
            print(f"   相关文档ID: {item['relevant_ids']}")

    print("\n" + "=" * 80)
    print("问答对生成测试完成！")
    print("=" * 80)

    return dataset


def test_qa_evaluation(dataset_size=10):
    """
    使用生成的问答对进行检索质量评估（旧版本，保留兼容）

    Args:
        dataset_size: 用于测试的问答对数量
    """
    print("\n" + "=" * 80)
    print("使用生成的问答对进行检索质量评估")
    print("=" * 80)

    # 加载评测集
    dataset = QAGenerator.load_dataset("data/eval_dataset.json")

    if not dataset:
        print("未找到评测集，请先运行 test_qa_generation()")
        return

    # 限制测试数量
    test_samples = dataset[:dataset_size]

    # 初始化向量存储
    vs = VectorStoreService()
    vs.load_documents()

    print(f"\n使用 {len(test_samples)} 个问题进行检索测试...\n")

    # 统计指标
    total_precision = 0
    total_recall = 0

    for idx, sample in enumerate(test_samples, 1):
        query = sample['query']
        relevant_ids = sample['relevant_ids']  # 现在是文件名

        print(f"[{idx}/{len(test_samples)}] 问题: {query}")

        # 执行检索
        retriever = vs.get_retriever()
        retrieved_docs = retriever.invoke(query)

        # 优化：提取检索结果的文件名进行对比
        retrieved_basenames = [os.path.basename(doc.metadata.get('source', '')) for doc in retrieved_docs]

        # 计算命中率
        hits = sum(1 for rid in relevant_ids if rid in retrieved_basenames)

        precision = hits / len(retrieved_docs) if retrieved_docs else 0
        recall = hits / len(relevant_ids) if relevant_ids else 0

        total_precision += precision
        total_recall += recall

        print(f"   检索到: {retrieved_basenames}")
        print(f"   目标文件: {relevant_ids}")
        print(f"   命中: {hits}/{len(relevant_ids)}, 精确率: {precision:.2f}, 召回率: {recall:.2f}")

    # 计算平均指标
    avg_precision = total_precision / len(test_samples)
    avg_recall = total_recall / len(test_samples)

    print("\n" + "=" * 80)
    print(f"评估结果 (基于{len(test_samples)}个样本):")
    print(f"  平均精确率: {avg_precision:.2%}")
    print(f"  平均召回率: {avg_recall:.2%}")
    print("=" * 80)


if __name__ == '__main__':
    import sys

    # 默认运行所有测试
    if len(sys.argv) == 1:
        # 测试混合检索
        test_hybrid_retrieval()

        # 测试 RAG 服务
        test_rag_with_hybrid()

        # 测试问答对生成
        test_qa_generation()

        # 🔥 新增：对比评测（核心功能）
        compare_retrievers(dataset_size=20, top_k=5)

    else:
        # 根据命令行参数运行特定测试
        test_name = sys.argv[1]

        if test_name == "hybrid":
            test_hybrid_retrieval()
        elif test_name == "rag":
            test_rag_with_hybrid()
        elif test_name == "qa_gen":
            test_qa_generation()
        elif test_name == "qa_eval":
            dataset_size = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            test_qa_evaluation(dataset_size)
        elif test_name == "compare":
            # 🔥 新增：对比评测命令
            dataset_size = int(sys.argv[2]) if len(sys.argv) > 2 else 20
            top_k = int(sys.argv[3]) if len(sys.argv) > 3 else 5
            compare_retrievers(dataset_size, top_k)
        else:
            print(f"未知测试: {test_name}")
            print("可用测试: hybrid, rag, qa_gen, qa_eval, compare")
