"""
对比混合检索和传统向量检索的效果
使用 Ragas 指标量化提升幅度
"""
import json
import time
from pathlib import Path
from typing import Dict, List
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_relevancy
from rag.rag_service import RagSummarizeService


class HybridVsTraditionalComparator:
    """对比混合检索和传统检索"""

    def __init__(self, dataset_path: str = "data/eval_dataset.json"):
        self.dataset_path = Path(dataset_path)
        self.results = {}

    def load_questions(self, max_samples: int = None) -> List[str]:
        """加载测试问题"""
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        questions = [item['query'] for item in data]

        if max_samples:
            questions = questions[:max_samples]

        return questions

    def evaluate_method(
        self,
        method_name: str,
        questions: List[str],
        use_hybrid: bool
    ) -> Dict:
        """
        评估特定检索方法

        Args:
            method_name: 方法名称
            questions: 问题列表
            use_hybrid: 是否使用混合检索
        """
        print(f"\n{'='*70}")
        print(f"🧪 测试方法: {method_name}")
        print(f"{'='*70}")
        print(f"混合检索: {'是' if use_hybrid else '否'}")

        # 初始化 RAG（注意：实际需要在 RagSummarizeService 中支持切换）
        rag = RagSummarizeService(
            use_langgraph=True,
            enable_structured_output=False
        )

        # 如果不需要混合检索，可以临时禁用
        if not use_hybrid and hasattr(rag.vector_store, 'hybrid_enabled'):
            original_hybrid = rag.vector_store.hybrid_enabled
            rag.vector_store.hybrid_enabled = False

        answers = []
        contexts = []

        for i, question in enumerate(questions, 1):
            print(f"  [{i}/{len(questions)}] {question[:50]}...", end=" ")

            try:
                start_time = time.time()
                result = rag.rag_summarize_with_history(
                    question,
                    thread_id=f"compare_{method_name}_{i}"
                )
                elapsed = time.time() - start_time

                answers.append(result.get('answer', ''))
                contexts.append([result.get('context', '')])

                retrieval_method = result.get('retrieval_metadata', {}).get('method', 'unknown')
                print(f"✅ ({elapsed:.2f}s, {retrieval_method})")

            except Exception as e:
                print(f"❌ {e}")
                answers.append("")
                contexts.append([""])

        # 恢复设置
        if not use_hybrid and hasattr(rag.vector_store, 'hybrid_enabled'):
            rag.vector_store.hybrid_enabled = original_hybrid

        # 构建数据集并评估
        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": [""] * len(questions)
        })

        print(f"\n  📊 计算 Ragas 指标...")
        start_time = time.time()

        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_relevancy]
        )

        elapsed = time.time() - start_time

        # 提取结果
        method_results = {
            "method": method_name,
            "use_hybrid": use_hybrid,
            "samples": len(questions),
            "elapsed_time": elapsed,
            "metrics": {}
        }

        for metric_name in ['faithfulness', 'answer_relevancy', 'context_relevancy']:
            scores = result[metric_name]
            valid_scores = [s for s in scores if s is not None]
            if valid_scores:
                method_results["metrics"][metric_name] = {
                    "average": sum(valid_scores) / len(valid_scores),
                    "scores": valid_scores
                }

        # 打印结果
        print(f"\n  📈 {method_name} 结果:")
        for metric, data in method_results['metrics'].items():
            print(f"     {metric:<25}: {data['average']:.3f}")
        print(f"     总耗时: {elapsed:.2f}秒")

        self.results[method_name] = method_results
        return method_results

    def compare(self, max_samples: int = 20):
        """执行对比实验"""
        questions = self.load_questions(max_samples)
        print(f"\n📋 加载了 {len(questions)} 个测试问题")

        # 评估传统检索
        self.evaluate_method(
            "traditional_vector",
            questions,
            use_hybrid=False
        )

        # 评估混合检索
        self.evaluate_method(
            "hybrid_search",
            questions,
            use_hybrid=True
        )

        # 生成对比报告
        self._generate_comparison_report()

    def _generate_comparison_report(self):
        """生成对比报告"""
        print(f"\n{'='*70}")
        print("📊 混合检索 vs 传统检索 对比报告")
        print(f"{'='*70}")

        if len(self.results) < 2:
            print("⚠️  需要两个版本的结果才能对比")
            return

        methods = list(self.results.keys())
        baseline = methods[0]
        optimized = methods[1]

        print(f"\n{'指标':<25} {'传统检索':>12} {'混合检索':>12} {'提升':>10}")
        print("-" * 70)

        metrics_to_compare = ['faithfulness', 'answer_relevancy', 'context_relevancy']

        improvements = {}

        for metric in metrics_to_compare:
            baseline_score = self.results[baseline]['metrics'][metric]['average']
            optimized_score = self.results[optimized]['metrics'][metric]['average']

            improvement = ((optimized_score - baseline_score) / baseline_score * 100) if baseline_score > 0 else 0
            improvements[metric] = improvement

            print(f"{metric:<25} {baseline_score:>12.3f} {optimized_score:>12.3f} {improvement:>+9.1f}%")

        print("-" * 70)

        # 总结
        avg_improvement = sum(improvements.values()) / len(improvements)
        print(f"\n平均提升: {avg_improvement:+.1f}%")

        if avg_improvement > 10:
            print("🎉 混合检索效果显著，建议在生产环境使用！")
        elif avg_improvement > 5:
            print("👍 混合检索有一定提升，可以考虑使用")
        else:
            print("⚠️  提升不明显，需要进一步优化")

        # 保存报告
        self._save_report()

    def _save_report(self):
        """保存对比报告"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"eval_results/hybrid_vs_traditional_{timestamp}.json")
        output_path.parent.mkdir(exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

        print(f"\n💾 对比报告已保存到: {output_path}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='混合检索 vs 传统检索对比')
    parser.add_argument('--samples', type=int, default=20,
                        help='测试样本数量（默认20）')

    args = parser.parse_args()

    print("=" * 70)
    print("🔄 混合检索 vs 传统检索 对比实验")
    print("=" * 70)
    print(f"\n将使用 {args.samples} 个问题进行对比测试")
    print("预计耗时: 5-10 分钟\n")

    comparator = HybridVsTraditionalComparator()
    comparator.compare(max_samples=args.samples)

    print("\n" + "=" * 70)
    print("✅ 对比实验完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()
