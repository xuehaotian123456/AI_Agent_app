"""
基于现有 eval_dataset.json 的 Ragas 自动化评估（适配 ragas>=0.2.0）
使用项目中已有的 39 个问题进行检索质量评估
使用 ContextPrecision / ContextRecall 指标（需要 LLM 作为裁判）

依赖版本：
- ragas (建议 >=0.2.0，已测试 0.4.3)
- langchain-core
- openai
- 使用项目配置的 ChatTongyi (qwen3-max) 作为评估 LLM
  通过 OpenAI 兼容接口调用（通义千问支持 OpenAI SDK）
"""
import json
import time
import os
from pathlib import Path
from typing import List, Dict, Optional
from datasets import Dataset
from tqdm import tqdm

# 导入新版 Ragas API
try:
    from ragas import evaluate
    from ragas.metrics import ContextPrecision, ContextRecall, AnswerRelevancy
    from ragas.llms import llm_factory
    from ragas.embeddings import LangchainEmbeddingsWrapper

except ImportError as e:
    print(f"❌ Ragas 导入失败: {e}")
    print("\n💡 解决方案:")
    print("   pip install ragas tqdm")
    raise

# 导入项目配置（请根据实际项目调整导入路径）
try:
    from utils.config_handler import rag_conf
    from rag.rag_service import RagSummarizeService
    from utils.logger_handler import logger
except ImportError:
    # 如果导入失败，提供模拟实现以便独立测试
    print("⚠️ 未找到项目模块，使用模拟配置。")
    class FakeConfig:
        def get(self, key, default=None):
            return "qwen-plus"
    rag_conf = FakeConfig()
    logger = type('Logger', (), {'info': print, 'warning': print, 'error': print})()
    RagSummarizeService = None


def create_ragas_eval_llm():
    """
    创建 Ragas 评估用 LLM（使用 llm_factory + 环境变量）

    通义千问支持 OpenAI 兼容协议，只需设置环境变量即可通过 OpenAI 协议调用。
    这是经过验证的、最简单可靠的通义千问接入方式。
    """
    # 1. 获取 API Key（优先从环境变量获取，也可从配置文件读取）
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")

    if not api_key:
        logger.warning("未找到 DASHSCOPE_API_KEY 或 OPENAI_API_KEY 环境变量")
        logger.warning("请设置环境变量: export DASHSCOPE_API_KEY=your-api-key")
        raise ValueError("缺少 API Key 配置")

    # 2. 获取模型名称（例如 qwen-plus, qwen-max, qwen3-max）
    model_name = rag_conf.get("chat_model_name", "qwen-plus")

    # 3. ✅ 核心：设置 OpenAI 兼容接口所需的环境变量
    #    Ragas 内部的 llm_factory 会读取这些变量并使用 OpenAI 客户端
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    logger.info(f"初始化 Ragas 评估 LLM: {model_name} (通过环境变量 OpenAI 兼容接口)")

    # 4. ✅ 直接调用 llm_factory，无需手动创建 client
    #    这是经过 test_tongyi.py 验证的成功方式
    eval_llm = llm_factory(model=model_name)

    logger.info("✅ Ragas 评估 LLM 初始化成功")
    return eval_llm





class ExistingDatasetEvaluator:
    """使用现有 eval_dataset.json 进行 Ragas 评估"""

    def __init__(
        self,
        dataset_path: str = "data/eval_dataset.json",
        prepared_dataset_path: str = "data/ragas_test_dataset.json",
        output_dir: str = "eval_results",
        use_references: bool = False
    ):
        self.dataset_path = Path(dataset_path)
        self.prepared_dataset_path = Path(prepared_dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.use_references = use_references

        # 初始化评估用 LLM
        self.eval_llm = create_ragas_eval_llm()

    def prepare_dataset_if_needed(self, chunk_size: int = 500) -> bool:
        """如果预处理数据集不存在，则生成"""
        if not self.prepared_dataset_path.exists():
            print("\n📝 预处理数据集不存在，正在生成...")
            try:
                from scripts.prepare_ragas_dataset import prepare_ragas_dataset
                count = prepare_ragas_dataset(
                    eval_json_path=str(self.dataset_path),
                    output_json_path=str(self.prepared_dataset_path),
                    chunk_size=chunk_size
                )
                return count > 0
            except ImportError:
                print("❌ 无法导入 prepare_ragas_dataset，请确保该模块存在")
                return False
        return True

    def load_prepared_questions(self) -> List[Dict]:
        """加载预处理后的测试问题"""
        if not self.prepared_dataset_path.exists():
            raise FileNotFoundError(f"预处理数据集不存在: {self.prepared_dataset_path}")

        with open(self.prepared_dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"✅ 加载了 {len(data)} 个测试问题（含 ground_truth_contexts）")
        return data

    def generate_answers_and_contexts(
        self,
        questions_data: List[Dict],
        max_samples: Optional[int] = None
    ) -> tuple:
        """
        为问题生成答案和检索上下文
        Returns:
            (questions, answers, contexts, ground_truth_contexts, references)
        """
        if RagSummarizeService is None:
            raise ImportError("RagSummarizeService 未正确导入，请检查项目结构")

        if max_samples:
            questions_data = questions_data[:max_samples]
            print(f"⚡ 快速模式：仅使用前 {max_samples} 个问题")

        rag = RagSummarizeService(
            use_langgraph=True,
            enable_structured_output=False
        )

        questions = []
        answers = []
        contexts = []
        ground_truth_contexts = []
        references = []

        print(f"\n🔄 正在为 {len(questions_data)} 个问题生成答案...")

        for i, item in enumerate(tqdm(questions_data, desc="生成 RAG 答案")):
            question = item['question']
            gt_contexts = item.get('ground_truth_contexts', [])
            reference = item.get('reference', '')

            # ✅ 核心修复：限制 ground_truth_contexts 数量
            MAX_GT_CONTEXTS = 3
            if len(gt_contexts) > MAX_GT_CONTEXTS:
                logger.info(f"问题 {i+1}: ground_truth_contexts 从 {len(gt_contexts)} 个截断为 {MAX_GT_CONTEXTS} 个")
                gt_contexts = gt_contexts[:MAX_GT_CONTEXTS]

            try:
                start_time = time.time()
                result = rag.rag_summarize_with_history(
                    question,
                    thread_id=f"eval_{i}"
                )
                elapsed = time.time() - start_time

                answer = result.get('answer', '')
                context_str = result.get('context', '')

                questions.append(question)

                # ✅ 确保 answer 非空
                if not answer or not answer.strip():
                    answer = "抱歉，无法生成答案。"
                answers.append(answer)

                # ✅ 核心修复：正确解析 context 字符串为文档列表
                context_list = []
                if context_str and context_str.strip():
                    # 按 "[参考资料：" 分割成多个文档
                    parts = context_str.split('[参考资料：')

                    for part in parts[1:]:  # 跳过第一个空元素
                        # 提取 "[参考资料]xxx |" 中的 xxx 部分
                        if '[参考资料]' in part and '|' in part:
                            start_idx = part.find('[参考资料]') + len('[参考资料]')
                            end_idx = part.find('|', start_idx)
                            if start_idx > 0 and end_idx > start_idx:
                                content = part[start_idx:end_idx].strip()
                                if content:
                                    context_list.append(content)

                    # 如果解析失败，降级使用整个字符串
                    if not context_list:
                        context_list = [context_str.strip()]

                # ✅ 确保 context_list 非空且每个元素都是非空字符串
                if not context_list:
                    context_list = [""]
                else:
                    # 过滤掉空字符串
                    context_list = [ctx for ctx in context_list if ctx and ctx.strip()]
                    if not context_list:
                        context_list = [""]

                contexts.append(context_list)

                # ✅ 确保 ground_truth_contexts 格式正确（保持嵌套结构）
                if not gt_contexts:
                    gt_contexts_processed = [""]
                else:
                    # 过滤掉空字符串
                    gt_contexts_filtered = [ctx for ctx in gt_contexts if ctx and ctx.strip()]
                    if not gt_contexts_filtered:
                        gt_contexts_processed = [""]
                    else:
                        # 截断过长的 ground_truth_contexts
                        gt_contexts_processed = self._truncate_contexts(gt_contexts_filtered, max_total_length=4000)
                        if not gt_contexts_processed:
                            gt_contexts_processed = [""]

                ground_truth_contexts.append(gt_contexts_processed)

                # 添加 reference
                if not reference or not reference.strip():
                    reference = ""
                references.append(reference)

                retrieval_method = result.get('retrieval_metadata', {}).get('method', 'unknown')

                # 调试：打印第一个样本
                if i == 0:
                    print(f"\n🔍 [调试] 第一个样本:")
                    print(f"   question: {question[:50]}...")
                    print(f"   answer 长度: {len(answer)}")
                    print(f"   context 原始长度: {len(context_str)}")
                    print(f"   解析后 contexts 数量: {len(context_list)}")
                    if context_list:
                        print(f"   第一个 context 长度: {len(context_list[0])}")
                        print(f"   第一个 context 前100字符: {context_list[0][:100]}")
                    print(f"   ground_truth_contexts 数量: {len(gt_contexts_processed)}")
                    if gt_contexts_processed:
                        print(f"   第一个 GT 长度: {len(gt_contexts_processed[0])}")
                    print(f"   reference 长度: {len(reference)}")

            except Exception as e:
                logger.error(f"处理问题失败: {question}, 错误: {e}")
                questions.append(question)
                answers.append("抱歉，处理时出现错误。")
                contexts.append([""])

                # 确保异常情况下也有有效数据
                if not gt_contexts:
                    gt_contexts_processed = [""]
                else:
                    gt_contexts_filtered = [ctx for ctx in gt_contexts if ctx and ctx.strip()]
                    if not gt_contexts_filtered:
                        gt_contexts_processed = [""]
                    else:
                        gt_contexts_processed = self._truncate_contexts(gt_contexts_filtered, max_total_length=4000)
                        if not gt_contexts_processed:
                            gt_contexts_processed = [""]

                ground_truth_contexts.append(gt_contexts_processed)
                references.append(reference if reference else "")

            print(f"\n✅ 成功生成 {len(questions)} 个样本的答案")

            # ✅ 最终验证：确保所有列表长度一致且非空
            assert len(questions) == len(answers) == len(contexts) == len(ground_truth_contexts) == len(references), \
                f"数据长度不一致: {len(questions)}, {len(answers)}, {len(contexts)}, {len(ground_truth_contexts)}, {len(references)}"

            # 检查是否有空的 contexts
            empty_contexts = sum(1 for ctx in contexts if ctx == [""] or ctx == [])
            if empty_contexts > 0:
                logger.warning(f"⚠️  有 {empty_contexts} 个样本的 contexts 为空")

            return questions, answers, contexts, ground_truth_contexts, references

    def _truncate_contexts(self, contexts: List[str], max_total_length: int = 4000) -> List[str]:
        """
        截断上下文列表以避免超出 LLM 上下文窗口

        Args:
            contexts: 上下文列表
            max_total_length: 最大总长度

        Returns:
            截断后的上下文列表
        """
        if not contexts:
            return contexts

        total_length = sum(len(ctx) for ctx in contexts)
        if total_length <= max_total_length:
            return contexts

        # 按比例截断每个上下文
        truncated = []
        remaining_length = max_total_length

        for ctx in contexts:
            if remaining_length <= 0:
                break

            if len(ctx) <= remaining_length:
                truncated.append(ctx)
                remaining_length -= len(ctx)
            else:
                # 截断当前上下文
                truncated_ctx = ctx[:remaining_length - 50] + "...(截断)"
                truncated.append(truncated_ctx)
                remaining_length = 0
                break

        return truncated if truncated else [""]

    def run_evaluation(
            self,
            max_samples: Optional[int] = None,
            save_results: bool = True
    ) -> Dict:
        """执行完整评估流程"""
        # ✅ 修复 1：创建指标（Ragas 0.3.5 不需要在初始化时传入 embeddings）
        context_precision_metric = ContextPrecision(llm=self.eval_llm)
        context_recall_metric = ContextRecall(llm=self.eval_llm)

        metrics_to_use = [context_precision_metric, context_recall_metric]
        metric_names = ['context_precision', 'context_recall']

        if self.use_references:
            try:
                answer_relevancy_metric = AnswerRelevancy(llm=self.eval_llm)
                metrics_to_use.append(answer_relevancy_metric)
                metric_names.append('answer_relevancy')
                print("✅ 已启用 Answer Relevancy 指标(需要 reference)")
            except Exception as e:
                logger.warning(f"无法启用 AnswerRelevancy: {e}")

        print("\n" + "=" * 70)
        print("🔍 基于现有数据集的 Ragas 评估 (ragas>=0.2.0)")
        print("=" * 70)
        print(f"📊 评估指标 ({len(metric_names)}个): {', '.join(metric_names)}")
        print(f"🤖 评估 LLM: {rag_conf.get('chat_model_name', 'unknown')}")
        if self.use_references:
            print("📝 使用标准答案 (reference) 进行评估")

        # Step 1: 准备数据集
        print("\n📂 Step 1: 准备测试数据集...")
        self.prepare_dataset_if_needed()

        # Step 2: 加载问题
        print("\n📋 Step 2: 加载测试问题...")
        questions_data = self.load_prepared_questions()

        # Step 3: 生成答案
        print("\n📝 Step 3: 生成答案和上下文...")
        questions, answers, contexts, gt_contexts, references = self.generate_answers_and_contexts(
            questions_data,
            max_samples=max_samples
        )

        # Step 4: 构建 Dataset
        print("\n📊 Step 4: 构建评估数据集...")
        dataset_dict = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth_contexts": gt_contexts
        }

        # Ragas 0.3.5+ 要求必须有 reference 字段
        if self.use_references:
            dataset_dict["reference"] = [ref if ref else "" for ref in references]
            valid_ref_count = sum(1 for ref in references if ref)
            if valid_ref_count > 0:
                print(f"✅ 添加了 {valid_ref_count} 个标准答案")
            else:
                print("⚠️  reference 字段已添加，但内容为空")
        else:
            dataset_dict["reference"] = [ref if ref else "" for ref in references]
            print("⚠️  reference 字段为空（未启用 use_references）")

        dataset = Dataset.from_dict(dataset_dict)

        # ✅ 调试：保存数据集到文件，检查格式
        df = dataset.to_pandas()
        debug_file = self.output_dir / "debug_dataset_format.csv"
        df.to_csv(debug_file, index=False, encoding='utf-8-sig')
        print(f"   📝 数据集已保存到: {debug_file}")
        print(f"   📊 数据集形状: {df.shape}")
        print(f"   📋 列名: {df.columns.tolist()}")
        print(f"\n   🔍 第一个样本详情:")
        for col in df.columns:
            val = df.iloc[0][col]
            if isinstance(val, list):
                print(f"      {col}: list[{len(val)}], 元素类型={[type(v).__name__ for v in val[:2]]}")
            else:
                print(f"      {col}: {type(val).__name__}, 值={str(val)[:100]}")

        # Step 5: 执行评估
        print("\n⚙️  Step 5: 计算 Ragas 指标...")
        print("   ⏳ 这可能需要几分钟时间（LLM 评估较慢）...")
        start_time = time.time()

        try:
            # ✅ 修复：移除 max_workers 参数（Ragas 0.3.5 不支持）
            result = evaluate(
                dataset=dataset,
                metrics=metrics_to_use,
                raise_exceptions=False  # 不因为单个样本失败而中断
            )
            elapsed_time = time.time() - start_time
            print(f"   ✅ 评估完成！耗时: {elapsed_time:.2f}秒")

            # 处理结果
            result_dict = self._process_results(result, dataset, elapsed_time, metric_names)

            # 打印报告
            self._print_report(result_dict)

            # 保存结果
            if save_results:
                self._save_results(result_dict, dataset)

            return result_dict

        except Exception as e:
            logger.error(f"评估失败: {e}", exc_info=True)
            raise

    def _process_results(self, result, dataset: Dataset, elapsed_time: float, metric_names: List[str]) -> Dict:
        """处理评估结果"""
        result_dict = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_size": len(dataset),
            "elapsed_time": elapsed_time,
            "metrics": {}
        }

        # Ragas 0.3.5+ 的 result 是 EvaluationResult 对象
        try:
            print(f"📊 评估结果类型: {type(result)}")
            print(f"📊 result 属性: {dir(result)}")

            # 尝试不同的访问方式
            if hasattr(result, 'scores'):
                scores_data = result.scores
                print(f"📊 scores 类型: {type(scores_data)}")

                # 如果是 list，直接处理
                if isinstance(scores_data, list):
                    print(f"📊 scores 长度: {len(scores_data)}")
                    if len(scores_data) > 0:
                        print(f"📊 第一个元素类型: {type(scores_data[0])}")
                        print(f"📊 第一个元素内容: {scores_data[0]}")

                    # list 中每个元素是一个字典，包含所有指标的分数
                    for metric_name in metric_names:
                        valid_scores = []
                        for sample_scores in scores_data:
                            if isinstance(sample_scores, dict) and metric_name in sample_scores:
                                score = sample_scores[metric_name]
                                if score is not None and not (isinstance(score, float) and score != score):
                                    valid_scores.append(score)

                        if valid_scores:
                            result_dict["metrics"][metric_name] = {
                                "average": sum(valid_scores) / len(valid_scores),
                                "min": min(valid_scores),
                                "max": max(valid_scores),
                                "scores": valid_scores
                            }
                            print(
                                f"✅ {metric_name}: 平均分={result_dict['metrics'][metric_name]['average']:.3f} ({len(valid_scores)}个样本)")
                        else:
                            logger.warning(f"⚠️  {metric_name} 没有有效分数")

                # 如果是 DataFrame
                elif hasattr(scores_data, 'columns'):
                    print(f"📊 分数 DataFrame 列名: {scores_data.columns.tolist()}")
                    for metric_name in metric_names:
                        if metric_name in scores_data.columns:
                            scores = scores_data[metric_name].dropna().tolist()
                            valid_scores = [
                                s for s in scores
                                if s is not None and not (isinstance(s, float) and s != s)
                            ]

                            if valid_scores:
                                result_dict["metrics"][metric_name] = {
                                    "average": sum(valid_scores) / len(valid_scores),
                                    "min": min(valid_scores),
                                    "max": max(valid_scores),
                                    "scores": valid_scores
                                }
                                print(f"✅ {metric_name}: 平均分={result_dict['metrics'][metric_name]['average']:.3f}")
                        else:
                            logger.warning(f"⚠️  指标 {metric_name} 不在结果中")
                else:
                    logger.error(f"❌ 未知的 scores 类型: {type(scores_data)}")
            else:
                logger.error("❌ result 对象没有 scores 属性")
                logger.error(f"result 内容: {result}")

        except Exception as e:
            logger.error(f"处理结果失败: {e}", exc_info=True)
            raise

        return result_dict

    def _print_report(self, results: Dict):
        """打印评估报告"""
        print("\n" + "=" * 70)
        print("📊 Ragas 综合评估报告")
        print("=" * 70)

        metrics_display = {
            'context_precision': '上下文精确率 (Context Precision)',
            'context_recall': '上下文召回率 (Context Recall)',
            'answer_relevancy': '答案相关性 (Answer Relevancy)'
        }

        print(f"\n{'指标':<40} {'平均分':>8} {'最低分':>8} {'最高分':>8}")
        print("-" * 70)

        for metric_key, display_name in metrics_display.items():
            if metric_key in results['metrics']:
                data = results['metrics'][metric_key]
                print(f"{display_name:<40} {data['average']:>8.3f} {data['min']:>8.3f} {data['max']:>8.3f}")

        print("-" * 70)
        print(f"样本数量: {results['dataset_size']}")
        print(f"总耗时: {results['elapsed_time']:.2f}秒")

        available_metrics = list(results['metrics'].keys())
        if available_metrics:
            overall_score = sum(
                results['metrics'][m]['average'] for m in available_metrics
            ) / len(available_metrics)

            print(f"\n综合评分: {overall_score:.3f}/1.000")
            print(f"评估指标数: {len(available_metrics)}")

            if overall_score >= 0.8:
                print("🎉 评级: 优秀 - RAG 系统表现非常出色！")
            elif overall_score >= 0.6:
                print("👍 评级: 良好 - RAG 系统工作正常，仍有优化空间")
            elif overall_score >= 0.4:
                print("⚠️  评级: 一般 - 建议优化检索策略或 Prompt")
            else:
                print("❌ 评级: 较差 - 需要重点优化 RAG 流程")

            print(f"\n💡 详细分析:")
            if 'context_precision' in results['metrics']:
                score = results['metrics']['context_precision']['average']
                if score < 0.6:
                    print("   ⚠️  Context Precision 较低，建议优化重排序或调整 top_k")

            if 'context_recall' in results['metrics']:
                score = results['metrics']['context_recall']['average']
                if score < 0.6:
                    print("   ⚠️  Context Recall 较低，建议增加检索数量或优化 Embedding")

            if 'answer_relevancy' in results['metrics']:
                score = results['metrics']['answer_relevancy']['average']
                if score < 0.6:
                    print("   ⚠️  Answer Relevancy 较低，建议优化生成 Prompt 或增加参考信息")

    def _save_results(self, results: Dict, dataset: Dataset):
        """保存详细结果"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"ragas_eval_{timestamp}.json"

        detailed_results = []
        for i in range(len(dataset)):
            sample = {
                "question": dataset[i]['question'],
                "answer": dataset[i]['answer'],
                "scores": {}
            }

            for metric_key in results['metrics'].keys():
                if metric_key in results['metrics']:
                    scores_list = results['metrics'][metric_key]['scores']
                    if i < len(scores_list):
                        sample["scores"][metric_key] = scores_list[i]

            detailed_results.append(sample)

        output_data = {
            "summary": {
                "timestamp": results['timestamp'],
                "dataset_size": results['dataset_size'],
                "elapsed_time": results['elapsed_time'],
                "averages": {
                    k: v['average'] for k, v in results['metrics'].items()
                }
            },
            "detailed_results": detailed_results
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"\n💾 详细结果已保存到: {output_file}")

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='基于现有数据集的 Ragas 评估')
    parser.add_argument('--samples', type=int, default=None,
                        help='样本数量（默认全部）')
    parser.add_argument('--quick', action='store_true',
                        help='快速模式：仅评估前5个样本')
    parser.add_argument('--chunk-size', type=int, default=500,
                        help='文档分块大小（默认500）')
    parser.add_argument('--dataset', type=str,
                        default='data/ragas_test_dataset_with_references.json',
                        help='数据集路径（默认使用带 reference 的数据集）')

    args = parser.parse_args()

    evaluator = ExistingDatasetEvaluator(
        prepared_dataset_path=args.dataset,
        use_references=True
    )

    if args.quick:
        print("⚡ 快速模式启用")
        results = evaluator.run_evaluation(max_samples=5, save_results=True)
    else:
        results = evaluator.run_evaluation(
            max_samples=args.samples,
            save_results=True
        )

    return results


if __name__ == '__main__':
    main()
