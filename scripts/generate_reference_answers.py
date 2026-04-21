"""
为 ragas_test_dataset.json 生成标准答案 (reference)

此脚本读取现有的测试问题数据集,使用 LLM 基于 ground_truth_contexts 生成高质量的标准答案,
并将结果保存回 JSON 文件。

使用方法:
    python generate_reference_answers.py [--samples N] [--quick]
"""
import json
import time
import os
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

# 导入项目模块
try:
    from model.factory import robust_llm_caller
    from utils.logger_handler import logger
    from utils.config_handler import rag_conf
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请确保在项目根目录运行此脚本")
    raise


def generate_reference_answer(question: str, contexts: List[str], max_context_length: int = 3000) -> str:
    """
    基于问题和上下文生成标准答案

    Args:
        question: 用户问题
        contexts: 相关的文档片段列表
        max_context_length: 上下文最大长度(避免超出 LLM 限制)

    Returns:
        生成的标准答案
    """
    # 截断过长的上下文
    combined_context = "\n\n".join(contexts)
    if len(combined_context) > max_context_length:
        combined_context = combined_context[:max_context_length] + "\n...(内容过长已截断)"

    prompt = f"""你是一个专业的问答助手。请根据提供的参考文档,准确、简洁地回答用户的问题。

要求:
1. 答案必须基于参考文档,不要编造信息
2. 答案要简洁明了,直接回答问题核心
3. 如果参考文档中没有相关信息,请回答"参考文档中未找到相关信息"
4. 答案长度控制在 100-300 字之间

参考文档:
{combined_context}

用户问题: {question}

请给出标准答案:"""

    try:
        response = robust_llm_caller(prompt)
        answer = response.content.strip() if hasattr(response, 'content') else str(response).strip()
        return answer
    except Exception as e:
        logger.error(f"生成答案失败: {e}")
        return ""


def load_dataset(dataset_path: str) -> List[Dict]:
    """加载数据集"""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def save_dataset(data: List[Dict], output_path: str):
    """保存数据集"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✅ 数据已保存到: {output_path}")


def generate_references_for_dataset(
    dataset_path: str = "data/ragas_test_dataset.json",
    output_path: str = "data/ragas_test_dataset_with_references.json",
    max_samples: Optional[int] = None,
    batch_size: int = 5,
    delay_between_batches: float = 1.0
) -> int:
    """
    为数据集中的所有问题生成标准答案

    Args:
        dataset_path: 输入数据集路径
        output_path: 输出数据集路径
        max_samples: 最大样本数(None 表示全部)
        batch_size: 批次大小
        delay_between_batches: 批次间延迟(秒)

    Returns:
        成功生成的答案数量
    """
    print("\n" + "=" * 70)
    print("📝 为标准答案生成工具")
    print("=" * 70)

    # 加载数据
    print(f"\n📂 正在加载数据集: {dataset_path}")
    data = load_dataset(dataset_path)

    if max_samples:
        data = data[:max_samples]
        print(f"⚡ 快速模式:仅处理前 {max_samples} 个样本")

    print(f"📊 共 {len(data)} 个问题需要处理")

    # 检查是否已有 reference
    has_existing_refs = any(item.get('reference', '').strip() for item in data)
    if has_existing_refs:
        print("⚠️  检测到部分样本已有 reference,将跳过这些样本")

    success_count = 0
    failed_count = 0

    # 使用进度条
    for i, item in enumerate(tqdm(data, desc="生成标准答案")):
        # 如果已有答案,跳过
        if item.get('reference', '').strip():
            continue

        question = item['question']
        contexts = item.get('ground_truth_contexts', [])

        if not contexts:
            logger.warning(f"样本 {i+1} 没有 ground_truth_contexts,跳过")
            failed_count += 1
            continue

        try:
            # 生成答案
            answer = generate_reference_answer(question, contexts)

            if answer:
                item['reference'] = answer
                success_count += 1
            else:
                item['reference'] = ""
                failed_count += 1
                logger.warning(f"样本 {i+1} 生成答案为空白")

            # 每 batch_size 个样本保存一次并延迟
            if (i + 1) % batch_size == 0:
                save_dataset(data, output_path)
                print(f"\n💾 中间结果已保存 (已完成 {i+1}/{len(data)})")
                time.sleep(delay_between_batches)

        except Exception as e:
            logger.error(f"处理样本 {i+1} 失败: {e}")
            item['reference'] = ""
            failed_count += 1

    # 最终保存
    save_dataset(data, output_path)

    print("\n" + "=" * 70)
    print("✅ 生成完成!")
    print("=" * 70)
    print(f"成功: {success_count}")
    print(f"失败: {failed_count}")
    print(f"总计: {len(data)}")
    print(f"输出文件: {output_path}")

    return success_count


def main():
    import argparse

    parser = argparse.ArgumentParser(description='为 Ragas 数据集生成标准答案')
    parser.add_argument('--samples', type=int, default=None,
                        help='处理的样本数量(默认全部)')
    parser.add_argument('--quick', action='store_true',
                        help='快速模式:仅处理前 5 个样本')
    parser.add_argument('--input', type=str, default='data/ragas_test_dataset.json',
                        help='输入数据集路径')
    parser.add_argument('--output', type=str, default='data/ragas_test_dataset_with_references.json',
                        help='输出数据集路径')
    parser.add_argument('--batch-size', type=int, default=5,
                        help='批次大小(默认 5)')

    args = parser.parse_args()

    max_samples = 5 if args.quick else args.samples

    count = generate_references_for_dataset(
        dataset_path=args.input,
        output_path=args.output,
        max_samples=max_samples,
        batch_size=args.batch_size
    )

    if count > 0:
        print(f"\n🎉 成功生成 {count} 个标准答案!")
        print(f"💡 下一步: 运行 'python Ragas自动化评估.py --use-references' 进行评估")
    else:
        print("\n❌ 未能生成任何答案,请检查日志")


if __name__ == '__main__':
    main()
