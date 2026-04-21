"""
准备 Ragas 评估数据集
从 eval_dataset.json 提取问题,并从源文档中获取 ground_truth_contexts
"""
import json
from pathlib import Path
from typing import List, Dict
from utils.file_handler import pdf_loader, txt_loader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def prepare_ragas_dataset(
    eval_json_path: str = "data/eval_dataset.json",
    output_json_path: str = "data/ragas_test_dataset_with_references.json",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    top_k_relevant: int = 3
):
    """
    从 eval_dataset.json 生成 Ragas 评估数据集

    Args:
        eval_json_path: 原始评估数据集路径
        output_json_path: 输出路径
        chunk_size: 文档分块大小（避免 token 超限）
        chunk_overlap: 分块重叠
        top_k_relevant: 每个问题保留的相关上下文数量
    """
    with open(eval_json_path, 'r', encoding='utf-8') as f:
        items = json.load(f)

    # 初始化文本分割器
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?"],
        length_function=len
    )

    prepared = []

    print(f"📂 开始处理 {len(items)} 个问题...")

    for i, item in enumerate(items, 1):
        source_path = Path(item["source"])

        if not source_path.exists():
            print(f"  [{i}/{len(items)}] ❌ 文件不存在: {source_path}")
            continue

        try:
            # 使用现有的 loader 函数
            if source_path.suffix.lower() == '.txt':
                documents = txt_loader(str(source_path))
            elif source_path.suffix.lower() == '.pdf':
                documents = pdf_loader(str(source_path))
            else:
                print(f"  [{i}/{len(items)}] ⚠️ 不支持的文件类型: {source_path.suffix}")
                continue

            if not documents:
                print(f"  [{i}/{len(items)}] ⚠️ 文件内容为空: {source_path.name}")
                continue

            # 分块处理（避免单个文档太大）
            split_docs = splitter.split_documents(documents)
            all_chunks = [doc.page_content for doc in split_docs]

            # ✅ 核心改进：基于简单关键词匹配筛选相关上下文
            question = item["query"]

            # 提取问题中的关键词（中文分词简化版：按字符和常见标点分割）
            question_keywords = set()
            for char in question:
                if char.isalnum() or '\u4e00' <= char <= '\u9fff':
                    question_keywords.add(char)

            # 计算每个 chunk 与问题的关键词重叠度
            chunk_scores = []
            for idx, chunk in enumerate(all_chunks):
                # 统计问题关键词在 chunk 中出现的次数
                score = sum(1 for keyword in question_keywords if keyword in chunk)
                # 归一化分数（考虑 chunk 长度）
                normalized_score = score / (len(chunk) ** 0.5) if len(chunk) > 0 else 0
                chunk_scores.append((idx, normalized_score))

            # 按分数降序排序，选择 top_k
            chunk_scores.sort(key=lambda x: x[1], reverse=True)
            top_indices = [idx for idx, score in chunk_scores[:top_k_relevant]]

            # 保持原始顺序（避免打乱文档结构）
            top_indices_sorted = sorted(top_indices)
            ground_truth_contexts = [all_chunks[idx] for idx in top_indices_sorted]

            print(f"  [{i}/{len(items)}] ✅ {source_path.name} (选中 {len(ground_truth_contexts)}/{len(all_chunks)} 个块)")

            prepared.append({
                "question": item["query"],
                "ground_truth_contexts": ground_truth_contexts,
                "ground_truth": "",
                "source_doc": str(source_path),
                "num_chunks": len(ground_truth_contexts),
                "reference": item.get("answer", "")  # 添加 reference 字段
            })

        except Exception as e:
            print(f"  [{i}/{len(items)}] ❌ 处理失败: {e}")
            continue

    # 保存结果
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(prepared, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 成功生成 {len(prepared)} 个测试样本")
    print(f"💾 保存至: {output_json_path}")

    return len(prepared)


if __name__ == "__main__":
    count = prepare_ragas_dataset()
    print(f"\n总计: {count} 个样本已准备好用于 Ragas 评估")
