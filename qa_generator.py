
"""
问答对自动生成服务
基于现有文档使用LLM自动生成评测集
"""
import json
import os
from typing import List, Dict, Optional
from langchain_core.documents import Document
from utils.logger_handler import logger
from model.factory import chat_model
from utils.file_handler import txt_loader, pdf_loader, listdir_with_allowed_type
from utils.config_handler import chroma_conf
from utils.path_tool import get_abs_path


class QAGenerator:
    """问答对生成器"""

    def __init__(self, llm=None):
        """
        初始化问答对生成器

        Args:
            llm: LLM实例，默认使用项目配置的chat_model
        """
        self.llm = llm or chat_model
        self.num_questions_per_doc = 3  # 每个文档生成的问题数量
        self.max_content_length = 2000  # 文档内容最大长度

    def generate_qa_from_document(
            self,
            doc: Document,
            num_questions: Optional[int] = None
    ) -> List[Dict]:
        """
        根据单个文档生成多个问题与相关文档ID的标注

        Args:
            doc: 文档对象
            num_questions: 生成的问题数量，默认使用类配置

        Returns:
            包含问题和相关文档ID的列表
        """
        if num_questions is None:
            num_questions = self.num_questions_per_doc

        content = doc.page_content[:self.max_content_length]
        # 优化：只取文件名作为 ID，避免路径差异导致的匹配失败
        full_path = doc.metadata.get("source", "unknown")
        doc_id = os.path.basename(full_path)

        prompt = f"""请阅读以下文档片段，生成 {num_questions} 个用户可能会问的问题。要求：
1. 问题必须能基于该片段内容回答。
2. 问题应模拟真实用户口吻，简短明确（不超过30个字）。
3. 问题类型多样化（包括是什么、为什么、怎么做、适用场景等）。
4. 输出格式为严格的JSON数组，每个元素只包含 "question" 字段。

文档内容：
{content}

输出示例（必须是合法JSON）：
[
    {{"question": "小户型适合哪些扫地机器人？"}},
    {{"question": "扫地机器人如何避障？"}},
    {{"question": "这款机器人的续航时间是多少？"}}
]

请只输出JSON数组，不要有其他文字："""

        try:
            response = self.llm.invoke(prompt)
            response_content = response.content if hasattr(response, 'content') else str(response)

            # 尝试解析JSON
            qa_list = self._parse_json_response(response_content)

            if not qa_list:
                logger.warning(f"[QA生成] JSON解析失败，尝试备用解析方法")
                qa_list = self._fallback_parse(response_content)

            # 为每个问题附加相关文档ID
            annotated_samples = []
            for item in qa_list:
                if isinstance(item, dict) and "question" in item:
                    annotated_samples.append({
                        "query": item["question"],
                        "relevant_ids": [doc_id],  # 这里现在存的是文件名
                        "source": full_path
                    })
                elif isinstance(item, str):
                    annotated_samples.append({
                        "query": item,
                        "relevant_ids": [doc_id],
                        "source": full_path
                    })

            logger.info(f"[QA生成] 从文档生成 {len(annotated_samples)} 个问题: {doc_id}")
            return annotated_samples

        except Exception as e:
            logger.error(f"[QA生成] 生成失败: {doc_id}, 错误: {str(e)}", exc_info=True)
            return []

    def _parse_json_response(self, response_content: str) -> List[Dict]:
        """解析JSON响应"""
        try:
            # 尝试直接解析
            return json.loads(response_content)
        except json.JSONDecodeError:
            # 尝试提取JSON部分
            try:
                start_idx = response_content.find('[')
                end_idx = response_content.rfind(']') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = response_content[start_idx:end_idx]
                    return json.loads(json_str)
            except:
                pass
        return []

    def _fallback_parse(self, response_content: str) -> List[Dict]:
        """备用解析方法：从文本中提取问题"""
        questions = []
        lines = response_content.split('\n')
        for line in lines:
            line = line.strip().strip('- ').strip('* ')
            # 移除可能的JSON格式
            line = line.replace('"question":', '').replace('"', '').strip()
            line = line.lstrip('{').rstrip('}').strip(',')

            # 如果包含问号或是明显的问题格式
            if '?' in line or '？' in line or len(line) > 5:
                questions.append({"question": line})

        return questions

    def build_eval_dataset_from_docs(
            self,
            documents: List[Document],
            num_questions: Optional[int] = None,
            output_path: Optional[str] = None
    ) -> List[Dict]:
        """
        从文档列表生成评测数据集

        Args:
            documents: 文档列表
            num_questions: 每个文档生成的问题数
            output_path: 输出文件路径（可选）

        Returns:
            评测数据集列表
        """
        if num_questions is None:
            num_questions = self.num_questions_per_doc

        dataset = []
        total_docs = len(documents)

        logger.info(f"[QA生成] 开始从 {total_docs} 个文档生成问答对...")

        for idx, doc in enumerate(documents, 1):
            logger.info(f"[QA生成] 处理进度: {idx}/{total_docs}")
            samples = self.generate_qa_from_document(doc, num_questions)
            dataset.extend(samples)

        logger.info(f"[QA生成] 完成！共生成 {len(dataset)} 个问答对")

        # 保存到文件
        if output_path:
            self.save_dataset(dataset, output_path)

        return dataset

    def build_eval_dataset_from_files(
            self,
            data_dir: Optional[str] = None,
            allowed_types: Optional[tuple] = None,
            num_questions: Optional[int] = None,
            output_path: Optional[str] = None
    ) -> List[Dict]:
        """
        从文件目录直接生成评测数据集

        Args:
            data_dir: 数据目录路径
            allowed_types: 允许的文件类型
            num_questions: 每个文档生成的问题数
            output_path: 输出文件路径

        Returns:
            评测数据集列表
        """
        if data_dir is None:
            data_dir = get_abs_path(chroma_conf["data_path"])

        if allowed_types is None:
            allowed_types = tuple(chroma_conf.get("allow_knowledge_file_type", [".txt", ".pdf"]))

        if num_questions is None:
            num_questions = self.num_questions_per_doc

        # 获取所有文件
        file_paths = listdir_with_allowed_type(data_dir, allowed_types)

        if not file_paths:
            logger.warning(f"[QA生成] 未找到任何文件: {data_dir}")
            return []

        logger.info(f"[QA生成] 找到 {len(file_paths)} 个文件")

        # 加载所有文档
        all_documents = []
        for file_path in file_paths:
            try:
                if file_path.endswith(".pdf"):
                    docs = pdf_loader(file_path)
                elif file_path.endswith(".txt"):
                    docs = txt_loader(file_path)
                else:
                    continue

                all_documents.extend(docs)
                logger.info(f"[QA生成] 已加载: {file_path}")
            except Exception as e:
                logger.error(f"[QA生成] 加载文件失败 {file_path}: {str(e)}")

        if not all_documents:
            logger.warning("[QA生成] 没有成功加载任何文档")
            return []

        # 生成评测集
        if output_path is None:
            output_path = get_abs_path("data/eval_dataset.json")

        return self.build_eval_dataset_from_docs(
            documents=all_documents,
            num_questions=num_questions,
            output_path=output_path
        )

    @staticmethod
    def save_dataset(dataset: List[Dict], output_path: str):
        """保存数据集到JSON文件"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            logger.info(f"[QA生成] 数据集已保存: {output_path} (共{len(dataset)}条)")
        except Exception as e:
            logger.error(f"[QA生成] 保存数据集失败: {str(e)}")

    @staticmethod
    def load_dataset(input_path: str) -> List[Dict]:
        """从JSON文件加载数据集"""
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            logger.info(f"[QA生成] 数据集已加载: {input_path} (共{len(dataset)}条)")
            return dataset
        except Exception as e:
            logger.error(f"[QA生成] 加载数据集失败: {str(e)}")
            return []


if __name__ == '__main__':
    # 测试生成问答对
    generator = QAGenerator()

    print("=" * 80)
    print("开始从文档生成问答对评测集")
    print("=" * 80)

    dataset = generator.build_eval_dataset_from_files(
        num_questions=3,
        output_path="data/eval_dataset.json"
    )

    print(f"\n生成了 {len(dataset)} 个问答对")
    if dataset:
        print("\n前5个示例:")
        for i, item in enumerate(dataset[:5], 1):
            print(f"{i}. 问题: {item['query']}")
            print(f"   来源: {item['source']}")
            print()
