import os
from typing import List
from langchain_core.documents import Document
from utils.logger_handler import logger
from utils.path_tool import get_abs_path
import jieba
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


class HybridRetriever:
    """混合检索器：向量检索 + BM25 + 重排序"""

    def __init__(self, vector_store, text_chunks: List[str], documents: List[Document]):
        """
        初始化混合检索器

        Args:
            vector_store: Chroma 向量存储实例
            text_chunks: 文本片段列表（用于BM25）
            documents: 原始文档对象列表（与text_chunks对应）
        """
        self.vector_store = vector_store
        self.original_documents = documents

        # 构建 BM25 索引
        logger.info("[混合检索] 正在构建 BM25 索引...")
        tokenized_chunks = [list(jieba.cut(chunk)) for chunk in text_chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)
        self.tokenized_chunks = tokenized_chunks
        logger.info(f"[混合检索] BM25 索引构建完成，共 {len(text_chunks)} 个文本块")

        # 初始化重排序模型 (使用本地路径)
        local_model_path = get_abs_path("models/bge-reranker-base")
        logger.info(f"[混合检索] 正在从本地加载重排序模型: {local_model_path}")
        try:
            if os.path.exists(local_model_path):
                self.reranker = CrossEncoder(local_model_path)
                logger.info("[混合检索] 本地重排序模型加载成功")
            else:
                logger.warning(f"[混合检索] 本地模型路径不存在: {local_model_path}，尝试在线加载...")
                self.reranker = CrossEncoder('BAAI/bge-reranker-base')
                logger.info("[混合检索] 在线重排序模型加载成功")
        except Exception as e:
            logger.warning(f"[混合检索] 重排序模型加载失败: {e}，将跳过重排序步骤")
            self.reranker = None

    def hybrid_search(self, query: str, top_k_retrieve: int = 10,
                      top_k_rerank: int = 3, alpha: float = 0.5) -> List[Document]:
        """
        执行混合检索

        Args:
            query: 查询文本
            top_k_retrieve: 初步检索返回的文档数量
            top_k_rerank: 重排序后返回的文档数量
            alpha: BM25权重 (0-1)，向量检索权重为 (1-alpha)

        Returns:
            重排序后的文档列表
        """
        try:
            # 1. 向量检索
            vector_docs = self.vector_store.similarity_search(query, k=top_k_retrieve)
            if not vector_docs:
                logger.warning("[混合检索] 向量检索未返回结果")
                return []

            logger.info(f"[混合检索] 向量检索返回 {len(vector_docs)} 个文档")

            # 获取向量检索的文档ID和分数（使用相似度作为分数）
            vector_results = {}
            for i, doc in enumerate(vector_docs):
                # 归一化分数：越靠前分数越高
                doc_id = id(doc)
                vector_results[doc_id] = {
                    'doc': doc,
                    'vector_score': 1 - (i / top_k_retrieve)
                }

            # 2. BM25 检索
            tokenized_query = list(jieba.cut(query))
            bm25_scores = self.bm25.get_scores(tokenized_query)

            # 获取BM25 top-k的索引
            bm25_top_indices = sorted(
                range(len(bm25_scores)),
                key=lambda i: bm25_scores[i],
                reverse=True
            )[:top_k_retrieve]

            logger.info(f"[混合检索] BM25检索完成")

            # 3. 合并两种检索结果
            # 注意：这里需要建立 BM25 索引与原始文档的映射关系
            combined_scores = {}

            # 添加向量检索结果
            for doc_id, data in vector_results.items():
                combined_scores[doc_id] = {
                    'doc': data['doc'],
                    'final_score': (1 - alpha) * data['vector_score']
                }

            # 添加 BM25 检索结果
            max_bm25_score = max(bm25_scores) if max(bm25_scores) > 0 else 1
            for idx in bm25_top_indices:
                if idx < len(self.original_documents):
                    doc = self.original_documents[idx]
                    doc_id = id(doc)
                    normalized_bm25_score = bm25_scores[idx] / max_bm25_score

                    if doc_id in combined_scores:
                        # 文档已存在，累加分数
                        combined_scores[doc_id]['final_score'] += alpha * normalized_bm25_score
                    else:
                        # 新文档
                        combined_scores[doc_id] = {
                            'doc': doc,
                            'final_score': alpha * normalized_bm25_score
                        }

            # 4. 按综合分数排序
            sorted_results = sorted(
                combined_scores.values(),
                key=lambda x: x['final_score'],
                reverse=True
            )[:top_k_retrieve]

            merged_docs = [item['doc'] for item in sorted_results]
            logger.info(f"[混合检索] 合并后得到 {len(merged_docs)} 个文档")

            # 5. 重排序（如果启用了重排序模型）
            if self.reranker and len(merged_docs) > 0:
                reranked_docs = self._rerank(query, merged_docs, top_k_rerank)
                logger.info(f"[混合检索] 重排序完成，返回 {len(reranked_docs)} 个文档")
                return reranked_docs
            else:
                # 返回 top_k_rerank 个文档
                return merged_docs[:top_k_rerank]

        except Exception as e:
            logger.error(f"[混合检索] 混合检索失败: {e}", exc_info=True)
            # 降级：使用纯向量检索
            logger.warning("[混合检索] 降级为纯向量检索")
            return self.vector_store.similarity_search(query, k=top_k_rerank)

    def invoke(self, query: str, **kwargs):
        """
        兼容 LangChain Retriever 的 invoke 接口
        允许像调用普通检索器一样调用混合检索器
        """
        # 获取检索数量 k，默认为 3
        k = kwargs.get('k', 3)
        # 调用内部的 hybrid_search 方法
        # 注意：召回数量通常设为最终返回数量的 2 倍，给重排序留足空间
        return self.hybrid_search(
            query=query,
            top_k_retrieve=k * 2,
            top_k_rerank=k
        )

    def _rerank(self, query: str, docs: List[Document], top_k: int) -> List[Document]:
        """
        使用交叉编码器重排序文档

        Args:
            query: 查询文本
            docs: 待重排序的文档列表
            top_k: 返回的文档数量

        Returns:
            重排序后的文档列表
        """
        try:
            # 构建查询-文档对
            pairs = [[query, doc.page_content] for doc in docs]

            # 计算相关性分数
            scores = self.reranker.predict(pairs)

            # 按分数排序
            sorted_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )

            # 返回 top-k 文档
            reranked_docs = [docs[i] for i in sorted_indices[:top_k]]

            # 记录重排序分数（用于调试）
            for i, idx in enumerate(sorted_indices[:top_k]):
                logger.debug(f"[重排序] 文档{i + 1}: 分数={scores[idx]:.4f}")

            return reranked_docs

        except Exception as e:
            logger.error(f"[重排序] 重排序失败: {e}", exc_info=True)
            # 失败时返回原始顺序的前top_k个
            return docs[:top_k]
