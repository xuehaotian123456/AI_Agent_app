# import os
# from re import search
#
# from langchain_core.documents import Document
#
# from utils.logger_handler import logger
# from utils.path_tool import get_abs_path
# from langchain_chroma import Chroma
# from utils.config_handler import chroma_conf
# from model.factory import embed_model
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from utils.file_handler import txt_loader, pdf_loader, listdir_with_allowed_type, get_file_md5_hex
#
#
# # from langchain_core.embeddings import Embeddings
# # from langchain_community.chat_models.tongyi import ChatTongyi
# # from langchain_community.embeddings import DashScopeEmbeddings
# # from langchain_community.chat_models.tongyi import BaseChatModel
#
# class VectorStoreService:
#     def __init__(self):
#         self.vector_store = Chroma(
#             collection_name=chroma_conf["collection_name"],
#             embedding_function=embed_model,
#             persist_directory=chroma_conf["persist_directory"],
#         )
#         self.spliter = RecursiveCharacterTextSplitter(
#             chunk_size=chroma_conf["chunk_size"],
#             chunk_overlap=chroma_conf["chunk_overlap"],
#             separators=chroma_conf["separators"],
#             length_function=len,
#         )
#
#         # 用于混合检索的文档缓存
#         self.all_text_chunks = []
#         self.all_documents = []
#         self.hybrid_retriever = None
#
#         # 检查是否启用混合检索
#         hybrid_config = chroma_conf.get("hybrid_search", {})
#         self.hybrid_enabled = hybrid_config.get("enabled", False)
#
#     def get_retriever(self):
#         """获取检索器（根据配置返回传统或混合检索器）"""
#         if self.hybrid_enabled and self.hybrid_retriever:
#             return self.hybrid_retriever
#         return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})
#
#     def load_documents(self):
#         # texts = self.spliter.split_documents(docs)
#         # self.vector_store.add_documents(texts)
#
#         def check_md5_hex(md5_for_check:str):
#             if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])):
#                 open(get_abs_path(chroma_conf["md5_hex_store"]), "w", encoding="utf-8").close()
#                 return False
#             with open(get_abs_path(chroma_conf["md5_hex_store"]), "r", encoding="utf-8") as f:
#
#                 for line in f.readlines():
#                     if line.strip() == md5_for_check:
#                         return True
#
#                 return False
#
#         def save_md5_hex(md5_for_check:str):
#             with open(get_abs_path(chroma_conf["md5_hex_store"]), "a", encoding="utf-8") as f:
#                 f.write(md5_for_check +"\n")
#
#         def get_file_documents(read_path:str):
#             if read_path.endswith(".pdf"):
#                 return pdf_loader(read_path)
#             if read_path.endswith(".txt"):
#                 return txt_loader(read_path)
#             return []
#
#         allowed_file_path :list[str] = listdir_with_allowed_type(
#             get_abs_path(chroma_conf["data_path"]),
#             tuple(chroma_conf["allow_knowledge_file_type"])
#         )
#
#         for path in allowed_file_path:
#             md5_hex = get_file_md5_hex(path)
#             if not md5_hex:
#                 continue
#             if check_md5_hex(md5_hex):
#                 logger.info(f"[加载知识库] 文件已存在知识库中，跳过：{path}")
#                 continue
#             try:
#                 documents :list[Document] = get_file_documents(path)
#                 if not documents:
#                     logger.warning(f"[加载知识库] 文件无法加载：无有效文本信息：地址{path}")
#                     continue
#                 split_document :list[Document]= self.spliter.split_documents(documents)
#
#                 if not split_document:
#                     logger.warning(f"[加载知识库] 文件无法加载：分片后无有效文本信息：地址{path}")
#                     continue
#                 self.vector_store.add_documents(split_document)
#                 save_md5_hex(md5_hex)
#
#                 if self.hybrid_enabled:
#                     for doc in split_document:
#                         self.all_text_chunks.append(doc.page_content)
#                         self.all_documents.append(doc)
#
#                 logger.info(f"[加载知识库] 文件已加载：{path}")
#             except Exception as e:
#                 logger.error(f"[加载知识库] 文件加载异常：{path},{str(e)}",exc_info= True)
#                 continue
#
#         if self.hybrid_enabled and self.all_text_chunks:
#             try:
#                 from rag.hybrid_retriever import HybridRetriever
#                 hybrid_config = chroma_conf.get("hybrid_search", {})
#                 self.hybrid_retriever = HybridRetriever(
#                     vector_store=self.vector_store,
#                     text_chunks=self.all_text_chunks,
#                     documents=self.all_documents
#                 )
#                 logger.info("[混合检索] 混合检索器初始化成功")
#             except Exception as e:
#                 logger.error(f"[混合检索] 混合检索器初始化失败: {e}", exc_info=True)
#                 self.hybrid_enabled = False
#
# if __name__ == '__main__':
#     vs = VectorStoreService()
#     vs.load_documents()
#     retriever = vs.get_retriever()
#     res = retriever.invoke("迷路")
#     for r in res :
#         print(r.page_content)
#         print("-"*20)


import os
from re import search

from langchain_core.documents import Document

from utils.logger_handler import logger
from utils.path_tool import get_abs_path
from langchain_chroma import Chroma
from utils.config_handler import chroma_conf
from model.factory import embed_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.file_handler import txt_loader, pdf_loader, listdir_with_allowed_type, get_file_md5_hex


# from langchain_core.embeddings import Embeddings
# from langchain_community.chat_models.tongyi import ChatTongyi
# from langchain_community.embeddings import DashScopeEmbeddings
# from langchain_community.chat_models.tongyi import BaseChatModel

class VectorStoreService:
    def __init__(self):
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embed_model,
            persist_directory=chroma_conf["persist_directory"],
        )
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )

        # 用于混合检索的文档缓存
        self.all_text_chunks = []
        self.all_documents = []
        self.hybrid_retriever = None

        # 检查是否启用混合检索
        hybrid_config = chroma_conf.get("hybrid_search", {})
        self.hybrid_enabled = hybrid_config.get("enabled", False)

    def get_retriever(self):
        """获取检索器（根据配置返回传统或混合检索器）"""
        if self.hybrid_enabled and self.hybrid_retriever:
            return self.hybrid_retriever
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})

    def load_documents(self):

        def check_md5_hex(md5_for_check:str):
            if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])):
                open(get_abs_path(chroma_conf["md5_hex_store"]), "w", encoding="utf-8").close()
                return False
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "r", encoding="utf-8") as f:

                for line in f.readlines():
                    if line.strip() == md5_for_check:
                        return True

                return False

        def save_md5_hex(md5_for_check:str):
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_for_check +"\n")

        def get_file_documents(read_path:str):
            if read_path.endswith(".pdf"):
                return pdf_loader(read_path)
            if read_path.endswith(".txt"):
                return txt_loader(read_path)
            return []

        allowed_file_path :list[str] = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"])
        )

        for path in allowed_file_path:
            md5_hex = get_file_md5_hex(path)
            if not md5_hex:
                continue
            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库] 文件已存在知识库中，跳过：{path}")

                # 🔥 优化：即使跳过加载，也要从向量库恢复文档用于混合检索
                if check_md5_hex(md5_hex):
                    logger.info(f"[加载知识库] 文件已存在知识库中，跳过：{path}")

                    # 🔥 优化：即使跳过加载，也要从向量库恢复文档用于混合检索
                    if self.hybrid_enabled:
                        try:
                            temp_collection = Chroma(
                                collection_name=chroma_conf["collection_name"],
                                embedding_function=embed_model,
                                persist_directory=chroma_conf["persist_directory"],
                            )

                            # 🔥 修复：使用文件 basename 进行查询（因为metadata中可能存储的是basename）
                            file_basename = os.path.basename(path)

                            # 先尝试用完整路径查询
                            results = temp_collection.get(
                                where={"source": path},
                                include=["documents", "metadatas"]
                            )

                            # 如果没找到，尝试用basename查询
                            if not results or not results.get('documents'):
                                logger.info(f"[混合检索] 完整路径未找到，尝试basename: {file_basename}")
                                results = temp_collection.get(
                                    where={"source": file_basename},
                                    include=["documents", "metadatas"]
                                )

                            # 如果还是没找到，尝试模糊匹配（包含文件名）
                            if not results or not results.get('documents'):
                                logger.info(f"[混合检索] 尝试获取所有文档进行过滤...")
                                all_results = temp_collection.get(
                                    include=["documents", "metadatas"]
                                )

                                if all_results and all_results['documents']:
                                    filtered_docs = []
                                    filtered_metadatas = []
                                    for doc_content, metadata in zip(all_results['documents'],
                                                                     all_results['metadatas']):
                                        source = metadata.get('source', '')
                                        if file_basename in source or path in source:
                                            filtered_docs.append(doc_content)
                                            filtered_metadatas.append(metadata)

                                    results = {
                                        'documents': filtered_docs,
                                        'metadatas': filtered_metadatas
                                    }

                            if results and results.get('documents'):
                                for doc_content, metadata in zip(results['documents'], results['metadatas']):
                                    self.all_text_chunks.append(doc_content)
                                    self.all_documents.append(Document(page_content=doc_content, metadata=metadata))
                                logger.info(
                                    f"[混合检索] 从向量库恢复 {len(results['documents'])} 个文档块: {file_basename}")
                            else:
                                logger.warning(f"[混合检索] 向量库中未找到 {file_basename} 的文档")
                        except Exception as e:
                            logger.warning(f"[混合检索] 从向量库恢复文档失败: {e}", exc_info=True)

                    continue

            try:
                documents :list[Document] = get_file_documents(path)
                if not documents:
                    logger.warning(f"[加载知识库] 文件无法加载：无有效文本信息：地址{path}")
                    continue
                split_document :list[Document]= self.spliter.split_documents(documents)

                if not split_document:
                    logger.warning(f"[加载知识库] 文件无法加载：分片后无有效文本信息：地址{path}")
                    continue
                self.vector_store.add_documents(split_document)
                save_md5_hex(md5_hex)

                if self.hybrid_enabled:
                    for doc in split_document:
                        self.all_text_chunks.append(doc.page_content)
                        self.all_documents.append(doc)
                    logger.info(f"[混合检索] 已缓存 {len(split_document)} 个文档块: {os.path.basename(path)}")

                logger.info(f"[加载知识库] 文件已加载：{path}")
            except Exception as e:
                logger.error(f"[加载知识库] 文件加载异常：{path},{str(e)}",exc_info= True)
                continue

        if self.hybrid_enabled and self.all_text_chunks:
            try:
                from rag.hybrid_retriever import HybridRetriever
                hybrid_config = chroma_conf.get("hybrid_search", {})
                self.hybrid_retriever = HybridRetriever(
                    vector_store=self.vector_store,
                    text_chunks=self.all_text_chunks,
                    documents=self.all_documents
                )
                logger.info(f"[混合检索] 混合检索器初始化成功，共 {len(self.all_text_chunks)} 个文档块")
            except Exception as e:
                logger.error(f"[混合检索] 混合检索器初始化失败: {e}", exc_info=True)
                self.hybrid_enabled = False
        elif self.hybrid_enabled and not self.all_text_chunks:
            logger.warning("[混合检索] 没有可用的文档块，混合检索未启用")
            self.hybrid_enabled = False

if __name__ == '__main__':
    vs = VectorStoreService()
    vs.load_documents()
    retriever = vs.get_retriever()
    res = retriever.invoke("迷路")
    for r in res :
        print(r.page_content)
        print("-"*20)
