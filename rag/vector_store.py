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

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})

    def load_documents(self):
        # texts = self.spliter.split_documents(docs)
        # self.vector_store.add_documents(texts)

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

                logger.info(f"[加载知识库] 文件已加载：{path}")
            except Exception as e:
                logger.error(f"[加载知识库] 文件加载异常：{path},{str(e)}",exc_info= True)
                continue


if __name__ == '__main__':
    vs = VectorStoreService()
    vs.load_documents()
    retriever = vs.get_retriever()
    res = retriever.invoke("迷路")
    for r in res :
        print(r.page_content)
        print("-"*20)