import os.path
import hashlib
from utils.logger_handler import logger
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader


def get_file_md5_hex(filepath:str): # 获取文件的md5的16进制表示字符串
    if not os.path.exists(filepath):
        logger.error(f"[md5计算] 文件不存在：{filepath}")
        return
    if not os.path.isfile(filepath):
        logger.error(f"[md5计算] 不是文件：{filepath}")
        return

    md5_obj = hashlib.md5()
    chunk_size = 4096 #4kb分片防止文件大爆内存
    try:
        with open(filepath, 'rb') as f:  #必须二进制读取
            while chunk := f.read(chunk_size):  #    := 赋值并判断
                md5_obj.update(chunk)
                """相当于 chunk = f.read(chunk_size)
                        while chunk:
                            md5_obj.update(chunk)
                """
            md5_hex = md5_obj.hexdigest()
            return md5_hex
    except Exception as e:
        logger.error(f"[md5计算] 文件读取异常：{filepath},{str(e)}")
        return None

def listdir_with_allowed_type(path: str, allowed_types: tuple[str]):  # 返回文件夹内的文件列表（允许的文件后缀）
    files = []

    if not os.path.isdir(path):
        logger.error(f"[listdir_with_allowed_type] 不是文件夹：{path}")
        return allowed_types

    for f in os.listdir(path):
        if f.endswith(allowed_types):
            files.append(os.path.join(path, f))

    return tuple(files)

def pdf_loader(filepath: str,password: str = None) -> list[Document]:
    return PyPDFLoader(filepath,password).load()

def txt_loader(filepath: str) -> list[Document]:
    return TextLoader(filepath, encoding="utf-8").load()
