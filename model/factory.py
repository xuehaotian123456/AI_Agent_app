from abc import ABC,abstractmethod
from typing import Optional
from contextvars import ContextVar
from langchain_chroma import Chroma
from utils.config_handler import chroma_conf, rag_conf
from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import BaseChatModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from utils.logger_handler import logger
import time

class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self)->Optional[Embeddings | BaseChatModel]:
        pass

class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return ChatTongyi(model=rag_conf["chat_model_name"])

class BackupChatModelFactory(BaseModelFactory):
    """备份模型工厂"""
    def generator(self) -> Optional[BaseChatModel]:
        backup_model_name = rag_conf.get("backup_chat_model_name", "qwen-plus")
        logger.info(f"初始化备份模型: {backup_model_name}")
        return ChatTongyi(model=backup_model_name)

class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return DashScopeEmbeddings(model=rag_conf["embedding_model_name"])

chat_model = ChatModelFactory().generator()
embed_model = EmbeddingsFactory().generator()

# 初始化备份模型（可选）
backup_model = None
try:
    if "backup_chat_model_name" in rag_conf:
        backup_model = BackupChatModelFactory().generator()
        logger.info("备份模型初始化成功")
except Exception as e:
    logger.warning(f"备份模型初始化失败: {e}，将不使用备份模型")

# 获取重试配置
retry_config = rag_conf.get("llm_retry", {})
MAX_ATTEMPTS = retry_config.get("max_attempts", 3)
MIN_WAIT = retry_config.get("min_wait", 2)
MAX_WAIT = retry_config.get("max_wait", 10)
MULTIPLIER = retry_config.get("multiplier", 1)

# 记录当前请求最终使用的模型（主模型/备份模型）
_last_model_used: ContextVar[str] = ContextVar("last_model_used", default="unknown")


def create_llm_caller_with_retry(primary_model: BaseChatModel,
                                  backup_model: Optional[BaseChatModel] = None,
                                  max_attempts: int = MAX_ATTEMPTS,
                                  min_wait: int = MIN_WAIT,
                                  max_wait: int = MAX_WAIT,
                                  multiplier: int = MULTIPLIER):
    """
    创建带有重试和降级机制的 LLM 调用器

    Args:
        primary_model: 主模型
        backup_model: 备份模型（可选）
        max_attempts: 最大重试次数
        min_wait: 最小等待时间（秒）
        max_wait: 最大等待时间（秒）
        multiplier: 指数退避乘数

    Returns:
        可调用函数，接受 prompt 并返回模型响应
    """

    @retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=multiplier, min=min_wait, max=max_wait),
        retry=retry_if_exception_type((Exception,)),
        reraise=False
    )
    def call_primary_model(prompt):
        """调用主模型，带重试机制"""
        start_time = time.time()
        try:
            logger.debug(f"[LLM调用] 尝试调用主模型...")
            response = primary_model.invoke(prompt)
            elapsed = time.time() - start_time
            logger.debug(f"[LLM调用] 主模型调用成功，耗时: {elapsed:.2f}s")
            return response
        except Exception as e:
            elapsed = time.time() - start_time
            logger.warning(f"[LLM调用] 主模型调用失败 (耗时: {elapsed:.2f}s): {type(e).__name__}: {e}")
            raise

    def robust_llm_call(prompt):
        """
        健壮的 LLM 调用：先尝试主模型（带重试），失败后降级到备份模型

        Args:
            prompt: 提示词

        Returns:
            模型响应对象
        """
        _last_model_used.set("unknown")

        # 尝试主模型（带重试）
        try:
            logger.info("[LLM调用] 开始调用主模型")
            response = call_primary_model(prompt)
            _last_model_used.set("primary")
            logger.info("[LLM调用] 主模型调用成功")
            return response
        except Exception as e:
            logger.error(f"[LLM调用] 主模型在所有重试后仍然失败: {type(e).__name__}: {e}")

            # 如果有备份模型，尝试降级
            if backup_model is not None:
                logger.warning(f"[LLM降级] 切换到备份模型...")
                try:
                    start_time = time.time()
                    response = backup_model.invoke(prompt)
                    elapsed = time.time() - start_time
                    _last_model_used.set("backup")
                    logger.info(f"[LLM降级] 备份模型调用成功，耗时: {elapsed:.2f}s")
                    return response
                except Exception as backup_error:
                    logger.error(f"[LLM降级] 备份模型也失败了: {type(backup_error).__name__}: {backup_error}")
                    raise RuntimeError(
                        f"主模型和备份模型均调用失败。主模型错误: {e}, 备份模型错误: {backup_error}"
                    ) from backup_error
            else:
                logger.error("[LLM降级] 未配置备份模型，无法降级")
                raise RuntimeError(f"主模型调用失败且无备份模型可用: {e}") from e

    return robust_llm_call


# 创建默认的健壮调用器
robust_llm_caller = create_llm_caller_with_retry(chat_model, backup_model)


def get_last_model_used() -> str:
    """获取最近一次 robust_llm_caller 调用使用的模型类型。"""
    return _last_model_used.get()

