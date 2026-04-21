"""
Redis 配置与缓存初始化
支持精确匹配缓存和语义缓存两种模式
"""
import os
import redis
import redis.asyncio as aioredis
from langchain.globals import set_llm_cache
from langchain_community.cache import RedisCache, RedisSemanticCache
from utils.logger_handler import logger


# ==================== Redis 连接配置 ====================
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

# 同步 Redis 客户端（用于缓存）
redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    password=REDIS_PASSWORD,
    decode_responses=True,
    socket_connect_timeout=5,
    socket_timeout=5,
)

# 异步 Redis 客户端（用于会话管理和限流）
async_redis_client = aioredis.from_url(
    f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}",
    password=REDIS_PASSWORD,
    encoding="utf-8",
    decode_responses=True,
)


# ==================== LLM 缓存配置 ====================
def setup_exact_cache():
    """
    设置精确匹配缓存
    适合完全相同的 prompt，速度快但不够灵活
    """
    try:
        # 测试连接
        redis_client.ping()
        set_llm_cache(RedisCache(redis_client))
        logger.info("[Redis] 精确匹配缓存已启用")
        return True
    except Exception as e:
        logger.error(f"[Redis] 精确匹配缓存启用失败: {e}")
        return False


def setup_semantic_cache(embedding_model=None):
    """
    设置语义缓存
    适合问题表述不同但语义相同的情况，需要 embedding 模型

    Args:
        embedding_model: embedding 模型实例，如果为 None 则使用项目中的 BGE embedding
    """
    try:
        # 测试连接
        redis_client.ping()

        if embedding_model is None:
            # 使用项目中已有的 DashScope embedding
            from model.factory import embed_model
            embedding_model = embed_model

        redis_url = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
        if REDIS_PASSWORD:
            redis_url = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"

        set_llm_cache(
            RedisSemanticCache(
                embedding=embedding_model,
                redis_url=redis_url,
                score_threshold=0.85,  # 相似度阈值，0.85 以上视为相同问题
            )
        )
        logger.info("[Redis] 语义缓存已启用（阈值: 0.85）")
        return True
    except Exception as e:
        logger.error(f"[Redis] 语义缓存启用失败: {e}")
        return False


def disable_cache():
    """禁用 LLM 缓存"""
    set_llm_cache(None)
    logger.info("[Redis] LLM 缓存已禁用")
