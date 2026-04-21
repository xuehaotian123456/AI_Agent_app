"""
检索结果缓存
缓存向量搜索结果，避免重复检索
"""
import hashlib
import json
from typing import List, Optional
from langchain_core.documents import Document
import redis.asyncio as redis
from utils.logger_handler import logger


class RetrievalCache:
    """检索结果缓存器"""

    def __init__(self, redis_client: redis.Redis, ttl_seconds: int = 3600):
        """
        Args:
            redis_client: 异步 Redis 客户端
            ttl_seconds: 缓存过期时间（秒），默认 1 小时
        """
        self.redis = redis_client
        self.ttl = ttl_seconds
        self.prefix = "retrieval:"

    def _hash_query(self, query: str) -> str:
        """对查询进行 MD5 哈希"""
        return hashlib.md5(query.encode('utf-8')).hexdigest()

    async def get(self, query: str) -> Optional[List[Document]]:
        """
        从缓存中获取检索结果

        Args:
            query: 查询文本

        Returns:
            缓存的文档列表，未命中返回 None
        """
        try:
            key = f"{self.prefix}{self._hash_query(query)}"
            data = await self.redis.get(key)

            if data:
                docs_data = json.loads(data)
                # 反序列化为 Document 对象
                documents = [
                    Document(
                        page_content=doc["page_content"],
                        metadata=doc.get("metadata", {})
                    )
                    for doc in docs_data
                ]
                logger.debug(f"[RetrievalCache] 缓存命中: {query[:30]}...")
                return documents

            logger.debug(f"[RetrievalCache] 缓存未命中: {query[:30]}...")
            return None
        except Exception as e:
            logger.warning(f"[RetrievalCache] 获取缓存失败: {e}")
            return None

    async def set(self, query: str, docs: List[Document]):
        """
        将检索结果存入缓存

        Args:
            query: 查询文本
            docs: 文档列表
        """
        try:
            key = f"{self.prefix}{self._hash_query(query)}"

            # 序列化 Document 对象
            docs_data = [
                {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in docs
            ]

            await self.redis.setex(
                key,
                self.ttl,
                json.dumps(docs_data, ensure_ascii=False)
            )
            logger.debug(f"[RetrievalCache] 缓存已设置: {query[:30]}..., docs={len(docs)}")
        except Exception as e:
            logger.warning(f"[RetrievalCache] 设置缓存失败: {e}")

    async def clear(self):
        """清除所有检索缓存"""
        try:
            pattern = f"{self.prefix}*"
            keys = []
            async for key in self.redis.scan_iter(match=pattern):
                keys.append(key)

            if keys:
                await self.redis.delete(*keys)
                logger.info(f"[RetrievalCache] 已清除 {len(keys)} 个缓存项")
        except Exception as e:
            logger.error(f"[RetrievalCache] 清除缓存失败: {e}")
