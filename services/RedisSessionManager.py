"""
Redis 会话管理器
支持多轮对话持久化、自动过期、历史截断
"""
import json
from typing import List, Dict, Optional
import redis.asyncio as redis
from datetime import timedelta
from utils.logger_handler import logger


class RedisSessionManager:
    """基于 Redis 的会话管理器"""

    def __init__(
        self,
        redis_client: redis.Redis,
        ttl_hours: int = 2,
        max_history: int = 20,
        key_prefix: str = "chat:session:"
    ):
        """
        Args:
            redis_client: 异步 Redis 客户端
            ttl_hours: 会话过期时间（小时）
            max_history: 最大保留消息数
            key_prefix: Redis key 前缀
        """
        self.redis = redis_client
        self.ttl = timedelta(hours=ttl_hours)
        self.max_history = max_history
        self.key_prefix = key_prefix

    def _get_key(self, session_id: str) -> str:
        """生成 Redis key"""
        return f"{self.key_prefix}{session_id}"

    async def get_history(self, session_id: str) -> List[Dict]:
        """
        获取会话历史

        Args:
            session_id: 会话 ID

        Returns:
            消息列表 [{"role": "user/assistant", "content": "..."}]
        """
        try:
            key = self._get_key(session_id)
            data = await self.redis.get(key)
            if data:
                return json.loads(data)
            return []
        except Exception as e:
            logger.error(f"[RedisSession] 获取历史失败: {e}")
            return []

    async def add_message(self, session_id: str, role: str, content: str):
        """
        添加一条消息到会话历史

        Args:
            session_id: 会话 ID
            role: 角色 ("user" 或 "assistant")
            content: 消息内容
        """
        try:
            key = self._get_key(session_id)
            history = await self.get_history(session_id)

            # 添加新消息
            history.append({"role": role, "content": content})

            # 截断历史（防止 context 过长）
            if len(history) > self.max_history:
                history = history[-self.max_history:]

            # 存入 Redis，设置过期时间
            await self.redis.setex(
                key,
                int(self.ttl.total_seconds()),
                json.dumps(history, ensure_ascii=False)
            )
        except Exception as e:
            logger.error(f"[RedisSession] 添加消息失败: {e}")

    async def clear_history(self, session_id: str):
        """
        清除会话历史

        Args:
            session_id: 会话 ID
        """
        try:
            key = self._get_key(session_id)
            await self.redis.delete(key)
            logger.info(f"[RedisSession] 会话 {session_id} 已清除")
        except Exception as e:
            logger.error(f"[RedisSession] 清除会话失败: {e}")

    async def get_context(
        self,
        session_id: str,
        max_tokens: int = 2048,
        include_system_prompt: bool = False
    ) -> str:
        """
        获取格式化的对话上下文，用于注入 prompt

        Args:
            session_id: 会话 ID
            max_tokens: 最大 token 数（简单估算：1 中文字符 ≈ 1 token）
            include_system_prompt: 是否包含系统提示

        Returns:
            格式化的对话上下文字符串
        """
        try:
            history = await self.get_history(session_id)

            # 从后往前遍历，确保最近的对话优先
            context_parts = []
            total_chars = 0

            for msg in reversed(history):
                msg_str = f"{msg['role']}: {msg['content']}\n"
                if total_chars + len(msg_str) > max_tokens:
                    break
                context_parts.insert(0, msg_str)
                total_chars += len(msg_str)

            context = "".join(context_parts).strip()

            if include_system_prompt and context:
                context = f"以下是历史对话记录：\n{context}\n\n请基于以上对话继续回答用户的问题。"

            return context
        except Exception as e:
            logger.error(f"[RedisSession] 获取上下文失败: {e}")
            return ""

    async def get_session_stats(self, session_id: str) -> Dict:
        """
        获取会话统计信息

        Args:
            session_id: 会话 ID

        Returns:
            统计信息 {"message_count": int, "ttl_remaining": int}
        """
        try:
            key = self._get_key(session_id)
            data = await self.redis.get(key)
            ttl = await self.redis.ttl(key)

            if data:
                history = json.loads(data)
                return {
                    "message_count": len(history),
                    "ttl_remaining": ttl if ttl > 0 else 0,
                    "exists": True
                }
            return {"message_count": 0, "ttl_remaining": 0, "exists": False}
        except Exception as e:
            logger.error(f"[RedisSession] 获取统计信息失败: {e}")
            return {"message_count": 0, "ttl_remaining": 0, "exists": False}
