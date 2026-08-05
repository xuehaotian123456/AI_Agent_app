"""
自进化数据飞轮服务 (Flywheel Service)

职责：
1. 记录失败案例（Reflection判定retry、检索空结果、解析fallback等）
2. 双重持久化：Redis LPUSH（热数据）+ JSONL文件（冷备）
3. 聚合指标查询：失败率、平均置信度、重试分布、高频失败模式
4. 检索策略自适应：同类型失败≥adapt_threshold时调整参数

面试要点：
"Agent自进化数据飞轮"的具体实现——
不是概念，是可运行的代码：log_event收集 → get_metrics分析 → adapt_retrieval_strategy调整
"""
import os
import json
import time
from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter

from utils.logger_handler import logger
from utils.path_tool import get_abs_path
from utils.config_handler import multi_agent_conf


class FlywheelService:
    """
    自进化数据飞轮

    双重存储：
    - Redis (热数据)：按日分key，LPUSH追加，自动过期（7天）
    - JSONL (冷备)：逐行JSON，永久保存，支持离线分析
    """

    def __init__(self, redis_client=None):
        flywheel_conf = multi_agent_conf.get("flywheel", {})
        self.enabled = flywheel_conf.get("enabled", True)
        self.redis_prefix = flywheel_conf.get("redis_key_prefix", "flywheel:")
        self.jsonl_path = get_abs_path(flywheel_conf.get("jsonl_path", "logs/flywheel_failures.jsonl"))
        self.adapt_threshold = flywheel_conf.get("adapt_threshold", 3)

        self.redis = redis_client

        # 内存中的简单统计（不依赖Redis时使用）
        self._inmemory_events: List[dict] = []
        self._max_inmemory = 500

        # 策略状态
        self._strategy: Dict[str, Any] = {
            "top_k_boost": 0,
            "kg_enabled": True,
            "rewrite_hints": [],
            "total_requests": 0,
            "total_failures": 0,
            "last_updated": "",
        }

        # 确保JSONL目录存在
        os.makedirs(os.path.dirname(self.jsonl_path), exist_ok=True)

        if self.redis:
            logger.info("[Flywheel] 已启用 (Redis + JSONL 双写)")
        else:
            logger.info("[Flywheel] 已启用 (内存 + JSONL 双写)")

    # ==================== 事件记录 ====================

    def log_event(self, event_type: str, payload: dict):
        """
        记录飞轮事件

        Args:
            event_type: 事件类型 (reflection_fail | retrieval_empty | parse_fallback | user_feedback | strategy_change)
            payload: 事件负载
        """
        if not self.enabled:
            return

        event = {
            "event_type": event_type,
            "payload": payload,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "date": time.strftime("%Y%m%d"),
        }

        # 1. 写入内存
        self._inmemory_events.append(event)
        if len(self._inmemory_events) > self._max_inmemory:
            self._inmemory_events = self._inmemory_events[-self._max_inmemory:]

        # 2. 写入 Redis（按日分key）
        if self.redis:
            try:
                import json as _json
                key = f"{self.redis_prefix}{event_type}:{event['date']}"
                self.redis.lpush(key, _json.dumps(event, ensure_ascii=False))
                self.redis.expire(key, 7 * 24 * 3600)  # 7天自动过期
            except Exception as e:
                logger.debug(f"[Flywheel] Redis写入失败: {e}")

        # 3. 写入 JSONL（冷备）
        try:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.debug(f"[Flywheel] JSONL写入失败: {e}")

        # 4. 更新策略计数
        if event_type in ("reflection_fail", "retrieval_empty"):
            self._strategy["total_failures"] += 1
        self._strategy["total_requests"] += 1

    # ==================== 策略自适应 ====================

    def adapt_retrieval_strategy(self, intent: Optional[str] = None) -> dict:
        """
        检索策略自适应

        规则：
        1. reflection_fail ≥ adapt_threshold → top_k + 5
        2. retrieval_empty ≥ adapt_threshold → 启用KG扩展
        3. 连续retry → 启用改写提示词补充

        Returns:
            调整后的检索参数，供 RetrieverNode 使用
        """
        recent_fails = self._count_recent("reflection_fail")
        recent_empty = self._count_recent("retrieval_empty")

        boost = 0

        if recent_fails >= self.adapt_threshold:
            boost += 5
            logger.info(
                f"[Flywheel] 策略自适应: top_k +5 (reflection_fail={recent_fails} >= {self.adapt_threshold})"
            )
            self._strategy["top_k_boost"] = boost

        if recent_empty >= self.adapt_threshold:
            self._strategy["kg_enabled"] = True
            logger.info(
                f"[Flywheel] 策略自适应: KG扩展已启用 (retrieval_empty={recent_empty} >= {self.adapt_threshold})"
            )

        self._strategy["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")

        # 记录策略变更
        if boost > 0:
            self.log_event("strategy_change", {
                "reason": f"连续失败: reflection_fail={recent_fails}, retrieval_empty={recent_empty}",
                "adjustments": {"top_k_boost": boost, "kg_enabled": self._strategy["kg_enabled"]},
            })

        return {
            "top_k_boost": self._strategy["top_k_boost"],
            "kg_enabled": self._strategy["kg_enabled"],
            "rewrite_hints": self._strategy["rewrite_hints"],
        }

    def get_current_strategy(self) -> dict:
        """获取当前策略状态"""
        return dict(self._strategy)

    # ==================== 指标查询 ====================

    def get_metrics(self, days: int = 7) -> dict:
        """
        获取飞轮聚合指标

        Args:
            days: 统计最近N天

        Returns:
            {
                total_requests, total_failures, failure_rate,
                avg_confidence, retry_distribution, top_failed_queries,
                events_by_type, current_strategy
            }
        """
        # 过滤最近N天的事件
        cutoff_date = time.strftime("%Y%m%d", time.localtime(time.time() - days * 86400))
        recent_events = [
            e for e in self._inmemory_events
            if e.get("date", "00000000") >= cutoff_date
        ]

        total = len(recent_events)
        if total == 0:
            return {
                "status": "no_data",
                "total_events": 0,
                "days": days,
                "current_strategy": self._strategy,
            }

        # 按类型统计
        by_type = Counter(e["event_type"] for e in recent_events)

        # 失败率
        failures = by_type.get("reflection_fail", 0) + by_type.get("retrieval_empty", 0)

        # 平均置信度（从reflection_fail事件中提取）
        confidences = []
        for e in recent_events:
            if e["event_type"] == "reflection_fail":
                conf = e.get("payload", {}).get("confidence", 0)
                if conf:
                    confidences.append(conf)

        avg_confidence = sum(confidences) / len(confidences) if confidences else 1.0

        # 重试分布
        retry_counts = Counter()
        for e in recent_events:
            if e["event_type"] == "reflection_fail":
                rc = e.get("payload", {}).get("retry_count", 0)
                retry_counts[rc] += 1

        # 高频失败查询（最近20条）
        failed_queries = []
        for e in recent_events:
            if e["event_type"] == "reflection_fail":
                q = e.get("payload", {}).get("original_query", "")
                if q:
                    failed_queries.append({
                        "query": q[:100],
                        "reason": e.get("payload", {}).get("reason", "")[:100],
                    })
                if len(failed_queries) >= 20:
                    break

        return {
            "status": "ok",
            "days": days,
            "total_events": total,
            "events_by_type": dict(by_type),
            "failure_rate": round(failures / total, 3) if total > 0 else 0,
            "avg_confidence": round(avg_confidence, 3),
            "retry_distribution": dict(retry_counts),
            "recent_failed_queries": failed_queries[:10],
            "current_strategy": self._strategy,
        }

    def get_recent_failures(self, limit: int = 50) -> List[dict]:
        """获取最近的失败事件"""
        fails = [
            e for e in self._inmemory_events
            if e["event_type"] in ("reflection_fail", "retrieval_empty")
        ]
        return fails[-limit:]

    def reset_strategy(self):
        """重置自适应策略"""
        self._strategy = {
            "top_k_boost": 0,
            "kg_enabled": True,
            "rewrite_hints": [],
            "total_requests": 0,
            "total_failures": 0,
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        logger.info("[Flywheel] 策略已重置")

    # ==================== 内部方法 ====================

    def _count_recent(self, event_type: str, window: int = 50) -> int:
        """统计最近的某种事件次数"""
        recent = self._inmemory_events[-window:]
        return sum(1 for e in recent if e["event_type"] == event_type)


# ==================== 全局单例 ====================

_global_flywheel: Optional[FlywheelService] = None


def get_flywheel(redis_client=None) -> FlywheelService:
    """获取全局飞轮服务单例"""
    global _global_flywheel
    if _global_flywheel is None:
        _global_flywheel = FlywheelService(redis_client=redis_client)
    return _global_flywheel
