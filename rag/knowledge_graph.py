"""
知识图谱模块 — 轻量级实现（无需neo4j等外部依赖）

基于内存中已有的 all_text_chunks 构建：
1. 实体提取：jieba分词 + TF筛选 + 词性过滤
2. 实体-文档索引：实体 → chunk_id 映射（用于直接命中检索）
3. 共现矩阵：实体在文档块中的共现频率（用于一跳扩散）

使用方式：
    kg = KnowledgeGraph()
    kg.build_index(text_chunks)             # 从文本块构建
    expanded = kg.one_hop_expand("石头P10") # 获取一跳扩散实体词
    info = kg.get_entity_info("石头P10")    # 获取实体详情
"""
import os
import re
import pickle
import time
from typing import List, Dict, Set, Optional, Tuple
from collections import Counter, defaultdict

import jieba

from utils.logger_handler import logger
from utils.path_tool import get_abs_path


# 停用词（扫地机器人领域的通用词，不适合作为实体）
_DEFAULT_STOPWORDS = {
    "可以", "使用", "需要", "一个", "进行", "这个", "那个",
    "什么", "怎么", "如何", "为什么", "哪些", "哪个",
    "是否", "没有", "已经", "还是", "或者", "以及",
    "因为", "所以", "但是", "而且", "然后", "如果",
    "我们", "他们", "它们", "这个", "这样", "那样",
    "通过", "根据", "对于", "关于", "经过", "由于",
    "目前", "现在", "以前", "以后", "之前", "之后",
    "比较", "非常", "特别", "更加", "一定", "可能",
    "包括", "含有", "具有", "拥有", "存在", "发生",
    "的", "了", "在", "是", "有", "和", "就",
    "不", "人", "都", "一", "很", "去", "能",
    "到", "说", "要", "会", "也", "着", "被",
    "扫地机器人", "扫拖", "机器人", "扫地", "扫拖一体",  # 太泛化
    "辅助", "不用", "之内", "清洗",
}

# 强制实体词（领域专有名词，即使被jieba拆开也要保留）
_FORCED_ENTITIES = [
    "石头P10", "石头P10 Pro", "科沃斯T30", "科沃斯X2", "追觅X30",
    "LDS激光导航", "dToF导航", "AI双目视觉", "HEPA滤网",
    "BGE-Reranker", "ChromaDB", "LangGraph", "LangChain",
    "通义千问", "DashScope",
]


class KnowledgeGraph:
    """
    轻量级知识图谱

    核心数据结构：
    - entity_to_chunks: 实体 → 所在chunk ID列表（PATH-C: 实体直接命中检索）
    - co_occurrence: 实体 → {邻居实体: 共现次数}（PATH-B: 一跳扩散）
    - chunk_entities: chunk ID → 所含实体列表
    - entity_freq: 实体 → 总出现次数（用于TF筛选和排序）
    """

    DATA_DIR = "rag/kg_data"
    INDEX_FILE = "entity_index.pkl"

    def __init__(self):
        self.entity_to_chunks: Dict[str, List[int]] = defaultdict(list)
        self.co_occurrence: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.chunk_entities: Dict[int, List[str]] = defaultdict(list)
        self.entity_freq: Counter = Counter()
        self._is_built = False
        self._total_chunks = 0

    # ==================== 索引构建 ====================

    def build_index(self, text_chunks: List[str], verbose: bool = True):
        """
        从文本块列表构建索引

        Args:
            text_chunks: 文本块内容列表
            verbose: 是否输出详细日志
        """
        start_time = time.time()
        self._total_chunks = len(text_chunks)

        if verbose:
            logger.info(f"[KG] 开始构建实体索引: {self._total_chunks} chunks")

        for chunk_id, text in enumerate(text_chunks):
            entities = self._extract_entities(text)

            if not entities:
                continue

            # 更新 chunk_entities
            self.chunk_entities[chunk_id] = list(entities)

            # 更新 entity_to_chunks 和 entity_freq
            for entity in entities:
                self.entity_to_chunks[entity].append(chunk_id)
                self.entity_freq[entity] += 1

            # 更新 co_occurrence（chunk内实体两两共现）
            entity_list = list(entities)
            for i in range(len(entity_list)):
                for j in range(i + 1, len(entity_list)):
                    e1, e2 = entity_list[i], entity_list[j]
                    self.co_occurrence[e1][e2] += 1
                    self.co_occurrence[e2][e1] += 1

            # 进度
            if verbose and (chunk_id + 1) % max(1, self._total_chunks // 5) == 0:
                logger.info(f"[KG] 索引构建进度: {chunk_id + 1}/{self._total_chunks}")

        self._is_built = True
        elapsed = time.time() - start_time

        if verbose:
            logger.info(
                f"[KG] 索引构建完成: {len(self.entity_to_chunks)} entities, "
                f"{self._total_chunks} chunks, elapsed={elapsed:.2f}s"
            )

    def load_or_build(self, text_chunks: Optional[List[str]] = None):
        """
        加载已有索引或构建新索引

        优先从 pickle 文件加载，不存在则从 text_chunks 构建
        """
        if self._try_load():
            return

        if text_chunks:
            self.build_index(text_chunks)
            self._try_save()

    # ==================== 查询接口 ====================

    def one_hop_expand(
        self,
        query: str,
        max_entities: int = 3,
        max_hops: int = 1
    ) -> List[str]:
        """
        一跳扩散：从查询中提取实体，获取其最高频的共现邻居

        用于扩展检索查询词（PATH-B: 查询扩展检索）

        Args:
            query: 用户查询文本
            max_entities: 从查询中提取的最大实体数
            max_hops: 扩散跳数（当前仅支持1）

        Returns:
            扩展词列表（可作为额外检索词使用）
        """
        if not self._is_built:
            return []

        # 1. 从查询中提取实体
        query_entities = self._extract_entities(query, top_k=max_entities)

        if not query_entities:
            return []

        # 2. 一跳扩散：收集所有共现邻居
        expanded = set()
        for entity in query_entities:
            neighbors = self.co_occurrence.get(entity, {})
            # 按共现频率排序，取top-3
            top_neighbors = sorted(neighbors.items(), key=lambda x: x[1], reverse=True)[:3]
            for neighbor, _ in top_neighbors:
                if neighbor not in query_entities:
                    expanded.add(neighbor)

        result = list(expanded)[:max_entities * 2]
        if result:
            logger.debug(f"[KG] 一跳扩散: {query_entities} → {result}")

        return result

    def get_entity_info(self, entity_name: str) -> Optional[dict]:
        """
        获取实体详细信息

        Args:
            entity_name: 实体名称

        Returns:
            {name, frequency, chunk_count, related_entities: [...], top_chunks: [...]}
        """
        if not self._is_built:
            return None

        chunks = self.entity_to_chunks.get(entity_name, [])
        if not chunks:
            # 模糊匹配
            for key in self.entity_to_chunks:
                if entity_name in key:
                    chunks = self.entity_to_chunks[key]
                    entity_name = key
                    break

        if not chunks:
            return None

        neighbors = self.co_occurrence.get(entity_name, {})
        top_neighbors = sorted(neighbors.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "name": entity_name,
            "frequency": self.entity_freq.get(entity_name, 0),
            "chunk_count": len(chunks),
            "related_entities": [
                {"entity": n, "co_occurrence": c}
                for n, c in top_neighbors
            ],
            "top_chunks": chunks[:5],
        }

    def get_chunks_by_entity(self, entity_name: str) -> List[int]:
        """
        按实体获取对应的chunk ID列表（PATH-C: 实体直接命中检索）

        Args:
            entity_name: 实体名称

        Returns:
            chunk ID列表
        """
        return self.entity_to_chunks.get(entity_name, [])

    def search_entities(self, keyword: str, limit: int = 10) -> List[dict]:
        """
        按关键词搜索实体

        Args:
            keyword: 搜索关键词
            limit: 返回数量

        Returns:
            匹配的实体列表 [{name, frequency, chunk_count}]
        """
        if not self._is_built:
            return []

        results = []
        for entity, freq in self.entity_freq.items():
            if keyword in entity:
                results.append({
                    "name": entity,
                    "frequency": freq,
                    "chunk_count": len(self.entity_to_chunks.get(entity, [])),
                })

        results.sort(key=lambda x: x["frequency"], reverse=True)
        return results[:limit]

    @property
    def is_built(self) -> bool:
        return self._is_built

    @property
    def entity_count(self) -> int:
        return len(self.entity_to_chunks)

    # ==================== 实体提取 ====================

    def _extract_entities(self, text: str, top_k: Optional[int] = None) -> Set[str]:
        """
        从文本中提取实体

        策略：
        1. 先匹配强制实体词（领域专有名词）
        2. jieba分词 + 词性过滤（名词）
        3. 停用词过滤
        4. 长度过滤（≥2字符）
        5. TF排序取top_k

        Args:
            text: 输入文本
            top_k: 返回top-k个实体（None=全部）

        Returns:
            实体集合
        """
        entities = set()

        # 1. 强制实体词匹配
        for forced in _FORCED_ENTITIES:
            if forced in text:
                entities.add(forced)

        # 2. jieba分词 + 词性过滤
        words = jieba.cut(text)
        for word in words:
            word = word.strip()
            # 长度过滤
            if len(word) < 2:
                continue
            # 停用词过滤
            if word in _DEFAULT_STOPWORDS:
                continue
            # 数字/纯标点过滤
            if word.isdigit() or re.match(r'^[^\w]+$', word):
                continue

            entities.add(word)

        # 3. 如果指定 top_k，按频率排序
        if top_k and len(entities) > top_k:
            entity_scores = [(e, self.entity_freq.get(e, 1)) for e in entities]
            entity_scores.sort(key=lambda x: x[1], reverse=True)
            entities = {e for e, _ in entity_scores[:top_k]}

        return entities

    # ==================== 持久化 ====================

    def _try_load(self) -> bool:
        """尝试从pickle文件加载索引"""
        try:
            path = get_abs_path(os.path.join(self.DATA_DIR, self.INDEX_FILE))
            if not os.path.exists(path):
                return False

            with open(path, "rb") as f:
                data = pickle.load(f)

            self.entity_to_chunks = data["entity_to_chunks"]
            self.co_occurrence = data["co_occurrence"]
            self.chunk_entities = data["chunk_entities"]
            self.entity_freq = data["entity_freq"]
            self._total_chunks = data.get("total_chunks", 0)
            self._is_built = True

            logger.info(
                f"[KG] 索引已从文件加载: {len(self.entity_to_chunks)} entities, "
                f"{self._total_chunks} chunks"
            )
            return True

        except Exception as e:
            logger.warning(f"[KG] 加载索引失败: {e}")
            return False

    def _try_save(self) -> bool:
        """尝试将索引保存到pickle文件"""
        try:
            dir_path = get_abs_path(self.DATA_DIR)
            os.makedirs(dir_path, exist_ok=True)

            path = get_abs_path(os.path.join(self.DATA_DIR, self.INDEX_FILE))
            data = {
                "entity_to_chunks": dict(self.entity_to_chunks),
                "co_occurrence": {k: dict(v) for k, v in self.co_occurrence.items()},
                "chunk_entities": dict(self.chunk_entities),
                "entity_freq": self.entity_freq,
                "total_chunks": self._total_chunks,
            }

            with open(path, "wb") as f:
                pickle.dump(data, f)

            logger.info(f"[KG] 索引已保存到: {path}")
            return True

        except Exception as e:
            logger.warning(f"[KG] 保存索引失败: {e}")
            return False


# ==================== 模块级便捷函数 ====================

_global_kg: Optional[KnowledgeGraph] = None


def get_knowledge_graph() -> KnowledgeGraph:
    """获取全局知识图谱单例"""
    global _global_kg
    if _global_kg is None:
        _global_kg = KnowledgeGraph()
    return _global_kg


def init_knowledge_graph(text_chunks: Optional[List[str]] = None):
    """初始化全局知识图谱"""
    kg = get_knowledge_graph()
    if not kg.is_built:
        kg.load_or_build(text_chunks)
    return kg
