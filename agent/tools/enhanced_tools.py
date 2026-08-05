"""
增强工具集 —— 企业级Function Calling工具实现

设计原则：
1. 每个工具用 Pydantic 定义 Input/Output 模型（强类型、可测试）
2. JSON Schema 自动从 Pydantic 生成（给LLM的工具描述）
3. 统一错误返回格式 {"success": bool, "data": ..., "error": ..., "_meta": {...}}
4. 每个工具独立可测试，不依赖全局状态
5. 与现有 RAG 服务、混合检索器、向量存储无缝集成

工具分类：
- search:      文档检索、实体查询、关键词匹配、相似推荐
- data_query:  用户使用记录、设备规格查询、型号对比
- external_api: 天气查询、保养周期、固件版本
- analysis:    使用习惯分析、耗材预估、故障诊断
- system:      用户上下文、反馈记录
"""
import json
import os
import random
import time
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from utils.logger_handler import logger


# ============================================================
# Pydantic 输入/输出模型 (JSON Schema 自动生成)
# ============================================================

# ---------- Search Tools ----------

class HybridSearchInput(BaseModel):
    """混合检索输入"""
    query: str = Field(..., description="搜索查询词，自然语言描述")
    top_k: int = Field(default=5, ge=1, le=20, description="返回结果数量")
    filters: Optional[Dict[str, str]] = Field(
        default=None,
        description="元数据过滤条件，如 {'category': '选购指南', 'source': 'knowledge_base'}"
    )

class SearchResult(BaseModel):
    """单条检索结果"""
    content: str = Field(..., description="文档内容片段")
    source: str = Field(..., description="来源文件路径")
    page: Optional[int] = Field(default=None, description="页码（如PDF）")
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0, description="相关性分数")

class HybridSearchOutput(BaseModel):
    """混合检索输出"""
    results: List[SearchResult] = Field(default_factory=list)
    total_found: int = Field(default=0, description="检索到的总文档数")
    retrieval_method: str = Field(default="hybrid", description="检索方式")


class EntityLookupInput(BaseModel):
    """实体查找输入"""
    entity_name: str = Field(..., description="实体名称，如'石头P10'、'科沃斯T30'")
    entity_type: Optional[str] = Field(
        default=None,
        description="实体类型，如'model'(型号)、'brand'(品牌)、'part'(配件)"
    )

class EntityLookupOutput(BaseModel):
    """实体查找输出"""
    entity: Dict[str, Any] = Field(default_factory=dict, description="实体属性字典")
    related_docs: List[Dict[str, str]] = Field(default_factory=list, description="关联文档列表")
    found: bool = Field(default=False)


class KeywordSearchInput(BaseModel):
    """关键词检索输入"""
    keywords: str = Field(..., description="关键词，空格分隔，如'小户型 推荐 静音'")
    top_k: int = Field(default=5, ge=1, le=20, description="返回结果数量")


class KeywordSearchOutput(BaseModel):
    """关键词检索输出"""
    results: List[SearchResult] = Field(default_factory=list)
    total_found: int = Field(default=0)
    method: str = Field(default="bm25")


class SimilarDocInput(BaseModel):
    """相似文档输入"""
    doc_id: str = Field(..., description="参考文档ID")
    top_k: int = Field(default=5, ge=1, le=10, description="推荐数量")


# ---------- Data Query Tools ----------

class QueryUserUsageInput(BaseModel):
    """用户使用记录查询输入"""
    user_id: str = Field(..., description="用户ID，如'1001'")
    month: str = Field(..., description="查询月份，格式YYYY-MM，如'2025-06'")
    metrics: List[str] = Field(
        default=["cleaning_area", "duration", "mode"],
        description="关注指标: cleaning_area(清洁面积), duration(工作时长), mode(清洁模式), efficiency(效率), consumable(耗材)"
    )

class UsageRecord(BaseModel):
    """使用记录"""
    user_id: str
    month: str
    cleaning_area: str = ""
    duration: str = ""
    mode: str = ""
    efficiency: str = ""
    consumable: str = ""
    comparison: str = ""


class GetDeviceSpecsInput(BaseModel):
    """设备规格查询输入"""
    model_name: str = Field(..., description="型号名称，如'石头P10'、'科沃斯T30'")


class DeviceSpecs(BaseModel):
    """设备规格"""
    model: str = Field(..., description="型号")
    brand: str = Field(default="", description="品牌")
    suction_power: str = Field(default="", description="吸力(Pa)")
    battery_capacity: str = Field(default="", description="电池容量(mAh)")
    dust_bin_volume: str = Field(default="", description="尘盒容量(mL)")
    water_tank_volume: str = Field(default="", description="水箱容量(mL)")
    noise_level: str = Field(default="", description="噪音(dB)")
    weight: str = Field(default="", description="重量(kg)")
    features: List[str] = Field(default_factory=list, description="特色功能列表")
    suitable_areas: List[str] = Field(default_factory=list, description="适用场景")


class CompareModelsInput(BaseModel):
    """型号对比输入"""
    model_list: List[str] = Field(..., min_length=1, max_length=5, description="待对比型号列表")

class ModelComparison(BaseModel):
    """型号对比结果"""
    models: List[DeviceSpecs] = Field(default_factory=list)
    comparison_table: Dict[str, Dict[str, str]] = Field(default_factory=dict)


# ---------- External API Tools ----------

class GetWeatherInput(BaseModel):
    """天气查询输入"""
    city: str = Field(..., description="城市名称，如'深圳'、'北京'")


class WeatherInfo(BaseModel):
    """天气信息"""
    city: str
    weather: str = ""
    temperature: str = ""
    humidity: str = ""
    wind: str = ""
    aqi: str = ""
    rain_probability: str = ""


class GetMaintenanceInput(BaseModel):
    """保养周期输入"""
    model_name: str = Field(..., description="设备型号")


class MaintenanceSchedule(BaseModel):
    """保养计划"""
    model: str
    items: List[Dict[str, str]] = Field(default_factory=list)


class CheckFirmwareInput(BaseModel):
    """固件版本查询输入"""
    model_name: str = Field(..., description="设备型号")


class FirmwareInfo(BaseModel):
    """固件信息"""
    model: str
    current_version: str = ""
    latest_version: str = ""
    update_available: bool = False
    release_notes: str = ""


# ---------- Analysis Tools ----------

class AnalyzeUsageInput(BaseModel):
    """使用习惯分析输入"""
    user_id: str = Field(..., description="用户ID")
    months: List[str] = Field(..., min_length=1, max_length=12, description="分析月份列表")

class UsagePattern(BaseModel):
    """使用模式"""
    avg_cleaning_area: str = ""
    preferred_mode: str = ""
    peak_usage_time: str = ""
    trend: str = ""  # increasing/stable/decreasing


class CalculateConsumableInput(BaseModel):
    """耗材预估输入"""
    part_name: str = Field(..., description="耗材名称，如'主刷'、'边刷'、'滤网'、'拖布'")
    usage_hours: float = Field(..., ge=0, description="累计使用小时数")

class ConsumableLife(BaseModel):
    """耗材寿命"""
    part: str
    estimated_total_life_hours: float
    used_hours: float
    remaining_hours: float
    remaining_percent: float
    recommendation: str


class TroubleshootInput(BaseModel):
    """故障诊断输入"""
    symptom: str = Field(..., description="故障现象描述，如'无法充电'、'噪音大'、'不扫拖'、'频繁报错'")
    model: Optional[str] = Field(default=None, description="设备型号（可选，有助精准诊断）")

class TSStep(BaseModel):
    """诊断步骤"""
    step_number: int
    action: str
    expected_result: str

class TroubleshootOutput(BaseModel):
    """故障诊断输出"""
    symptom: str
    model: Optional[str] = None
    possible_causes: List[str] = Field(default_factory=list)
    diagnostic_steps: List[Dict[str, Any]] = Field(default_factory=list)
    solutions: List[str] = Field(default_factory=list)


# ---------- System Tools ----------

class UserContext(BaseModel):
    """用户上下文"""
    user_id: str
    city: str = ""
    devices: List[str] = Field(default_factory=list)
    preferences: Dict[str, str] = Field(default_factory=dict)


class FeedbackInput(BaseModel):
    """用户反馈输入"""
    query_id: str = Field(..., description="回答ID")
    rating: int = Field(..., ge=1, le=5, description="评分1-5")
    comment: Optional[str] = Field(default=None, description="反馈备注")


# ============================================================
# 工具处理函数
# ============================================================

# ---- 全局变量：延迟初始化，避免导入时构造重服务 ----
_rag_service = None
_kg_instance = None
_flywheel_instance = None

_device_specs_db: Dict[str, dict] = {}
_external_data_cache: Dict[str, Dict[str, dict]] = {}


def _get_rag_service():
    """延迟获取 RAG 服务实例"""
    global _rag_service
    if _rag_service is None:
        try:
            from rag.rag_service import RagSummarizeService
            _rag_service = RagSummarizeService(use_langgraph=True, enable_structured_output=True)
            logger.info("[增强工具] RAG服务已延迟初始化")
        except Exception as e:
            logger.error(f"[增强工具] RAG服务初始化失败: {e}")
    return _rag_service


def _get_kg():
    """延迟获取知识图谱实例"""
    global _kg_instance
    if _kg_instance is None:
        try:
            from rag.knowledge_graph import KnowledgeGraph
            _kg_instance = KnowledgeGraph()
            logger.info("[增强工具] 知识图谱已延迟初始化")
        except Exception as e:
            logger.debug(f"[增强工具] 知识图谱不可用: {e}")
    return _kg_instance


def _get_flywheel():
    """延迟获取飞轮服务实例"""
    global _flywheel_instance
    if _flywheel_instance is None:
        try:
            from services.FlywheelService import FlywheelService
            _flywheel_instance = FlywheelService()
            logger.info("[增强工具] 飞轮服务已延迟初始化")
        except Exception as e:
            logger.debug(f"[增强工具] 飞轮服务不可用: {e}")
    return _flywheel_instance


def _init_device_specs():
    """初始化扫地机器人设备规格数据库"""
    global _device_specs_db
    if not _device_specs_db:
        _device_specs_db = {
            "石头P10": {
                "model": "石头P10",
                "brand": "石头/roborock",
                "suction_power": "5500Pa",
                "battery_capacity": "5200mAh",
                "dust_bin_volume": "400mL",
                "water_tank_volume": "200mL",
                "noise_level": "63dB",
                "weight": "3.8kg",
                "features": ["LDS激光导航", "超声波地毯识别", "智能回充", "APP远程控制", "语音控制"],
                "suitable_areas": ["中小户型", "硬地板", "短毛地毯", "有宠物家庭"],
            },
            "石头P10 Pro": {
                "model": "石头P10 Pro",
                "brand": "石头/roborock",
                "suction_power": "7000Pa",
                "battery_capacity": "5200mAh",
                "dust_bin_volume": "400mL",
                "water_tank_volume": "200mL",
                "noise_level": "65dB",
                "weight": "4.0kg",
                "features": ["LDS激光导航", "AI避障", "热风烘干", "自动集尘", "语音控制"],
                "suitable_areas": ["中大户型", "多种地板", "有宠物家庭", "复杂环境"],
            },
            "科沃斯T30": {
                "model": "科沃斯T30",
                "brand": "科沃斯/ECOVACS",
                "suction_power": "5000Pa",
                "battery_capacity": "4800mAh",
                "dust_bin_volume": "350mL",
                "water_tank_volume": "180mL",
                "noise_level": "60dB",
                "weight": "3.5kg",
                "features": ["dToF导航", "AI视觉识别", "OTA升级", "YIKO语音助手"],
                "suitable_areas": ["中小户型", "硬地板", "低矮家具"],
            },
            "科沃斯X2": {
                "model": "科沃斯X2",
                "brand": "科沃斯/ECOVACS",
                "suction_power": "8000Pa",
                "battery_capacity": "6400mAh",
                "dust_bin_volume": "420mL",
                "water_tank_volume": "220mL",
                "noise_level": "67dB",
                "weight": "4.2kg",
                "features": ["半固态激光雷达", "AI双目视觉", "自动洗拖布", "自动集尘", "热风烘干", "全链路自清洁"],
                "suitable_areas": ["大户型", "所有地板类型", "商用环境"],
            },
            "追觅X30": {
                "model": "追觅X30",
                "brand": "追觅/Dreame",
                "suction_power": "8300Pa",
                "battery_capacity": "5200mAh",
                "dust_bin_volume": "300mL",
                "water_tank_volume": "200mL",
                "noise_level": "68dB",
                "weight": "4.0kg",
                "features": ["LDS导航", "AI避障2.0", "自动洗拖布", "自动集尘", "银离子除菌"],
                "suitable_areas": ["所有户型", "硬地板", "有宠物家庭"],
            },
        }


def _init_external_data():
    """初始化外部使用数据"""
    global _external_data_cache
    if not _external_data_cache:
        try:
            from utils.config_handler import agent_conf
            from utils.path_tool import get_abs_path
            data_path = get_abs_path(agent_conf.get("external_data_path", "data/external/records.csv"))
            if os.path.exists(data_path):
                with open(data_path, "r", encoding="utf-8") as f:
                    for line in f.readlines()[1:]:
                        arr = line.strip().replace('"', '').split(",")
                        if len(arr) >= 6:
                            uid, feat, eff, cons, comp, mon = arr[0], arr[1], arr[2], arr[3], arr[4], arr[5]
                            _external_data_cache.setdefault(uid, {})[mon] = {
                                "特征": feat, "效率": eff, "耗材": cons, "对比": comp
                            }
                logger.info(f"[增强工具] 外部数据已加载，共 {len(_external_data_cache)} 个用户")
        except Exception as e:
            logger.warning(f"[增强工具] 外部数据加载失败: {e}")


# ============================================================
# Search Tools (检索类)
# ============================================================

def hybrid_search_tool(query: str, top_k: int = 5, filters: dict = None) -> dict:
    """
    执行混合检索（向量 + BM25 + BGE重排序）

    这是核心检索工具，整合了项目现有的 HybridRetriever 能力。
    支持元数据过滤（按category/source等字段筛选）。
    """
    try:
        rag = _get_rag_service()
        if rag is None:
            return {"success": False, "data": None, "error": "RAG服务未初始化"}

        # 使用现有的混合检索器
        docs = rag.retriever_docs(query)

        # 可选：元数据过滤
        if filters:
            filtered_docs = []
            for doc in docs:
                metadata = doc.metadata
                match = all(metadata.get(k) == v for k, v in filters.items())
                if match:
                    filtered_docs.append(doc)
            docs = filtered_docs

        # 截断到 top_k
        docs = docs[:top_k]

        results = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "未知来源")
            page = doc.metadata.get("page")
            results.append(SearchResult(
                content=doc.page_content[:500],
                source=source,
                page=int(page) if page else None,
                relevance_score=round(1.0 - (i * 0.1), 2),  # 位置作为近似分数
            ))

        return {
            "success": True,
            "data": HybridSearchOutput(
                results=results,
                total_found=len(docs),
                retrieval_method="hybrid"
            ).model_dump(),
            "error": None,
        }
    except Exception as e:
        logger.error(f"[hybrid_search] 检索失败: {e}", exc_info=True)
        return {"success": False, "data": None, "error": str(e)}


def entity_lookup_tool(entity_name: str, entity_type: str = None) -> dict:
    """
    知识图谱实体精确查找

    从设备规格库和KG索引中查找实体信息，
    返回实体属性 + 关联文档列表
    """
    try:
        _init_device_specs()
        entity_data = {}
        related_docs = []
        found = False

        # 1. 先查设备规格库
        if entity_name in _device_specs_db:
            entity_data = {"type": "device_model", **_device_specs_db[entity_name]}
            found = True

        # 2. 扩展属性（从KG获取）
        kg = _get_kg()
        if kg is not None:
            try:
                kg_info = kg.get_entity_info(entity_name)
                if kg_info:
                    entity_data["kg_relations"] = kg_info
                    if not found:
                        entity_data["name"] = entity_name
                        found = True
            except Exception:
                pass

        # 3. 检索关联文档
        rag = _get_rag_service()
        if rag is not None:
            try:
                docs = rag.retriever_docs(entity_name)
                for doc in docs[:3]:
                    related_docs.append({
                        "content": doc.page_content[:300],
                        "source": doc.metadata.get("source", "未知"),
                    })
            except Exception:
                pass

        return {
            "success": True,
            "data": EntityLookupOutput(
                entity=entity_data,
                related_docs=related_docs,
                found=found,
            ).model_dump(),
            "error": None,
        }
    except Exception as e:
        logger.error(f"[entity_lookup] 实体查找失败: {e}", exc_info=True)
        return {"success": False, "data": None, "error": str(e)}


def keyword_search_tool(keywords: str, top_k: int = 5) -> dict:
    """
    纯BM25关键词检索

    绕过向量检索，直接使用BM25算法进行关键词匹配，
    适合精确术语搜索
    """
    try:
        rag = _get_rag_service()
        if rag is None:
            return {"success": False, "data": None, "error": "RAG服务未初始化"}

        # 通过现有混合检索器的BM25路径
        vs = rag.vector_store
        if hasattr(vs, 'hybrid_retriever') and vs.hybrid_retriever:
            # 设置 alpha=1.0 即纯BM25
            docs = vs.hybrid_retriever.hybrid_search(
                query=keywords,
                top_k_retrieve=top_k * 2,
                top_k_rerank=top_k,
                alpha=1.0  # 纯BM25
            )
        else:
            docs = vs.vector_store.similarity_search(keywords, k=top_k)

        results = [
            SearchResult(
                content=doc.page_content[:500],
                source=doc.metadata.get("source", "未知来源"),
                relevance_score=round(0.9 - (i * 0.1), 2),
            ).model_dump()
            for i, doc in enumerate(docs[:top_k])
        ]

        return {
            "success": True,
            "data": KeywordSearchOutput(
                results=[SearchResult(**r) for r in results],
                total_found=len(docs),
                method="bm25"
            ).model_dump(),
            "error": None,
        }
    except Exception as e:
        logger.error(f"[keyword_search] 检索失败: {e}", exc_info=True)
        return {"success": False, "data": None, "error": str(e)}


def similar_doc_search_tool(doc_id: str, top_k: int = 5) -> dict:
    """
    相似文档推荐

    基于向量相似度查找与指定文档最相似的其他文档
    """
    try:
        rag = _get_rag_service()
        if rag is None:
            return {"success": False, "data": None, "error": "RAG服务未初始化"}

        vs = rag.vector_store
        # 通过向量库查询相似文档
        results = vs.vector_store.similarity_search_by_vector(
            embedding=vs.vector_store._collection.get(ids=[doc_id])['embeddings'][0],
            k=top_k + 1  # +1 排除自身
        )
        results = [r for r in results if r.metadata.get("id") != doc_id][:top_k]

        docs = [
            {"content": r.page_content[:300], "source": r.metadata.get("source", "未知")}
            for r in results
        ]
        return {"success": True, "data": {"similar_docs": docs, "count": len(docs)}, "error": None}
    except Exception as e:
        logger.error(f"[similar_doc_search] 检索失败: {e}", exc_info=True)
        return {"success": False, "data": None, "error": str(e)}


# ============================================================
# Data Query Tools (数据查询类)
# ============================================================

def query_user_usage_tool(user_id: str, month: str, metrics: List[str] = None) -> dict:
    """
    查询用户使用记录

    从外部数据源查询指定用户在指定月份的使用数据
    """
    try:
        _init_external_data()
        metrics = metrics or ["cleaning_area", "duration", "mode"]

        if user_id not in _external_data_cache:
            return {
                "success": True,
                "data": {"user_id": user_id, "month": month, "found": False, "records": {}},
                "error": f"用户 {user_id} 无使用记录",
            }

        user_data = _external_data_cache[user_id]
        if month not in user_data:
            return {
                "success": True,
                "data": {"user_id": user_id, "month": month, "found": False, "records": {}},
                "error": f"用户 {user_id} 在 {month} 无使用记录",
            }

        record = user_data[month]
        filtered_record = {k: v for k, v in record.items() if k in metrics or k in ["特征", "效率", "耗材", "对比"]}

        return {
            "success": True,
            "data": {
                "user_id": user_id,
                "month": month,
                "found": True,
                "records": filtered_record,
            },
            "error": None,
        }
    except Exception as e:
        logger.error(f"[query_user_usage] 查询失败: {e}", exc_info=True)
        return {"success": False, "data": None, "error": str(e)}


def get_device_specs_tool(model_name: str) -> dict:
    """
    查询设备规格参数

    从内置设备数据库中查询扫地机器人的详细规格
    """
    try:
        _init_device_specs()

        # 模糊匹配
        matched = None
        for key in _device_specs_db:
            if model_name in key or key in model_name:
                matched = key
                break

        if matched:
            specs = DeviceSpecs(**_device_specs_db[matched])
            return {
                "success": True,
                "data": specs.model_dump(),
                "error": None,
            }

        return {
            "success": True,
            "data": None,
            "error": f"未找到型号 '{model_name}' 的规格信息。已知型号: {list(_device_specs_db.keys())}",
        }
    except Exception as e:
        logger.error(f"[get_device_specs] 查询失败: {e}", exc_info=True)
        return {"success": False, "data": None, "error": str(e)}


def compare_models_tool(model_list: List[str]) -> dict:
    """
    多型号对比分析

    对多个扫地机器人型号进行参数对比，帮助用户决策
    """
    try:
        _init_device_specs()

        specs_list = []
        for model in model_list:
            matched = None
            for key in _device_specs_db:
                if model in key or key in model:
                    matched = key
                    break
            if matched:
                specs_list.append(DeviceSpecs(**_device_specs_db[matched]))

        if not specs_list:
            return {"success": False, "data": None, "error": "未找到任何匹配型号"}

        # 构建对比表
        compare_fields = ["suction_power", "battery_capacity", "dust_bin_volume", "water_tank_volume", "noise_level", "weight"]
        comparison = {}
        for field in compare_fields:
            row = {}
            for spec in specs_list:
                row[spec.model] = getattr(spec, field, "N/A")
            comparison[field] = row

        result = ModelComparison(
            models=specs_list,
            comparison_table=comparison
        )
        return {"success": True, "data": result.model_dump(), "error": None}
    except Exception as e:
        logger.error(f"[compare_models] 对比失败: {e}", exc_info=True)
        return {"success": False, "data": None, "error": str(e)}


# ============================================================
# External API Tools (外部接口类)
# ============================================================

def get_weather_tool(city: str) -> dict:
    """
    获取城市天气信息

    返回天气摘要，供Agent判断环境是否适合扫地机器人使用
    """
    try:
        # 模拟天气数据（保留原有工具的模式）
        weather_options = {
            "深圳": {"weather": "晴天", "temperature": "28°C", "humidity": "65%", "wind": "南风2级", "aqi": "35", "rain": "极低"},
            "北京": {"weather": "多云", "temperature": "18°C", "humidity": "30%", "wind": "北风3级", "aqi": "72", "rain": "无"},
            "上海": {"weather": "阴天", "temperature": "22°C", "humidity": "70%", "wind": "东风2级", "aqi": "55", "rain": "中等"},
            "杭州": {"weather": "小雨", "temperature": "20°C", "humidity": "85%", "wind": "西风1级", "aqi": "28", "rain": "高"},
            "广州": {"weather": "晴天", "temperature": "30°C", "humidity": "75%", "wind": "南风1级", "aqi": "42", "rain": "低"},
            "成都": {"weather": "阴天", "temperature": "19°C", "humidity": "80%", "wind": "无持续风向", "aqi": "60", "rain": "中等"},
        }

        info = weather_options.get(city, {"weather": "晴天", "temperature": "26°C", "humidity": "50%", "wind": "南风1级", "aqi": "21", "rain": "极低"})

        return {
            "success": True,
            "data": WeatherInfo(
                city=city,
                weather=info["weather"],
                temperature=info["temperature"],
                humidity=info["humidity"],
                wind=info["wind"],
                aqi=info["aqi"],
                rain_probability=info["rain"],
            ).model_dump(),
            "error": None,
        }
    except Exception as e:
        return {"success": False, "data": None, "error": str(e)}


def get_maintenance_schedule_tool(model_name: str) -> dict:
    """
    获取保养周期建议

    返回扫地机器人各部件的保养频率和操作方法
    """
    try:
        schedule = {
            "model": model_name,
            "items": [
                {"part": "主刷", "frequency": "每周清理缠绕毛发", "method": "使用清洁工具去除缠绕物"},
                {"part": "边刷", "frequency": "每2周检查磨损", "method": "螺丝刀更换，寿命约6-12个月"},
                {"part": "滤网(HEPA)", "frequency": "每周水洗晾干", "method": "清水冲洗，晾干后装回，寿命约3-6个月"},
                {"part": "拖布", "frequency": "每次使用后清洗", "method": "清水或中性洗涤剂清洗，寿命约2-3个月"},
                {"part": "传感器", "frequency": "每月清洁", "method": "干软布擦拭，保持传感器窗口清洁"},
                {"part": "充电触点", "frequency": "每月清洁", "method": "干布擦拭正负极触点"},
                {"part": "万向轮", "frequency": "每月清理", "method": "清除缠绕物，检查转动是否顺畅"},
            ],
        }
        return {"success": True, "data": schedule, "error": None}
    except Exception as e:
        return {"success": False, "data": None, "error": str(e)}


def check_firmware_version_tool(model_name: str) -> dict:
    """
    检查固件版本

    返回当前固件版本和是否有可用更新
    """
    try:
        firmware_db = {
            "石头P10": ("v4.18.0", "v4.20.1", True, "优化避障算法，提升地毯识别准确率"),
            "石头P10 Pro": ("v5.2.0", "v5.2.0", False, "已是最新版本"),
            "科沃斯T30": ("v3.8.5", "v3.9.2", True, "新增智能分区清洁功能，修复偶发断连问题"),
            "科沃斯X2": ("v6.1.0", "v6.1.0", False, "已是最新版本"),
            "追觅X30": ("v1.5.3", "v1.6.0", True, "提升拖地压力控制精度，优化回充效率"),
        }

        for key in firmware_db:
            if model_name in key or key in model_name:
                cur, latest, update_avail, notes = firmware_db[key]
                return {
                    "success": True,
                    "data": FirmwareInfo(
                        model=key,
                        current_version=cur,
                        latest_version=latest,
                        update_available=update_avail,
                        release_notes=notes,
                    ).model_dump(),
                    "error": None,
                }

        return {"success": True, "data": None, "error": f"未找到 {model_name} 的固件信息"}
    except Exception as e:
        return {"success": False, "data": None, "error": str(e)}


# ============================================================
# Analysis Tools (分析类)
# ============================================================

def analyze_usage_pattern_tool(user_id: str, months: List[str]) -> dict:
    """
    分析用户使用习惯

    基于多个月份的使用数据，分析使用模式和趋势
    """
    try:
        _init_external_data()

        if user_id not in _external_data_cache:
            return {"success": False, "data": None, "error": f"用户 {user_id} 无使用记录"}

        records = []
        for month in months:
            if month in _external_data_cache[user_id]:
                records.append(_external_data_cache[user_id][month])

        if not records:
            return {"success": False, "data": None, "error": "选定月份无使用记录"}

        # 简单分析
        efficiency_values = []
        modes = {}
        for r in records:
            eff = r.get("效率", "")
            if eff:
                try:
                    efficiency_values.append(float(eff.replace("%", "")))
                except ValueError:
                    pass
            feature = r.get("特征", "")
            if feature:
                modes[feature] = modes.get(feature, 0) + 1

        avg_eff = sum(efficiency_values) / len(efficiency_values) if efficiency_values else 0
        preferred = max(modes, key=modes.get) if modes else "未知"
        trend = "稳定" if len(efficiency_values) < 2 else (
            "提升" if efficiency_values[-1] > efficiency_values[0] else "下降"
        )

        return {
            "success": True,
            "data": UsagePattern(
                avg_cleaning_area=f"{avg_eff:.1f}%",
                preferred_mode=preferred,
                peak_usage_time="周末上午",
                trend=trend,
            ).model_dump(),
            "error": None,
        }
    except Exception as e:
        logger.error(f"[analyze_usage_pattern] 分析失败: {e}", exc_info=True)
        return {"success": False, "data": None, "error": str(e)}


def calculate_consumable_life_tool(part_name: str, usage_hours: float) -> dict:
    """
    耗材寿命预估

    根据使用时长估算耗材剩余寿命
    """
    try:
        life_db = {
            "主刷": 300,      # 约300小时
            "边刷": 200,
            "滤网": 150,
            "HEPA滤网": 150,
            "拖布": 100,
            "万向轮": 500,
            "电池": 1000,
        }

        total_life = life_db.get(part_name, 200)
        remaining = max(0, total_life - usage_hours)
        percent = round(remaining / total_life * 100, 1)

        if percent > 50:
            rec = f"{part_name}状态良好，无需更换"
        elif percent > 20:
            rec = f"{part_name}剩余寿命约{remaining:.0f}小时，建议开始准备备件"
        elif percent > 0:
            rec = f"{part_name}即将耗尽，建议尽快更换"
        else:
            rec = f"{part_name}已超过建议使用寿命，请立即更换"

        return {
            "success": True,
            "data": ConsumableLife(
                part=part_name,
                estimated_total_life_hours=total_life,
                used_hours=usage_hours,
                remaining_hours=remaining,
                remaining_percent=percent,
                recommendation=rec,
            ).model_dump(),
            "error": None,
        }
    except Exception as e:
        return {"success": False, "data": None, "error": str(e)}


def troubleshoot_issue_tool(symptom: str, model: str = None) -> dict:
    """
    故障诊断决策树

    输入故障现象，返回可能的诊断步骤和解决方案
    """
    try:
        # 故障决策树
        trouble_db = {
            "无法充电": {
                "causes": ["充电底座接触不良", "适配器损坏", "电池老化", "充电触点氧化"],
                "steps": [
                    {"step": 1, "action": "检查充电底座指示灯是否亮起", "expected": "指示灯应常亮"},
                    {"step": 2, "action": "用干布清洁充电触点和机器人底部触点", "expected": "触点光亮无氧化"},
                    {"step": 3, "action": "尝试直连适配器充电（如支持）", "expected": "机器人充电指示灯亮起"},
                    {"step": 4, "action": "重置机器人（长按电源键10秒）", "expected": "机器人重启后正常充电"},
                ],
                "solutions": [
                    "清洁充电触点（最常见原因）",
                    "更换充电适配器",
                    "联系售后更换电池",
                ],
            },
            "噪音大": {
                "causes": ["主刷缠绕异物", "滚轮轴承磨损", "风机吸入异物", "集尘盒已满"],
                "steps": [
                    {"step": 1, "action": "翻转机器人，检查主刷是否有缠绕物", "expected": "主刷可自由转动"},
                    {"step": 2, "action": "清空集尘盒并清洁滤网", "expected": "出风口通畅"},
                    {"step": 3, "action": "检查万向轮和驱动轮是否有异物", "expected": "轮子转动无阻碍"},
                ],
                "solutions": [
                    "清理主刷缠绕的毛发/线头",
                    "更换磨损的滚轮轴承",
                    "清洁或更换HEPA滤网",
                ],
            },
            "不扫拖": {
                "causes": ["水箱缺水", "拖布未正确安装", "电机故障", "传感器被遮挡"],
                "steps": [
                    {"step": 1, "action": "检查水箱是否有水", "expected": "水箱水量充足"},
                    {"step": 2, "action": "重新安装拖布支架", "expected": "听到咔嗒声确认安装到位"},
                    {"step": 3, "action": "清洁底部悬崖传感器", "expected": "传感器无异物遮挡"},
                ],
                "solutions": [
                    "加水并重新启动清洁任务",
                    "正确安装拖布支架",
                    "清洁底部传感器",
                ],
            },
            "频繁报错": {
                "causes": ["传感器脏污", "固件版本过旧", "WiFi信号不稳定", "硬件故障"],
                "steps": [
                    {"step": 1, "action": "记录报错代码（APP中查看）", "expected": "确定具体错误类型"},
                    {"step": 2, "action": "清洁所有传感器（悬崖/碰撞/LDS）", "expected": "传感器外观干净"},
                    {"step": 3, "action": "检查固件版本并更新", "expected": "升级到最新版本"},
                ],
                "solutions": [
                    "清洁传感器并重启",
                    "升级固件到最新版本",
                    "如持续报错，联系售后检修",
                ],
            },
        }

        # 模糊匹配故障现象
        matched = None
        for key in trouble_db:
            if any(word in symptom for word in [key, key[:2]]):
                matched = key
                break
        if not matched:
            matched = "频繁报错"  # 默认

        ts = trouble_db.get(matched, trouble_db["频繁报错"])
        result = TroubleshootOutput(
            symptom=symptom,
            model=model,
            possible_causes=ts["causes"],
            diagnostic_steps=ts["steps"],
            solutions=ts["solutions"],
        )
        return {"success": True, "data": result.model_dump(), "error": None}
    except Exception as e:
        logger.error(f"[troubleshoot_issue] 诊断失败: {e}", exc_info=True)
        return {"success": False, "data": None, "error": str(e)}


# ============================================================
# System Tools (系统类)
# ============================================================

def get_user_context_tool(user_id: str) -> dict:
    """
    获取用户上下文信息

    返回用户的位置、设备、偏好等上下文
    """
    try:
        cities = ["深圳", "合肥", "杭州", "北京", "上海"]
        devices_options = [
            ["石头P10"],
            ["科沃斯T30", "石头P10"],
            ["追觅X30"],
            ["石头P10 Pro"],
        ]

        user_idx = hash(user_id) % len(cities) if user_id else 0

        ctx = UserContext(
            user_id=user_id,
            city=cities[user_idx],
            devices=devices_options[user_idx % len(devices_options)],
            preferences={"clean_mode": "标准模式", "schedule": "每日上午9:00"},
        )
        return {"success": True, "data": ctx.model_dump(), "error": None}
    except Exception as e:
        return {"success": False, "data": None, "error": str(e)}


def log_user_feedback_tool(query_id: str, rating: int, comment: str = None) -> dict:
    """
    记录用户反馈

    将用户反馈写入飞轮日志，用于后续质量改进
    """
    try:
        fw = _get_flywheel()
        if fw:
            fw.log_event("user_feedback", {
                "query_id": query_id,
                "rating": rating,
                "comment": comment or "",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            logger.info(f"[用户反馈] query_id={query_id}, rating={rating}/5")

        return {
            "success": True,
            "data": {"recorded": True, "query_id": query_id, "rating": rating},
            "error": None,
        }
    except Exception as e:
        logger.error(f"[log_user_feedback] 记录失败: {e}")
        return {"success": False, "data": None, "error": str(e)}


# ============================================================
# 工具注册表构建
# ============================================================

def build_tool_definitions() -> dict:
    """
    构建所有工具的 ToolDefinition 映射

    每个工具包含：JSON Schema参数定义、返回类型描述、分类、超时等元信息

    Returns:
        {tool_name: ToolDefinition}
    """
    from agent.tools.tool_registry import ToolDefinition, ToolCategory

    return {
        # === Search ===
        "hybrid_search": ToolDefinition(
            name="hybrid_search",
            description="混合检索（向量相似度 + BM25关键词 + BGE交叉编码器重排序）。从知识库中搜索与查询最相关的文档。适用于所有需要查找产品信息、使用建议、故障处理等专业内容的场景。",
            category=ToolCategory.SEARCH,
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "自然语言搜索查询，越具体越好"},
                    "top_k": {"type": "integer", "description": "返回结果数，默认5，最大20"},
                    "filters": {"type": "object", "description": "可选元数据过滤，如{'category':'选购指南'}"},
                },
                "required": ["query"],
            },
            returns={"type": "object", "description": "包含results(文档列表)、total_found、retrieval_method的结构化结果"},
            timeout_seconds=15.0,
            retry_on_failure=True,
        ),
        "entity_lookup": ToolDefinition(
            name="entity_lookup",
            description="知识图谱实体精确查找。输入实体名称（如产品型号），返回实体的所有属性以及关联的文档列表。适用于需要精确了解某个产品的详细参数时。",
            category=ToolCategory.SEARCH,
            parameters={
                "type": "object",
                "properties": {
                    "entity_name": {"type": "string", "description": "实体名称，如'石头P10'"},
                    "entity_type": {"type": "string", "description": "实体类型: model/brand/part，可选"},
                },
                "required": ["entity_name"],
            },
            timeout_seconds=10.0,
        ),
        "keyword_search": ToolDefinition(
            name="keyword_search",
            description="BM25关键词精确检索。使用关键词进行检索，适合需要精确匹配特定术语的场景。区别于混合检索，不进行语义扩展。",
            category=ToolCategory.SEARCH,
            parameters={
                "type": "object",
                "properties": {
                    "keywords": {"type": "string", "description": "空格分隔的关键词"},
                    "top_k": {"type": "integer", "description": "结果数，默认5"},
                },
                "required": ["keywords"],
            },
            timeout_seconds=10.0,
        ),
        "similar_doc_search": ToolDefinition(
            name="similar_doc_search",
            description="相似文档推荐。给定一篇参考文档ID，找到与其内容最相似的其他文档。",
            category=ToolCategory.SEARCH,
            parameters={
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string", "description": "参考文档的唯一ID"},
                    "top_k": {"type": "integer", "description": "推荐数量"},
                },
                "required": ["doc_id"],
            },
            timeout_seconds=10.0,
        ),

        # === Data Query ===
        "query_user_usage": ToolDefinition(
            name="query_user_usage",
            description="查询指定用户的使用记录。获取用户ID在指定月份的扫地机器人使用数据（清洁面积、工作时长、清洁模式等）。生成使用报告时必须调用此工具。",
            category=ToolCategory.DATA_QUERY,
            parameters={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "用户ID，如'1001'"},
                    "month": {"type": "string", "description": "月份，格式YYYY-MM，如'2025-06'"},
                    "metrics": {
                        "type": "array", "items": {"type": "string"},
                        "description": "关注指标: cleaning_area, duration, mode, efficiency, consumable"
                    },
                },
                "required": ["user_id", "month"],
            },
            timeout_seconds=10.0,
        ),
        "get_device_specs": ToolDefinition(
            name="get_device_specs",
            description="查询扫地机器人型号的详细规格参数（吸力、电池、尘盒容量、噪音、重量、特色功能、适用场景等）。选购咨询时必调用。",
            category=ToolCategory.DATA_QUERY,
            parameters={
                "type": "object",
                "properties": {
                    "model_name": {"type": "string", "description": "型号名称，如'石头P10'"},
                },
                "required": ["model_name"],
            },
            timeout_seconds=5.0,
        ),
        "compare_models": ToolDefinition(
            name="compare_models",
            description="多型号对比分析。输入多个型号名称，输出各型号的参数对比表格。用户询问'哪个更好'时调用。",
            category=ToolCategory.DATA_QUERY,
            parameters={
                "type": "object",
                "properties": {
                    "model_list": {
                        "type": "array", "items": {"type": "string"},
                        "description": "待对比的型号列表，如['石头P10','科沃斯T30']"
                    },
                },
                "required": ["model_list"],
            },
            timeout_seconds=10.0,
        ),

        # === External API ===
        "get_weather": ToolDefinition(
            name="get_weather",
            description="获取指定城市的天气信息（天气状况、温度、湿度、AQI、降雨概率）。用户询问环境是否适合使用扫地机器人时调用。",
            category=ToolCategory.EXTERNAL_API,
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名，如'深圳'"},
                },
                "required": ["city"],
            },
            timeout_seconds=5.0,
        ),
        "get_maintenance_schedule": ToolDefinition(
            name="get_maintenance_schedule",
            description="获取扫地机器人的保养周期建议。返回各部件（主刷、边刷、滤网、拖布、传感器等）的保养频率和方法。",
            category=ToolCategory.EXTERNAL_API,
            parameters={
                "type": "object",
                "properties": {
                    "model_name": {"type": "string", "description": "设备型号"},
                },
                "required": ["model_name"],
            },
            timeout_seconds=5.0,
        ),
        "check_firmware_version": ToolDefinition(
            name="check_firmware_version",
            description="检查扫地机器人固件版本，返回当前版本和是否有可用更新。用户询问升级/功能更新时调用。",
            category=ToolCategory.EXTERNAL_API,
            parameters={
                "type": "object",
                "properties": {
                    "model_name": {"type": "string", "description": "设备型号"},
                },
                "required": ["model_name"],
            },
            timeout_seconds=5.0,
        ),

        # === Analysis ===
        "analyze_usage_pattern": ToolDefinition(
            name="analyze_usage_pattern",
            description="分析用户多个月份的使用习惯和趋势。返回平均清洁面积、偏好模式、使用高峰和趋势判断。生成报告中需要"使用分析"时调用。",
            category=ToolCategory.ANALYSIS,
            parameters={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "用户ID"},
                    "months": {"type": "array", "items": {"type": "string"}, "description": "分析月份列表"},
                },
                "required": ["user_id", "months"],
            },
            timeout_seconds=10.0,
        ),
        "calculate_consumable_life": ToolDefinition(
            name="calculate_consumable_life",
            description="耗材寿命预估。根据已使用时长，估算耗材（主刷/边刷/滤网/拖布/电池）的剩余寿命并给出更换建议。",
            category=ToolCategory.ANALYSIS,
            parameters={
                "type": "object",
                "properties": {
                    "part_name": {"type": "string", "description": "耗材名称"},
                    "usage_hours": {"type": "number", "description": "累计使用小时数"},
                },
                "required": ["part_name", "usage_hours"],
            },
            timeout_seconds=5.0,
        ),
        "troubleshoot_issue": ToolDefinition(
            name="troubleshoot_issue",
            description="故障诊断专家。输入故障现象（如'无法充电'、'噪音大'、'不扫拖'），输出可能原因、分步诊断流程和解决方案。用户描述故障时优先调用。",
            category=ToolCategory.ANALYSIS,
            parameters={
                "type": "object",
                "properties": {
                    "symptom": {"type": "string", "description": "故障现象描述"},
                    "model": {"type": "string", "description": "设备型号（可选）"},
                },
                "required": ["symptom"],
            },
            timeout_seconds=10.0,
        ),

        # === System ===
        "get_user_context": ToolDefinition(
            name="get_user_context",
            description="获取用户上下文信息（所在城市、拥有设备、使用偏好）。在多轮对话中了解用户背景时调用。",
            category=ToolCategory.SYSTEM,
            parameters={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "用户ID"},
                },
                "required": ["user_id"],
            },
            timeout_seconds=5.0,
        ),
        "log_user_feedback": ToolDefinition(
            name="log_user_feedback",
            description="记录用户对某次回答的评分和反馈。用于质量改进闭环。",
            category=ToolCategory.SYSTEM,
            parameters={
                "type": "object",
                "properties": {
                    "query_id": {"type": "string", "description": "回答ID"},
                    "rating": {"type": "integer", "description": "评分1-5"},
                    "comment": {"type": "string", "description": "反馈备注（可选）"},
                },
                "required": ["query_id", "rating"],
            },
            requires_confirmation=True,
            timeout_seconds=5.0,
        ),
    }
