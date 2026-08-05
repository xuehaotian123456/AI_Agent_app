# 基于LangGraph的多Agent知识问答系统

**四智能体闭环（Planner → Retriever → Reflector → Summarizer）** + 混合检索 + Function Calling工具库 + 自进化数据飞轮

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2+-green)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-teal)](https://fastapi.tiangolo.com/)

---

## 架构概览

```
                         ┌──────────────────────────────────────┐
                         │   MultiAgentState (TypedDict)         │
                         │   + Annotated reducer 跨轮累积        │
                         └──────────────────────────────────────┘
                                         │
   ┌─────────────────────────────────────▼──────────────────────────────┐
   │  START                                                              │
   │    │                                                                │
   │    ▼                                                                │
   │  ┌────────────┐ requires_retrieval=true    ┌──────────────────┐    │
   │  │  planner   ├───────────────────────────►│    retriever     │    │
   │  │ (规划·工具选择)│                          │ (检索·工具编排)   │    │
   │  └─────┬──────┘                            │ hybrid + KG扩散   │    │
   │        │ requires_retrieval=false          └───────┬──────────┘    │
   │        ▼                                           │               │
   │  ┌────────────┐                                   ▼               │
   │  │ summarizer │◄──── route="pass" ────┌──────────────────────┐    │
   │  │ (总结·最终) │     or retry >= max    │     reflector        │    │
   │  └─────┬──────┘                       │ (反思·置信度自检)     │    │
   │        │                              └──────────┬───────────┘    │
   │        ▼                                         │ route="retry"  │
   │       END ◄──────────────────────────────────────┘ (改写查询+1)   │
   └────────────────────────────────────────────────────────────────────┘
         外部依赖: Redis · LangFuse · ChromaDB · BM25 · BGE-Reranker · KG索引
```

### 四智能体职责

| Agent | 角色 | 核心能力 |
|-------|------|---------|
| **Planner** | 规划智能体 | 意图识别 → 任务拆解 → 工具选择（从ToolRegistry按意图推荐） |
| **Retriever** | 检索智能体 | 工具调用编排 → 三路多路召回（hybrid+KG+关键词）→ BGE重排序 |
| **Reflector** | 反思智能体 | 置信度自检 → 质量判定 → 失败改写查询 → 图原生条件重试 |
| **Summarizer** | 总结智能体 | 多轮证据链综合 → 结构化输出（含引用） → Markdown格式化 |

---

## Function Calling 工具库

基于自研 `ToolRegistry` 工具注册中心，标准化管理 **5类15+工具**：

| 类别 | 工具 | 说明 |
|------|------|------|
| 🔍 search | `hybrid_search`, `entity_lookup`, `keyword_search`, `similar_doc_search` | 混合检索·实体查询·关键词匹配 |
| 📊 data_query | `query_user_usage`, `get_device_specs`, `compare_models` | 使用记录·设备规格·型号对比 |
| 🌐 external_api | `get_weather`, `get_maintenance_schedule`, `check_firmware_version` | 天气·保养·固件 |
| 📈 analysis | `analyze_usage_pattern`, `calculate_consumable_life`, `troubleshoot_issue` | 使用分析·耗材预估·故障诊断 |
| ⚙️ system | `get_user_context`, `log_user_feedback` | 上下文·反馈 |

每个工具：Pydantic强类型入参/出参 + JSON Schema自动导出 + 统一调用入口（参数校验→超时→异常→追踪）。

---

## 快速开始

### 前置条件

```bash
# 环境变量
export DASHSCOPE_API_KEY=your_api_key
# 可选
export LANGFUSE_PUBLIC_KEY=your_public_key
export LANGFUSE_SECRET_KEY=your_secret_key
```

### Docker 一键部署

```bash
cp .env.example .env
# 编辑 .env 填入 DASHSCOPE_API_KEY

docker-compose up -d
# API: http://localhost:8010
# Health: http://localhost:8010/health
```

### 本地运行

```bash
pip install -r requirements.txt
python api.py
```

---

## API 端点

### v2: Multi-Agent 四智能体闭环

| 端点 | 说明 |
|------|------|
| `POST /api/v2/chat` | Multi-Agent问答（返回Plan/Reflection/检索轮次/工具调用轨迹） |
| `POST /api/v2/chat/stream` | 流式SSE（逐阶段+逐token） |
| `GET /api/v2/flywheel/metrics` | 飞轮聚合指标 |
| `GET /api/v2/flywheel/failures` | 飞轮失败案例 |
| `GET /api/v2/tools/stats` | 工具调用统计 |
| `GET /api/v2/kg/entities` | KG实体搜索 |

### v1: 原有端点（向后兼容）

| 端点 | 说明 |
|------|------|
| `POST /api/v1/chat` | CorrectiveRAG问答 |
| `POST /api/v1/chat/stream` | 流式SSE |
| `GET /health` | 健康检查 |

### 请求示例

```bash
# v2 Multi-Agent 问答
curl -X POST http://localhost:8010/api/v2/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "小户型适合哪种扫地机器人？需要静音的"}'

# 响应包含完整决策轨迹：
# { answer, citations, plan, reflection, retrieval_rounds, tool_calls, ... }
```

---

## 项目结构

```
langgraph/
├── agent/
│   ├── __init__.py                     # Agent模块导出
│   ├── react_agent.py                  # 原有ReAct Agent（Streamlit）
│   ├── multi_agent/                    # ★ Multi-Agent核心
│   │   ├── __init__.py
│   │   ├── models.py                   # Pydantic数据模型
│   │   ├── state.py                    # MultiAgentState (Annotated reducer)
│   │   ├── planner.py                  # PlannerNode（规划+工具选择）
│   │   ├── retriever.py                # RetrieverNode（检索+工具调用编排）
│   │   ├── reflection.py               # ReflectionNode（置信度自检+重试）
│   │   ├── summarizer.py               # SummarizerNode（证据链综合）
│   │   └── graph.py                    # 四节点条件图组装
│   ├── multi_agent_orchestrator.py     # 编排器（同步+流式）
│   └── tools/
│       ├── __init__.py                 # 工具模块注册
│       ├── tool_registry.py            # ★ ToolRegistry 工具注册中心
│       ├── enhanced_tools.py           # ★ 5类15+增强工具
│       └── middleware.py               # Agent中间件
├── rag/
│   ├── rag_service.py                  # RAG服务（2节点图）
│   ├── hybrid_retriever.py             # 混合检索器
│   ├── self_corrector.py               # 自我纠错器
│   ├── query_rewriter.py               # 查询改写
│   ├── vector_store.py                 # Chroma向量库
│   ├── knowledge_graph.py              # ★ 轻量知识图谱
│   └── output_models.py                # 结构化输出模型
├── services/
│   ├── RedisSessionManager.py          # Redis会话管理
│   ├── RetrievalCache.py               # 检索缓存
│   └── FlywheelService.py              # ★ 自进化数据飞轮
├── model/factory.py                    # LLM工厂（主模型+备份降级）
├── config/
│   ├── rag.yml / chroma.yml / agent.yml
│   └── multi_agent.yml                 # ★ Multi-Agent配置
├── prompts/                            # 提示词模板
├── api.py                              # FastAPI (v1+v2)
├── app.py                              # Streamlit UI
├── docker-compose.yml                  # Docker部署
└── requirements.txt                    # Python依赖
```

---

## 技术栈

| 层次 | 技术 |
|------|------|
| **Agent框架** | LangGraph (StateGraph) + LangChain |
| **LLM** | 通义千问 (DashScope) · deepseek-v3.2 主模型 · qwen-plus 备份降级 |
| **向量存储** | ChromaDB |
| **检索** | 向量 + BM25 (jieba) + BGE-Reranker 三路融合 |
| **知识图谱** | 轻量自研KG（实体共现 + 一跳扩散） |
| **缓存** | Redis（LLM精确缓存 + 检索结果缓存） |
| **可观测性** | LangFuse（4节点+工具调用全链路追踪） |
| **API** | FastAPI + SSE流式 |
| **类型校验** | Pydantic v2 |
| **部署** | Docker Compose (API + Redis) |

---

## 混合检索

融合三种检索路径，通过 BGE-Reranker 交叉编码器统一重排序：

1. **向量语义检索** (ChromaDB + DashScope text-embedding-v4) — 语义相似度匹配
2. **BM25关键词检索** (jieba分词) — 精确术语匹配
3. **KG实体扩散检索** — 从查询中提取实体 → 一跳共现扩展 → 附加检索词

**基准指标**: MRR@5 85% · Precision@5 58.3% · Hit Rate@5 95%

---

## 自进化数据飞轮

```
Reflection判定失败 → FlywheelService.log_event()
                    ├─ Redis LPUSH (热数据, 7天TTL)
                    ├─ JSONL 文件 (冷备, 永久保留)
                    └─ adapt_threshold 触发 → 检索策略自适应
                                               ├─ top_k 增大
                                               ├─ KG扩展启用
                                               └─ 改写提示词补充
```

---

## 许可证

MIT
