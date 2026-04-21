# LangGraph RAG项目
基于LangGraph构建的检索增强生成（RAG）系统，支持问答生成、混合检索等功能。

## Docker 一键部署

已提供容器化配置，可直接使用 Docker / Docker Compose 运行。

- 说明文档：`DOCKER_DEPLOY_GUIDE.md`
- API 默认端口：`8010`

## 可观测性（LangFuse）

已集成 LangFuse 追踪，可查看请求耗时、检索命中、纠错重试、模型用量等指标。

- 说明文档：`OBSERVABILITY_LANGFUSE.md`

## 项目结构
- agent/: 智能体核心逻辑
- config/: 配置文件
- data/: 数据集
- logs/: 日志文件
- model/: 模型文件
- prompts/: 提示词模板
- rag/: RAG核心模块
- utils/: 工具函数
- app.py: 项目入口
- qa_generator.py: 问答生成脚本