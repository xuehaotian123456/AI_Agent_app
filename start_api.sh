#!/bin/bash
echo "🚀 启动智扫通 RAG Agent API 服务..."

# 检查 Python 环境
if ! command -v python &> /dev/null; then
    echo "❌ 错误: 未找到 Python"
    exit 1
fi

# 安装依赖
echo "📦 检查依赖..."
pip install -r requirements_api.txt

# 启动服务
echo "🌐 启动服务..."
uvicorn api:app --host 0.0.0.0 --port 8010 --reload --log-level info
