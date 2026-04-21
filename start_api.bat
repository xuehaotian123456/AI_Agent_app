@echo off
echo 🚀 启动智扫通 RAG Agent API 服务...

REM 检查 Python 环境
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 错误: 未找到 Python
    pause
    exit /b 1
)

REM 安装依赖
echo 📦 检查依赖...
pip install -r requirements_api.txt

REM 启动服务
echo 🌐 启动服务...
uvicorn api:app --host 0.0.0.0 --port 8010 --reload --log-level info

pause
