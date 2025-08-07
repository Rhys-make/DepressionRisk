@echo off
echo 正在启动FastAPI后端服务...
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误：未找到Python，请先安装Python 3.8+
    pause
    exit /b 1
)

REM 检查是否已安装依赖
if not exist "venv" (
    echo 正在创建虚拟环境...
    python -m venv venv
    echo.
)

echo 激活虚拟环境...
call venv\Scripts\activate.bat

echo 安装依赖包...
pip install -r requirements.txt
echo.

echo 启动FastAPI服务器...
echo 服务将在 http://localhost:8000 启动
echo 按 Ctrl+C 停止服务器
echo.
python app/main.py 