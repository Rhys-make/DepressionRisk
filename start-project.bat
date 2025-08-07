@echo off
echo ========================================
echo     Vue3 + FastAPI 登录系统启动器
echo ========================================
echo.

echo 请确保您已经：
echo 1. 安装了 Node.js
echo 2. 安装了 Python 3.8+
echo 3. 安装了 PostgreSQL 数据库
echo 4. 配置了数据库连接信息
echo.

echo 启动顺序：
echo 1. 启动后端服务 (FastAPI)
echo 2. 启动前端服务 (Vue3)
echo.

echo 按任意键开始启动...
pause >nul

echo.
echo 🚀 启动后端服务...
start "Backend Server" cmd /k "cd backend && start.bat"

echo 等待后端服务启动...
timeout /t 5 /nobreak >nul

echo.
echo 🎨 启动前端服务...
start "Frontend Server" cmd /k "cd frontend && start.bat"

echo.
echo ✅ 服务启动完成！
echo.
echo 📊 后端服务: http://localhost:8000 (本地) / http://192.168.10.69:8000 (网络)
echo 🎨 前端服务: http://localhost:3000 (本地) / http://192.168.10.69:3000 (网络)
echo 📚 API文档: http://localhost:8000/docs (本地) / http://192.168.10.69:8000/docs (网络)
echo.
echo 🔧 运行 test-connection.bat 测试网络连接
echo.
echo 按任意键退出...
pause >nul 