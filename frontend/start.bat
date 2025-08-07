@echo off
echo 正在启动Vue3登录系统...
echo.
echo 请确保您已经安装了Node.js
echo.

REM 检查是否已安装依赖
if not exist "node_modules" (
    echo 正在安装依赖...
    npm install
    echo.
)

echo 启动开发服务器...
echo 应用将在 http://localhost:3000 启动
echo 按 Ctrl+C 停止服务器
echo.
npm run dev 