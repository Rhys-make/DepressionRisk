@echo off
echo ========================================
echo     网络连接测试工具
echo ========================================
echo.

echo 正在测试后端连接...
echo.

REM 测试后端健康检查
echo 1. 测试后端健康检查接口...
curl -s http://192.168.10.69:8000/api/health
if %errorlevel% equ 0 (
    echo ✅ 后端连接成功！
) else (
    echo ❌ 后端连接失败！
    echo 请检查：
    echo - 后端服务是否启动
    echo - 防火墙是否允许8000端口
    echo - 网络连接是否正常
)
echo.

REM 测试API文档
echo 2. 测试API文档访问...
curl -s -o nul http://192.168.10.69:8000/docs
if %errorlevel% equ 0 (
    echo ✅ API文档访问成功！
) else (
    echo ❌ API文档访问失败！
)
echo.

REM 测试前端连接
echo 3. 测试前端连接...
curl -s -o nul http://192.168.10.69:3000
if %errorlevel% equ 0 (
    echo ✅ 前端连接成功！
) else (
    echo ❌ 前端连接失败！
    echo 请检查前端服务是否启动
)
echo.

echo ========================================
echo 测试完成！
echo ========================================
echo.
echo 如果所有测试都通过，说明网络连接正常。
echo 如果出现错误，请检查相应的服务状态。
echo.
pause 