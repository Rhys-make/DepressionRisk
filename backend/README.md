# 用户注册登录系统

这是一个完整的Python Flask后端服务，提供用户注册和登录功能，支持数据库存储。

## 功能特性

- ✅ 用户注册功能
- ✅ 用户登录验证
- ✅ 密码加密存储
- ✅ 数据库持久化
- ✅ 跨域请求支持
- ✅ 健康检查接口
- ✅ JSON格式响应
- ✅ 中文友好界面

## 数据库信息

- 数据库文件: `users.db`
- 表名: `users`
- 字段: `id`, `username`, `password`, `created_at`
- 密码加密: SHA256哈希

## 安装依赖

```bash
pip install -r requirements.txt
```

## 启动服务

### 方法1: 直接运行
```bash
python app.py
```

### 方法2: 使用启动脚本
```bash
python run.py
```

### 方法3: 测试功能
```bash
python test_register_login.py
```

服务将在 `http://localhost:5000` 启动

## API接口

### 1. 用户注册接口

**URL:** `POST /api/register`

**请求体:**
```json
{
    "username": "newuser",
    "password": "123456",
    "confirm_password": "123456"
}
```

**成功响应:**
```json
{
    "success": true,
    "message": "注册成功！",
    "data": {
        "username": "newuser"
    }
}
```

**失败响应:**
```json
{
    "success": false,
    "message": "用户名已存在，请选择其他用户名"
}
```

### 2. 用户登录接口

**URL:** `POST /api/login`

**请求体:**
```json
{
    "username": "newuser",
    "password": "123456"
}
```

**成功响应:**
```json
{
    "success": true,
    "message": "登录成功！",
    "data": {
        "username": "newuser"
    }
}
```

**失败响应:**
```json
{
    "success": false,
    "message": "用户名或密码错误"
}
```

### 2. 健康检查接口

**URL:** `GET /api/health`

**响应:**
```json
{
    "success": true,
    "message": "服务器运行正常"
}
```

### 3. 用户列表接口

**URL:** `GET /api/users`

**响应:**
```json
{
    "success": true,
    "message": "获取用户列表成功",
    "data": {
        "users": [
            {
                "username": "user1",
                "created_at": "2024-01-01 12:00:00"
            }
        ],
        "total": 1
    }
}
```

### 4. 健康检查接口

**URL:** `GET /api/health`

**响应:**
```json
{
    "success": true,
    "message": "服务器运行正常",
    "status": "在线"
}
```

### 5. 根路径

**URL:** `GET /`

**响应:**
```json
{
    "message": "欢迎使用用户注册登录系统",
    "version": "2.0.0",
    "description": "这是一个支持用户注册和登录的后端服务",
    "endpoints": {
        "用户注册": "/api/register (POST)",
        "用户登录": "/api/login (POST)",
        "用户列表": "/api/users (GET)",
        "健康检查": "/api/health (GET)"
    }
}
```

## 测试示例

### 使用curl测试注册接口

```bash
# 成功注册
curl -X POST http://localhost:5000/api/register \
  -H "Content-Type: application/json" \
  -d '{"username": "newuser", "password": "123456", "confirm_password": "123456"}'

# 密码不匹配
curl -X POST http://localhost:5000/api/register \
  -H "Content-Type: application/json" \
  -d '{"username": "testuser", "password": "123456", "confirm_password": "654321"}'
```

### 使用curl测试登录接口

```bash
# 成功登录
curl -X POST http://localhost:5000/api/login \
  -H "Content-Type: application/json" \
  -d '{"username": "newuser", "password": "123456"}'

# 失败登录
curl -X POST http://localhost:5000/api/login \
  -H "Content-Type: application/json" \
  -d '{"username": "newuser", "password": "wrong_password"}'
```

### 使用curl测试用户列表

```bash
curl http://localhost:5000/api/users
```

### 使用curl测试健康检查

```bash
curl http://localhost:5000/api/health
```

## 项目结构

```
backend/
├── app.py                    # 主应用文件
├── test_register_login.py    # 注册登录测试脚本
├── requirements.txt          # 依赖文件
├── users.db                 # SQLite数据库文件（自动生成）
└── README.md                # 说明文档
```

## 注意事项

1. 这是一个演示项目，使用SQLite数据库存储用户信息
2. 密码使用SHA256哈希加密存储，提高安全性
3. 用户名唯一性验证，防止重复注册
4. 密码确认验证，确保用户输入正确
5. 所有接口响应均为中文，便于理解和使用
6. 建议在生产环境中使用更安全的密码加密方式（如bcrypt）
7. 可以添加JWT token认证机制增强安全性