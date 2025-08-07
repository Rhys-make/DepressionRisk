# Vue3 + FastAPI 登录系统

一个基于Vue3前端和FastAPI后端的完整用户认证系统。

## 🚀 技术栈

### 前端 (Vue3)
- **框架**: Vue 3.3.4 + Composition API
- **构建工具**: Vite 4.4.9
- **状态管理**: Pinia 2.1.6
- **路由**: Vue Router 4.2.4
- **HTTP客户端**: Axios 1.5.0
- **样式**: CSS3 + 毛玻璃效果

### 后端 (FastAPI)
- **框架**: FastAPI 0.104.1
- **数据库**: PostgreSQL
- **ORM**: psycopg2
- **数据验证**: Pydantic 2.5.0
- **服务器**: Uvicorn 0.24.0

## 📁 项目结构

```
exercise/
├── frontend/                 # Vue3前端
│   ├── src/
│   │   ├── api/             # API接口
│   │   ├── stores/          # 状态管理
│   │   ├── views/           # 页面组件
│   │   ├── router/          # 路由配置
│   │   └── style.css        # 全局样式
│   ├── package.json
│   └── start.bat            # 前端启动脚本
├── backend/                  # FastAPI后端
│   ├── app/
│   │   └── main.py          # 主应用文件
│   ├── requirements.txt     # Python依赖
│   └── start.bat            # 后端启动脚本
└── start-project.bat        # 项目启动脚本
```

## 🛠️ 安装和运行

### 前置要求
1. **Node.js** (v16+)
2. **Python** (v3.8+)
3. **PostgreSQL** 数据库

### 快速启动

#### 方法一：一键启动（推荐）
```bash
# 双击运行
start-project.bat
```

#### 方法二：分别启动

**启动后端：**
```bash
cd backend
start.bat
```

**启动前端：**
```bash
cd frontend
start.bat
```

### 数据库配置

1. 安装PostgreSQL数据库
2. 创建数据库：
```sql
CREATE DATABASE register;
```

3. 修改数据库连接配置（`backend/app/main.py`）：
```python
DATABASE_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'user': 'your_username',
    'password': 'your_password',
    'database': 'register'
}
```

## 🌐 服务地址

### 本地访问
- **前端应用**: http://localhost:3000
- **后端API**: http://localhost:8000
- **API文档**: http://localhost:8000/docs
- **健康检查**: http://localhost:8000/api/health

### 网络访问
- **前端应用**: http://192.168.10.69:3000
- **后端API**: http://192.168.10.69:8000
- **API文档**: http://192.168.10.69:8000/docs
- **健康检查**: http://192.168.10.69:8000/api/health

## 📋 API接口

### 用户注册
```http
POST /api/register
Content-Type: application/json

{
  "username": "testuser",
  "password": "123456",
  "confirm_password": "123456"
}
```

### 用户登录
```http
POST /api/login
Content-Type: application/json

{
  "username": "testuser",
  "password": "123456"
}
```

### 获取用户列表
```http
GET /api/users
```

### 健康检查
```http
GET /api/health
```

## 🎨 功能特性

### 前端特性
- ✅ 响应式设计，适配各种屏幕尺寸
- ✅ 毛玻璃效果登录界面
- ✅ 表单验证和错误提示
- ✅ 用户状态管理
- ✅ 路由守卫和权限控制
- ✅ 美观的UI动画效果

### 后端特性
- ✅ RESTful API设计
- ✅ 用户注册和登录
- ✅ 密码SHA256加密
- ✅ PostgreSQL数据库集成
- ✅ 自动API文档生成
- ✅ CORS跨域支持
- ✅ 错误处理和日志

## 🔐 安全特性

- 密码使用SHA256哈希加密存储
- 表单输入验证和清理
- CORS跨域安全配置
- 数据库连接池管理
- 错误信息不暴露敏感数据

## 🧪 测试账号

系统内置测试账号：
- **用户名**: admin
- **密码**: 123456

## 📝 开发说明

### 前端开发
```bash
cd frontend
npm install
npm run dev
```

### 后端开发
```bash
cd backend
pip install -r requirements.txt
python app/main.py
```

### 数据库操作
```bash
# 连接到PostgreSQL
psql -U postgres -d register

# 查看用户表
SELECT * FROM users;
```

## 🐛 故障排除

### 常见问题

1. **数据库连接失败**
   - 检查PostgreSQL服务是否启动
   - 验证数据库连接配置
   - 确认数据库和用户权限

2. **前端无法连接后端**
   - 确认后端服务在8000端口运行
   - 检查CORS配置
   - 验证API地址配置

3. **依赖安装失败**
   - 更新Node.js和Python版本
   - 清除缓存重新安装
   - 检查网络连接

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交Issue和Pull Request！

---

**注意**: 这是一个演示项目，生产环境使用前请加强安全措施。 