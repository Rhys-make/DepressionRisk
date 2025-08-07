from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import hashlib
import psycopg2
from typing import List, Optional
from datetime import datetime

app = FastAPI(
    title="用户注册登录系统",
    description="基于PostgreSQL的用户注册和登录后端服务",
    version="4.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头部
)

# PostgreSQL数据库配置
DATABASE_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'user': 'postgres',
    'password': 'Zhengnan4568',
    'database': 'register'
}

# Pydantic模型
class UserRegister(BaseModel):
    username: str
    password: str
    confirm_password: str

class UserLogin(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    username: str
    created_at: str

class ApiResponse(BaseModel):
    success: bool
    message: str
    data: Optional[dict] = None

def get_database_connection():
    """获取PostgreSQL数据库连接"""
    return psycopg2.connect(
        host=DATABASE_CONFIG['host'],
        port=DATABASE_CONFIG['port'],
        user=DATABASE_CONFIG['user'],
        password=DATABASE_CONFIG['password'],
        database=DATABASE_CONFIG['database']
    )

def init_database():
    """初始化数据库，创建用户表"""
    conn = get_database_connection()
    cursor = conn.cursor()
    
    # PostgreSQL创建表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(50) UNIQUE NOT NULL,
            password VARCHAR(255) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ PostgreSQL数据库初始化完成")

def hash_password(password: str) -> str:
    """对密码进行哈希加密"""
    return hashlib.sha256(password.encode()).hexdigest()

def check_user_exists(username: str) -> bool:
    """检查用户是否已存在"""
    conn = get_database_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT username FROM users WHERE username = %s', (username,))
    result = cursor.fetchone()
    conn.close()
    return result is not None

def create_user(username: str, password: str) -> bool:
    """创建新用户"""
    hashed_password = hash_password(password)
    conn = get_database_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('INSERT INTO users (username, password) VALUES (%s, %s)', 
                      (username, hashed_password))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        conn.close()
        print(f"创建用户失败: {e}")
        return False

def verify_user(username: str, password: str) -> bool:
    """验证用户登录"""
    hashed_password = hash_password(password)
    conn = get_database_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT username FROM users WHERE username = %s AND password = %s', 
                  (username, hashed_password))
    result = cursor.fetchone()
    conn.close()
    return result is not None

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化数据库"""
    init_database()

@app.get("/", response_model=dict)
async def index():
    """根路径"""
    return {
        "message": "欢迎使用用户注册登录系统",
        "version": "4.0.0",
        "description": "这是一个基于PostgreSQL的用户注册和登录后端服务",
        "database": {
            "type": "PostgreSQL",
            "config": DATABASE_CONFIG
        },
        "endpoints": {
            "用户注册": "/api/register (POST)",
            "用户登录": "/api/login (POST)",
            "用户列表": "/api/users (GET)",
            "健康检查": "/api/health (GET)"
        },
        "使用说明": {
            "注册": "发送POST请求到/api/register，包含用户名、密码和确认密码",
            "登录": "发送POST请求到/api/login，包含用户名和密码",
            "密码安全": "密码使用SHA256加密存储",
            "数据库": "使用PostgreSQL数据库"
        }
    }

@app.post("/api/register", response_model=ApiResponse)
async def register(user_data: UserRegister):
    """
    用户注册接口
    接收用户名、密码和确认密码，验证后存储到数据库
    """
    try:
        username = user_data.username.strip()
        password = user_data.password
        confirm_password = user_data.confirm_password
        
        # 验证输入数据
        if not username or not password or not confirm_password:
            raise HTTPException(status_code=400, detail="用户名、密码和确认密码不能为空")
        
        if len(username) < 3:
            raise HTTPException(status_code=400, detail="用户名长度至少为3个字符")
        
        if len(password) < 6:
            raise HTTPException(status_code=400, detail="密码长度至少为6个字符")
        
        if password != confirm_password:
            raise HTTPException(status_code=400, detail="两次输入的密码不一致")
        
        # 检查用户是否已存在
        if check_user_exists(username):
            raise HTTPException(status_code=400, detail="用户名已存在，请选择其他用户名")
        
        # 创建新用户
        if create_user(username, password):
            return ApiResponse(
                success=True,
                message="注册成功！",
                data={"username": username}
            )
        else:
            raise HTTPException(status_code=500, detail="注册失败，请稍后重试")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器错误: {str(e)}")

@app.post("/api/login", response_model=ApiResponse)
async def login(user_data: UserLogin):
    """
    用户登录接口
    验证用户名和密码，从数据库中查询
    """
    try:
        username = user_data.username.strip()
        password = user_data.password
        
        if not username or not password:
            raise HTTPException(status_code=400, detail="用户名和密码不能为空")
        
        # 验证用户名和密码
        if verify_user(username, password):
            return ApiResponse(
                success=True,
                message="登录成功！",
                data={"username": username}
            )
        else:
            raise HTTPException(status_code=401, detail="用户名或密码错误")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器错误: {str(e)}")

@app.get("/api/users", response_model=ApiResponse)
async def get_users():
    """
    获取所有用户列表（仅用于测试）
    """
    try:
        conn = get_database_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT username, created_at FROM users')
        users = cursor.fetchall()
        conn.close()
        
        user_list = [{"username": user[0], "created_at": str(user[1])} for user in users]
        
        return ApiResponse(
            success=True,
            message="获取用户列表成功",
            data={
                "users": user_list,
                "total": len(user_list)
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器错误: {str(e)}")

@app.get("/api/health", response_model=ApiResponse)
async def health_check():
    """
    健康检查接口
    """
    try:
        # 测试数据库连接
        conn = get_database_connection()
        conn.close()
        db_status = "正常"
    except Exception as e:
        db_status = f"异常: {str(e)}"
    
    return ApiResponse(
        success=True,
        message="服务器运行正常",
        data={
            "status": "在线",
            "database": {
                "type": "PostgreSQL",
                "status": db_status,
                "config": {
                    "host": DATABASE_CONFIG['host'],
                    "port": DATABASE_CONFIG['port'],
                    "database": DATABASE_CONFIG['database']
                }
            }
        }
    )

if __name__ == "__main__":
    import uvicorn
    print("🚀 启动用户注册登录系统...")
    print("📊 数据库类型: PostgreSQL")
    print("🌐 服务地址: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
