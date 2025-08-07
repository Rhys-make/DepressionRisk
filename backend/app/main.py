from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import os
import hashlib
import psycopg2
from typing import List, Optional
from datetime import datetime, timedelta
from jose import JWTError, jwt
app = FastAPI(
    title="用户注册登录系统",
    description="基于PostgreSQL的用户注册和登录后端服务",
    version="4.0.0"
)

# JWT配置
SECRET_KEY = "your-secret-key-here-make-it-long-and-secure-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# 安全配置
security = HTTPBearer()

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

class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int

class TokenData(BaseModel):
    username: Optional[str] = None

class ApiResponse(BaseModel):
    success: bool
    message: str
    data: Optional[dict] = None

# JWT相关函数
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """创建JWT访问令牌"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Optional[str]:
    """验证JWT令牌并返回用户名"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None
        return username
    except JWTError:
        return None

# 新增：根据token获取用户名

def get_username_by_token(token: str) -> Optional[str]:
    """通过token获取用户名"""
    return verify_token(token)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """获取当前用户（从JWT令牌中）"""
    token = credentials.credentials
    username = verify_token(token)
    if username is None:
        raise HTTPException(
            status_code=401,
            detail="无效的认证凭据",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return username

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

# 删除 index 根路径接口

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
    验证用户名和密码，生成JWT token
    """
    try:
        username = user_data.username.strip()
        password = user_data.password
        
        if not username or not password:
            raise HTTPException(status_code=400, detail="用户名和密码不能为空")
        
        # 验证用户名和密码
        if verify_user(username, password):
            # 生成JWT token
            access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
            access_token = create_access_token(
                data={"sub": username}, expires_delta=access_token_expires
            )
            # 新增：通过token获取用户名
            checked_username = get_username_by_token(access_token)
            return ApiResponse(
                success=True,
                message="登录成功！",
                data={
                    "username": checked_username,
                    "access_token": access_token,
                    "token_type": "bearer",
                    "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60  # 秒
                }
            )
        else:
            raise HTTPException(status_code=401, detail="用户名或密码错误")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器错误: {str(e)}")

@app.get("/api/me", response_model=ApiResponse)
async def get_current_user_info(current_user: str = Depends(get_current_user)):
    """
    获取当前用户信息
    需要JWT token认证
    """
    try:
        conn = get_database_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT username, created_at FROM users WHERE username = %s', (current_user,))
        user = cursor.fetchone()
        conn.close()
        
        if user:
            return ApiResponse(
                success=True,
                message="获取用户信息成功",
                data={
                    "username": user[0],
                    "created_at": str(user[1])
                }
            )
        else:
            raise HTTPException(status_code=404, detail="用户不存在")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器错误: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("🚀 启动用户注册登录系统...")
    print("📊 数据库类型: PostgreSQL")
    print("🔐 JWT认证: 已启用")
    print("🌐 服务地址: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
