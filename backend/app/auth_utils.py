from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional

# JWT配置（需与main.py保持一致）
SECRET_KEY = "your-secret-key-here-make-it-long-and-secure-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

security = HTTPBearer()

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

