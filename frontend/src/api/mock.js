// 模拟API服务 - 用于测试登录功能
import axios from 'axios'

// 创建模拟axios实例
const mockApi = axios.create({
  timeout: 1000
})

// 模拟用户数据
const mockUsers = [
  {
    id: 1,
    username: 'admin',
    password: '123456',
    email: 'admin@example.com',
    role: 'admin'
  },
  {
    id: 2,
    username: 'user',
    password: '123456',
    email: 'user@example.com',
    role: 'user'
  }
]

// 模拟登录API
export const mockLoginApi = async (credentials) => {
  // 模拟网络延迟
  await new Promise(resolve => setTimeout(resolve, 800))
  
  const { username, password } = credentials
  
  // 查找用户
  const user = mockUsers.find(u => 
    u.username === username && u.password === password
  )
  
  if (user) {
    // 模拟成功响应
    const { password: _, ...userWithoutPassword } = user
    return {
      data: {
        success: true,
        message: '登录成功',
        token: `mock-token-${user.id}-${Date.now()}`,
        user: userWithoutPassword
      }
    }
  } else {
    // 模拟失败响应
    throw {
      response: {
        status: 401,
        data: {
          success: false,
          message: '用户名或密码错误'
        }
      }
    }
  }
}

// 模拟获取用户信息API
export const mockGetUserInfoApi = async () => {
  await new Promise(resolve => setTimeout(resolve, 500))
  
  const token = localStorage.getItem('token')
  if (!token) {
    throw {
      response: {
        status: 401,
        data: {
          success: false,
          message: '未授权访问'
        }
      }
    }
  }
  
  // 从token中提取用户ID（模拟）
  const userId = parseInt(token.split('-')[2])
  const user = mockUsers.find(u => u.id === userId)
  
  if (user) {
    const { password: _, ...userWithoutPassword } = user
    return {
      data: {
        success: true,
        user: userWithoutPassword
      }
    }
  } else {
    throw {
      response: {
        status: 401,
        data: {
          success: false,
          message: '用户不存在'
        }
      }
    }
  }
}

export default mockApi 