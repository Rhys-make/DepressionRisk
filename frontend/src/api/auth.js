import axios from 'axios'

// 创建axios实例
const api = axios.create({
  baseURL: 'http://localhost:3001/api', // 后端API地址
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json'
  }
})

// 请求拦截器
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// 响应拦截器
api.interceptors.response.use(
  (response) => {
    return response
  },
  (error) => {
    if (error.response?.status === 401) {
      // token过期或无效，清除本地存储并跳转到登录页
      localStorage.removeItem('token')
      localStorage.removeItem('user')
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)

// 登录API
export const loginApi = (credentials) => {
  return api.post('/auth/login', credentials)
}

// 注册API
export const registerApi = (userData) => {
  return api.post('/auth/register', userData)
}

// 获取用户信息API
export const getUserInfoApi = () => {
  return api.get('/auth/me')
}

// 刷新token API
export const refreshTokenApi = () => {
  return api.post('/auth/refresh')
}

export default api 