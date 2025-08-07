import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { loginApi, registerApi } from '@/api/auth'

export const useAuthStore = defineStore('auth', () => {
  // 状态
  const user = ref(null)
  const token = ref(localStorage.getItem('token') || null)
  const loading = ref(false)
  const error = ref(null)

  // 计算属性
  const isAuthenticated = computed(() => !!token.value)
  const userInfo = computed(() => user.value)

  // 方法
  const login = async (credentials) => {
    loading.value = true
    error.value = null
    
    try {
      const response = await loginApi(credentials)
      const { success, message, data } = response.data
      
      if (success) {
        // 后端返回成功，保存用户信息
        const userData = {
          username: data.username,
          email: data.username + '@example.com', // 模拟邮箱
          role: 'user'
        }
        
        // 生成模拟token（实际项目中后端应该返回JWT token）
        const authToken = `token-${data.username}-${Date.now()}`
        
        // 保存token和用户信息
        token.value = authToken
        user.value = userData
        localStorage.setItem('token', authToken)
        localStorage.setItem('user', JSON.stringify(userData))
        
        return { success: true }
      } else {
        error.value = message || '登录失败'
        return { success: false, error: error.value }
      }
    } catch (err) {
      error.value = err.response?.data?.detail || '登录失败，请重试'
      return { success: false, error: error.value }
    } finally {
      loading.value = false
    }
  }

  const register = async (userData) => {
    loading.value = true
    error.value = null
    
    try {
      const response = await registerApi(userData)
      const { success, message, data } = response.data
      
      if (success) {
        return { success: true, message }
      } else {
        error.value = message || '注册失败'
        return { success: false, error: error.value }
      }
    } catch (err) {
      error.value = err.response?.data?.detail || '注册失败，请重试'
      return { success: false, error: error.value }
    } finally {
      loading.value = false
    }
  }

  const logout = () => {
    token.value = null
    user.value = null
    error.value = null
    localStorage.removeItem('token')
    localStorage.removeItem('user')
  }

  const clearError = () => {
    error.value = null
  }

  const initializeAuth = () => {
    const savedToken = localStorage.getItem('token')
    const savedUser = localStorage.getItem('user')
    
    if (savedToken && savedUser) {
      token.value = savedToken
      user.value = JSON.parse(savedUser)
    }
  }

  return {
    // 状态
    user,
    token,
    loading,
    error,
    
    // 计算属性
    isAuthenticated,
    userInfo,
    
    // 方法
    login,
    register,
    logout,
    clearError,
    initializeAuth
  }
}) 