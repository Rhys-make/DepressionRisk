import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { loginApi } from '@/api/auth'
import { mockLoginApi } from '@/api/mock'

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
      // 使用模拟API进行测试，您可以切换回真实API
      const response = await mockLoginApi(credentials)
      const { token: authToken, user: userData } = response.data
      
      // 保存token和用户信息
      token.value = authToken
      user.value = userData
      localStorage.setItem('token', authToken)
      localStorage.setItem('user', JSON.stringify(userData))
      
      return { success: true }
    } catch (err) {
      error.value = err.response?.data?.message || '登录失败，请重试'
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
    logout,
    clearError,
    initializeAuth
  }
}) 