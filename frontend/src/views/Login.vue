<template>
  <div class="container">
    <div class="login-card">
      <!-- 登录头部 -->
      <div class="login-header">
        <h1 class="login-title">Login</h1>
        <p class="login-subtitle">欢迎回来，请登录您的账户</p>
      </div>

      <!-- 登录表单 -->
      <form @submit.prevent="handleLogin" class="login-form">
        <div class="form-group">
          <label for="username" class="form-label">Username</label>
          <input
            id="username"
            v-model="form.username"
            type="text"
            class="form-input"
            :class="{ error: errors.username }"
            placeholder="请输入用户名"
            @input="clearError('username')"
            @blur="validateField('username')"
          />
          <div v-if="errors.username" class="error-message">
            {{ errors.username }}
          </div>
        </div>

        <div class="form-group">
          <label for="password" class="form-label">Password</label>
          <input
            id="password"
            v-model="form.password"
            :type="showPassword ? 'text' : 'password'"
            class="form-input"
            :class="{ error: errors.password }"
            placeholder="请输入密码"
            @input="clearError('password')"
            @blur="validateField('password')"
          />
          <div v-if="errors.password" class="error-message">
            {{ errors.password }}
          </div>
        </div>

        <div class="form-group">
          <label class="checkbox-container">
            <input
              v-model="form.rememberMe"
              type="checkbox"
              class="checkbox-input"
            />
            <span class="checkbox-label">Remember me</span>
          </label>
        </div>

        <button
          type="submit"
          class="login-button"
          :disabled="loading || !isFormValid"
        >
          <span v-if="loading" class="loading-spinner"></span>
          {{ loading ? 'Logging in...' : 'Login' }}
        </button>
      </form>

      <!-- 错误信息显示 -->
      <div v-if="authStore.error" class="error-message" style="text-align: center; margin-top: 15px;">
        {{ authStore.error }}
      </div>

      <!-- 登录页脚 -->
      <div class="login-footer">
        <a href="#" class="forgot-password" @click.prevent="handleForgotPassword">
          Forgot password?
        </a>
        <div class="signup-link">
          <span>Don't have an account? </span>
          <a href="#" @click.prevent="handleSignup">Signup now</a>
        </div>
      </div>

      <!-- 社交登录 -->
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'

// 路由和状态管理
const router = useRouter()
const authStore = useAuthStore()

// 表单数据
const form = reactive({
  username: '',
  password: '',
  rememberMe: false
})

// 表单验证错误
const errors = reactive({
  username: '',
  password: ''
})

// 响应式数据
const loading = computed(() => authStore.loading)
const showPassword = ref(false)

// 表单验证
const isFormValid = computed(() => {
  return form.username.trim() && form.password.trim() && !errors.username && !errors.password
})

// 验证字段
const validateField = (field) => {
  errors[field] = ''
  
  if (field === 'username') {
    if (!form.username.trim()) {
      errors.username = '用户名不能为空'
    } else if (form.username.length < 3) {
      errors.username = '用户名至少3个字符'
    }
  }
  
  if (field === 'password') {
    if (!form.password.trim()) {
      errors.password = '密码不能为空'
    } else if (form.password.length < 6) {
      errors.password = '密码至少6个字符'
    }
  }
}

// 清除错误
const clearError = (field) => {
  errors[field] = ''
  authStore.clearError()
}

// 处理登录
const handleLogin = async () => {
  // 验证所有字段
  validateField('username')
  validateField('password')
  
  if (!isFormValid.value) {
    return
  }
  
  try {
    const result = await authStore.login({
      username: form.username.trim(),
      password: form.password
    })
    
    if (result.success) {
      // 登录成功，跳转到成功页面
      router.push('/success')
    }
  } catch (error) {
    console.error('登录失败:', error)
  }
}

// 处理忘记密码
const handleForgotPassword = () => {
  alert('忘记密码功能正在开发中...')
}

// 处理注册
const handleSignup = () => {
  router.push('/register')
}

// 处理社交登录
const handleSocialLogin = (provider) => {
  alert(`${provider}登录功能正在开发中...`)
}

// 组件挂载时初始化
onMounted(() => {
  authStore.initializeAuth()
})
</script>

<style scoped>
.checkbox-container {
  display: flex;
  align-items: center;
  cursor: pointer;
  font-size: 14px;
  color: #666;
}

.checkbox-input {
  margin-right: 8px;
  width: 16px;
  height: 16px;
  accent-color: #667eea;
}

.checkbox-label {
  user-select: none;
}

.login-form {
  margin-bottom: 20px;
}
</style> 