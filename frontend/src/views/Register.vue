<template>
  <div class="container">
    <div class="login-card">
      <!-- 注册头部 -->
      <div class="login-header">
        <h1 class="login-title">Register</h1>
        <p class="login-subtitle">创建您的账户</p>
      </div>

      <!-- 注册表单 -->
      <form @submit.prevent="handleRegister" class="login-form">
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
          <label for="confirmPassword" class="form-label">Confirm Password</label>
          <input
            id="confirmPassword"
            v-model="form.confirmPassword"
            :type="showPassword ? 'text' : 'password'"
            class="form-input"
            :class="{ error: errors.confirmPassword }"
            placeholder="请再次输入密码"
            @input="clearError('confirmPassword')"
            @blur="validateField('confirmPassword')"
          />
          <div v-if="errors.confirmPassword" class="error-message">
            {{ errors.confirmPassword }}
          </div>
        </div>

        <button
          type="submit"
          class="login-button"
          :disabled="loading || !isFormValid"
        >
          <span v-if="loading" class="loading-spinner"></span>
          {{ loading ? 'Registering...' : 'Register' }}
        </button>
      </form>

      <!-- 错误信息 -->
      <div v-if="authStore.error" class="error-message">
        {{ authStore.error }}
      </div>

      <!-- 注册页脚 -->
      <div class="login-footer">
        <div class="signup-link">
          <span>Already have an account? </span>
          <a href="#" @click.prevent="goToLogin">Login now</a>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'

// 路由和状态管理
const router = useRouter()
const authStore = useAuthStore()

// 表单数据
const form = ref({
  username: '',
  password: '',
  confirmPassword: ''
})

// 表单验证错误
const errors = ref({
  username: '',
  password: '',
  confirmPassword: ''
})

// 显示密码状态
const showPassword = ref(false)

// 计算属性
const loading = computed(() => authStore.loading)
const isFormValid = computed(() => {
  return form.value.username && 
         form.value.password && 
         form.value.confirmPassword &&
         !errors.value.username &&
         !errors.value.password &&
         !errors.value.confirmPassword
})

// 表单验证
const validateField = (field) => {
  errors.value[field] = ''
  
  switch (field) {
    case 'username':
      if (!form.value.username) {
        errors.value.username = '用户名不能为空'
      } else if (form.value.username.length < 3) {
        errors.value.username = '用户名长度至少为3个字符'
      }
      break
      
    case 'password':
      if (!form.value.password) {
        errors.value.password = '密码不能为空'
      } else if (form.value.password.length < 6) {
        errors.value.password = '密码长度至少为6个字符'
      }
      break
      
    case 'confirmPassword':
      if (!form.value.confirmPassword) {
        errors.value.confirmPassword = '确认密码不能为空'
      } else if (form.value.password !== form.value.confirmPassword) {
        errors.value.confirmPassword = '两次输入的密码不一致'
      }
      break
  }
}

// 清除错误
const clearError = (field) => {
  errors.value[field] = ''
  authStore.clearError()
}

// 处理注册
const handleRegister = async () => {
  // 验证所有字段
  validateField('username')
  validateField('password')
  validateField('confirmPassword')
  
  if (!isFormValid.value) {
    return
  }
  
  const result = await authStore.register({
    username: form.value.username,
    password: form.value.password,
    confirm_password: form.value.confirmPassword
  })
  
  if (result.success) {
    alert('注册成功！请登录')
    router.push('/login')
  }
}

// 跳转到登录页
const goToLogin = () => {
  router.push('/login')
}
</script>

<style scoped>
/* 继承全局样式，无需额外样式 */
</style> 