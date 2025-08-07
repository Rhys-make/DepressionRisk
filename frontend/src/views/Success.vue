<template>
  <div class="success-page">
    <div class="success-card">
      <div class="success-icon">✅</div>
      <h1 class="success-title">登录成功！</h1>
      <p class="success-message">欢迎回来，{{ userInfo?.username || '用户' }}</p>
      <div class="user-info">
        <p><strong>用户名：</strong>{{ userInfo?.username }}</p>
        <p><strong>邮箱：</strong>{{ userInfo?.email }}</p>
        <p><strong>角色：</strong>{{ userInfo?.role }}</p>
        <p><strong>登录时间：</strong>{{ loginTime }}</p>
      </div>
      <button class="logout-button" @click="handleLogout">
        退出登录
      </button>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'

// 路由和状态管理
const router = useRouter()
const authStore = useAuthStore()

// 计算属性
const userInfo = computed(() => authStore.userInfo)
const loginTime = computed(() => {
  return new Date().toLocaleString('zh-CN')
})

// 处理退出登录
const handleLogout = () => {
  authStore.logout()
  router.push('/login')
}
</script>

<style scoped>
.success-page {
  min-height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 20px;
}

.success-card {
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  border-radius: 20px;
  padding: 40px;
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.2);
  text-align: center;
  max-width: 500px;
  width: 100%;
}

.success-icon {
  font-size: 60px;
  margin-bottom: 20px;
}

.success-title {
  font-size: 28px;
  font-weight: 600;
  color: #333;
  margin-bottom: 10px;
}

.success-message {
  color: #666;
  font-size: 16px;
  margin-bottom: 30px;
}

.user-info {
  background: #f8f9fa;
  padding: 20px;
  border-radius: 10px;
  margin-bottom: 30px;
  text-align: left;
}

.user-info p {
  margin: 8px 0;
  color: #333;
  font-size: 14px;
}

.logout-button {
  background: #ff4757;
  color: white;
  border: none;
  padding: 12px 24px;
  border-radius: 8px;
  cursor: pointer;
  font-size: 16px;
  font-weight: 500;
  transition: background-color 0.3s ease;
}

.logout-button:hover {
  background: #ff3742;
}

@media (max-width: 480px) {
  .success-card {
    padding: 30px 20px;
  }
  
  .success-title {
    font-size: 24px;
  }
  
  .success-icon {
    font-size: 50px;
  }
}
</style> 