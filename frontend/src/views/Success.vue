<template>
  <div class="success-page">
    <div class="success-card">
      <div class="success-icon">✨</div>
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
  background: url('https://pic3.zhimg.com/v2-defdc242772bbc870cfa1db4453cb082_r.jpg') no-repeat center center fixed;
  background-size: cover;
  background-color: #f0f8ff;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 20px;
  position: relative;
  overflow: hidden;
}

/* 背景遮罩层 */
.success-page::before {
  content: '';
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.2);
  z-index: -1;
}

.success-card {
  background: rgba(255, 255, 255, 0.2);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border-radius: 20px;
  padding: 50px;
  box-shadow: 
    0 25px 45px rgba(0, 0, 0, 0.2),
    inset 0 1px 0 rgba(255, 255, 255, 0.2);
  border: 1px solid rgba(255, 255, 255, 0.3);
  text-align: center;
  max-width: 600px;
  width: 100%;
  position: relative;
  overflow: hidden;
}

/* 卡片内部光效 */
.success-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: -100%;
  width: 100%;
  height: 100%;
  background: linear-gradient(
    90deg,
    transparent,
    rgba(255, 255, 255, 0.1),
    transparent
  );
  transition: left 0.5s;
}

.success-card:hover::before {
  left: 100%;
}

.success-icon {
  font-size: 60px;
  margin-bottom: 20px;
  animation: glow 2s ease-in-out infinite alternate;
}

@keyframes glow {
  from { text-shadow: 0 0 20px rgba(255, 255, 255, 0.5); }
  to { text-shadow: 0 0 30px rgba(255, 255, 255, 0.8), 0 0 40px rgba(102, 126, 234, 0.5); }
}

.success-title {
  font-size: 32px;
  font-weight: 700;
  color: #fff;
  margin-bottom: 10px;
  text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
}

.success-message {
  color: rgba(255, 255, 255, 0.8);
  font-size: 18px;
  margin-bottom: 30px;
  font-weight: 300;
}

.user-info {
  background: rgba(255, 255, 255, 0.1);
  padding: 25px;
  border-radius: 15px;
  margin-bottom: 30px;
  text-align: left;
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.1);
}

.user-info p {
  margin: 12px 0;
  color: rgba(255, 255, 255, 0.9);
  font-size: 14px;
  line-height: 1.6;
}

.user-info strong {
  color: #667eea;
  text-shadow: 0 0 10px rgba(102, 126, 234, 0.3);
}

.logout-button {
  background: linear-gradient(135deg, #ff4757 0%, #ff3742 100%);
  color: white;
  border: none;
  padding: 14px 28px;
  border-radius: 12px;
  cursor: pointer;
  font-size: 16px;
  font-weight: 600;
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
  text-transform: uppercase;
  letter-spacing: 1px;
}

.logout-button::before {
  content: '';
  position: absolute;
  top: 0;
  left: -100%;
  width: 100%;
  height: 100%;
  background: linear-gradient(
    90deg,
    transparent,
    rgba(255, 255, 255, 0.2),
    transparent
  );
  transition: left 0.5s;
}

.logout-button:hover::before {
  left: 100%;
}

.logout-button:hover {
  transform: translateY(-2px);
  box-shadow: 
    0 15px 30px rgba(255, 71, 87, 0.4),
    0 0 30px rgba(255, 71, 87, 0.3);
}

@media (max-width: 480px) {
  .success-card {
    padding: 30px 25px;
  }
  
  .success-title {
    font-size: 28px;
  }
  
  .success-icon {
    font-size: 50px;
  }
  
  .user-info {
    padding: 20px;
  }
}
</style> 