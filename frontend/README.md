# Vue3 登录系统

一个基于 Vue3 + Vite + Pinia 的现代化登录页面应用。

## 功能特性

- 🎨 现代化的UI设计，支持响应式布局
- 🔐 完整的登录/登出功能
- 📱 移动端适配
- 🛡️ 路由守卫和认证保护
- 📦 基于 Pinia 的状态管理
- 🔄 自动token管理和API拦截器
- ✨ 表单验证和错误处理
- 🎯 社交登录入口（可扩展）

## 技术栈

- **前端框架**: Vue 3.3.4
- **构建工具**: Vite 4.4.9
- **状态管理**: Pinia 2.1.6
- **路由管理**: Vue Router 4.2.4
- **HTTP客户端**: Axios 1.5.0
- **样式**: CSS3 + 响应式设计

## 项目结构

```
frontend/
├── src/
│   ├── api/           # API接口
│   ├── components/    # 公共组件
│   ├── router/        # 路由配置
│   ├── stores/        # Pinia状态管理
│   ├── views/         # 页面组件
│   ├── App.vue        # 根组件
│   ├── main.js        # 应用入口
│   └── style.css      # 全局样式
├── index.html         # HTML模板
├── package.json       # 项目配置
├── vite.config.js     # Vite配置
└── README.md          # 项目说明
```

## 快速开始

### 1. 安装依赖

```bash
npm install
```

### 2. 启动开发服务器

```bash
npm run dev
```

应用将在 `http://localhost:3000` 启动

### 3. 构建生产版本

```bash
npm run build
```

### 4. 预览生产版本

```bash
npm run preview
```

## 使用说明

### 登录功能

1. 访问登录页面
2. 输入用户名和密码
3. 点击登录按钮
4. 登录成功后自动跳转到成功页面

### 测试账号

由于目前没有后端服务，您可以：
- 在浏览器控制台查看API调用
- 修改 `src/api/auth.js` 中的API地址指向您的后端服务
- 或者添加模拟数据进行测试

### 路由说明

- `/` - 重定向到登录页
- `/login` - 登录页面
- `/success` - 登录成功页面（需要认证）

## 自定义配置

### 修改API地址

编辑 `src/api/auth.js` 文件中的 `baseURL`：

```javascript
const api = axios.create({
  baseURL: 'http://your-backend-url/api', // 修改为您的后端地址
  // ...
})
```

### 修改样式主题

编辑 `src/style.css` 文件中的CSS变量来修改主题色彩。

## 开发指南

### 添加新页面

1. 在 `src/views/` 目录下创建新的Vue组件
2. 在 `src/router/index.js` 中添加路由配置
3. 根据需要添加路由守卫

### 添加新功能

1. 在 `src/stores/` 目录下创建新的Pinia store
2. 在 `src/api/` 目录下添加相应的API接口
3. 在组件中使用store和API

## 浏览器支持

- Chrome >= 87
- Firefox >= 78
- Safari >= 14
- Edge >= 88

## 许可证

MIT License