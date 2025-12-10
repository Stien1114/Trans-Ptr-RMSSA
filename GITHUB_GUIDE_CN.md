# 如何将代码开源到 GitHub

本文档提供将 Trans-Ptr-RMSSA 项目开源到 GitHub 的完整步骤指南。

## 准备工作

### 1. 注册 GitHub 账号
如果还没有 GitHub 账号，请访问 https://github.com 注册。

### 2. 安装 Git
```bash
# Ubuntu/Debian
sudo apt-get install git

# Windows
# 下载安装 Git for Windows: https://git-scm.com/download/win

# macOS
brew install git
```

### 3. 配置 Git
```bash
git config --global user.name "Sibo Chen"
git config --global user.email "chensibo1114@outlook.com"
```

## 创建 GitHub 仓库

### 方法一：通过 GitHub 网页界面（推荐）

1. 登录 GitHub
2. 点击右上角 "+" → "New repository"
3. 填写仓库信息：
   - Repository name: `Trans-Ptr-RMSSA`
   - Description: `A Transformer-Pointer DRL Model for Static Resource Allocation in SDM-EONs`
   - 选择 **Public**
   - **不要** 勾选 "Add a README file"（我们已经准备好了）
   - **不要** 选择 .gitignore 模板（我们已经准备好了）
   - License: 选择 MIT（或跳过，我们已经准备好了）
4. 点击 "Create repository"

## 上传代码

### 步骤 1：初始化本地仓库

在项目目录下执行：

```bash
cd Trans-Ptr-RMSSA
git init
```

### 步骤 2：添加所有文件

```bash
git add .
```

### 步骤 3：创建首次提交

```bash
git commit -m "Initial release: Trans-Ptr RMSSA model

- Transformer encoder with multi-head self-attention
- Pointer network decoder for request ordering
- First-Fit optical network simulator
- Support for NSF, N6S9, EURO16 topologies
- 11-dimensional enhanced features"
```

### 步骤 4：关联远程仓库

```bash
# 将 YOUR_USERNAME 替换为你的 GitHub 用户名
git remote add origin https://github.com/YOUR_USERNAME/Trans-Ptr-RMSSA.git
```

### 步骤 5：推送代码

```bash
git branch -M main
git push -u origin main
```

如果使用 HTTPS 推送时需要认证：
- GitHub 现在不支持密码认证
- 需要创建 Personal Access Token (PAT)
- 访问 Settings → Developer settings → Personal access tokens → Tokens (classic)
- 生成新 token，勾选 `repo` 权限
- 推送时用 token 替代密码

## 项目文件清单

确保以下文件都已包含：

```
Trans-Ptr-RMSSA/
├── README.md              ✓ 项目说明文档
├── LICENSE                ✓ MIT 开源许可证
├── requirements.txt       ✓ Python 依赖
├── .gitignore            ✓ Git 忽略规则
├── model.py              ✓ 神经网络模型
├── trainer.py            ✓ 训练脚本
├── RMSSA_environment.py  ✓ 光网络仿真环境
├── RMSSA_function.py     ✓ 数据集和奖励函数
├── topology_loader.py    ✓ 拓扑加载器
└── ksp_cache.py          ✓ K最短路径缓存
```

## 后续维护

### 添加新版本标签

```bash
git tag -a v1.0.0 -m "First public release"
git push origin v1.0.0
```

### 创建 Release

1. 在 GitHub 仓库页面点击 "Releases"
2. 点击 "Create a new release"
3. 选择刚创建的 tag
4. 填写 Release 标题和说明
5. 发布

### 回复审稿人

在审稿意见回复中，可以这样填写 Response 3：

```
Response 3:
We thank the reviewer for this constructive suggestion regarding code availability.
We have made our implementation publicly available at:

https://github.com/YOUR_USERNAME/Trans-Ptr-RMSSA

The repository includes:
- Complete source code for the Transformer-Pointer network
- Training and evaluation scripts  
- Network topology definitions (NSF, N6S9, EURO16)
- Pre-trained model checkpoints (optional)
- Detailed documentation and usage instructions

Additionally, we have added a Data Availability statement in the revised 
manuscript (Section X) specifying the repository URL.
```

## 可选：添加预训练模型

如果想分享预训练模型权重：

1. 在 GitHub 创建 Release
2. 上传 `.pt` 模型文件作为 Release Assets
3. 或使用 Git LFS (Large File Storage) 管理大文件

```bash
# 安装 Git LFS
git lfs install

# 追踪 .pt 文件
git lfs track "*.pt"

# 添加 .gitattributes
git add .gitattributes

# 然后正常添加和提交模型文件
git add checkpoints/
git commit -m "Add pre-trained model checkpoints"
git push
```

## 常见问题

### Q: 推送时报错 "Permission denied"
A: 确认已正确配置 SSH key 或 Personal Access Token

### Q: 文件太大无法推送
A: 使用 Git LFS 或将大文件添加到 .gitignore

### Q: 如何添加协作者
A: Repository Settings → Collaborators → Add people

## 最佳实践

1. **保持 README 更新**: 及时更新使用说明和实验结果
2. **语义化版本**: 使用 v1.0.0, v1.1.0 等格式
3. **清晰的提交信息**: 描述每次改动的内容
4. **Issue 管理**: 用 GitHub Issues 追踪 bug 和功能请求
5. **CI/CD（可选）**: 配置 GitHub Actions 自动测试
