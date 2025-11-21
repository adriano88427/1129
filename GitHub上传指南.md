# GitHub 代码上传指南

## 准备工作

您的代码已经准备好上传到GitHub，包括：
- Git仓库已初始化
- 已创建初始提交
- 已添加.gitignore文件排除不必要的文件
- 已创建README.md文件描述项目

## 上传步骤

### 方法1：使用提供的批处理脚本（推荐）

1. 双击运行 `upload_to_github.bat` 文件
2. 按照脚本提示操作：
   - 访问GitHub并创建新仓库
   - 输入仓库URL
   - 脚本会自动完成剩余步骤

### 方法2：手动执行命令

1. 访问 https://github.com 并登录您的账户
2. 点击右上角的 "+" 按钮，选择 "New repository"
3. 填写仓库信息：
   - 仓库名称：`factor-analysis-project`
   - 描述：`因子分析项目 - 包含因子分析代码和数据处理功能`
   - 可见性：选择Public或Private
   - 不要勾选"Add a README file"（我们已经创建了）
4. 点击"Create repository"
5. 在新创建的仓库页面，复制仓库URL（类似：`https://github.com/您的用户名/factor-analysis-project.git`）
6. 在命令行中执行以下命令：

```bash
git remote add origin https://github.com/您的用户名/factor-analysis-project.git
git branch -M main
git push -u origin main
```

## 注意事项

- 如果遇到身份验证问题，您可能需要配置GitHub的Personal Access Token
- 确保您的网络连接正常
- 如果推送失败，可能需要先执行 `git pull origin main --allow-unrelated-histories` 然后再推送

## 上传完成后

上传完成后，您将能够：
- 在GitHub上查看您的代码
- 与他人分享项目链接
- 使用GitHub的协作功能
- 创建Issues和Pull Requests

## 项目结构

上传到GitHub的项目将包含以下主要内容：
- `yinzifenxi1119.py` - 主要的因子分析代码
- `README.md` - 项目说明文档
- `.gitignore` - Git忽略文件配置
- 各种分析报告和日志文件
- MCP相关配置文件

祝您使用愉快！