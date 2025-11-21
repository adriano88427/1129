@echo off
echo ===== GitHub 代码上传脚本 =====
echo.
echo 请按照以下步骤操作：
echo.
echo 1. 访问 https://github.com 并登录您的账户
echo 2. 点击右上角的 "+" 按钮，选择 "New repository"
echo 3. 输入仓库名称: factor-analysis-project
echo 4. 添加描述: 因子分析项目 - 包含因子分析代码和数据处理功能
echo 5. 选择 "Public" 或 "Private" (根据您的需求)
echo 6. 不要勾选 "Add a README file" (我们已经创建了)
echo 7. 点击 "Create repository"
echo.
echo 创建完成后，GitHub会显示快速设置页面，复制其中的仓库URL
echo.
set /p repo_url=请粘贴GitHub仓库URL: 

echo.
echo 正在添加远程仓库...
git remote add origin %repo_url%

echo.
echo 正在推送代码到GitHub...
git branch -M main
git push -u origin main

echo.
echo ===== 上传完成 =====
echo 您的代码已成功上传到GitHub！
pause