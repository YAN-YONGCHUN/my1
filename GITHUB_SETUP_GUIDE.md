# GitHub 仓库设置指南

本指南将帮助您在GitHub上创建一个新的仓库并上传手势识别项目文件。

## 步骤一：创建GitHub仓库

1. **登录GitHub账号**：
   - 访问 [GitHub官网](https://github.com)
   - 登录您的GitHub账号

2. **创建新仓库**：
   - 点击页面右上角的加号（+）图标
   - 选择 "New repository"

3. **填写仓库信息**：
   - **Repository name**: 建议使用 `hand-gesture-recognition`
   - **Description**: 输入 "基于YOLOv8和OpenCV的实时手势识别系统，可识别8种手势（数字1-5、剪刀、锤头、布）"
   - **Visibility**: 选择 "Public" 或 "Private"（根据您的需求）
   - 不要勾选 "Add a README file"、"Add .gitignore" 和 "Choose a license"（因为我们的项目已经包含了这些文件）
   - 点击 "Create repository"

## 步骤二：上传项目文件

项目压缩包已经创建完成，您可以通过以下两种方式上传：

### 方式一：直接上传压缩包（简单）

1. 在新创建的仓库页面，点击 "Add file" 下拉菜单
2. 选择 "Upload files"
3. 拖拽 `hand-gesture-recognition.zip` 文件到上传区域，或点击 "choose your files" 选择文件
4. 滚动到页面底部，在 "Commit changes" 部分：
   - **Commit message**: 输入 "Initial commit - 上传手势识别项目文件"
   - 点击 "Commit changes"

### 方式二：克隆仓库并上传（推荐，适合后续维护）

如果您计划后续持续维护此项目，建议安装Git并使用这种方式：

1. **安装Git**（如果尚未安装）：
   - 访问 [Git官网](https://git-scm.com/downloads) 下载适合您系统的Git安装包
   - 按照安装向导完成安装
   - 安装完成后，打开命令提示符（CMD）或PowerShell，输入 `git --version` 验证安装成功

2. **配置Git**：
   ```bash
   git config --global user.name "您的GitHub用户名"
   git config --global user.email "您的GitHub邮箱"
   ```

3. **克隆仓库**：
   ```bash
   git clone https://github.com/您的用户名/hand-gesture-recognition.git
   cd hand-gesture-recognition
   ```

4. **解压并复制文件**：
   - 解压 `hand-gesture-recognition.zip` 文件
   - 将解压后的所有文件复制到克隆的仓库目录中

5. **提交和推送更改**：
   ```bash
   git add .
   git commit -m "Initial commit - 手势识别项目文件"
   git push origin main
   ```

## 步骤三：仓库优化设置

1. **设置 README**：
   - 如果您使用方式一上传了压缩包，需要解压文件并更新README
   - 确保README中包含项目简介、安装说明和使用方法

2. **设置 .gitignore**：
   - 确保项目根目录包含 `.gitignore` 文件
   - 建议添加以下内容到 `.gitignore`：
     ```
     # Python
     __pycache__/
     *.py[cod]
     *$py.class
     
     # Environment
     .env
     venv/
     env/
     
     # IDE
     .idea/
     .vscode/
     *.swp
     *.swo
     
     # OS
     .DS_Store
     Thumbs.db
     
     # Model files (large)
     *.pt
     ```

3. **设置仓库描述和标签**：
   - 在仓库页面，点击 "About" 部分右侧的齿轮图标
   - 添加相关标签如：`python`, `opencv`, `yolo`, `computer-vision`, `gesture-recognition`

## 步骤四：项目使用说明

在README.md中确保包含以下使用信息：

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行项目
```bash
python hand_gesture_recognition.py
```

### 可选命令行参数
- `-c` 或 `--camera`：指定摄像头ID，默认为0
- `-v` 或 `--video`：指定视频文件路径（如果使用视频而不是摄像头）
- `-m` 或 `--model`：指定模型文件路径，默认为当前目录下的yolov8n.pt

## 常见问题解决

1. **无法运行cv2.imshow**：
   - 如果遇到cv2.imshow错误，请参考项目中的解决文档
   - 可以使用终端模式运行程序，通过命令行输出查看识别结果

2. **模型加载失败**：
   - 确保yolov8n.pt文件存在且路径正确
   - 检查ultralytics库是否正确安装

3. **手势识别不准确**：
   - 确保光线充足
   - 调整摄像头距离和角度
   - 可以考虑重新训练模型以提高特定环境下的识别准确率

## 后续维护建议

1. **定期更新依赖**：
   ```bash
   pip install -r requirements.txt --upgrade
   ```

2. **改进模型性能**：
   - 收集更多数据进行模型训练
   - 调整模型参数以提高准确率

3. **扩展功能**：
   - 添加更多手势识别
   - 实现手势控制应用程序的功能
   - 优化实时性能

---

祝您的GitHub项目顺利！如有任何问题，请参考GitHub官方文档或在项目Issues中提出。