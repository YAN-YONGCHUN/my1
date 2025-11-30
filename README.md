# 实时手势识别系统

基于YOLOv8和OpenCV的实时手势识别系统，能够识别8种特定手势：数字1、2、3、4、5，以及剪刀、锤头、布（石头）手势。

## 功能特点

- ✅ 实时摄像头视频流捕获
- ✅ 基于YOLOv8的高效手势检测与分类
- ✅ 实时标注识别结果（手势名称和置信度）
- ✅ 帧率显示（目标：≥24fps）
- ✅ 支持自定义模型训练
- ✅ 简单易用的命令行界面

## 技术栈

- **YOLOv8**：用于目标检测和分类
- **OpenCV**：用于视频捕获和图像处理
- **Python**：主要开发语言

## 安装步骤

### 1. 克隆或下载项目

将项目文件下载到本地目录。

### 2. 安装依赖

使用pip安装所需依赖：

```bash
pip install -r requirements.txt
```

## 使用方法

### 运行实时手势识别

```bash
python hand_gesture_recognition.py
```

程序将启动摄像头，实时显示手势识别结果。按 `q` 键退出程序。

### 训练自定义模型

1. **准备数据集**

   按照以下结构组织数据集：
   ```
   datasets/
   ├── train/
   │   ├── images/      # 训练集图像
   │   └── labels/      # 训练集标注文件
   ├── val/
   │   ├── images/      # 验证集图像
   │   └── labels/      # 验证集标注文件
   └── test/ (可选)
       ├── images/      # 测试集图像
       └── labels/      # 测试集标注文件
   ```

   标注文件格式为YOLO格式（txt文件）：
   ```
   <class_id> <x_center> <y_center> <width> <height>
   ```

2. **配置数据文件**

   修改 `hand_gesture_data.yaml` 文件，确保路径正确。

3. **运行训练**

   在代码中调用 `train_model` 方法，或修改主函数进行训练：

   ```python
   def main():
       recognizer = HandGestureRecognizer()
       recognizer.train_model('hand_gesture_data.yaml', epochs=100)
   ```

4. **使用训练好的模型**

   将训练好的模型文件（如 `runs/detect/hand_gesture_model/weights/best.pt`）路径传递给 `HandGestureRecognizer` 构造函数：

   ```python
   recognizer = HandGestureRecognizer(model_path='path/to/best.pt')
   ```

## 手势类别

| 类别ID | 手势名称 |
|--------|----------|
| 0      | 数字1    |
| 1      | 数字2    |
| 2      | 数字3    |
| 3      | 数字4    |
| 4      | 数字5    |
| 5      | 剪刀     |
| 6      | 锤头     |
| 7      | 布       |

## 性能优化

- 降低摄像头分辨率可以提高帧率
- 调整模型的 `conf` 参数可以平衡准确率和速度
- 使用更小的YOLOv8模型（如nano）可以提高速度

## 注意事项

1. 确保摄像头能够正常工作
2. 在光线充足的环境下使用效果更佳
3. 训练模型需要大量标注数据才能达到90%以上的准确率
4. 首次运行时会自动下载YOLOv8预训练模型

## 项目结构

```
hand_gesture_recognition/
├── hand_gesture_recognition.py  # 主程序文件
├── hand_gesture_data.yaml       # 数据配置文件
├── requirements.txt             # 依赖列表
└── README.md                    # 项目说明
```

## 许可证

MIT License
