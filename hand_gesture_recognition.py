import cv2
import numpy as np
from ultralytics import YOLO
import time
from PIL import Image, ImageDraw, ImageFont

# 手势类别映射
gesture_classes = {
    0: "数字1",
    1: "数字2",
    2: "数字3",
    3: "数字4",
    4: "数字5",
    5: "剪刀",
    6: "锤头",
    7: "布"
}

# 加载中文字体
try:
    # 尝试加载Windows系统字体
    FONT_PATH = "C:/Windows/Fonts/simhei.ttf"  # 黑体
    FONT = ImageFont.truetype(FONT_PATH, 24)
    print(f"✅ 成功加载中文字体: {FONT_PATH}")
except Exception as e:
    print(f"❌ 无法加载指定字体: {e}")
    print("💡 尝试使用默认字体")
    FONT = ImageFont.load_default()

class HandGestureRecognizer:
    def __init__(self, model_path=None):
        # 加载YOLOv8模型
        if model_path:
            self.model = YOLO(model_path)
        else:
            # 使用预训练模型，后续可以替换为自定义训练的模型
            self.model = YOLO('yolov8n.pt')
        
        # 打开摄像头
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("无法打开摄像头")
        
        # 设置摄像头分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # 用于计算帧率
        self.prev_time = 0
        self.fps = 0
    
    def preprocess_frame(self, frame):
        """预处理帧图像"""
        # YOLOv8会自动处理图像，这里可以添加额外的预处理步骤
        return frame
    
    def detect_gestures(self, frame):
        """检测手势"""
        # 调整YOLOv8模型参数，提高识别准确率
        # conf: 置信度阈值，提高到0.6减少误检
        # iou: IOU阈值，控制重叠检测框的合并
        # imgsz: 输入图像大小，调整为320x320提高速度
        results = self.model(frame, 
                           conf=0.6,  # 提高置信度阈值，减少误检
                           iou=0.5,   # IOU阈值，控制重叠检测框
                           imgsz=320, # 输入图像大小，平衡速度和准确率
                           verbose=False)
        return results
    
    def draw_results(self, frame, results):
        """在帧上绘制检测结果"""
        # 将OpenCV图像转换为PIL图像，用于绘制中文
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        # 解析检测结果
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 获取边界框坐标
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # 获取置信度
                conf = float(box.conf[0])
                # 获取类别
                cls = int(box.cls[0])
                
                # 检查类别是否在手势映射中
                if cls in gesture_classes:
                    gesture_name = gesture_classes[cls]
                    # 绘制边界框
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # 绘制类别名称和置信度（使用PIL绘制中文）
                    label = f"{gesture_name}: {conf:.2f}"
                    
                    # 将PIL图像转换回OpenCV图像，绘制中文
                    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(pil_img)
                    draw.text((x1, y1 - 30), label, font=FONT, fill=(0, 255, 0))
                    frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        # 绘制帧率（使用PIL绘制中文）
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        fps_text = f"FPS: {self.fps:.1f}"
        draw.text((10, 10), fps_text, font=FONT, fill=(0, 255, 0))
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        return frame
    
    def run(self):
        """运行实时手势识别"""
        print("实时手势识别系统已启动")
        print("💡 提示：")
        print("   - 按 'q' 键退出（如果窗口可用）")
        print("   - 或按 Ctrl+C 退出")
        print()
        
        # 窗口可用性标志
        window_available = True
        
        while True:
            try:
                # 读取帧
                ret, frame = self.cap.read()
                if not ret:
                    print("无法读取摄像头帧")
                    break
                
                # 镜像翻转帧
                frame = cv2.flip(frame, 1)
                
                # 预处理帧
                processed_frame = self.preprocess_frame(frame)
                
                # 检测手势
                results = self.detect_gestures(processed_frame)
                
                # 计算帧率
                current_time = time.time()
                self.fps = 1 / (current_time - self.prev_time) if (current_time - self.prev_time) > 0 else 0
                self.prev_time = current_time
                
                # 绘制结果
                output_frame = self.draw_results(frame, results)
                
                # 提取识别结果（用于终端输出）
                detected_gestures = []
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        if cls in gesture_classes:
                            gesture_name = gesture_classes[cls]
                            detected_gestures.append(f"{gesture_name} ({conf:.2f})")
                
                # 尝试显示帧
                if window_available:
                    try:
                        cv2.imshow("实时手势识别", output_frame)
                        # 按 'q' 键退出
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    except Exception as e:
                        # 窗口显示失败，切换到终端输出模式
                        window_available = False
                        print("⚠️  窗口显示不可用，切换到终端输出模式")
                        print("📋 识别结果将输出到终端")
                        print()
                else:
                    # 终端输出模式
                    if detected_gestures:
                        print(f"[{time.strftime('%H:%M:%S')}] FPS: {self.fps:.1f} | 识别结果: {', '.join(detected_gestures)}")
                    else:
                        print(f"[{time.strftime('%H:%M:%S')}] FPS: {self.fps:.1f} | 未检测到手势")
                    
                    # 短暂延迟，避免输出过快
                    time.sleep(0.1)
                    
            except KeyboardInterrupt:
                # 捕获 Ctrl+C 退出
                print()
                print("🔄 正在退出系统...")
                break
        
        # 释放资源
        self.cap.release()
        try:
            cv2.destroyAllWindows()
        except:
            pass
        print("✅ 实时手势识别系统已关闭")
    
    def train_model(self, data_yaml, epochs=100, imgsz=640):
        """训练自定义手势识别模型"""
        # 加载YOLOv8模型进行训练
        model = YOLO('yolov8n.pt')
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=16,
            name='hand_gesture_model'
        )
        return results

def main():
    """主函数"""
    print("=" * 50)
    print("实时手势识别系统 v1.0")
    print("基于 YOLOv8 和 OpenCV")
    print("=" * 50)
    print()
    
    try:
        # 创建手势识别器实例
        recognizer = HandGestureRecognizer()
        # 运行实时识别
        recognizer.run()
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        print()
        print("💡 常见问题解决方法:")
        print("1. 模型下载失败:")
        print("   - 手动下载模型: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt")
        print("   - 将模型文件放在当前目录")
        print()
        print("2. 摄像头无法打开:")
        print("   - 确保摄像头未被其他程序占用")
        print("   - 检查摄像头驱动是否正常")
        print("   - 尝试使用不同的摄像头索引 (修改代码中的 cv2.VideoCapture(0) 为 1, 2 等)")
        print()
        print("3. OpenCV 窗口显示错误:")
        print("   - 这是由于 OpenCV 编译配置问题")
        print("   - 尝试使用其他环境运行")
        print("   - 或使用远程桌面等工具")
        print()
        print("4. 性能问题:")
        print("   - 降低摄像头分辨率")
        print("   - 提高置信度阈值 conf")
        print("   - 使用更小的模型 (如 yolov8n.pt)")
        print()
        print("📚 详细使用说明请查看 README.md 文件")

if __name__ == "__main__":
    main()
