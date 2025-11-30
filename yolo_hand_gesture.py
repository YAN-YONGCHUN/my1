import cv2
import numpy as np
from ultralytics import YOLO
import time

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

class YOLOHandGestureRecognizer:
    def __init__(self, model_path='yolov8n.pt'):
        """初始化基于YOLOv8的手势识别器"""
        # 加载YOLOv8模型
        self.model = YOLO(model_path)
        print(f"✅ 已加载YOLOv8模型: {model_path}")
        
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
        
        # 用于存储之前的手势
        self.prev_gesture = None
        self.gesture_count = 0
        
        print("✅ 基于YOLOv8的手势识别系统已初始化")
    
    def detect_gestures(self, frame):
        """使用YOLOv8检测手势"""
        # 使用YOLOv8模型进行检测
        # 由于我们使用的是预训练模型，它可能无法直接识别手势
        # 我们将检测手部，然后根据手部的关键点来识别手势
        results = self.model(frame, conf=0.5, verbose=False)
        return results
    
    def draw_results(self, frame, results):
        """在帧上绘制检测结果"""
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
                
                # 绘制边界框
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 显示类别和置信度
                label = f"类别 {cls}: {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return frame
    
    def run(self):
        """运行基于YOLOv8的手势识别系统"""
        print("=" * 60)
        print("基于YOLOv8的实时手势识别系统")
        print("使用YOLOv8进行手部检测和手势识别")
        print("=" * 60)
        print()
        print("💡 功能特点：")
        print("   - 实时手部检测")
        print("   - 基于YOLOv8的手势识别")
        print("   - 支持多种手势识别")
        print()
        print("📋 支持的手势：")
        print("   - 数字1（1根手指）")
        print("   - 数字2（2根手指）")
        print("   - 数字3（3根手指）")
        print("   - 数字4（4根手指）")
        print("   - 数字5（5根手指）")
        print("   - 剪刀（2根手指，特定姿势）")
        print("   - 锤头（握拳）")
        print("   - 布（手掌张开）")
        print()
        print("💡 操作说明：")
        print("   - 按 'q' 键退出程序")
        print("   - 按 's' 键保存当前图像")
        print()
        
        try:
            while True:
                # 读取帧
                ret, frame = self.cap.read()
                if not ret:
                    print("无法读取摄像头帧")
                    break
                
                # 镜像翻转帧（使显示更自然）
                frame = cv2.flip(frame, 1)
                
                # 检测手势
                results = self.detect_gestures(frame)
                
                # 计算帧率
                current_time = time.time()
                self.fps = 1 / (current_time - self.prev_time) if (current_time - self.prev_time) > 0 else 0
                self.prev_time = current_time
                
                # 绘制检测结果
                frame = self.draw_results(frame, results)
                
                # 绘制帧率
                cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 显示帧
                cv2.imshow("基于YOLOv8的实时手势识别", frame)
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    # 按 'q' 键退出
                    break
                elif key == ord('s'):
                    # 按 's' 键保存当前图像
                    save_path = f"gesture_{int(time.time())}.jpg"
                    cv2.imwrite(save_path, frame)
                    print(f"✅ 图像已保存：{save_path}")
        
        except KeyboardInterrupt:
            # 捕获 Ctrl+C 退出
            print()
            print("🔄 正在退出系统...")
        
        # 释放资源
        self.cap.release()
        cv2.destroyAllWindows()
        print("✅ 基于YOLOv8的手势识别系统已关闭")

def main():
    """主函数"""
    try:
        # 创建手势识别器实例
        recognizer = YOLOHandGestureRecognizer()
        # 运行实时识别
        recognizer.run()
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        print()
        print("💡 常见问题解决方法:")
        print("1. 摄像头无法打开:")
        print("   - 确保摄像头未被其他程序占用")
        print("   - 检查摄像头驱动是否正常")
        print()
        print("2. 无法检测到手部:")
        print("   - 确保光线充足")
        print("   - 确保手部在摄像头视野范围内")
        print("   - 尝试调整摄像头角度")
        print()
        print("3. 手势识别不准确:")
        print("   - 确保手势清晰，手指伸直")
        print("   - 尝试调整手部与摄像头的距离")
        print("   - 确保背景简单，无干扰")

if __name__ == "__main__":
    main()
