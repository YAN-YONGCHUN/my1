import cv2
import numpy as np
import mediapipe as mp
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

class OptimizedHandGestureRecognizer:
    def __init__(self):
        """初始化优化后的手势识别器"""
        # 初始化MediaPipe手部检测
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # 配置手部检测模型
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
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
        
        print("✅ 优化后的手势识别系统已初始化")
    
    def count_fingers(self, hand_landmarks):
        """根据手部关键点计算手指数量"""
        # 手指尖端关键点索引
        finger_tips = [8, 12, 16, 20]  # 食指、中指、无名指、小指
        thumb_tip = 4  # 拇指
        
        # 手指根部关键点索引（用于判断手指是否伸直）
        finger_bases = [6, 10, 14, 18]  # 食指、中指、无名指、小指
        
        # 获取手腕关键点（用于参考）
        wrist = hand_landmarks.landmark[0]
        
        # 计数伸直的手指
        finger_count = 0
        
        # 检查拇指
        # 拇指的判断比较特殊，需要考虑左右手
        # 对于右手，拇指尖端x坐标小于拇指根部x坐标时，拇指伸直
        # 对于左手，拇指尖端x坐标大于拇指根部x坐标时，拇指伸直
        thumb_base = hand_landmarks.landmark[2]
        if hand_landmarks.landmark[thumb_tip].x < hand_landmarks.landmark[thumb_base].x:
            # 右手拇指伸直
            finger_count += 1
        elif hand_landmarks.landmark[thumb_tip].x > hand_landmarks.landmark[thumb_base].x:
            # 左手拇指伸直
            finger_count += 1
        
        # 检查其他四根手指
        for tip, base in zip(finger_tips, finger_bases):
            # 如果手指尖端y坐标小于手指根部y坐标，说明手指伸直
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[base].y:
                finger_count += 1
        
        return finger_count
    
    def recognize_gesture(self, hand_landmarks):
        """根据手部关键点识别手势"""
        # 计算手指数量
        finger_count = self.count_fingers(hand_landmarks)
        
        # 根据手指数量识别基本手势
        if finger_count == 1:
            return "数字1", 0.95
        elif finger_count == 2:
            # 检查是否是剪刀手势（食指和中指伸直，其他手指弯曲）
            # 获取食指和中指尖端
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            ring_tip = hand_landmarks.landmark[16]
            pinky_tip = hand_landmarks.landmark[20]
            
            # 检查无名指和小指是否弯曲
            if ring_tip.y > hand_landmarks.landmark[14].y and pinky_tip.y > hand_landmarks.landmark[18].y:
                return "剪刀", 0.90
            else:
                return "数字2", 0.90
        elif finger_count == 3:
            return "数字3", 0.85
        elif finger_count == 4:
            return "数字4", 0.85
        elif finger_count == 5:
            return "数字5", 0.90
        else:
            # 检查是否是锤头手势（握拳）
            # 检查所有手指尖端是否都低于手指根部
            all_bent = True
            for tip in [4, 8, 12, 16, 20]:
                if tip == 4:  # 拇指
                    if hand_landmarks.landmark[tip].x < hand_landmarks.landmark[2].x or \
                       hand_landmarks.landmark[tip].x > hand_landmarks.landmark[2].x:
                        all_bent = False
                        break
                else:  # 其他手指
                    finger_index = tip // 4  # 0:食指, 1:中指, 2:无名指, 3:小指
                    base_index = 6 + finger_index * 4
                    if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[base_index].y:
                        all_bent = False
                        break
            
            if all_bent:
                return "锤头", 0.85
            else:
                return "布", 0.85
    
    def draw_hand_landmarks(self, frame, hand_landmarks):
        """绘制手部关键点和轮廓"""
        # 绘制手部关键点
        self.mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing_styles.get_default_hand_landmarks_style(),
            self.mp_drawing_styles.get_default_hand_connections_style()
        )
        
        # 绘制手指轮廓（连接手指尖端）
        # 获取手指尖端坐标
        finger_tips = [4, 8, 12, 16, 20]  # 拇指、食指、中指、无名指、小指
        h, w, _ = frame.shape
        
        # 转换关键点坐标到图像坐标系
        tip_points = []
        for tip in finger_tips:
            x = int(hand_landmarks.landmark[tip].x * w)
            y = int(hand_landmarks.landmark[tip].y * h)
            tip_points.append((x, y))
            # 绘制指尖点
            cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
        
        # 绘制手指轮廓线（连接指尖）
        if len(tip_points) >= 2:
            # 连接拇指到食指
            cv2.line(frame, tip_points[0], tip_points[1], (0, 255, 255), 2)
            # 连接食指到中指
            cv2.line(frame, tip_points[1], tip_points[2], (0, 255, 255), 2)
            # 连接中指到无名指
            cv2.line(frame, tip_points[2], tip_points[3], (0, 255, 255), 2)
            # 连接无名指到小指
            cv2.line(frame, tip_points[3], tip_points[4], (0, 255, 255), 2)
        
        return frame
    
    def run(self):
        """运行优化后的手势识别系统"""
        print("=" * 60)
        print("优化后的实时手势识别系统 v2.0")
        print("基于 MediaPipe 和 OpenCV")
        print("=" * 60)
        print()
        print("💡 功能特点：")
        print("   - 实时手部检测和关键点识别")
        print("   - 手指轮廓标注")
        print("   - 基于手指数量和位置的手势分类")
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
                
                # 转换为RGB格式（MediaPipe需要RGB输入）
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 处理帧，检测手部
                results = self.hands.process(rgb_frame)
                
                # 计算帧率
                current_time = time.time()
                self.fps = 1 / (current_time - self.prev_time) if (current_time - self.prev_time) > 0 else 0
                self.prev_time = current_time
                
                # 绘制帧率
                cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 如果检测到手部
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # 绘制手部关键点和轮廓
                        frame = self.draw_hand_landmarks(frame, hand_landmarks)
                        
                        # 识别手势
                        gesture_name, confidence = self.recognize_gesture(hand_landmarks)
                        
                        # 获取手部边界框
                        h, w, _ = frame.shape
                        x_min = w
                        y_min = h
                        x_max = 0
                        y_max = 0
                        
                        for landmark in hand_landmarks.landmark:
                            x, y = int(landmark.x * w), int(landmark.y * h)
                            if x < x_min:
                                x_min = x
                            if y < y_min:
                                y_min = y
                            if x > x_max:
                                x_max = x
                            if y > y_max:
                                y_max = y
                        
                        # 添加边界框和手势标签
                        cv2.rectangle(frame, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), 
                                     (0, 255, 0), 2)
                        
                        # 显示手势名称和置信度
                        label = f"{gesture_name}: {confidence:.2f}"
                        cv2.putText(frame, label, (x_min - 20, y_min - 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # 显示帧
                cv2.imshow("优化后的实时手势识别", frame)
                
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
        self.hands.close()
        print("✅ 优化后的手势识别系统已关闭")

def main():
    """主函数"""
    try:
        # 创建手势识别器实例
        recognizer = OptimizedHandGestureRecognizer()
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
