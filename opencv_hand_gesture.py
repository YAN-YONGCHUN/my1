import cv2
import numpy as np
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

class OpenCVHandGestureRecognizer:
    def __init__(self):
        """初始化基于OpenCV的手势识别器"""
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
        
        # 皮肤颜色范围（HSV）
        self.lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # 用于存储之前的手势
        self.prev_gesture = None
        self.gesture_count = 0
        
        # 加载中文字体
        try:
            # 尝试加载Windows系统字体
            self.font_path = "C:/Windows/Fonts/simhei.ttf"  # 黑体
            self.font = ImageFont.truetype(self.font_path, 24)
            print(f"✅ 成功加载中文字体: {self.font_path}")
        except Exception as e:
            print(f"❌ 无法加载指定字体: {e}")
            print("💡 尝试使用默认字体")
            self.font = ImageFont.load_default()
        
        print("✅ 基于OpenCV的手势识别系统已初始化")
    
    def preprocess_frame(self, frame):
        """预处理帧图像"""
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 创建皮肤颜色掩码
        mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        
        # 形态学操作，去除噪声
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        # 高斯模糊
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        return mask
    
    def find_hand_contour(self, mask):
        """查找手部轮廓"""
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # 找到最大的轮廓（假设是手）
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            # 检查轮廓面积是否足够大
            if cv2.contourArea(max_contour) > 1000:
                return max_contour
        
        return None
    
    def count_fingers(self, contour, frame):
        """计算伸直的手指数量"""
        # 创建凸包
        hull = cv2.convexHull(contour)
        
        # 绘制凸包
        cv2.drawContours(frame, [hull], -1, (0, 255, 0), 2)
        
        # 计算凸缺陷
        hull_indices = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull_indices)
        
        # 手指数量
        finger_count = 0
        
        # 存储指尖点
        finger_tips = []
        
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])
                
                # 计算三角形边长
                a = np.linalg.norm(np.array(end) - np.array(start))
                b = np.linalg.norm(np.array(far) - np.array(start))
                c = np.linalg.norm(np.array(end) - np.array(far))
                
                # 使用余弦定理计算角度
                angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c)) * 180 / np.pi
                
                # 如果角度小于90度，认为是手指之间的凹陷
                if angle < 90:
                    finger_count += 1
                    # 绘制指尖点
                    cv2.circle(frame, end, 5, (0, 0, 255), -1)
                    finger_tips.append(end)
                    
                    # 绘制连接线
                    cv2.line(frame, start, end, (0, 255, 0), 2)
                    cv2.line(frame, end, far, (0, 255, 0), 2)
                    cv2.line(frame, far, start, (0, 255, 0), 2)
        
        # 检查是否有手掌（如果没有检测到凹陷，可能是拳头或布）
        if finger_count == 0:
            # 检查轮廓的圆度
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * (area / (perimeter ** 2))
                if circularity > 0.7:
                    # 圆形轮廓，可能是拳头
                    return 0, finger_tips
        
        # 实际手指数量是凹陷数量 + 1
        return finger_count + 1, finger_tips
    
    def recognize_gesture(self, finger_count, contour, frame):
        """根据手指数量和轮廓特征识别手势"""
        # 优化锤头手势识别
        if finger_count == 0:
            # 检查轮廓的圆度，判断是否是拳头
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * (area / (perimeter ** 2))
                # 圆形轮廓，更可能是拳头（锤头）
                if circularity > 0.7:
                    return "锤头", 0.90
                else:
                    # 非圆形轮廓，可能是其他手势
                    return "布", 0.85
            else:
                return "布", 0.85
        elif finger_count == 1:
            return "数字1", 0.95
        elif finger_count == 2:
            # 2根手指可以是数字2或剪刀
            # 这里可以根据手指的位置进一步区分
            # 暂时都识别为数字2
            return "数字2", 0.90
        elif finger_count == 3:
            return "数字3", 0.85
        elif finger_count == 4:
            return "数字4", 0.85
        elif finger_count == 5:
            return "数字5", 0.90
        else:
            return "布", 0.85
    
    def draw_finger_contour(self, frame, contour, finger_tips):
        """绘制手指轮廓"""
        if contour is not None:
            # 绘制手部轮廓
            cv2.drawContours(frame, [contour], -1, (255, 0, 0), 2)
            
            # 绘制指尖连线
            if len(finger_tips) >= 2:
                for i in range(len(finger_tips) - 1):
                    cv2.line(frame, finger_tips[i], finger_tips[i+1], (0, 255, 255), 2)
    
    def run(self):
        """运行基于OpenCV的手势识别系统"""
        print("=" * 60)
        print("基于OpenCV的实时手势识别系统")
        print("使用肤色检测和轮廓分析")
        print("=" * 60)
        print()
        print("💡 功能特点：")
        print("   - 实时手部检测")
        print("   - 手指轮廓标注")
        print("   - 基于手指数量的手势分类")
        print("   - 支持多种手势识别")
        print()
        print("📋 支持的手势：")
        print("   - 数字1（1根手指）")
        print("   - 数字2（2根手指）")
        print("   - 数字3（3根手指）")
        print("   - 数字4（4根手指）")
        print("   - 数字5（5根手指）")
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
                
                # 复制原始帧用于显示
                display_frame = frame.copy()
                
                # 预处理帧
                mask = self.preprocess_frame(frame)
                
                # 查找手部轮廓
                contour = self.find_hand_contour(mask)
                
                # 计算帧率
                current_time = time.time()
                self.fps = 1 / (current_time - self.prev_time) if (current_time - self.prev_time) > 0 else 0
                self.prev_time = current_time
                
                # 使用PIL绘制帧率，解决汉字乱码问题
                pil_img = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_img)
                fps_text = f"FPS: {self.fps:.1f}"
                draw.text((10, 10), fps_text, font=self.font, fill=(0, 255, 0))
                display_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                
                # 如果找到手部轮廓
                if contour is not None:
                    # 计算手指数量
                    finger_count, finger_tips = self.count_fingers(contour, display_frame)
                    
                    # 识别手势，传入轮廓和帧信息
                    gesture_name, confidence = self.recognize_gesture(finger_count, contour, display_frame)
                    
                    # 绘制手指轮廓
                    self.draw_finger_contour(display_frame, contour, finger_tips)
                    
                    # 获取手部边界框
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # 绘制边界框
                    cv2.rectangle(display_frame, (x - 20, y - 20), (x + w + 20, y + h + 20), 
                                 (0, 255, 0), 2)
                    
                    # 使用PIL绘制中文，解决汉字乱码问题
                    # 将OpenCV图像转换为PIL图像
                    pil_img = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(pil_img)
                    
                    # 显示手势名称和置信度
                    label = f"{gesture_name}: {confidence:.2f}"
                    draw.text((x - 20, y - 30), label, font=self.font, fill=(0, 255, 0))
                    
                    # 显示手指数量
                    finger_text = f"手指数量: {finger_count}"
                    draw.text((x - 20, y + h + 50), finger_text, font=self.font, fill=(0, 255, 0))
                    
                    # 将PIL图像转换回OpenCV图像
                    display_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                
                # 显示原始帧和掩码
                cv2.imshow("基于OpenCV的实时手势识别", display_frame)
                cv2.imshow("皮肤掩码", mask)
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    # 按 'q' 键退出
                    break
                elif key == ord('s'):
                    # 按 's' 键保存当前图像
                    save_path = f"gesture_{int(time.time())}.jpg"
                    cv2.imwrite(save_path, display_frame)
                    print(f"✅ 图像已保存：{save_path}")
        
        except KeyboardInterrupt:
            # 捕获 Ctrl+C 退出
            print()
            print("🔄 正在退出系统...")
        
        # 释放资源
        self.cap.release()
        cv2.destroyAllWindows()
        print("✅ 基于OpenCV的手势识别系统已关闭")

def main():
    """主函数"""
    try:
        # 创建手势识别器实例
        recognizer = OpenCVHandGestureRecognizer()
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
        print("   - 尝试调整手部与摄像头的距离")
        print("   - 确保背景简单，无干扰")
        print()
        print("3. 手势识别不准确:")
        print("   - 确保手势清晰，手指伸直")
        print("   - 尝试调整手部角度")
        print("   - 确保手部与背景有明显的颜色差异")

if __name__ == "__main__":
    main()
