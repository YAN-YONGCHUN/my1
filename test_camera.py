import cv2
import time

"""
摄像头测试脚本
用于验证摄像头和OpenCV是否正常工作
"""

def test_camera():
    print("正在测试摄像头...")
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return False
    
    # 设置摄像头分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("✅ 摄像头已打开")
    print("📷 正在显示摄像头画面...")
    print("💡 按 'q' 键退出测试")
    
    # 用于计算帧率
    prev_time = 0
    fps = 0
    
    while True:
        # 读取帧
        ret, frame = cap.read()
        if not ret:
            print("❌ 无法读取摄像头帧")
            break
        
        # 镜像翻转帧
        frame = cv2.flip(frame, 1)
        
        # 计算帧率
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time
        
        # 绘制测试信息
        cv2.putText(frame, "摄像头测试", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "按 'q' 键退出", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 显示帧
        cv2.imshow("摄像头测试", frame)
        
        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("✅ 摄像头测试完成")
    return True

def main():
    try:
        test_camera()
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    main()
