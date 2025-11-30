import cv2
import numpy as np

"""
OpenCV窗口测试脚本
用于验证OpenCV窗口显示功能是否正常工作
"""

def test_opencv_window():
    print("正在测试OpenCV窗口显示功能...")
    
    # 创建一个简单的测试图像
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    img.fill(255)  # 白色背景
    
    # 绘制测试内容
    cv2.putText(img, "OpenCV窗口测试", (50, 100), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    cv2.putText(img, "如果能看到这个窗口，说明OpenCV窗口功能正常", (50, 150), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "按 'q' 键退出", (50, 200), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    try:
        # 尝试显示窗口
        cv2.namedWindow("OpenCV窗口测试", cv2.WINDOW_NORMAL)
        cv2.imshow("OpenCV窗口测试", img)
        print("✅ 窗口已创建，正在显示测试图像...")
        print("💡 按 'q' 键退出测试")
        
        # 等待按键
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # 释放资源
        cv2.destroyAllWindows()
        print("✅ OpenCV窗口测试成功")
        return True
        
    except Exception as e:
        print(f"❌ OpenCV窗口测试失败: {e}")
        print("💡 错误原因：")
        print("   - 当前环境可能没有图形界面支持")
        print("   - 或OpenCV编译时没有包含窗口支持")
        print("   - 或其他系统配置问题")
        return False

def main():
    test_opencv_window()

if __name__ == "__main__":
    main()
