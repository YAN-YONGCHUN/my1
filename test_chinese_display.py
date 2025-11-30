import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

"""
中文显示测试脚本
用于验证和修复OpenCV窗口中文显示问题
"""

def test_cv2_chinese():
    """测试OpenCV原生中文显示"""
    print("测试OpenCV原生中文显示...")
    
    # 创建测试图像
    img = np.zeros((300, 600, 3), dtype=np.uint8)
    img.fill(255)
    
    # 尝试使用OpenCV默认字体显示中文
    cv2.putText(img, "测试中文显示", (50, 100), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    
    # 显示图像
    cv2.imshow("OpenCV中文显示测试", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_pil_chinese():
    """使用PIL库显示中文"""
    print("测试PIL库中文显示...")
    
    # 创建测试图像
    img = np.zeros((300, 600, 3), dtype=np.uint8)
    img.fill(255)
    
    # 将OpenCV图像转换为PIL图像
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    
    # 尝试加载系统字体
    try:
        # 尝试加载Windows系统字体
        font_path = "C:/Windows/Fonts/simhei.ttf"  # 黑体
        font = ImageFont.truetype(font_path, 36)
        print(f"✅ 成功加载字体: {font_path}")
    except Exception as e:
        print(f"❌ 无法加载指定字体: {e}")
        print("💡 尝试使用默认字体")
        font = ImageFont.load_default()
    
    # 使用PIL绘制中文
    draw.text((50, 100), "测试中文显示", font=font, fill=(255, 0, 0))
    draw.text((50, 150), "数字1: 0.95", font=font, fill=(0, 255, 0))
    draw.text((50, 200), "剪刀: 0.88", font=font, fill=(0, 0, 255))
    
    # 将PIL图像转换回OpenCV图像
    img_with_chinese = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    # 显示图像
    cv2.imshow("PIL中文显示测试", img_with_chinese)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    print("中文显示测试")
    print("=" * 30)
    
    # 测试OpenCV原生中文显示
    test_cv2_chinese()
    
    # 测试PIL库中文显示
    test_pil_chinese()

if __name__ == "__main__":
    main()
