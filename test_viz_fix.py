"""
测试可视化修复效果
"""

import numpy as np
import cv2
from viz import plot_wire_detection_results, plot_stereo_matches, plot_3d_points, smart_show, is_headless

def create_test_data():
    """
    创建测试数据
    """
    # 创建测试图像
    imgL = np.zeros((480, 640), dtype=np.uint8)
    imgR = np.zeros((480, 640), dtype=np.uint8)
    
    # 在左图像中画线
    cv2.line(imgL, (100, 200), (500, 300), 255, 2)
    # 在右图像中画对应的线（有视差）
    cv2.line(imgR, (80, 200), (480, 300), 255, 2)
    
    # 创建导线像素
    wire_pixelsL = np.array([[200, 250], [300, 270], [400, 290], [150, 230], [350, 280]])
    wire_pixelsR = np.array([[170, 250], [270, 270], [370, 290], [120, 230], [320, 280]])
    
    # 创建匹配点对
    matches = [(wire_pixelsL[i], wire_pixelsR[i]) for i in range(len(wire_pixelsL))]
    
    # 创建3D点
    points_3d = np.random.randn(10, 3) * 10
    
    return imgL, imgR, wire_pixelsL, wire_pixelsR, matches, points_3d

def test_visualization_functions():
    """
    测试可视化函数
    """
    print("=" * 60)
    print("测试可视化修复效果")
    print("=" * 60)
    
    print(f"当前环境: {'无头环境' if is_headless() else '有显示环境'}")
    
    # 创建测试数据
    imgL, imgR, wire_pixelsL, wire_pixelsR, matches, points_3d = create_test_data()
    
    print("\n1. 测试导线检测可视化...")
    try:
        plot_wire_detection_results(imgL, imgR, wire_pixelsL, wire_pixelsR, "测试导线检测")
        print("✅ 导线检测可视化成功")
    except Exception as e:
        print(f"❌ 导线检测可视化失败: {e}")
    
    print("\n2. 测试双目匹配可视化...")
    try:
        plot_stereo_matches(imgL, imgR, matches, "测试双目匹配")
        print("✅ 双目匹配可视化成功")
    except Exception as e:
        print(f"❌ 双目匹配可视化失败: {e}")
    
    print("\n3. 测试3D点云可视化...")
    try:
        plot_3d_points(points_3d, title="测试3D点云")
        print("✅ 3D点云可视化成功")
    except Exception as e:
        print(f"❌ 3D点云可视化失败: {e}")
    
    print("\n4. 测试智能显示函数...")
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        plt.plot([1, 2, 3, 4], [1, 4, 2, 3], 'bo-')
        plt.title('测试智能显示')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        
        smart_show(title="test_smart_show")
        print("✅ 智能显示函数成功")
    except Exception as e:
        print(f"❌ 智能显示函数失败: {e}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
    
    if is_headless():
        print("在无头环境下，图表已自动保存到 output/ 目录")
        print("可以使用以下命令查看生成的图片：")
        print("ls -la output/*.png")
    else:
        print("在有显示环境下，图表应该已经显示在窗口中")

if __name__ == "__main__":
    test_visualization_functions()
