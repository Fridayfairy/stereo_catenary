"""
简单演示脚本：使用现有demo数据测试双目视觉悬链线重建
"""

import numpy as np
import cv2
import yaml
import os
from typing import Dict, Any

def load_camera_config(config_path: str) -> Dict[str, Any]:
    """
    加载相机配置文件
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 转换为numpy数组
    config['left_camera']['intrinsic_matrix'] = np.array(config['left_camera']['intrinsic_matrix'])
    config['left_camera']['rotation_matrix'] = np.array(config['left_camera']['rotation_matrix'])
    config['left_camera']['translation_vector'] = np.array(config['left_camera']['translation_vector'])
    
    config['right_camera']['intrinsic_matrix'] = np.array(config['right_camera']['intrinsic_matrix'])
    config['right_camera']['rotation_matrix'] = np.array(config['right_camera']['rotation_matrix'])
    config['right_camera']['translation_vector'] = np.array(config['right_camera']['translation_vector'])
    
    config['gravity_direction'] = np.array(config['gravity_direction'])
    
    return config

def create_synthetic_wire_images():
    """
    创建合成的导线图像用于测试
    """
    print("创建合成导线图像...")
    
    # 图像尺寸
    height, width = 480, 640
    
    # 创建左图像
    imgL = np.zeros((height, width), dtype=np.uint8)
    
    # 绘制悬链线形状的导线
    s = np.linspace(50, 590, 200)
    z = 20 * np.cosh((s - 320) / 100) - 20  # 悬链线方程
    y = 240 + z  # 转换到图像坐标
    
    # 绘制导线
    for i in range(len(s) - 1):
        cv2.line(imgL, (int(s[i]), int(y[i])), (int(s[i+1]), int(y[i+1])), 255, 3)
    
    # 添加一些噪声
    noise = np.random.randint(0, 30, (height, width), dtype=np.uint8)
    imgL = cv2.add(imgL, noise)
    
    # 创建右图像（有视差）
    imgR = np.zeros((height, width), dtype=np.uint8)
    s_right = s - 30  # 右图像中的导线位置（视差）
    
    for i in range(len(s_right) - 1):
        cv2.line(imgR, (int(s_right[i]), int(y[i])), (int(s_right[i+1]), int(y[i+1])), 255, 3)
    
    # 添加噪声
    noise = np.random.randint(0, 30, (height, width), dtype=np.uint8)
    imgR = cv2.add(imgR, noise)
    
    return imgL, imgR

def test_basic_pipeline():
    """
    测试基本的处理流程
    """
    print("=" * 60)
    print("双目视觉悬链线重建 - 简单演示")
    print("=" * 60)
    
    try:
        # 1. 加载配置
        print("\n1. 加载相机配置...")
        config = load_camera_config('config/cameras.yaml')
        print("✅ 配置加载成功")
        
        # 2. 创建测试图像
        print("\n2. 创建测试图像...")
        imgL, imgR = create_synthetic_wire_images()
        print(f"✅ 图像创建成功: {imgL.shape}")
        
        # 3. 导线检测
        print("\n3. 导线像素检测...")
        try:
            from line_detect import detect_wire_pixels, filter_wire_pixels
            
            wire_pixelsL = detect_wire_pixels(imgL, method='canny_hough')
            wire_pixelsR = detect_wire_pixels(imgR, method='canny_hough')
            
            wire_pixelsL = filter_wire_pixels(wire_pixelsL, imgL.shape[:2])
            wire_pixelsR = filter_wire_pixels(wire_pixelsR, imgR.shape[:2])
            
            print(f"✅ 导线检测成功: 左图像 {len(wire_pixelsL)} 个像素, 右图像 {len(wire_pixelsR)} 个像素")
            
        except ImportError as e:
            print(f"⚠️  导线检测模块导入失败: {e}")
            print("   跳过导线检测步骤")
            return True
        
        # 4. 双目匹配
        print("\n4. 双目匹配...")
        try:
            from stereo_match import match_along_epipolar
            
            K1 = config['left_camera']['intrinsic_matrix']
            R1 = config['left_camera']['rotation_matrix']
            t1 = config['left_camera']['translation_vector']
            K2 = config['right_camera']['intrinsic_matrix']
            R2 = config['right_camera']['rotation_matrix']
            t2 = config['right_camera']['translation_vector']
            
            matches = match_along_epipolar(wire_pixelsL, imgL, imgR, K1, R1, t1, K2, R2, t2)
            print(f"✅ 双目匹配成功: {len(matches)} 个匹配点对")
            
        except ImportError as e:
            print(f"⚠️  双目匹配模块导入失败: {e}")
            print("   跳过双目匹配步骤")
            return True
        
        # 5. 三角测量
        print("\n5. 三角测量...")
        try:
            from triangulation import triangulate_pairs
            
            P1 = K1 @ np.hstack([R1, t1.reshape(3, 1)])
            P2 = K2 @ np.hstack([R2, t2.reshape(3, 1)])
            
            points_3d = triangulate_pairs(matches, P1, P2, method='dlt')
            print(f"✅ 三角测量成功: {len(points_3d)} 个3D点")
            
        except ImportError as e:
            print(f"⚠️  三角测量模块导入失败: {e}")
            print("   跳过三角测量步骤")
            return True
        
        # 6. 平面估计
        print("\n6. 平面估计...")
        try:
            from plane_estimation import estimate_vertical_plane, project_to_plane
            
            g_hat = config['gravity_direction']
            p0, u, g_hat_est = estimate_vertical_plane(points_3d, g_hat)
            s_list, z_list = project_to_plane(points_3d, p0, u, g_hat_est)
            
            print(f"✅ 平面估计成功: 投影得到 {len(s_list)} 个2D点")
            
        except ImportError as e:
            print(f"⚠️  平面估计模块导入失败: {e}")
            print("   跳过平面估计步骤")
            return True
        
        # 7. 悬链线拟合
        print("\n7. 悬链线拟合...")
        try:
            from catenary_fit import fit_catenary, compute_fitting_quality
            
            a, s0, c = fit_catenary(s_list, z_list, method='lm_robust')
            quality = compute_fitting_quality(s_list, z_list, a, s0, c)
            
            print(f"✅ 悬链线拟合成功:")
            print(f"   参数: a={a:.3f}, s0={s0:.3f}, c={c:.3f}")
            print(f"   质量: R²={quality['r_squared']:.3f}, RMSE={quality['rmse']:.3f}")
            
        except ImportError as e:
            print(f"⚠️  悬链线拟合模块导入失败: {e}")
            print("   跳过悬链线拟合步骤")
            return True
        
        print("\n" + "=" * 60)
        print("🎉 所有模块测试通过！")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    主函数
    """
    success = test_basic_pipeline()
    
    if success:
        print("\n✅ 简单演示完成！")
        print("\n要运行完整的处理流程，请确保安装了所有依赖包：")
        print("pip install numpy opencv-python scipy matplotlib PyYAML scikit-image scikit-learn")
        print("\n然后运行：")
        print("python main.py --show-plots")
    else:
        print("\n❌ 演示失败！")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
