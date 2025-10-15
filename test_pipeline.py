"""
测试脚本：验证双目视觉悬链线重建流程的基本功能
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from main import run_stereo_catenary_pipeline, load_camera_config
import os


def create_synthetic_images():
    """
    创建合成的测试图像
    """
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
        cv2.line(imgL, (int(s[i]), int(y[i])), (int(s[i+1]), int(y[i+1])), 255, 2)
    
    # 添加一些噪声
    noise = np.random.randint(0, 50, (height, width), dtype=np.uint8)
    imgL = cv2.add(imgL, noise)
    
    # 创建右图像（有视差）
    imgR = np.zeros((height, width), dtype=np.uint8)
    s_right = s - 30  # 右图像中的导线位置（视差）
    
    for i in range(len(s_right) - 1):
        cv2.line(imgR, (int(s_right[i]), int(y[i])), (int(s_right[i+1]), int(y[i+1])), 255, 2)
    
    # 添加噪声
    noise = np.random.randint(0, 50, (height, width), dtype=np.uint8)
    imgR = cv2.add(imgR, noise)
    
    return imgL, imgR


def create_test_config():
    """
    创建测试用的相机配置
    """
    config = {
        'left_camera': {
            'intrinsic_matrix': np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float32),
            'rotation_matrix': np.eye(3, dtype=np.float32),
            'translation_vector': np.array([0, 0, 0], dtype=np.float32)
        },
        'right_camera': {
            'intrinsic_matrix': np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float32),
            'rotation_matrix': np.eye(3, dtype=np.float32),
            'translation_vector': np.array([-50, 0, 0], dtype=np.float32)  # 50mm基线
        },
        'gravity_direction': np.array([0, 0, -1], dtype=np.float32),
        'image_size': {'width': 640, 'height': 480}
    }
    return config


def test_individual_modules():
    """
    测试各个模块的功能
    """
    print("测试各个模块功能...")
    
    # 创建测试图像
    imgL, imgR = create_synthetic_images()
    
    # 测试导线检测
    from line_detect import detect_wire_pixels
    wire_pixelsL = detect_wire_pixels(imgL, method='canny_hough')
    wire_pixelsR = detect_wire_pixels(imgR, method='canny_hough')
    
    print(f"导线检测: 左图像 {len(wire_pixelsL)} 个像素, 右图像 {len(wire_pixelsR)} 个像素")
    
    # 测试双目匹配
    config = create_test_config()
    K1 = config['left_camera']['intrinsic_matrix']
    R1 = config['left_camera']['rotation_matrix']
    t1 = config['left_camera']['translation_vector']
    K2 = config['right_camera']['intrinsic_matrix']
    R2 = config['right_camera']['rotation_matrix']
    t2 = config['right_camera']['translation_vector']
    
    from stereo_match import match_along_epipolar
    matches = match_along_epipolar(wire_pixelsL, imgL, imgR, K1, R1, t1, K2, R2, t2)
    
    print(f"双目匹配: {len(matches)} 个匹配点对")
    
    # 测试三角测量
    from triangulation import triangulate_pairs
    P1 = K1 @ np.hstack([R1, t1.reshape(3, 1)])
    P2 = K2 @ np.hstack([R2, t2.reshape(3, 1)])
    
    points_3d = triangulate_pairs(matches, P1, P2, method='dlt')
    
    print(f"三角测量: {len(points_3d)} 个3D点")
    
    # 测试平面估计
    from plane_estimation import estimate_vertical_plane, project_to_plane
    g_hat = config['gravity_direction']
    p0, u, g_hat_est = estimate_vertical_plane(points_3d, g_hat)
    s_list, z_list = project_to_plane(points_3d, p0, u, g_hat_est)
    
    print(f"平面估计: 投影得到 {len(s_list)} 个2D点")
    
    # 测试悬链线拟合
    from catenary_fit import fit_catenary, compute_fitting_quality
    a, s0, c = fit_catenary(s_list, z_list, method='lm_robust')
    quality = compute_fitting_quality(s_list, z_list, a, s0, c)
    
    print(f"悬链线拟合: a={a:.3f}, s0={s0:.3f}, c={c:.3f}")
    print(f"拟合质量: R²={quality['r_squared']:.3f}, RMSE={quality['rmse']:.3f}")
    
    return True


def test_full_pipeline():
    """
    测试完整的处理流程
    """
    print("\n测试完整处理流程...")
    
    # 创建测试数据
    imgL, imgR = create_synthetic_images()
    config = create_test_config()
    
    try:
        # 运行完整流程
        results = run_stereo_catenary_pipeline(imgL, imgR, config, verbose=True)
        
        # 检查结果
        assert len(results['wire_pixelsL']) > 0, "未检测到导线像素"
        assert len(results['matches']) > 0, "未找到匹配点对"
        assert len(results['points_3d']) > 0, "三角测量失败"
        assert len(results['s_list']) > 0, "平面投影失败"
        
        params = results['catenary_params']
        quality = results['fitting_quality']
        
        print(f"\n最终结果:")
        print(f"悬链线参数: a={params['a']:.3f}, s0={params['s0']:.3f}, c={params['c']:.3f}")
        print(f"拟合质量: R²={quality['r_squared']:.3f}, RMSE={quality['rmse']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"完整流程测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def visualize_test_results():
    """
    可视化测试结果
    """
    print("\n生成测试结果可视化...")
    
    # 创建测试数据
    imgL, imgR = create_synthetic_images()
    config = create_test_config()
    
    # 运行流程
    results = run_stereo_catenary_pipeline(imgL, imgR, config, verbose=False)
    
    # 可视化
    from viz import plot_pipeline_summary
    
    plot_pipeline_summary(
        imgL, imgR,
        results['wire_pixelsL'], results['wire_pixelsR'],
        results['matches'],
        results['points_3d'],
        results['s_list'], results['z_list'],
        results['catenary_params']['a'],
        results['catenary_params']['s0'],
        results['catenary_params']['c'],
        results['fitting_quality']
    )
    
    plt.show()


def main():
    """
    主测试函数
    """
    print("=" * 60)
    print("双目视觉悬链线重建项目测试")
    print("=" * 60)
    
    # 测试各个模块
    success1 = test_individual_modules()
    
    # 测试完整流程
    success2 = test_full_pipeline()
    
    # 可视化结果
    if success1 and success2:
        print("\n所有测试通过！")
        try:
            visualize_test_results()
        except Exception as e:
            print(f"可视化失败: {e}")
    else:
        print("\n测试失败！")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
