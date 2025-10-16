"""
双目视觉悬链线重建主程序
给定双目相机采集的两张图和相机内外参，计算导线的空间悬链线方程
"""

import numpy as np
import cv2
import yaml
import argparse
import os
from typing import Tuple, Dict, Any
import warnings

# 导入自定义模块
from line_detect import detect_wire_pixels, filter_wire_pixels
from stereo_match import match_along_epipolar, filter_matches_by_sampson_error, filter_matches_by_right_uniqueness
from triangulation import triangulate_pairs, compute_reprojection_error, filter_outliers_by_reprojection
from plane_estimation import estimate_vertical_plane, project_to_plane, compute_plane_quality
from catenary_fit import fit_catenary, compose_catenary_3d, compute_fitting_quality, ransac_catenary_fit
from viz import (plot_wire_detection_results, plot_stereo_matches, plot_3d_points, 
                plot_plane_projection, plot_catenary_fitting, plot_3d_catenary_curve, 
                plot_pipeline_summary, plot_pipeline_summary_from_results)


def load_camera_config(config_path: str) -> Dict[str, Any]:
    """
    加载相机配置文件
    
    参数:
        config_path: 配置文件路径
        
    返回:
        config: 相机配置字典
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


def load_images(left_path: str, right_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载左右图像
    
    参数:
        left_path: 左图像路径
        right_path: 右图像路径
        
    返回:
        imgL: 左图像
        imgR: 右图像
    """
    imgL = cv2.imread(left_path)
    imgR = cv2.imread(right_path)
    
    if imgL is None:
        raise FileNotFoundError(f"无法加载左图像: {left_path}")
    if imgR is None:
        raise FileNotFoundError(f"无法加载右图像: {right_path}")
    
    return imgL, imgR


def run_stereo_catenary_pipeline(imgL: np.ndarray, 
                                imgR: np.ndarray,
                                config: Dict[str, Any],
                                verbose: bool = True,
                                detect_method: str = 'canny_hough') -> Dict[str, Any]:
    """
    运行完整的双目视觉悬链线重建流程
    
    参数:
        imgL: 左图像
        imgR: 右图像
        config: 相机配置
        verbose: 是否显示详细信息
        
    返回:
        results: 包含所有结果的字典
    """
    results = {}
    
    if verbose:
        print("=" * 60)
        print("双目视觉悬链线重建流程开始")
        print("=" * 60)
    
    # 1. 导线像素检测
    if verbose:
        print("\n1. 导线像素检测...")
    
    wire_pixelsL = detect_wire_pixels(imgL, method=detect_method)
    wire_pixelsR = detect_wire_pixels(imgR, method=detect_method)
    
    # 过滤噪声像素
    wire_pixelsL = filter_wire_pixels(wire_pixelsL, imgL.shape[:2])
    wire_pixelsR = filter_wire_pixels(wire_pixelsR, imgR.shape[:2])
    
    results['wire_pixelsL'] = wire_pixelsL
    results['wire_pixelsR'] = wire_pixelsR
    
    if verbose:
        print(f"   左图像检测到 {len(wire_pixelsL)} 个导线像素")
        print(f"   右图像检测到 {len(wire_pixelsR)} 个导线像素")
    
    if len(wire_pixelsL) == 0 or len(wire_pixelsR) == 0:
        raise ValueError("未检测到足够的导线像素")
    
    # 2. 双目匹配
    if verbose:
        print("\n2. 双目匹配...")
    
    # 提取相机参数
    K1 = config['left_camera']['intrinsic_matrix']
    R1 = config['left_camera']['rotation_matrix']
    t1 = config['left_camera']['translation_vector']
    K2 = config['right_camera']['intrinsic_matrix']
    R2 = config['right_camera']['rotation_matrix']
    t2 = config['right_camera']['translation_vector']
    
    # 进行匹配
    matches = match_along_epipolar(wire_pixelsL, imgL, imgR, K1, R1, t1, K2, R2, t2)
    
    if verbose:
        print(f"   找到 {len(matches)} 个匹配点对")
    
    if len(matches) < 3:
        raise ValueError("匹配点对数量不足，无法进行三角测量")
    
    # 使用Sampson误差与右图唯一性过滤匹配
    from stereo_match import compute_fundamental_matrix
    F = compute_fundamental_matrix(K1, R1, t1, K2, R2, t2)
    matches = filter_matches_by_sampson_error(matches, F, threshold=config.get('sampson_threshold', 1.2))
    matches = filter_matches_by_right_uniqueness(matches, F, bin_tolerance_px=config.get('right_uniqueness_bin', 2.0))
    
    results['matches'] = matches
    
    if verbose:
        print(f"   过滤后剩余 {len(matches)} 个匹配点对")
    
    # 3. 三角测量
    if verbose:
        print("\n3. 三角测量...")
    
    # 构建投影矩阵并三角测量
    P1 = K1 @ np.hstack([R1, t1.reshape(3, 1)])
    P2 = K2 @ np.hstack([R2, t2.reshape(3, 1)])
    points_3d = triangulate_pairs(matches, P1, P2, method='dlt')
    
    # 过滤重投影误差过大的点
    points_3d, matches = filter_outliers_by_reprojection(points_3d, matches, P1, P2, threshold=3.0)
    
    results['points_3d'] = points_3d
    results['filtered_matches'] = matches
    
    if verbose:
        print(f"   三角测量得到 {len(points_3d)} 个3D点")
        
        # 计算重投影误差
        reproj_error = compute_reprojection_error(points_3d, matches, P1, P2)
        print(f"   平均重投影误差: {reproj_error:.3f} 像素")
    
    if len(points_3d) < 3:
        raise ValueError("3D点数量不足，无法估计平面")
    
    # 4. 垂直平面估计
    if verbose:
        print("\n4. 垂直平面估计...")
    
    g_hat = config['gravity_direction']
    p0, u, g_hat_est = estimate_vertical_plane(points_3d, g_hat)
    
    # 计算平面质量
    plane_quality = compute_plane_quality(points_3d, p0, u, g_hat_est)
    
    results['plane_params'] = {'p0': p0, 'u': u, 'g_hat': g_hat_est}
    results['plane_quality'] = plane_quality
    
    if verbose:
        print(f"   平面估计质量: {plane_quality:.3f}")
        print(f"   参考点: ({p0[0]:.2f}, {p0[1]:.2f}, {p0[2]:.2f})")
        print(f"   导线方向: ({u[0]:.2f}, {u[1]:.2f}, {u[2]:.2f})")
    
    # 自检：点到 n_perp 的平均距离与相机基线
    try:
        n_perp = np.cross(u, g_hat_est)
        n_perp = n_perp / (np.linalg.norm(n_perp) + 1e-12)
        distances = np.abs((points_3d - p0.reshape(1, 3)) @ n_perp)
        mean_dist = float(np.mean(distances))
        std_dist = float(np.std(distances))
        # 基线长度（由 t 推回相机中心）
        C1 = -R1.T @ t1
        C2 = -R2.T @ t2
        baseline = float(np.linalg.norm(C1 - C2))
        if verbose:
            print(f"   平面一致性自检: mean|n_perp·(X-p0)| = {mean_dist:.4f} (std={std_dist:.4f})")
            print(f"   相机中心: C1={C1}, C2={C2}, 基线长度={baseline:.4f}")
    except Exception as _:
        pass

    # 若偏差过大，使用点云自适应平面替代以保证可视化/拟合在同一平面
    adapt_threshold = float(config.get('plane_adapt_threshold', 0.05))
    if 'mean_dist' in locals() and mean_dist > adapt_threshold:
        if verbose:
            print(f"   警告: 点云偏离估计平面较大(>{adapt_threshold}), 使用点云自适应平面进行后续投影/拟合")
        # 使用 PCA 的第一主方向作为导线方向，法向为 n_perp，自适应得到新的 (p0,u,g_hat')
        # 令 g_hat' 为原 g_hat 在平面内的分量归一化
        # 重新计算 p0 为点云投到自适应平面的质心
        # 这里直接复用 estimate_vertical_plane 的结果作为基准，仅修改 g_hat 为其在平面内分量
        g_proj = g_hat_est - np.dot(g_hat_est, n_perp) * n_perp
        if np.linalg.norm(g_proj) > 1e-12:
            g_hat_adapt = g_proj / np.linalg.norm(g_proj)
            g_hat_use = g_hat_adapt
        else:
            g_hat_use = g_hat_est
        p0_use, u_use, g_hat_use = p0, u, g_hat_use
    else:
        p0_use, u_use, g_hat_use = p0, u, g_hat_est

    # 5. 平面投影
    if verbose:
        print("\n5. 平面投影...")
    
    s_list, z_list = project_to_plane(points_3d, p0_use, u_use, g_hat_use)
    # 投影后按 s 分桶剔除“竖直列”：若同一 s 桶内 z 方差过大则删除
    s_list, z_list = _remove_vertical_stripes(s_list, z_list)
    
    results['s_list'] = s_list
    results['z_list'] = z_list
    results['plane_params'] = {'p0': p0_use, 'u': u_use, 'g_hat': g_hat_use}
    
    if verbose:
        print(f"   投影得到 {len(s_list)} 个2D点")
        print(f"   s范围: [{np.min(s_list):.2f}, {np.max(s_list):.2f}]")
        print(f"   z范围: [{np.min(z_list):.2f}, {np.max(z_list):.2f}]")
    
    # 6. 悬链线拟合
    if verbose:
        print("\n6. 悬链线拟合...")
    
    # 使用RANSAC进行鲁棒拟合
    a, s0, c, inliers = ransac_catenary_fit(s_list, z_list, 
                                           min_samples=3, 
                                           max_trials=100, 
                                           residual_threshold=0.5)
    
    # 使用内点进行最终拟合
    if np.sum(inliers) >= 3:
        s_inliers = s_list[inliers]
        z_inliers = z_list[inliers]
        a, s0, c = fit_catenary(s_inliers, z_inliers, method='lm_robust')
        
        # 计算拟合质量
        quality = compute_fitting_quality(s_inliers, z_inliers, a, s0, c)
    else:
        # 如果RANSAC失败，使用所有点
        a, s0, c = fit_catenary(s_list, z_list, method='lm_robust')
        quality = compute_fitting_quality(s_list, z_list, a, s0, c)
        inliers = np.ones(len(s_list), dtype=bool)
    
    results['catenary_params'] = {'a': a, 's0': s0, 'c': c}
    results['fitting_quality'] = quality
    results['inliers'] = inliers
    
    if verbose:
        print(f"   悬链线参数:")
        print(f"     a = {a:.3f}")
        print(f"     s₀ = {s0:.3f}")
        print(f"     c = {c:.3f}")
        print(f"   拟合质量:")
        print(f"     R² = {quality['r_squared']:.3f}")
        print(f"     RMSE = {quality['rmse']:.3f}")
        print(f"     MAE = {quality['mae']:.3f}")
        print(f"   内点数量: {np.sum(inliers)}/{len(s_list)}")
    
    # 7. 构建3D悬链线函数
    catenary_3d = compose_catenary_3d(p0, u, g_hat_est, a, s0, c)
    results['catenary_3d'] = catenary_3d
    
    if verbose:
        print("\n" + "=" * 60)
        print("处理完成！")
        print("=" * 60)
    
    return results


def _remove_vertical_stripes(s_list: np.ndarray, z_list: np.ndarray,
                             bin_width: float = 10.0,
                             var_threshold: float = 200.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    按 s 分桶，删除同一桶内 z 方差异常大的列，抑制“竖直散点线”。
    bin_width: s 方向分桶宽度（与数据尺度相关，可根据场景调整）
    var_threshold: z 方差阈值（像素或物理单位平方）
    """
    if len(s_list) == 0:
        return s_list, z_list
    # 构建桶
    s0 = np.min(s_list)
    keys = ((s_list - s0) / max(1e-6, bin_width)).astype(int)
    keep_mask = np.ones_like(s_list, dtype=bool)
    for k in np.unique(keys):
        idx = (keys == k)
        if np.sum(idx) < 3:
            continue
        varz = np.var(z_list[idx])
        if varz > var_threshold:
            keep_mask[idx] = False
    return s_list[keep_mask], z_list[keep_mask]


def save_results(results: Dict[str, Any], output_dir: str) -> None:
    """
    保存处理结果
    
    参数:
        results: 处理结果字典
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存悬链线参数
    params = results['catenary_params']
    quality = results['fitting_quality']
    
    with open(os.path.join(output_dir, 'catenary_params.txt'), 'w', encoding='utf-8') as f:
        f.write("悬链线参数:\n")
        f.write(f"a = {params['a']:.6f}\n")
        f.write(f"s0 = {params['s0']:.6f}\n")
        f.write(f"c = {params['c']:.6f}\n\n")
        f.write("拟合质量:\n")
        f.write(f"R² = {quality['r_squared']:.6f}\n")
        f.write(f"RMSE = {quality['rmse']:.6f}\n")
        f.write(f"MAE = {quality['mae']:.6f}\n")
        f.write(f"内点数量 = {np.sum(results['inliers'])}\n")
        f.write(f"总点数 = {len(results['s_list'])}\n")
    
    # 保存3D点
    np.savetxt(os.path.join(output_dir, 'points_3d.txt'), results['points_3d'], 
               header='X Y Z', fmt='%.6f')
    
    # 保存2D投影点
    projection_data = np.column_stack([results['s_list'], results['z_list']])
    np.savetxt(os.path.join(output_dir, 'projection_2d.txt'), projection_data, 
               header='s z', fmt='%.6f')
    
    print(f"结果已保存到: {output_dir}")


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='双目视觉悬链线重建')
    parser.add_argument('--left', type=str, default='./demo/simulation/Outputs/left.png', help='左图像路径')
    parser.add_argument('--right', type=str, default='./demo/simulation/Outputs/right.png', help='右图像路径')
    parser.add_argument('--config', type=str, default='./config/simulation.yaml', help='相机配置文件路径')
    parser.add_argument('--output', type=str, default='output', help='输出目录')
    parser.add_argument('--detect-method', type=str, default='catenary', choices=['canny_hough', 'ridge', 'skeleton', 'catenary'], help='导线像素检测方法')
    parser.add_argument('--verbose', action='store_true', default=True, help='显示详细信息')
    parser.add_argument('--sampson-th', type=float, default=1.2, help='Sampson误差阈值')
    parser.add_argument('--uniq-bin', type=float, default=2.0, help='右图唯一性分桶像素宽度')
    
    args = parser.parse_args()
    
    try:
        # 加载配置
        print("加载相机配置...")
        config = load_camera_config(args.config)
        
        # 加载图像
        print("加载图像...")
        imgL, imgR = load_images(args.left, args.right)
        print(f"左图像尺寸: {imgL.shape}")
        print(f"右图像尺寸: {imgR.shape}")
        
        # 运行处理流程（将阈值注入配置）
        config['sampson_threshold'] = args.sampson_th
        config['right_uniqueness_bin'] = args.uniq_bin
        results = run_stereo_catenary_pipeline(imgL, imgR, config, verbose=args.verbose, detect_method=args.detect_method)
        
        # 保存结果
        save_results(results, args.output)
        
        # 生成并保存可视化结果
        print("\n生成可视化图表并保存...")
        plot_pipeline_summary_from_results(imgL, imgR, results)
        import matplotlib.pyplot as plt
        os.makedirs(args.output, exist_ok=True)
        plt.savefig(os.path.join(args.output, 'pipeline_summary.png'), 
                   dpi=300, bbox_inches='tight')
        print(f"可视化图表已保存到: {os.path.join(args.output, 'pipeline_summary.png')}")
        
        # 输出最终结果
        params = results['catenary_params']
        print(f"\n最终悬链线方程:")
        print(f"X(s) = p0 + s*u + [a*cosh((s-s0)/a) + c]*g_hat")
        print(f"其中:")
        print(f"  p0 = ({results['plane_params']['p0'][0]:.3f}, {results['plane_params']['p0'][1]:.3f}, {results['plane_params']['p0'][2]:.3f})")
        print(f"  u = ({results['plane_params']['u'][0]:.3f}, {results['plane_params']['u'][1]:.3f}, {results['plane_params']['u'][2]:.3f})")
        print(f"  g_hat = ({results['plane_params']['g_hat'][0]:.3f}, {results['plane_params']['g_hat'][1]:.3f}, {results['plane_params']['g_hat'][2]:.3f})")
        print(f"  a = {params['a']:.6f}")
        print(f"  s0 = {params['s0']:.6f}")
        print(f"  c = {params['c']:.6f}")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
