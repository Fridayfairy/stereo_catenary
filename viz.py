"""
可视化模块
提供2D和3D可视化功能，包括导线检测、匹配、三角测量和悬链线拟合结果
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from typing import List, Tuple, Optional, Callable
import matplotlib.patches as patches
import os

# 检查是否在无头环境中运行
def is_headless():
    """检查是否在无头环境中运行"""
    return os.environ.get('DISPLAY') is None or 'headless' in os.environ.get('MATPLOTLIB_BACKEND', '').lower()

# 设置合适的后端
if is_headless():
    matplotlib.use('Agg')
    print("检测到无头环境，使用Agg后端")
else:
    try:
        matplotlib.use('TkAgg')
        print("使用TkAgg后端")
    except:
        matplotlib.use('Agg')
        print("TkAgg不可用，使用Agg后端")

# 设置中文字体支持
def setup_chinese_font():
    """设置中文字体支持"""
    try:
        # 尝试使用系统中文字体
        import matplotlib.font_manager as fm
        
        # 常见的中文字体列表
        chinese_fonts = [
            'AR PL UKai CN',    # 文鼎PL中楷
            'AR PL UMing CN',   # 文鼎PL明体
            'Noto Sans CJK JP', # Google Noto字体（日文版支持中文）
            'Noto Serif CJK JP', # Google Noto字体（日文版支持中文）
            'SimHei',           # 黑体
            'Microsoft YaHei',  # 微软雅黑
            'WenQuanYi Micro Hei',  # 文泉驿微米黑
            'DejaVu Sans',      # 默认字体
            'Arial Unicode MS', # Arial Unicode
            'Noto Sans CJK SC', # Google Noto字体
            'Source Han Sans SC' # 思源黑体
        ]
        
        # 查找可用的中文字体
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        for font_name in chinese_fonts:
            if font_name in available_fonts:
                plt.rcParams['font.sans-serif'] = [font_name]
                plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
                print(f"使用中文字体: {font_name}")
                return True
        
        # 如果没有找到中文字体，使用默认字体但禁用警告
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        print("未找到中文字体，使用默认字体")
        return False
        
    except Exception as e:
        print(f"字体设置失败: {e}")
        return False

# 初始化中文字体
setup_chinese_font()

def smart_show(fig=None, filename=None, title="plot"):
    """
    智能显示函数：在无头环境下自动保存图片，在有显示环境下显示窗口
    """
    if fig is None:
        fig = plt.gcf()
    
    if is_headless():
        if filename is None:
            filename = f"output/{title.replace(' ', '_')}.png"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {filename}")
    else:
        smart_show(title="wire_detection")


def plot_wire_detection_results(imgL: np.ndarray, 
                               imgR: np.ndarray,
                               wire_pixelsL: np.ndarray,
                               wire_pixelsR: np.ndarray,
                               title: str = "导线检测结果") -> None:
    """
    可视化导线检测结果
    
    参数:
        imgL: 左图像
        imgR: 右图像
        wire_pixelsL: 左图像中的导线像素
        wire_pixelsR: 右图像中的导线像素
        title: 图像标题
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 左图像
    if len(imgL.shape) == 3:
        imgL_display = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
        axes[0].imshow(imgL_display)
    else:
        axes[0].imshow(imgL, cmap='gray')
    
    if len(wire_pixelsL) > 0:
        axes[0].scatter(wire_pixelsL[:, 0], wire_pixelsL[:, 1], 
                       c='red', s=1, alpha=0.8, label=f'导线像素 ({len(wire_pixelsL)})')
    axes[0].set_title('左图像导线检测')
    axes[0].axis('off')
    axes[0].legend()
    
    # 右图像
    if len(imgR.shape) == 3:
        imgR_display = cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB)
        axes[1].imshow(imgR_display)
    else:
        axes[1].imshow(imgR, cmap='gray')
    
    if len(wire_pixelsR) > 0:
        axes[1].scatter(wire_pixelsR[:, 0], wire_pixelsR[:, 1], 
                       c='red', s=1, alpha=0.8, label=f'导线像素 ({len(wire_pixelsR)})')
    axes[1].set_title('右图像导线检测')
    axes[1].axis('off')
    axes[1].legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    smart_show(title="wire_detection")


def plot_stereo_matches(imgL: np.ndarray, 
                       imgR: np.ndarray,
                       matches: List[Tuple[np.ndarray, np.ndarray]],
                       title: str = "双目匹配结果",
                       max_matches: int = 50) -> None:
    """
    可视化双目匹配结果
    
    参数:
        imgL: 左图像
        imgR: 右图像
        matches: 匹配点对列表
        title: 图像标题
        max_matches: 最大显示匹配数
    """
    fig, ax = plt.subplots(1, 1, figsize=(15, 6))
    
    h1, w1 = imgL.shape[:2]
    h2, w2 = imgR.shape[:2]
    
    # 创建拼接图像
    combined_img = np.zeros((max(h1, h2), w1 + w2), dtype=np.uint8)
    
    # 处理彩色图像
    if len(imgL.shape) == 3:
        imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    else:
        imgL_gray = imgL
    
    if len(imgR.shape) == 3:
        imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    else:
        imgR_gray = imgR
    
    combined_img[:h1, :w1] = imgL_gray
    combined_img[:h2, w1:] = imgR_gray
    
    ax.imshow(combined_img, cmap='gray')
    
    # 绘制匹配线
    display_matches = matches[:max_matches] if len(matches) > max_matches else matches
    
    for i, (ptL, ptR) in enumerate(display_matches):
        color = plt.cm.tab10(i % 10)
        
        # 绘制点
        ax.scatter(ptL[0], ptL[1], c=[color], s=30, marker='o')
        ax.scatter(ptR[0] + w1, ptR[1], c=[color], s=30, marker='o')
        
        # 绘制连线
        ax.plot([ptL[0], ptR[0] + w1], [ptL[1], ptR[1]], 
               color=color, linewidth=1, alpha=0.7)
    
    ax.set_title(f'{title} (显示{len(display_matches)}/{len(matches)}个匹配)')
    ax.axis('off')
    
    plt.tight_layout()
    smart_show(title="wire_detection")


def plot_3d_points(points_3d: np.ndarray,
                  camera_centers: Optional[List[np.ndarray]] = None,
                  title: str = "3D点云",
                  show_cameras: bool = True) -> None:
    """
    可视化3D点云
    
    参数:
        points_3d: 3D点坐标数组 (N, 3)
        camera_centers: 相机中心位置列表
        title: 图像标题
        show_cameras: 是否显示相机位置
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    if len(points_3d) > 0:
        # 绘制3D点
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
                  c='blue', s=50, alpha=0.7, label=f'3D点 ({len(points_3d)})')
        
        # 绘制点云边界框
        if len(points_3d) > 1:
            min_coords = np.min(points_3d, axis=0)
            max_coords = np.max(points_3d, axis=0)
            
            # 绘制边界框
            for i in range(3):
                for j in range(3):
                    if i != j:
                        for k in [0, 1]:
                            if k == 0:
                                coords = min_coords.copy()
                            else:
                                coords = max_coords.copy()
                            
                            # 创建边界线
                            line_coords = np.zeros((2, 3))
                            line_coords[0] = coords
                            line_coords[1] = coords
                            line_coords[1, i] = max_coords[i] if k == 0 else min_coords[i]
                            
                            ax.plot(line_coords[:, 0], line_coords[:, 1], line_coords[:, 2], 
                                   'k-', alpha=0.3, linewidth=0.5)
    
    # 绘制相机位置
    if show_cameras and camera_centers is not None:
        for i, center in enumerate(camera_centers):
            ax.scatter(center[0], center[1], center[2], 
                      c='red', s=200, marker='^', label=f'相机{i+1}')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title(title)
    
    # 设置相等的坐标轴比例
    if len(points_3d) > 0:
        max_range = np.array([points_3d[:, 0].max() - points_3d[:, 0].min(),
                             points_3d[:, 1].max() - points_3d[:, 1].min(),
                             points_3d[:, 2].max() - points_3d[:, 2].min()]).max() / 2.0
        
        mid_x = (points_3d[:, 0].max() + points_3d[:, 0].min()) * 0.5
        mid_y = (points_3d[:, 1].max() + points_3d[:, 1].min()) * 0.5
        mid_z = (points_3d[:, 2].max() + points_3d[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    smart_show(title="wire_detection")


def plot_plane_projection(s_list: np.ndarray, 
                         z_list: np.ndarray,
                         p0: np.ndarray,
                         u: np.ndarray,
                         g_hat: np.ndarray,
                         title: str = "平面投影结果") -> None:
    """
    可视化平面投影结果
    
    参数:
        s_list: 沿导线方向的坐标
        z_list: 重力方向的坐标
        p0: 平面参考点
        u: 导线方向向量
        g_hat: 重力方向向量
        title: 图像标题
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 2D投影图
    ax1.scatter(s_list, z_list, c='blue', alpha=0.7, s=30, label='投影点')
    ax1.set_xlabel('s (沿导线方向)')
    ax1.set_ylabel('z (重力方向)')
    ax1.set_title('2D平面投影')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 添加坐标轴方向指示
    if len(s_list) > 0:
        s_range = np.max(s_list) - np.min(s_list)
        z_range = np.max(z_list) - np.min(z_list)
        
        # 绘制方向箭头
        s_center = np.mean(s_list)
        z_center = np.mean(z_list)
        
        arrow_length = min(s_range, z_range) * 0.1
        ax1.arrow(s_center, z_center, arrow_length, 0, 
                 head_width=arrow_length*0.1, head_length=arrow_length*0.1, 
                 fc='red', ec='red', label='导线方向')
        ax1.arrow(s_center, z_center, 0, arrow_length, 
                 head_width=arrow_length*0.1, head_length=arrow_length*0.1, 
                 fc='green', ec='green', label='重力方向')
    
    # 3D可视化（显示平面）
    ax2 = fig.add_subplot(122, projection='3d')
    
    if len(s_list) > 0:
        # 将2D点转换回3D
        points_3d = p0.reshape(1, 3) + s_list.reshape(-1, 1) * u.reshape(1, 3) + z_list.reshape(-1, 1) * g_hat.reshape(1, 3)
        
        # 绘制3D点
        ax2.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
                   c='blue', s=50, alpha=0.7, label='3D点')
        
        # 绘制平面
        s_range = np.linspace(np.min(s_list), np.max(s_list), 20)
        z_range = np.linspace(np.min(z_list), np.max(z_list), 20)
        S, Z = np.meshgrid(s_range, z_range)
        
        X_plane = p0[0] + S * u[0] + Z * g_hat[0]
        Y_plane = p0[1] + S * u[1] + Z * g_hat[1]
        Z_plane = p0[2] + S * u[2] + Z * g_hat[2]
        
        ax2.plot_surface(X_plane, Y_plane, Z_plane, alpha=0.3, color='cyan')
        
        # 绘制方向向量
        scale = np.max(np.linalg.norm(points_3d - p0, axis=1)) * 0.2
        ax2.quiver(p0[0], p0[1], p0[2], 
                  u[0] * scale, u[1] * scale, u[2] * scale, 
                  color='red', arrow_length_ratio=0.1, label='导线方向')
        ax2.quiver(p0[0], p0[1], p0[2], 
                  g_hat[0] * scale, g_hat[1] * scale, g_hat[2] * scale, 
                  color='green', arrow_length_ratio=0.1, label='重力方向')
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend()
    ax2.set_title('3D平面可视化')
    
    plt.suptitle(title)
    plt.tight_layout()
    smart_show(title="wire_detection")


def plot_catenary_fitting(s: np.ndarray, 
                         z: np.ndarray,
                         a: float, 
                         s0: float, 
                         c: float,
                         quality: Optional[dict] = None,
                         title: str = "悬链线拟合结果") -> None:
    """
    可视化悬链线拟合结果
    
    参数:
        s: 输入s坐标
        z: 输入z坐标
        a, s0, c: 拟合参数
        quality: 拟合质量指标
        title: 图像标题
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 拟合结果图
    ax1.scatter(s, z, c='blue', alpha=0.7, s=30, label='数据点')
    
    # 绘制拟合曲线
    s_plot = np.linspace(np.min(s), np.max(s), 100)
    z_plot = a * np.cosh((s_plot - s0) / a) + c
    ax1.plot(s_plot, z_plot, 'r-', linewidth=2, label='拟合悬链线')
    
    ax1.set_xlabel('s (沿导线方向)')
    ax1.set_ylabel('z (重力方向)')
    ax1.set_title('悬链线拟合')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 添加参数信息
    param_text = f'参数:\na = {a:.3f}\ns₀ = {s0:.3f}\nc = {c:.3f}'
    if quality is not None:
        param_text += f'\n\n质量指标:\nR² = {quality["r_squared"]:.3f}\nRMSE = {quality["rmse"]:.3f}'
    ax1.text(0.02, 0.98, param_text, transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 残差图
    if len(s) > 0:
        z_pred = a * np.cosh((s - s0) / a) + c
        residuals = z - z_pred
        
        ax2.scatter(s, residuals, c='red', alpha=0.7, s=30, label='残差')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('s (沿导线方向)')
        ax2.set_ylabel('残差')
        ax2.set_title('拟合残差')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 添加残差统计信息
        if quality is not None:
            residual_text = f'残差统计:\nMAE = {quality["mae"]:.3f}\nRMSE = {quality["rmse"]:.3f}'
            ax2.text(0.02, 0.98, residual_text, transform=ax2.transAxes, 
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle(title)
    plt.tight_layout()
    smart_show(title="wire_detection")


def plot_3d_catenary_curve(catenary_3d: Callable[[np.ndarray], np.ndarray],
                          s_range: Tuple[float, float],
                          points_3d: Optional[np.ndarray] = None,
                          title: str = "3D悬链线曲线") -> None:
    """
    可视化3D悬链线曲线
    
    参数:
        catenary_3d: 3D悬链线函数
        s_range: s坐标范围 (s_min, s_max)
        points_3d: 原始3D点（可选）
        title: 图像标题
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 生成悬链线曲线
    s_curve = np.linspace(s_range[0], s_range[1], 100)
    curve_3d = catenary_3d(s_curve)
    
    # 绘制悬链线曲线
    ax.plot(curve_3d[:, 0], curve_3d[:, 1], curve_3d[:, 2], 
           'r-', linewidth=3, label='悬链线曲线')
    
    # 绘制原始3D点（如果提供）
    if points_3d is not None and len(points_3d) > 0:
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
                  c='blue', s=50, alpha=0.7, label='原始3D点')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title(title)
    
    # 设置相等的坐标轴比例
    if points_3d is not None and len(points_3d) > 0:
        all_points = np.vstack([curve_3d, points_3d])
    else:
        all_points = curve_3d
    
    max_range = np.array([all_points[:, 0].max() - all_points[:, 0].min(),
                         all_points[:, 1].max() - all_points[:, 1].min(),
                         all_points[:, 2].max() - all_points[:, 2].min()]).max() / 2.0
    
    mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
    mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
    mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    smart_show(title="wire_detection")


def plot_pipeline_summary(imgL: np.ndarray,
                         imgR: np.ndarray,
                         wire_pixelsL: np.ndarray,
                         wire_pixelsR: np.ndarray,
                         matches: List[Tuple[np.ndarray, np.ndarray]],
                         points_3d: np.ndarray,
                         s_list: np.ndarray,
                         z_list: np.ndarray,
                         a: float,
                         s0: float,
                         c: float,
                         quality: Optional[dict] = None) -> None:
    """
    绘制完整的处理流程总结图
    
    参数:
        imgL, imgR: 左右图像
        wire_pixelsL, wire_pixelsR: 导线像素
        matches: 匹配点对
        points_3d: 3D点
        s_list, z_list: 平面投影坐标
        a, s0, c: 悬链线参数
        quality: 拟合质量
    """
    fig = plt.figure(figsize=(20, 12))
    
    # 1. 导线检测结果
    ax1 = plt.subplot(2, 4, 1)
    if len(imgL.shape) == 3:
        imgL_display = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
        ax1.imshow(imgL_display)
    else:
        ax1.imshow(imgL, cmap='gray')
    if len(wire_pixelsL) > 0:
        ax1.scatter(wire_pixelsL[:, 0], wire_pixelsL[:, 1], c='red', s=1, alpha=0.8)
    ax1.set_title(f'左图像导线检测\n({len(wire_pixelsL)}个像素)')
    ax1.axis('off')
    
    ax2 = plt.subplot(2, 4, 2)
    if len(imgR.shape) == 3:
        imgR_display = cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB)
        ax2.imshow(imgR_display)
    else:
        ax2.imshow(imgR, cmap='gray')
    if len(wire_pixelsR) > 0:
        ax2.scatter(wire_pixelsR[:, 0], wire_pixelsR[:, 1], c='red', s=1, alpha=0.8)
    ax2.set_title(f'右图像导线检测\n({len(wire_pixelsR)}个像素)')
    ax2.axis('off')
    
    # 2. 双目匹配结果
    ax3 = plt.subplot(2, 4, 3)
    h1, w1 = imgL.shape[:2]
    h2, w2 = imgR.shape[:2]
    combined_img = np.zeros((max(h1, h2), w1 + w2), dtype=np.uint8)
    
    # 处理彩色图像
    if len(imgL.shape) == 3:
        imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    else:
        imgL_gray = imgL
    
    if len(imgR.shape) == 3:
        imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    else:
        imgR_gray = imgR
    
    combined_img[:h1, :w1] = imgL_gray
    combined_img[:h2, w1:] = imgR_gray
    ax3.imshow(combined_img, cmap='gray')
    
    # 绘制部分匹配线
    display_matches = matches[:50] if len(matches) > 50 else matches
    for i, (ptL, ptR) in enumerate(display_matches):
        color = plt.cm.tab10(i % 10)
        ax3.plot([ptL[0], ptR[0] + w1], [ptL[1], ptR[1]], color=color, linewidth=2, alpha=0.6)
    
    ax3.set_title(f'双目匹配\n({len(matches)}个匹配)')
    ax3.axis('off')
    
    # 3. 3D点云
    ax4 = plt.subplot(2, 4, 4, projection='3d')
    if len(points_3d) > 0:
        ax4.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='blue', s=20, alpha=0.7)
    ax4.set_title(f'3D点云\n({len(points_3d)}个点)')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    
    # 4. 平面投影
    ax5 = plt.subplot(2, 4, 5)
    ax5.scatter(s_list, z_list, c='blue', alpha=0.7, s=20)
    ax5.set_xlabel('s (沿导线方向)')
    ax5.set_ylabel('z (重力方向)')
    ax5.set_title('平面投影')
    ax5.grid(True, alpha=0.3)
    
    # 5. 悬链线拟合
    ax6 = plt.subplot(2, 4, 6)
    ax6.scatter(s_list, z_list, c='blue', alpha=0.7, s=20, label='数据点')
    
    if len(s_list) > 0:
        s_plot = np.linspace(np.min(s_list), np.max(s_list), 100)
        z_plot = a * np.cosh((s_plot - s0) / a) + c
        ax6.plot(s_plot, z_plot, 'r-', linewidth=2, label='拟合悬链线')
    
    ax6.set_xlabel('s')
    ax6.set_ylabel('z')
    ax6.set_title('悬链线拟合')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 6. 拟合质量
    ax7 = plt.subplot(2, 4, 7)
    ax7.axis('off')
    
    quality_text = f'悬链线参数:\n\na = {a:.3f}\ns₀ = {s0:.3f}\nc = {c:.3f}'
    
    if quality is not None:
        quality_text += f'\n\n拟合质量:\n\nR² = {quality["r_squared"]:.3f}\nRMSE = {quality["rmse"]:.3f}\nMAE = {quality["mae"]:.3f}'
    
    ax7.text(0.1, 0.9, quality_text, transform=ax7.transAxes, 
             verticalalignment='top', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax7.set_title('拟合结果')
    
    # 7. 残差分析
    ax8 = plt.subplot(2, 4, 8)
    if len(s_list) > 0:
        z_pred = a * np.cosh((s_list - s0) / a) + c
        residuals = z_list - z_pred
        ax8.scatter(s_list, residuals, c='red', alpha=0.7, s=20)
        ax8.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax8.set_xlabel('s')
        ax8.set_ylabel('残差')
        ax8.set_title('拟合残差')
        ax8.grid(True, alpha=0.3)
    
    plt.suptitle('双目视觉悬链线重建完整流程', fontsize=16)
    plt.tight_layout()
    smart_show(title="wire_detection")


if __name__ == "__main__":
    # 测试代码
    print("可视化模块测试")
    
    # 创建测试数据
    imgL = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    imgR = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    
    # 添加一些线条
    cv2.line(imgL, (100, 200), (500, 300), 255, 2)
    cv2.line(imgR, (80, 200), (480, 300), 255, 2)
    
    # 测试数据
    wire_pixelsL = np.random.randint(100, 500, (50, 2))
    wire_pixelsR = np.random.randint(80, 480, (50, 2))
    
    matches = [(np.random.rand(2) * 100 + 100, np.random.rand(2) * 100 + 80) for _ in range(20)]
    
    points_3d = np.random.randn(20, 3) * 10
    
    s_list = np.linspace(-20, 20, 20)
    z_list = 5 * np.cosh(s_list / 5) + 10 + np.random.normal(0, 0.5, 20)
    
    a, s0, c = 5.0, 0.0, 10.0
    quality = {'r_squared': 0.95, 'rmse': 0.3, 'mae': 0.2}
    
    # 测试各个可视化函数
    print("测试导线检测可视化...")
    plot_wire_detection_results(imgL, imgR, wire_pixelsL, wire_pixelsR)
    
    print("测试双目匹配可视化...")
    plot_stereo_matches(imgL, imgR, matches)
    
    print("测试3D点云可视化...")
    plot_3d_points(points_3d)
    
    print("测试悬链线拟合可视化...")
    plot_catenary_fitting(s_list, z_list, a, s0, c, quality)
    
    print("测试完整流程可视化...")
    plot_pipeline_summary(imgL, imgR, wire_pixelsL, wire_pixelsR, matches, 
                         points_3d, s_list, z_list, a, s0, c, quality)
