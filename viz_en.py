"""
英文版可视化模块
提供2D和3D可视化功能，使用英文标签
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from typing import List, Tuple, Optional, Callable
import matplotlib.patches as patches
import os

# 设置合适的后端
matplotlib.use('Agg')

def smart_show(fig=None, filename=None, title="plot"):
    """
    智能显示函数：在无头环境下自动保存图片，在有显示环境下显示窗口
    """
    if fig is None:
        fig = plt.gcf()
    
    if filename is None:
        filename = f"output/{title.replace(' ', '_')}.png"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Chart saved to: {filename}")


def plot_wire_detection_results(imgL: np.ndarray, 
                               imgR: np.ndarray,
                               wire_pixelsL: np.ndarray,
                               wire_pixelsR: np.ndarray,
                               title: str = "Wire Detection Results") -> None:
    """
    可视化导线检测结果
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
                       c='red', s=1, alpha=0.8, label=f'Wire pixels ({len(wire_pixelsL)})')
    axes[0].set_title('Left Image Wire Detection')
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
                       c='red', s=1, alpha=0.8, label=f'Wire pixels ({len(wire_pixelsR)})')
    axes[1].set_title('Right Image Wire Detection')
    axes[1].axis('off')
    axes[1].legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    smart_show(title="wire_detection_en")


def plot_stereo_matches(imgL: np.ndarray, 
                       imgR: np.ndarray,
                       matches: List[Tuple[np.ndarray, np.ndarray]],
                       title: str = "Stereo Matching Results",
                       max_matches: int = 50) -> None:
    """
    可视化双目匹配结果
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
    
    ax.set_title(f'{title} (showing {len(display_matches)}/{len(matches)} matches)')
    ax.axis('off')
    
    plt.tight_layout()
    smart_show(title="stereo_matches_en")


def plot_3d_points(points_3d: np.ndarray,
                  camera_centers: Optional[List[np.ndarray]] = None,
                  title: str = "3D Point Cloud",
                  show_cameras: bool = True) -> None:
    """
    可视化3D点云
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    if len(points_3d) > 0:
        # 绘制3D点
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
                  c='blue', s=50, alpha=0.7, label=f'3D Points ({len(points_3d)})')
        
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
                      c='red', s=200, marker='^', label=f'Camera {i+1}')
    
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
    
    smart_show(title="3d_points_en")


def plot_catenary_fitting(s: np.ndarray, 
                         z: np.ndarray,
                         a: float, 
                         s0: float, 
                         c: float,
                         quality: Optional[dict] = None,
                         title: str = "Catenary Fitting Results") -> None:
    """
    可视化悬链线拟合结果
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 拟合结果图
    ax1.scatter(s, z, c='blue', alpha=0.7, s=30, label='Data points')
    
    # 绘制拟合曲线
    s_plot = np.linspace(np.min(s), np.max(s), 100)
    z_plot = a * np.cosh((s_plot - s0) / a) + c
    ax1.plot(s_plot, z_plot, 'r-', linewidth=2, label='Fitted catenary')
    
    ax1.set_xlabel('s (along wire direction)')
    ax1.set_ylabel('z (gravity direction)')
    ax1.set_title('Catenary Fitting')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 添加参数信息
    param_text = f'Parameters:\na = {a:.3f}\ns₀ = {s0:.3f}\nc = {c:.3f}'
    if quality is not None:
        param_text += f'\n\nQuality:\nR² = {quality["r_squared"]:.3f}\nRMSE = {quality["rmse"]:.3f}'
    ax1.text(0.02, 0.98, param_text, transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 残差图
    if len(s) > 0:
        z_pred = a * np.cosh((s - s0) / a) + c
        residuals = z - z_pred
        
        ax2.scatter(s, residuals, c='red', alpha=0.7, s=30, label='Residuals')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('s (along wire direction)')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Fitting Residuals')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 添加残差统计信息
        if quality is not None:
            residual_text = f'Residual Statistics:\nMAE = {quality["mae"]:.3f}\nRMSE = {quality["rmse"]:.3f}'
            ax2.text(0.02, 0.98, residual_text, transform=ax2.transAxes, 
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle(title)
    plt.tight_layout()
    smart_show(title="catenary_fitting_en")


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
    绘制完整的处理流程总结图（英文版）
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
    ax1.set_title(f'Left Wire Detection\n({len(wire_pixelsL)} pixels)')
    ax1.axis('off')
    
    ax2 = plt.subplot(2, 4, 2)
    if len(imgR.shape) == 3:
        imgR_display = cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB)
        ax2.imshow(imgR_display)
    else:
        ax2.imshow(imgR, cmap='gray')
    if len(wire_pixelsR) > 0:
        ax2.scatter(wire_pixelsR[:, 0], wire_pixelsR[:, 1], c='red', s=1, alpha=0.8)
    ax2.set_title(f'Right Wire Detection\n({len(wire_pixelsR)} pixels)')
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
    display_matches = matches[:20] if len(matches) > 20 else matches
    for i, (ptL, ptR) in enumerate(display_matches):
        color = plt.cm.tab10(i % 10)
        ax3.plot([ptL[0], ptR[0] + w1], [ptL[1], ptR[1]], color=color, linewidth=1, alpha=0.7)
    
    ax3.set_title(f'Stereo Matching\n({len(matches)} matches)')
    ax3.axis('off')
    
    # 3. 3D点云
    ax4 = plt.subplot(2, 4, 4, projection='3d')
    if len(points_3d) > 0:
        ax4.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='blue', s=20, alpha=0.7)
    ax4.set_title(f'3D Point Cloud\n({len(points_3d)} points)')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    
    # 4. 平面投影
    ax5 = plt.subplot(2, 4, 5)
    ax5.scatter(s_list, z_list, c='blue', alpha=0.7, s=20)
    ax5.set_xlabel('s (along wire)')
    ax5.set_ylabel('z (gravity)')
    ax5.set_title('Plane Projection')
    ax5.grid(True, alpha=0.3)
    
    # 5. 悬链线拟合
    ax6 = plt.subplot(2, 4, 6)
    ax6.scatter(s_list, z_list, c='blue', alpha=0.7, s=20, label='Data points')
    
    if len(s_list) > 0:
        s_plot = np.linspace(np.min(s_list), np.max(s_list), 100)
        z_plot = a * np.cosh((s_plot - s0) / a) + c
        ax6.plot(s_plot, z_plot, 'r-', linewidth=2, label='Fitted catenary')
    
    ax6.set_xlabel('s')
    ax6.set_ylabel('z')
    ax6.set_title('Catenary Fitting')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 6. 拟合质量
    ax7 = plt.subplot(2, 4, 7)
    ax7.axis('off')
    
    quality_text = f'Catenary Parameters:\n\na = {a:.3f}\ns₀ = {s0:.3f}\nc = {c:.3f}'
    
    if quality is not None:
        quality_text += f'\n\nFitting Quality:\n\nR² = {quality["r_squared"]:.3f}\nRMSE = {quality["rmse"]:.3f}\nMAE = {quality["mae"]:.3f}'
    
    ax7.text(0.1, 0.9, quality_text, transform=ax7.transAxes, 
             verticalalignment='top', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax7.set_title('Fitting Results')
    
    # 7. 残差分析
    ax8 = plt.subplot(2, 4, 8)
    if len(s_list) > 0:
        z_pred = a * np.cosh((s_list - s0) / a) + c
        residuals = z_list - z_pred
        ax8.scatter(s_list, residuals, c='red', alpha=0.7, s=20)
        ax8.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax8.set_xlabel('s')
        ax8.set_ylabel('Residuals')
        ax8.set_title('Fitting Residuals')
        ax8.grid(True, alpha=0.3)
    
    plt.suptitle('Stereo Vision Catenary Reconstruction Pipeline', fontsize=16)
    plt.tight_layout()
    smart_show(title="pipeline_summary_en")
