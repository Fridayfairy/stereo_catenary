"""
垂直平面估计模块
估计包含重力方向和导线方向的垂直平面
"""

import numpy as np
from typing import Tuple, Optional
from sklearn.decomposition import PCA
from scipy.optimize import minimize


def estimate_vertical_plane(Xs: np.ndarray, 
                          g_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    估计包含重力方向和导线方向的垂直平面
    
    参数:
        Xs: 3D点坐标数组 (N, 3)
        g_hat: 重力方向单位向量 (3,)
        
    返回:
        p0: 平面上的参考点 (3,)
        u: 沿导线方向的单位向量 (3,)
        g_hat: 重力方向单位向量 (3,)
    """
    if len(Xs) < 3:
        raise ValueError("至少需要3个点来估计平面")
    
    # 1. 使用PCA找到导线的主要方向
    pca = PCA(n_components=3)
    pca.fit(Xs)
    
    # 主成分方向
    v1 = pca.components_[0]  # 第一主成分（最大方差方向）
    v2 = pca.components_[1]  # 第二主成分
    v3 = pca.components_[2]  # 第三主成分
    
    # 2. 计算沿导线方向的单位向量u
    # u = normalize(v1 - (v1·g_hat) * g_hat)
    # 这确保u在垂直于重力的平面内
    v1_proj_g = np.dot(v1, g_hat) * g_hat
    u = v1 - v1_proj_g
    u = u / np.linalg.norm(u)
    
    # 3. 选择平面上的参考点p0
    # 正确做法：让平面通过点云质心，平面法向量 n = u × g_hat
    centroid = np.mean(Xs, axis=0)
    n = np.cross(u, g_hat)
    n = n / np.linalg.norm(n)
    # 令 p0 为质心本身（而不是投影到过原点的平面），避免引入全局偏移
    p0 = centroid
    
    return p0, u, g_hat


def project_to_plane(Xs: np.ndarray, 
                    p0: np.ndarray, 
                    u: np.ndarray, 
                    g_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    将3D点投影到垂直平面上，得到2D坐标(s, z)
    
    参数:
        Xs: 3D点坐标数组 (N, 3)
        p0: 平面参考点 (3,)
        u: 沿导线方向的单位向量 (3,)
        g_hat: 重力方向单位向量 (3,)
        
    返回:
        s_list: 沿导线方向的坐标 (N,)
        z_list: 重力方向的坐标 (N,)
    """
    if len(Xs) == 0:
        return np.array([]), np.array([])
    
    # 计算每个点相对于参考点的向量
    vectors = Xs - p0
    
    # 投影到导线方向 (s坐标)
    s_list = np.dot(vectors, u)
    
    # 投影到重力方向 (z坐标)
    z_list = np.dot(vectors, g_hat)
    
    return s_list, z_list


def refine_plane_estimation(Xs: np.ndarray, 
                          g_hat: np.ndarray,
                          initial_p0: Optional[np.ndarray] = None,
                          initial_u: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    使用优化方法精化平面估计
    
    参数:
        Xs: 3D点坐标数组 (N, 3)
        g_hat: 重力方向单位向量 (3,)
        initial_p0: 初始参考点
        initial_u: 初始导线方向向量
        
    返回:
        p0: 优化后的参考点
        u: 优化后的导线方向向量
        g_hat: 重力方向向量
    """
    if len(Xs) < 3:
        raise ValueError("至少需要3个点来估计平面")
    
    # 获取初始估计
    if initial_p0 is None or initial_u is None:
        p0_init, u_init, _ = estimate_vertical_plane(Xs, g_hat)
    else:
        p0_init, u_init = initial_p0, initial_u
    
    # 定义目标函数：最小化点到平面的距离平方和
    def objective(params):
        # 参数: [p0_x, p0_y, p0_z, u_x, u_y, u_z]
        p0 = params[:3]
        u = params[3:6]
        
        # 归一化u
        u = u / np.linalg.norm(u)
        
        # 确保u垂直于g_hat
        u_proj_g = np.dot(u, g_hat) * g_hat
        u = u - u_proj_g
        u = u / np.linalg.norm(u)
        
        # 计算平面法向量
        n = np.cross(u, g_hat)
        n = n / np.linalg.norm(n)
        
        # 计算所有点到平面的距离平方和
        vectors = Xs - p0
        distances = np.abs(np.dot(vectors, n))
        
        return np.sum(distances**2)
    
    # 初始参数
    x0 = np.concatenate([p0_init, u_init])
    
    # 约束条件：u必须是单位向量
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.linalg.norm(x[3:6]) - 1.0}
    ]
    
    # 优化
    try:
        result = minimize(objective, x0, method='SLSQP', constraints=constraints)
        if result.success:
            p0_opt = result.x[:3]
            u_opt = result.x[3:6]
            u_opt = u_opt / np.linalg.norm(u_opt)
            
            # 确保u垂直于g_hat
            u_proj_g = np.dot(u_opt, g_hat) * g_hat
            u_opt = u_opt - u_proj_g
            u_opt = u_opt / np.linalg.norm(u_opt)
            
            return p0_opt, u_opt, g_hat
        else:
            return p0_init, u_init, g_hat
    except:
        return p0_init, u_init, g_hat


def compute_plane_quality(Xs: np.ndarray, 
                         p0: np.ndarray, 
                         u: np.ndarray, 
                         g_hat: np.ndarray) -> float:
    """
    计算平面估计的质量
    
    参数:
        Xs: 3D点坐标数组
        p0: 平面参考点
        u: 导线方向向量
        g_hat: 重力方向向量
        
    返回:
        quality: 平面质量分数 (0-1，1表示完美)
    """
    if len(Xs) < 3:
        return 0.0
    
    # 计算平面法向量
    n = np.cross(u, g_hat)
    n = n / np.linalg.norm(n)
    
    # 计算所有点到平面的距离
    vectors = Xs - p0
    distances = np.abs(np.dot(vectors, n))
    
    # 计算距离的标准差
    std_distance = np.std(distances)
    
    # 计算点云的跨度
    span = np.max(distances) - np.min(distances)
    
    # 质量分数：标准差越小，质量越高
    if span > 0:
        quality = 1.0 - (std_distance / span)
    else:
        quality = 1.0
    
    return max(0.0, min(1.0, quality))


def filter_points_by_plane_distance(Xs: np.ndarray, 
                                  p0: np.ndarray, 
                                  u: np.ndarray, 
                                  g_hat: np.ndarray,
                                  threshold: float = 0.1) -> np.ndarray:
    """
    根据到平面的距离过滤点
    
    参数:
        Xs: 3D点坐标数组
        p0: 平面参考点
        u: 导线方向向量
        g_hat: 重力方向向量
        threshold: 距离阈值
        
    返回:
        filtered_Xs: 过滤后的3D点
    """
    if len(Xs) == 0:
        return Xs
    
    # 计算平面法向量
    n = np.cross(u, g_hat)
    n = n / np.linalg.norm(n)
    
    # 计算所有点到平面的距离
    vectors = Xs - p0
    distances = np.abs(np.dot(vectors, n))
    
    # 过滤距离小于阈值的点
    mask = distances < threshold
    return Xs[mask]


def visualize_plane_estimation(Xs: np.ndarray, 
                             p0: np.ndarray, 
                             u: np.ndarray, 
                             g_hat: np.ndarray,
                             title: str = "平面估计结果") -> None:
    """
    可视化平面估计结果
    
    参数:
        Xs: 3D点坐标数组
        p0: 平面参考点
        u: 导线方向向量
        g_hat: 重力方向向量
        title: 图像标题
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制3D点
    ax.scatter(Xs[:, 0], Xs[:, 1], Xs[:, 2], c='blue', s=50, alpha=0.6, label='3D点')
    
    # 绘制参考点
    ax.scatter(p0[0], p0[1], p0[2], c='red', s=200, marker='o', label='参考点p0')
    
    # 绘制方向向量
    scale = 50
    ax.quiver(p0[0], p0[1], p0[2], 
              u[0] * scale, u[1] * scale, u[2] * scale, 
              color='green', arrow_length_ratio=0.1, label='导线方向u')
    
    ax.quiver(p0[0], p0[1], p0[2], 
              g_hat[0] * scale, g_hat[1] * scale, g_hat[2] * scale, 
              color='orange', arrow_length_ratio=0.1, label='重力方向g')
    
    # 绘制平面
    n = np.cross(u, g_hat)
    n = n / np.linalg.norm(n)
    
    # 创建平面网格
    s_range = np.linspace(-100, 100, 20)
    z_range = np.linspace(-50, 50, 20)
    S, Z = np.meshgrid(s_range, z_range)
    
    # 将2D网格转换为3D坐标
    X_plane = p0[0] + S * u[0] + Z * g_hat[0]
    Y_plane = p0[1] + S * u[1] + Z * g_hat[1]
    Z_plane = p0[2] + S * u[2] + Z * g_hat[2]
    
    ax.plot_surface(X_plane, Y_plane, Z_plane, alpha=0.3, color='cyan')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title(title)
    
    plt.show()


if __name__ == "__main__":
    # 测试代码
    import matplotlib.pyplot as plt
    
    # 创建测试数据：模拟悬链线形状的3D点
    s = np.linspace(-50, 50, 20)
    z = 10 * np.cosh(s / 10) - 10  # 悬链线方程
    
    # 添加一些噪声
    s += np.random.normal(0, 0.5, len(s))
    z += np.random.normal(0, 0.2, len(z))
    
    # 转换为3D坐标
    g_hat = np.array([0, 0, -1])  # 重力向下
    u = np.array([1, 0, 0])       # 导线沿x方向
    p0 = np.array([0, 0, 0])      # 原点
    
    Xs = np.column_stack([
        s * u[0] + p0[0],
        s * u[1] + p0[1], 
        z * g_hat[2] + p0[2]
    ])
    
    print(f"测试数据: {len(Xs)} 个3D点")
    
    # 估计垂直平面
    p0_est, u_est, g_hat_est = estimate_vertical_plane(Xs, g_hat)
    
    print(f"估计的参考点: ({p0_est[0]:.2f}, {p0_est[1]:.2f}, {p0_est[2]:.2f})")
    print(f"估计的导线方向: ({u_est[0]:.2f}, {u_est[1]:.2f}, {u_est[2]:.2f})")
    
    # 投影到平面
    s_proj, z_proj = project_to_plane(Xs, p0_est, u_est, g_hat_est)
    
    # 计算平面质量
    quality = compute_plane_quality(Xs, p0_est, u_est, g_hat_est)
    print(f"平面估计质量: {quality:.3f}")
    
    # 可视化结果
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(s_proj, z_proj, c='blue', alpha=0.7, label='投影点')
    plt.plot(s, z, 'r-', linewidth=2, label='真实悬链线')
    plt.xlabel('s (沿导线方向)')
    plt.ylabel('z (重力方向)')
    plt.title('2D投影结果')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.scatter(s, z, c='blue', alpha=0.7, label='原始数据')
    plt.scatter(s_proj, z_proj, c='red', alpha=0.7, label='投影数据')
    plt.xlabel('s')
    plt.ylabel('z')
    plt.title('原始vs投影')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 3D可视化
    visualize_plane_estimation(Xs, p0_est, u_est, g_hat_est)
