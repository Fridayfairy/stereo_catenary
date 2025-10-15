"""
三角测量模块
将双目匹配点对转换为3D空间坐标
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from scipy.optimize import minimize


def triangulate_pairs(pairs: List[Tuple[np.ndarray, np.ndarray]], 
                     P1: np.ndarray, 
                     P2: np.ndarray,
                     method: str = 'dlt') -> np.ndarray:
    """
    将匹配点对三角测量为3D点
    
    参数:
        pairs: 匹配点对列表 [(pL, pR), ...]
        P1: 左相机投影矩阵 (3x4)
        P2: 右相机投影矩阵 (3x4)
        method: 三角测量方法 ('dlt', 'midpoint', 'optimal')
        
    返回:
        points_3d: 3D点坐标数组 (N, 3)
    """
    if len(pairs) == 0:
        return np.array([]).reshape(0, 3)
    
    points_3d = []
    
    for ptL, ptR in pairs:
        if method == 'dlt':
            point_3d = triangulate_dlt(ptL, ptR, P1, P2)
        elif method == 'midpoint':
            point_3d = triangulate_midpoint(ptL, ptR, P1, P2)
        elif method == 'optimal':
            point_3d = triangulate_optimal(ptL, ptR, P1, P2)
        else:
            raise ValueError(f"不支持的三角测量方法: {method}")
        
        if point_3d is not None:
            points_3d.append(point_3d)
    
    return np.array(points_3d)


def triangulate_dlt(ptL: np.ndarray, 
                   ptR: np.ndarray, 
                   P1: np.ndarray, 
                   P2: np.ndarray) -> Optional[np.ndarray]:
    """
    使用直接线性变换(DLT)进行三角测量
    
    参数:
        ptL: 左图像中的点
        ptR: 右图像中的点
        P1: 左相机投影矩阵
        P2: 右相机投影矩阵
        
    返回:
        point_3d: 3D点坐标，如果失败则返回None
    """
    # 构建线性方程组 Ax = 0
    A = np.zeros((4, 4))
    
    # 左图像约束
    A[0] = ptL[0] * P1[2] - P1[0]
    A[1] = ptL[1] * P1[2] - P1[1]
    
    # 右图像约束
    A[2] = ptR[0] * P2[2] - P2[0]
    A[3] = ptR[1] * P2[2] - P2[1]
    
    # 使用SVD求解
    try:
        _, _, V = np.linalg.svd(A)
        X = V[-1]  # 最小特征值对应的特征向量
        
        # 转换为非齐次坐标
        if abs(X[3]) > 1e-8:
            point_3d = X[:3] / X[3]
        else:
            return None
        
        return point_3d
    except:
        return None


def triangulate_midpoint(ptL: np.ndarray, 
                        ptR: np.ndarray, 
                        P1: np.ndarray, 
                        P2: np.ndarray) -> Optional[np.ndarray]:
    """
    使用射线中点法进行三角测量
    
    参数:
        ptL: 左图像中的点
        ptR: 右图像中的点
        P1: 左相机投影矩阵
        P2: 右相机投影矩阵
        
    返回:
        point_3d: 3D点坐标，如果失败则返回None
    """
    # 计算相机中心
    C1 = compute_camera_center(P1)
    C2 = compute_camera_center(P2)
    
    # 计算射线方向
    ray1 = compute_ray_direction(ptL, P1)
    ray2 = compute_ray_direction(ptR, P2)
    
    if ray1 is None or ray2 is None:
        return None
    
    # 计算两条射线的最短距离点
    point_3d = compute_ray_midpoint(C1, ray1, C2, ray2)
    
    return point_3d


def triangulate_optimal(ptL: np.ndarray, 
                       ptR: np.ndarray, 
                       P1: np.ndarray, 
                       P2: np.ndarray) -> Optional[np.ndarray]:
    """
    使用最优三角测量方法
    
    参数:
        ptL: 左图像中的点
        ptR: 右图像中的点
        P1: 左相机投影矩阵
        P2: 右相机投影矩阵
        
    返回:
        point_3d: 3D点坐标，如果失败则返回None
    """
    # 使用DLT作为初始估计
    initial_point = triangulate_dlt(ptL, ptR, P1, P2)
    if initial_point is None:
        return None
    
    # 定义目标函数：重投影误差
    def objective(X):
        # 投影到左图像
        projL = P1 @ np.append(X, 1)
        if abs(projL[2]) < 1e-8:
            return float('inf')
        projL = projL[:2] / projL[2]
        
        # 投影到右图像
        projR = P2 @ np.append(X, 1)
        if abs(projR[2]) < 1e-8:
            return float('inf')
        projR = projR[:2] / projR[2]
        
        # 计算重投影误差
        errorL = np.sum((projL - ptL)**2)
        errorR = np.sum((projR - ptR)**2)
        
        return errorL + errorR
    
    # 使用优化算法求解
    try:
        result = minimize(objective, initial_point, method='BFGS')
        if result.success:
            return result.x
        else:
            return initial_point
    except:
        return initial_point


def compute_camera_center(P: np.ndarray) -> np.ndarray:
    """
    计算相机中心
    
    参数:
        P: 投影矩阵 (3x4)
        
    返回:
        C: 相机中心坐标 (3,)
    """
    # 分解投影矩阵 P = K[R|t]
    # 相机中心 C = -R^T * t
    M = P[:, :3]
    t = P[:, 3]
    
    try:
        R = np.linalg.inv(M)
        C = -R @ t
        return C
    except:
        # 如果M不可逆，使用SVD分解
        _, _, V = np.linalg.svd(P)
        C = V[-1, :3] / V[-1, 3]
        return C


def compute_ray_direction(pt: np.ndarray, P: np.ndarray) -> Optional[np.ndarray]:
    """
    计算从相机中心通过图像点的射线方向
    
    参数:
        pt: 图像点坐标
        P: 投影矩阵
        
    返回:
        direction: 射线方向向量，如果失败则返回None
    """
    # 计算相机中心
    C = compute_camera_center(P)
    
    # 计算射线上的另一个点
    # 在深度为1的平面上
    pt_homo = np.array([pt[0], pt[1], 1.0])
    
    # 求解 P * X = pt_homo，其中 X[2] = 1
    # 这相当于求解 [P[:,:2], P[:,3]] * [X[0], X[1], 1] = pt_homo
    A = P[:, :2]
    b = pt_homo - P[:, 3]
    
    try:
        X_2d = np.linalg.solve(A, b)
        X = np.array([X_2d[0], X_2d[1], 1.0])
        
        # 计算射线方向
        direction = X - C
        direction = direction / np.linalg.norm(direction)
        
        return direction
    except:
        return None


def compute_ray_midpoint(C1: np.ndarray, 
                        ray1: np.ndarray, 
                        C2: np.ndarray, 
                        ray2: np.ndarray) -> np.ndarray:
    """
    计算两条射线的最短距离中点
    
    参数:
        C1: 第一条射线的起点
        ray1: 第一条射线的方向
        C2: 第二条射线的起点
        ray2: 第二条射线的方向
        
    返回:
        midpoint: 最短距离的中点
    """
    # 计算两条射线的最短距离点
    # 设第一条射线上的点为 C1 + t1 * ray1
    # 设第二条射线上的点为 C2 + t2 * ray2
    # 最小化距离 ||(C1 + t1 * ray1) - (C2 + t2 * ray2)||^2
    
    # 构建线性方程组
    w0 = C1 - C2
    a = np.dot(ray1, ray1)
    b = np.dot(ray1, ray2)
    c = np.dot(ray2, ray2)
    d = np.dot(ray1, w0)
    e = np.dot(ray2, w0)
    
    # 求解 t1 和 t2
    denom = a * c - b * b
    if abs(denom) < 1e-8:
        # 射线平行，使用中点
        t1 = 0
        t2 = 0
    else:
        t1 = (b * e - c * d) / denom
        t2 = (a * e - b * d) / denom
    
    # 计算两个点
    point1 = C1 + t1 * ray1
    point2 = C2 + t2 * ray2
    
    # 返回中点
    return (point1 + point2) / 2


def compute_reprojection_error(points_3d: np.ndarray,
                             pairs: List[Tuple[np.ndarray, np.ndarray]],
                             P1: np.ndarray,
                             P2: np.ndarray) -> float:
    """
    计算重投影误差
    
    参数:
        points_3d: 3D点坐标
        pairs: 匹配点对
        P1: 左相机投影矩阵
        P2: 右相机投影矩阵
        
    返回:
        error: 平均重投影误差
    """
    if len(points_3d) == 0:
        return 0.0
    
    total_error = 0.0
    count = 0
    
    for i, (ptL, ptR) in enumerate(pairs):
        if i >= len(points_3d):
            break
        
        point_3d = points_3d[i]
        
        # 投影到左图像
        projL = P1 @ np.append(point_3d, 1)
        if abs(projL[2]) > 1e-8:
            projL = projL[:2] / projL[2]
            errorL = np.sum((projL - ptL)**2)
            total_error += errorL
            count += 1
        
        # 投影到右图像
        projR = P2 @ np.append(point_3d, 1)
        if abs(projR[2]) > 1e-8:
            projR = projR[:2] / projR[2]
            errorR = np.sum((projR - ptR)**2)
            total_error += errorR
            count += 1
    
    return total_error / count if count > 0 else 0.0


def filter_outliers_by_reprojection(points_3d: np.ndarray,
                                  pairs: List[Tuple[np.ndarray, np.ndarray]],
                                  P1: np.ndarray,
                                  P2: np.ndarray,
                                  threshold: float = 2.0) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
    """
    根据重投影误差过滤异常点
    
    参数:
        points_3d: 3D点坐标
        pairs: 匹配点对
        P1: 左相机投影矩阵
        P2: 右相机投影矩阵
        threshold: 重投影误差阈值
        
    返回:
        filtered_points: 过滤后的3D点
        filtered_pairs: 过滤后的匹配点对
    """
    if len(points_3d) == 0:
        return points_3d, pairs
    
    filtered_points = []
    filtered_pairs = []
    
    for i, (point_3d, (ptL, ptR)) in enumerate(zip(points_3d, pairs)):
        # 计算重投影误差
        projL = P1 @ np.append(point_3d, 1)
        projR = P2 @ np.append(point_3d, 1)
        
        if abs(projL[2]) > 1e-8 and abs(projR[2]) > 1e-8:
            projL = projL[:2] / projL[2]
            projR = projR[:2] / projR[2]
            
            errorL = np.sum((projL - ptL)**2)
            errorR = np.sum((projR - ptR)**2)
            
            if errorL < threshold and errorR < threshold:
                filtered_points.append(point_3d)
                filtered_pairs.append((ptL, ptR))
    
    return np.array(filtered_points), filtered_pairs


if __name__ == "__main__":
    # 测试代码
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # 创建测试数据
    # 简单的相机参数
    K1 = K2 = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float32)
    R1 = R2 = np.eye(3, dtype=np.float32)
    t1 = np.array([0, 0, 0], dtype=np.float32)
    t2 = np.array([-50, 0, 0], dtype=np.float32)
    
    # 构建投影矩阵
    P1 = K1 @ np.hstack([R1, t1.reshape(3, 1)])
    P2 = K2 @ np.hstack([R2, t2.reshape(3, 1)])
    
    # 测试点对
    pairs = [
        (np.array([200, 250]), np.array([150, 250])),
        (np.array([300, 270]), np.array([250, 270])),
        (np.array([400, 290]), np.array([350, 290]))
    ]
    
    # 进行三角测量
    points_3d = triangulate_pairs(pairs, P1, P2, method='dlt')
    
    print(f"三角测量得到 {len(points_3d)} 个3D点")
    print("3D点坐标:")
    for i, point in enumerate(points_3d):
        print(f"  点{i+1}: ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})")
    
    # 计算重投影误差
    error = compute_reprojection_error(points_3d, pairs, P1, P2)
    print(f"平均重投影误差: {error:.4f}")
    
    # 可视化3D点
    if len(points_3d) > 0:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
                  c='red', s=100, label='3D点')
        
        # 绘制相机位置
        C1 = compute_camera_center(P1)
        C2 = compute_camera_center(P2)
        ax.scatter([C1[0], C2[0]], [C1[1], C2[1]], [C1[2], C2[2]], 
                  c='blue', s=200, marker='^', label='相机位置')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.set_title('三角测量结果')
        
        plt.show()
