"""
导线像素提取模块
使用细线检测算法从图像中提取导线的像素坐标
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from scipy import ndimage
from skimage import filters, morphology


def detect_wire_pixels(img: np.ndarray, 
                      method: str = 'canny_hough',
                      min_line_length: int = 50,
                      max_line_gap: int = 10) -> np.ndarray:
    """
    从图像中检测导线像素
    
    参数:
        img: 输入图像 (灰度图或彩色图)
        method: 检测方法 ('canny_hough', 'ridge', 'skeleton', 'catenary')
        min_line_length: 霍夫变换最小线段长度
        max_line_gap: 霍夫变换最大线段间隙
        
    返回:
        wire_pixels: 导线像素坐标数组 (N, 2)，格式为 [x, y]
    """
    # 转换为灰度图
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    if method == 'canny_hough':
        return _detect_with_canny_hough(gray, min_line_length, max_line_gap)
    elif method == 'ridge':
        return _detect_with_ridge(gray)
    elif method == 'skeleton':
        return _detect_with_skeleton(gray)
    elif method == 'catenary':
        return _detect_with_catenary(gray)
    else:
        raise ValueError(f"不支持的检测方法: {method}")


def _detect_with_canny_hough(gray: np.ndarray, 
                            min_line_length: int, 
                            max_line_gap: int) -> np.ndarray:
    """
    使用Canny边缘检测 + 霍夫变换检测直线
    """
    # 高斯滤波去噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny边缘检测
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    
    # 霍夫变换检测直线
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                           minLineLength=min_line_length, maxLineGap=max_line_gap)
    
    if lines is None:
        return np.array([]).reshape(0, 2)
    
    # 提取直线上的所有像素点
    wire_pixels = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # 使用Bresenham算法获取直线上的所有像素
        pixels = _get_line_pixels(x1, y1, x2, y2)
        wire_pixels.extend(pixels)
    
    return np.array(wire_pixels)


def _detect_with_ridge(gray: np.ndarray) -> np.ndarray:
    """
    使用脊线检测算法
    """
    # 使用高斯滤波
    smoothed = filters.gaussian(gray, sigma=1.0)
    
    # 计算梯度
    grad_x = filters.sobel_h(smoothed)
    grad_y = filters.sobel_v(smoothed)
    
    # 计算Hessian矩阵的特征值
    grad_xx = filters.sobel_h(grad_x)
    grad_yy = filters.sobel_v(grad_y)
    grad_xy = filters.sobel_h(grad_y)
    
    # 计算特征值
    trace = grad_xx + grad_yy
    det = grad_xx * grad_yy - grad_xy**2
    
    # 脊线条件：小特征值接近0，大特征值为负
    ridge_mask = (det > 0) & (trace < 0) & (np.abs(trace) > 0.1)
    
    # 形态学操作连接断开的脊线
    ridge_mask = morphology.binary_closing(ridge_mask, morphology.disk(2))
    
    # 提取脊线像素坐标
    y_coords, x_coords = np.where(ridge_mask)
    return np.column_stack([x_coords, y_coords])


def _detect_with_skeleton(gray: np.ndarray) -> np.ndarray:
    """
    使用骨架化算法
    """
    # 二值化
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 反转图像（骨架化需要白色前景）
    binary = 255 - binary
    
    # 骨架化
    skeleton = morphology.skeletonize(binary > 0)
    
    # 提取骨架像素坐标
    y_coords, x_coords = np.where(skeleton)
    return np.column_stack([x_coords, y_coords])


def _detect_with_catenary(gray: np.ndarray) -> np.ndarray:
    """
    使用适合悬链线形态的鲁棒二次曲线拟合进行导线像素检测。
    流程：
      1) 软化 + Canny 获取候选边缘点；
      2) 在 y=f(x) 与 x=f(y) 两种模型上分别做RANSAC二次拟合；
      3) 选择内点数最多且残差更小的模型，输出内点；
      4) 沿拟合曲线进行稠密采样以增强连续性。
    返回坐标为 [x, y]。
    """
    # 预处理
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    # 候选点（边缘点）
    ys, xs = np.nonzero(edges)
    if len(xs) < 30:
        return np.array([]).reshape(0, 2)
    points = np.column_stack([xs, ys])  # [x, y]

    # RANSAC 设置
    rng = np.random.default_rng(42)
    max_trials = 300
    residual_threshold = 1.5  # 像素

    def fit_quadratic_xy(sample_pts: np.ndarray):
        # 拟合 y = a x^2 + b x + c
        X = sample_pts[:, 0]
        Y = sample_pts[:, 1]
        A = np.column_stack([X**2, X, np.ones_like(X)])
        try:
            coef, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)
            return coef  # a, b, c
        except np.linalg.LinAlgError:
            return None

    def residuals_xy(coef, pts: np.ndarray):
        a, b, c = coef
        X = pts[:, 0]
        Y = pts[:, 1]
        Y_pred = a * X**2 + b * X + c
        return np.abs(Y - Y_pred)

    def fit_quadratic_yx(sample_pts: np.ndarray):
        # 拟合 x = a y^2 + b y + c
        X = sample_pts[:, 0]
        Y = sample_pts[:, 1]
        A = np.column_stack([Y**2, Y, np.ones_like(Y)])
        try:
            coef, _, _, _ = np.linalg.lstsq(A, X, rcond=None)
            return coef  # a, b, c
        except np.linalg.LinAlgError:
            return None

    def residuals_yx(coef, pts: np.ndarray):
        a, b, c = coef
        Y = pts[:, 1]
        X = pts[:, 0]
        X_pred = a * Y**2 + b * Y + c
        return np.abs(X - X_pred)

    def ransac(points_arr: np.ndarray, model_fit_fn, resid_fn):
        n_points = len(points_arr)
        best_inliers = None
        best_coef = None
        best_score = -1
        for _ in range(max_trials):
            # 至少3点拟合二次曲线
            sample_idx = rng.choice(n_points, size=6, replace=False) if n_points >= 6 else np.arange(n_points)
            sample = points_arr[sample_idx]
            coef = model_fit_fn(sample)
            if coef is None:
                continue
            res = resid_fn(coef, points_arr)
            inliers_mask = res < residual_threshold
            num_inliers = np.sum(inliers_mask)
            if num_inliers > best_score:
                best_score = num_inliers
                best_inliers = inliers_mask
                best_coef = coef
        # 用内点做一次精拟合
        if best_inliers is None or np.sum(best_inliers) < 6:
            return None, None
        refined_coef = model_fit_fn(points_arr[best_inliers])
        if refined_coef is None:
            refined_coef = best_coef
        return refined_coef, best_inliers

    # 在两种模型上做RANSAC
    coef_xy, inliers_xy = ransac(points, fit_quadratic_xy, residuals_xy)
    coef_yx, inliers_yx = ransac(points, fit_quadratic_yx, residuals_yx)

    # 选择更优模型
    def score(inliers):
        return -1 if inliers is None else int(np.sum(inliers))
    score_xy = score(inliers_xy)
    score_yx = score(inliers_yx)

    if score_xy <= 0 and score_yx <= 0:
        return np.array([]).reshape(0, 2)

    use_xy = score_xy >= score_yx
    if use_xy:
        a, b, c = coef_xy
        inliers_pts = points[inliers_xy]
        # 稠密采样
        x_min, x_max = int(np.min(inliers_pts[:, 0])), int(np.max(inliers_pts[:, 0]))
        xs_dense = np.linspace(x_min, x_max, num=max(50, x_max - x_min + 1))
        ys_dense = a * xs_dense**2 + b * xs_dense + c
        dense_pts = np.column_stack([xs_dense, ys_dense])
    else:
        a, b, c = coef_yx
        inliers_pts = points[inliers_yx]
        y_min, y_max = int(np.min(inliers_pts[:, 1])), int(np.max(inliers_pts[:, 1]))
        ys_dense = np.linspace(y_min, y_max, num=max(50, y_max - y_min + 1))
        xs_dense = a * ys_dense**2 + b * ys_dense + c
        dense_pts = np.column_stack([xs_dense, ys_dense])

    # 合并内点与稠密点，并裁剪到图像范围
    all_pts = np.vstack([inliers_pts, dense_pts])
    all_pts[:, 0] = np.clip(np.round(all_pts[:, 0]), 0, gray.shape[1] - 1)
    all_pts[:, 1] = np.clip(np.round(all_pts[:, 1]), 0, gray.shape[0] - 1)
    all_pts = np.unique(all_pts.astype(int), axis=0)

    return all_pts


def _get_line_pixels(x1: int, y1: int, x2: int, y2: int) -> List[Tuple[int, int]]:
    """
    使用Bresenham算法获取直线上的所有像素坐标
    """
    pixels = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    
    err = dx - dy
    x, y = x1, y1
    
    while True:
        pixels.append((x, y))
        
        if x == x2 and y == y2:
            break
            
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    
    return pixels


def filter_wire_pixels(pixels: np.ndarray, 
                      img_shape: Tuple[int, int],
                      min_length: int = 20) -> np.ndarray:
    """
    过滤导线像素，去除噪声和短线段
    
    参数:
        pixels: 原始像素坐标 (N, 2)
        img_shape: 图像尺寸 (height, width)
        min_length: 最小线段长度
        
    返回:
        filtered_pixels: 过滤后的像素坐标
    """
    if len(pixels) == 0:
        return pixels
    
    # 边界检查
    valid_mask = ((pixels[:, 0] >= 0) & (pixels[:, 0] < img_shape[1]) &
                  (pixels[:, 1] >= 0) & (pixels[:, 1] < img_shape[0]))
    pixels = pixels[valid_mask]
    
    if len(pixels) == 0:
        return pixels
    
    # 使用DBSCAN聚类去除孤立点
    from sklearn.cluster import DBSCAN
    
    clustering = DBSCAN(eps=5, min_samples=3).fit(pixels)
    labels = clustering.labels_
    
    # 保留最大的聚类
    unique_labels = np.unique(labels)
    if len(unique_labels) <= 1:
        return pixels
    
    # 计算每个聚类的点数（排除噪声标签-1）
    valid_labels = unique_labels[unique_labels != -1]
    if len(valid_labels) == 0:
        return pixels
    
    cluster_sizes = [np.sum(labels == label) for label in valid_labels]
    if not cluster_sizes:
        return pixels
    
    # 找到最大聚类对应的标签
    largest_cluster_idx = np.argmax(cluster_sizes)
    largest_cluster_label = valid_labels[largest_cluster_idx]
    filtered_pixels = pixels[labels == largest_cluster_label]
    
    return filtered_pixels


def visualize_wire_detection(img: np.ndarray, 
                           wire_pixels: np.ndarray,
                           title: str = "导线检测结果") -> np.ndarray:
    """
    可视化导线检测结果
    
    参数:
        img: 原始图像
        wire_pixels: 检测到的导线像素
        title: 图像标题
        
    返回:
        vis_img: 可视化图像
    """
    vis_img = img.copy()
    if len(vis_img.shape) == 2:
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)
    
    # 在检测到的像素上绘制红色点
    if len(wire_pixels) > 0:
        for x, y in wire_pixels:
            cv2.circle(vis_img, (int(x), int(y)), 1, (0, 0, 255), -1)
    
    return vis_img


if __name__ == "__main__":
    # 测试代码
    import matplotlib.pyplot as plt
    
    # 创建测试图像
    test_img = np.zeros((480, 640), dtype=np.uint8)
    cv2.line(test_img, (100, 200), (500, 300), 255, 2)
    cv2.line(test_img, (200, 100), (400, 400), 255, 2)
    
    # 检测导线像素
    pixels = detect_wire_pixels(test_img, method='canny_hough')
    print(f"检测到 {len(pixels)} 个导线像素")
    
    # 可视化结果
    vis_img = visualize_wire_detection(test_img, pixels)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(test_img, cmap='gray')
    plt.title('原始图像')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    plt.title('检测结果')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
