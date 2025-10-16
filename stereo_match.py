"""
极线约束匹配模块
在双目图像中沿极线进行导线像素的匹配
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from scipy.spatial.distance import cdist


def match_along_epipolar(ptsL: np.ndarray, 
                        imgL: np.ndarray, 
                        imgR: np.ndarray,
                        K1: np.ndarray, 
                        R1: np.ndarray, 
                        t1: np.ndarray,
                        K2: np.ndarray, 
                        R2: np.ndarray, 
                        t2: np.ndarray,
                        max_disparity: int = 200,
                        window_size: int = 15) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    沿极线进行导线像素匹配
    
    参数:
        ptsL: 左图像中的导线像素坐标 (N, 2)
        imgL: 左图像
        imgR: 右图像  
        K1, R1, t1: 左相机内参、旋转矩阵、平移向量
        K2, R2, t2: 右相机内参、旋转矩阵、平移向量
        max_disparity: 最大视差值
        window_size: 匹配窗口大小
        
    返回:
        matches: 匹配点对列表 [(pL, pR), ...]
    """
    if len(ptsL) == 0:
        return []
    
    # 计算基础矩阵F
    F = compute_fundamental_matrix(K1, R1, t1, K2, R2, t2)
    
    # 计算极线
    epilines = cv2.computeCorrespondEpilines(ptsL.reshape(-1, 1, 2), 1, F)
    epilines = epilines.reshape(-1, 3)
    
    matches = []
    
    for i, (ptL, epiline) in enumerate(zip(ptsL, epilines)):
        # 在右图像中沿极线搜索匹配点
        ptR = search_along_epiline(ptL, epiline, imgL, imgR, 
                                 max_disparity, window_size)
        
        if ptR is not None:
            # 验证匹配质量
            if verify_match(ptL, ptR, imgL, imgR, window_size):
                matches.append((ptL, ptR))
    
    return matches


def compute_fundamental_matrix(K1: np.ndarray, R1: np.ndarray, t1: np.ndarray,
                             K2: np.ndarray, R2: np.ndarray, t2: np.ndarray) -> np.ndarray:
    """
    计算基础矩阵F
    
    参数:
        K1, R1, t1: 左相机参数
        K2, R2, t2: 右相机参数
        
    返回:
        F: 基础矩阵 (3x3)
    """
    # 计算投影矩阵
    P1 = K1 @ np.hstack([R1, t1.reshape(3, 1)])
    P2 = K2 @ np.hstack([R2, t2.reshape(3, 1)])
    
    # 计算相机中心
    C1 = -R1.T @ t1
    C2 = -R2.T @ t2
    
    # 计算基础矩阵
    # P2: 右相机投影矩阵
    # C1: 左相机中心
    # e2: 右图像中的极点
    # e2_skew: 右图像中的极点对应的反对称矩阵
    e2 = P2 @ np.hstack([C1, 1])  # 右图像中的极点
    e2_skew = np.array([[0, -e2[2], e2[1]],
                       [e2[2], 0, -e2[0]],
                       [-e2[1], e2[0], 0]])
    
    # 计算伪逆
    P1_pinv = np.linalg.pinv(P1)
    
    F = e2_skew @ P2 @ P1_pinv
    
    return F


def search_along_epiline(ptL: np.ndarray, 
                        epiline: np.ndarray,
                        imgL: np.ndarray, 
                        imgR: np.ndarray,
                        max_disparity: int,
                        window_size: int) -> Optional[np.ndarray]:
    """
    沿极线搜索最佳匹配点
    
    参数:
        ptL: 左图像中的点
        epiline: 对应的极线 [a, b, c] (ax + by + c = 0)
        imgL: 左图像
        imgR: 右图像
        max_disparity: 最大视差
        window_size: 匹配窗口大小
        
    返回:
        ptR: 右图像中的匹配点，如果未找到则返回None
    """
    h, w = imgR.shape[:2]
    
    # 计算极线与图像边界的交点
    intersections = []
    
    # 与左边界 (x=0) 的交点
    if abs(epiline[1]) > 1e-6:
        y = -epiline[2] / epiline[1]
        if 0 <= y < h:
            intersections.append((0, y))
    
    # 与右边界 (x=w-1) 的交点
    if abs(epiline[1]) > 1e-6:
        y = -(epiline[0] * (w-1) + epiline[2]) / epiline[1]
        if 0 <= y < h:
            intersections.append((w-1, y))
    
    # 与上边界 (y=0) 的交点
    if abs(epiline[0]) > 1e-6:
        x = -epiline[2] / epiline[0]
        if 0 <= x < w:
            intersections.append((x, 0))
    
    # 与下边界 (y=h-1) 的交点
    if abs(epiline[0]) > 1e-6:
        x = -(epiline[1] * (h-1) + epiline[2]) / epiline[0]
        if 0 <= x < w:
            intersections.append((x, h-1))
    
    if len(intersections) < 2:
        return None
    
    # 选择极线在图像内的部分
    x_min = max(0, min(intersections[0][0], intersections[1][0]))
    x_max = min(w-1, max(intersections[0][0], intersections[1][0]))
    
    # 沿极线采样点
    search_points = []
    for x in range(int(x_min), int(x_max) + 1):
        if abs(epiline[1]) > 1e-6:
            y = -(epiline[0] * x + epiline[2]) / epiline[1]
            if 0 <= y < h:
                search_points.append((x, y))
    
    if len(search_points) == 0:
        return None
    
    # 计算左图像中的模板
    template = extract_template(imgL, ptL, window_size)
    if template is None:
        return None
    
    # 在右图像中搜索最佳匹配
    best_match = None
    best_score = float('inf')
    
    for ptR in search_points:
        # 检查视差约束
        disparity = abs(ptL[0] - ptR[0])
        if disparity > max_disparity:
            continue
        
        # 提取右图像中的窗口
        candidate = extract_template(imgR, ptR, window_size)
        if candidate is None:
            continue
        
        # 计算匹配分数 (使用归一化互相关)
        score = compute_ncc_score(template, candidate)
        
        if score < best_score:
            best_score = score
            best_match = np.array(ptR)
    
    # 如果匹配分数足够好，返回最佳匹配点
    if best_match is not None and best_score < 0.8:  # 阈值可调
        return best_match
    
    return None


def extract_template(img: np.ndarray, 
                    center: np.ndarray, 
                    window_size: int) -> Optional[np.ndarray]:
    """
    从图像中提取模板窗口
    
    参数:
        img: 输入图像
        center: 中心点坐标
        window_size: 窗口大小
        
    返回:
        template: 提取的模板，如果超出边界则返回None
    """
    h, w = img.shape[:2]
    half_size = window_size // 2
    
    x1 = int(center[0] - half_size)
    y1 = int(center[1] - half_size)
    x2 = int(center[0] + half_size + 1)
    y2 = int(center[1] + half_size + 1)
    
    # 检查边界
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        return None
    
    return img[y1:y2, x1:x2]


def compute_ncc_score(template1: np.ndarray, template2: np.ndarray) -> float:
    """
    计算归一化互相关分数
    
    参数:
        template1: 模板1
        template2: 模板2
        
    返回:
        score: NCC分数 (越小越好)
    """
    # 转换为浮点数
    t1 = template1.astype(np.float32)
    t2 = template2.astype(np.float32)
    
    # 归一化
    t1 = (t1 - np.mean(t1)) / (np.std(t1) + 1e-8)
    t2 = (t2 - np.mean(t2)) / (np.std(t2) + 1e-8)
    
    # 计算相关系数
    correlation = np.sum(t1 * t2) / (t1.size)
    
    # 返回1 - correlation作为距离分数
    return 1.0 - correlation


def verify_match(ptL: np.ndarray, 
                ptR: np.ndarray,
                imgL: np.ndarray, 
                imgR: np.ndarray,
                window_size: int) -> bool:
    """
    验证匹配点的质量
    
    参数:
        ptL: 左图像中的点
        ptR: 右图像中的点
        imgL: 左图像
        imgR: 右图像
        window_size: 窗口大小
        
    返回:
        is_valid: 是否为有效匹配
    """
    # 提取模板
    templateL = extract_template(imgL, ptL, window_size)
    templateR = extract_template(imgR, ptR, window_size)
    
    if templateL is None or templateR is None:
        return False
    
    # 计算NCC分数
    ncc_score = compute_ncc_score(templateL, templateR)
    
    # 检查视差合理性
    disparity = abs(ptL[0] - ptR[0])
    
    # 验证条件
    return ncc_score < 0.5 and disparity > 0 and disparity < 200


def filter_matches_by_sampson_error(matches: List[Tuple[np.ndarray, np.ndarray]],
                                  F: np.ndarray,
                                  threshold: float = 1.0) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    使用Sampson误差过滤匹配点
    
    参数:
        matches: 匹配点对列表
        F: 基础矩阵
        threshold: Sampson误差阈值
        
    返回:
        filtered_matches: 过滤后的匹配点对
    """
    if len(matches) == 0:
        return matches
    
    filtered_matches = []
    
    for ptL, ptR in matches:
        # 计算Sampson误差
        error = compute_sampson_error(ptL, ptR, F)
        
        if error < threshold:
            filtered_matches.append((ptL, ptR))
    
    return filtered_matches


def filter_matches_by_right_uniqueness(matches: List[Tuple[np.ndarray, np.ndarray]],
                                       F: np.ndarray,
                                       bin_tolerance_px: float = 1.0) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    右图唯一性过滤：如果多个左点匹配到右图近同一列（x 接近），
    仅保留 Sampson 误差最小的那个，抑制“竖直条带”式错误匹配。

    参数:
        matches: [(ptL, ptR), ...]
        F: 基础矩阵（用于计算 Sampson 误差）
        bin_tolerance_px: 将右图 x 轴按该像素宽度分桶

    返回:
        过滤后的匹配
    """
    if len(matches) <= 1:
        return matches

    # 分桶键：右图 x / 宽度
    buckets = {}
    for ptL, ptR in matches:
        key = int(round(ptR[0] / max(1e-6, bin_tolerance_px)))
        if key not in buckets:
            buckets[key] = []
        buckets[key].append((ptL, ptR))

    kept: List[Tuple[np.ndarray, np.ndarray]] = []
    for key, pairs in buckets.items():
        if len(pairs) == 1:
            kept.append(pairs[0])
            continue
        # 选择 Sampson 误差最小的匹配
        best_pair = None
        best_err = float('inf')
        for ptL, ptR in pairs:
            err = compute_sampson_error(ptL, ptR, F)
            if err < best_err:
                best_err = err
                best_pair = (ptL, ptR)
        if best_pair is not None:
            kept.append(best_pair)

    return kept


def compute_sampson_error(pt1: np.ndarray, pt2: np.ndarray, F: np.ndarray) -> float:
    """
    计算Sampson误差
    
    参数:
        pt1: 第一个点
        pt2: 第二个点
        F: 基础矩阵
        
    返回:
        error: Sampson误差
    """
    # 转换为齐次坐标
    p1 = np.array([pt1[0], pt1[1], 1.0])
    p2 = np.array([pt2[0], pt2[1], 1.0])
    
    # 计算极线
    l2 = F @ p1  # 右图像中的极线
    l1 = F.T @ p2  # 左图像中的极线
    
    # 计算点到极线的距离
    d1 = abs(l1[0] * p1[0] + l1[1] * p1[1] + l1[2]) / np.sqrt(l1[0]**2 + l1[1]**2)
    d2 = abs(l2[0] * p2[0] + l2[1] * p2[1] + l2[2]) / np.sqrt(l2[0]**2 + l2[1]**2)
    
    # Sampson误差
    error = (d1**2 + d2**2) / (l1[0]**2 + l1[1]**2 + l2[0]**2 + l2[1]**2)
    
    return error


def visualize_matches(imgL: np.ndarray, 
                     imgR: np.ndarray,
                     matches: List[Tuple[np.ndarray, np.ndarray]],
                     title: str = "双目匹配结果") -> np.ndarray:
    """
    可视化匹配结果
    
    参数:
        imgL: 左图像
        imgR: 右图像
        matches: 匹配点对列表
        title: 图像标题
        
    返回:
        vis_img: 可视化图像
    """
    h1, w1 = imgL.shape[:2]
    h2, w2 = imgR.shape[:2]
    
    # 创建拼接图像
    vis_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    
    # 放置左右图像
    if len(imgL.shape) == 2:
        vis_img[:h1, :w1] = cv2.cvtColor(imgL, cv2.COLOR_GRAY2BGR)
    else:
        vis_img[:h1, :w1] = imgL
    
    if len(imgR.shape) == 2:
        vis_img[:h2, w1:] = cv2.cvtColor(imgR, cv2.COLOR_GRAY2BGR)
    else:
        vis_img[:h2, w1:] = imgR
    
    # 绘制匹配线
    for i, (ptL, ptR) in enumerate(matches):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        
        # 绘制点
        cv2.circle(vis_img, (int(ptL[0]), int(ptL[1])), 3, color, -1)
        cv2.circle(vis_img, (int(ptR[0] + w1), int(ptR[1])), 3, color, -1)
        
        # 绘制连线
        cv2.line(vis_img, (int(ptL[0]), int(ptL[1])), 
                (int(ptR[0] + w1), int(ptR[1])), color, 1)
    
    return vis_img


if __name__ == "__main__":
    # 测试代码
    import matplotlib.pyplot as plt
    
    # 创建测试图像
    imgL = np.zeros((480, 640), dtype=np.uint8)
    imgR = np.zeros((480, 640), dtype=np.uint8)
    
    # 在左图像中画线
    cv2.line(imgL, (100, 200), (500, 300), 255, 2)
    # 在右图像中画对应的线（有视差）
    cv2.line(imgR, (80, 200), (480, 300), 255, 2)
    
    # 测试点
    ptsL = np.array([[200, 250], [300, 270], [400, 290]])
    
    # 简单的相机参数
    K1 = K2 = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float32)
    R1 = R2 = np.eye(3, dtype=np.float32)
    t1 = np.array([0, 0, 0], dtype=np.float32)
    t2 = np.array([-50, 0, 0], dtype=np.float32)
    
    # 进行匹配
    matches = match_along_epipolar(ptsL, imgL, imgR, K1, R1, t1, K2, R2, t2)
    
    print(f"找到 {len(matches)} 个匹配点对")
    
    # 可视化结果
    vis_img = visualize_matches(imgL, imgR, matches)
    
    plt.figure(figsize=(15, 6))
    plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    plt.title('双目匹配结果')
    plt.axis('off')

    plt.show()
