"""
悬链线拟合模块
使用Levenberg-Marquardt算法拟合悬链线方程，包含解析雅可比矩阵和鲁棒损失函数
"""

import numpy as np
from typing import Tuple, Optional, Callable
from scipy.optimize import least_squares
from scipy.linalg import inv
import warnings


def fit_catenary(s: np.ndarray, 
                z: np.ndarray,
                method: str = 'lm_robust',
                initial_params: Optional[np.ndarray] = None,
                loss_function: str = 'huber',
                huber_threshold: float = 1.0) -> Tuple[float, float, float]:
    """
    拟合悬链线方程 z(s) = a * cosh((s - s0) / a) + c
    
    参数:
        s: 沿导线方向的坐标 (N,)
        z: 重力方向的坐标 (N,)
        method: 拟合方法 ('lm_robust', 'lm_standard', 'parabola_init')
        initial_params: 初始参数 [a, s0, c]，如果为None则自动估计
        loss_function: 损失函数类型 ('huber', 'soft_l1', 'linear')
        huber_threshold: Huber损失函数的阈值
        
    返回:
        a: 悬链线参数a
        s0: 悬链线参数s0
        c: 悬链线参数c
    """
    if len(s) != len(z):
        raise ValueError("s和z的长度必须相等")
    
    if len(s) < 3:
        raise ValueError("至少需要3个点来拟合悬链线")
    
    # 获取初始参数
    if initial_params is None:
        a_init, s0_init, c_init = estimate_initial_params(s, z)
    else:
        a_init, s0_init, c_init = initial_params
    
    # 检查初始参数的合理性
    if a_init <= 0:
        a_init = 1.0
    if np.isnan(a_init) or np.isnan(s0_init) or np.isnan(c_init):
        a_init, s0_init, c_init = 1.0, 0.0, np.mean(z)
    
    initial_params = np.array([a_init, s0_init, c_init])
    
    if method == 'lm_robust':
        return _fit_with_lm_robust(s, z, initial_params, loss_function, huber_threshold)
    elif method == 'lm_standard':
        return _fit_with_lm_standard(s, z, initial_params)
    elif method == 'parabola_init':
        return _fit_with_parabola_init(s, z)
    else:
        raise ValueError(f"不支持的拟合方法: {method}")


def estimate_initial_params(s: np.ndarray, z: np.ndarray) -> Tuple[float, float, float]:
    """
    使用抛物线拟合估计悬链线的初始参数
    
    参数:
        s: 沿导线方向的坐标
        z: 重力方向的坐标
        
    返回:
        a_init: 初始参数a
        s0_init: 初始参数s0  
        c_init: 初始参数c
    """
    # 使用抛物线 z = α*s² + β*s + γ 作为初始估计
    # 对于小角度，cosh(x) ≈ 1 + x²/2
    # 所以 a*cosh((s-s0)/a) + c ≈ a + (s-s0)²/(2a) + c
    # 即 z ≈ (1/(2a))*s² - (s0/a)*s + (a + s0²/(2a) + c)
    
    try:
        # 拟合抛物线
        A = np.column_stack([s**2, s, np.ones_like(s)])
        coeffs = np.linalg.lstsq(A, z, rcond=None)[0]
        alpha, beta, gamma = coeffs
        
        # 从抛物线参数推导悬链线参数
        if abs(alpha) > 1e-8:
            a_init = 1.0 / (2.0 * alpha)
            s0_init = -beta / (2.0 * alpha)
            c_init = gamma - a_init
        else:
            # 如果抛物线拟合失败，使用默认值
            a_init = 1.0
            s0_init = np.mean(s)
            c_init = np.mean(z) - a_init
        
        # 确保参数合理性
        a_init = max(0.1, abs(a_init))  # a必须为正
        s0_init = np.clip(s0_init, np.min(s), np.max(s))  # s0在数据范围内
        
    except:
        # 如果拟合失败，使用简单的启发式方法
        a_init = 1.0
        s0_init = np.mean(s)
        c_init = np.mean(z) - a_init
    
    return a_init, s0_init, c_init


def _fit_with_lm_robust(s: np.ndarray, 
                       z: np.ndarray, 
                       initial_params: np.ndarray,
                       loss_function: str,
                       huber_threshold: float) -> Tuple[float, float, float]:
    """
    使用带鲁棒损失函数的Levenberg-Marquardt算法拟合
    """
    def residual_function(params):
        a, s0, c = params
        if a <= 0:
            return np.full_like(s, 1e6)  # 惩罚负的a值
        
        # 计算悬链线值
        u = (s - s0) / a
        z_pred = a * np.cosh(u) + c
        
        return z - z_pred
    
    def jacobian_function(params):
        a, s0, c = params
        if a <= 0:
            return np.zeros((len(s), 3))
        
        u = (s - s0) / a
        C = np.cosh(u)
        S = np.sinh(u)
        
        # 解析雅可比矩阵
        # ∂r/∂a = -[C - u*S]
        # ∂r/∂s0 = -S  
        # ∂r/∂c = -1
        
        jac = np.zeros((len(s), 3))
        jac[:, 0] = -(C - u * S)  # ∂r/∂a
        jac[:, 1] = -S            # ∂r/∂s0
        jac[:, 2] = -np.ones_like(s)  # ∂r/∂c
        
        return jac
    
    # 设置参数边界
    bounds = ([0.01, -np.inf, -np.inf], [np.inf, np.inf, np.inf])
    
    # 使用least_squares进行鲁棒拟合
    try:
        result = least_squares(
            residual_function, 
            initial_params,
            jac=jacobian_function,
            bounds=bounds,
            loss=loss_function,
            f_scale=huber_threshold,
            max_nfev=1000
        )
        
        if result.success:
            a, s0, c = result.x
        else:
            # 如果优化失败，使用初始参数
            a, s0, c = initial_params
            
    except Exception as e:
        warnings.warn(f"鲁棒拟合失败: {e}，使用初始参数")
        a, s0, c = initial_params
    
    return a, s0, c


def _fit_with_lm_standard(s: np.ndarray, 
                         z: np.ndarray, 
                         initial_params: np.ndarray) -> Tuple[float, float, float]:
    """
    使用标准Levenberg-Marquardt算法拟合
    """
    def residual_function(params):
        a, s0, c = params
        if a <= 0:
            return np.full_like(s, 1e6)
        
        u = (s - s0) / a
        z_pred = a * np.cosh(u) + c
        
        return z - z_pred
    
    def jacobian_function(params):
        a, s0, c = params
        if a <= 0:
            return np.zeros((len(s), 3))
        
        u = (s - s0) / a
        C = np.cosh(u)
        S = np.sinh(u)
        
        jac = np.zeros((len(s), 3))
        jac[:, 0] = -(C - u * S)
        jac[:, 1] = -S
        jac[:, 2] = -np.ones_like(s)
        
        return jac
    
    bounds = ([0.01, -np.inf, -np.inf], [np.inf, np.inf, np.inf])
    
    try:
        result = least_squares(
            residual_function, 
            initial_params,
            jac=jacobian_function,
            bounds=bounds,
            loss='linear',
            max_nfev=1000
        )
        
        if result.success:
            a, s0, c = result.x
        else:
            a, s0, c = initial_params
            
    except Exception as e:
        warnings.warn(f"标准拟合失败: {e}，使用初始参数")
        a, s0, c = initial_params
    
    return a, s0, c


def _fit_with_parabola_init(s: np.ndarray, z: np.ndarray) -> Tuple[float, float, float]:
    """
    使用抛物线拟合作为最终结果（适用于小角度情况）
    """
    a_init, s0_init, c_init = estimate_initial_params(s, z)
    return a_init, s0_init, c_init


def compose_catenary_3d(p0: np.ndarray, 
                       u: np.ndarray, 
                       g_hat: np.ndarray,
                       a: float, 
                       s0: float, 
                       c: float) -> Callable[[np.ndarray], np.ndarray]:
    """
    组合3D悬链线函数 X(s) = p0 + s*u + [a*cosh((s-s0)/a) + c]*g_hat
    
    参数:
        p0: 平面参考点 (3,)
        u: 沿导线方向的单位向量 (3,)
        g_hat: 重力方向单位向量 (3,)
        a: 悬链线参数a
        s0: 悬链线参数s0
        c: 悬链线参数c
        
    返回:
        catenary_3d: 3D悬链线函数，输入s坐标，输出3D坐标
    """
    def catenary_3d(s_values):
        """
        计算3D悬链线坐标
        
        参数:
            s_values: s坐标值（标量或数组）
            
        返回:
            X: 3D坐标 (3,) 或 (N, 3)
        """
        s_values = np.asarray(s_values)
        is_scalar = s_values.ndim == 0
        s_values = np.atleast_1d(s_values)
        
        # 计算悬链线高度
        u_cat = (s_values - s0) / a
        z_cat = a * np.cosh(u_cat) + c
        
        # 计算3D坐标
        X = p0.reshape(1, 3) + s_values.reshape(-1, 1) * u.reshape(1, 3) + z_cat.reshape(-1, 1) * g_hat.reshape(1, 3)
        
        if is_scalar:
            return X[0]
        else:
            return X
    
    return catenary_3d


def compute_fitting_quality(s: np.ndarray, 
                          z: np.ndarray, 
                          a: float, 
                          s0: float, 
                          c: float) -> dict:
    """
    计算拟合质量指标
    
    参数:
        s: 输入s坐标
        z: 输入z坐标
        a, s0, c: 拟合参数
        
    返回:
        quality: 包含各种质量指标的字典
    """
    # 计算预测值
    u = (s - s0) / a
    z_pred = a * np.cosh(u) + c
    
    # 计算残差
    residuals = z - z_pred
    
    # 计算各种质量指标
    mse = np.mean(residuals**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residuals))
    
    # 计算R²
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((z - np.mean(z))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # 计算调整后的R²
    n = len(s)
    p = 3  # 参数个数
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1) if n > p + 1 else r_squared
    
    # 计算参数的不确定性（使用雅可比矩阵）
    try:
        u = (s - s0) / a
        C = np.cosh(u)
        S = np.sinh(u)
        
        jac = np.zeros((len(s), 3))
        jac[:, 0] = -(C - u * S)
        jac[:, 1] = -S
        jac[:, 2] = -np.ones_like(s)
        
        # 计算协方差矩阵
        cov_matrix = inv(jac.T @ jac) * mse
        param_std = np.sqrt(np.diag(cov_matrix))
        
        a_std, s0_std, c_std = param_std
    except:
        a_std = s0_std = c_std = np.nan
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r_squared': r_squared,
        'adj_r_squared': adj_r_squared,
        'a_std': a_std,
        's0_std': s0_std,
        'c_std': c_std,
        'n_points': n,
        'residuals': residuals
    }


def ransac_catenary_fit(s: np.ndarray, 
                       z: np.ndarray,
                       min_samples: int = 3,
                       max_trials: int = 100,
                       residual_threshold: float = 0.1) -> Tuple[float, float, float, np.ndarray]:
    """
    使用RANSAC算法进行鲁棒悬链线拟合
    
    参数:
        s: 输入s坐标
        z: 输入z坐标
        min_samples: 最小样本数
        max_trials: 最大迭代次数
        residual_threshold: 残差阈值
        
    返回:
        a, s0, c: 拟合参数
        inliers: 内点掩码
    """
    if len(s) < min_samples:
        return 1.0, np.mean(s), np.mean(z), np.ones(len(s), dtype=bool)
    
    best_a, best_s0, best_c = 1.0, np.mean(s), np.mean(z)
    best_inliers = np.zeros(len(s), dtype=bool)
    best_inlier_count = 0
    
    for trial in range(max_trials):
        # 随机选择最小样本
        if len(s) >= min_samples:
            sample_indices = np.random.choice(len(s), min_samples, replace=False)
            s_sample = s[sample_indices]
            z_sample = z[sample_indices]
        else:
            continue
        
        try:
            # 拟合悬链线
            a, s0, c = fit_catenary(s_sample, z_sample, method='parabola_init')
            
            # 计算所有点的残差
            u = (s - s0) / a
            z_pred = a * np.cosh(u) + c
            residuals = np.abs(z - z_pred)
            
            # 确定内点
            inliers = residuals < residual_threshold
            inlier_count = np.sum(inliers)
            
            # 更新最佳结果
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_a, best_s0, best_c = a, s0, c
                best_inliers = inliers
                
        except:
            continue
    
    # 如果有足够的内点，使用所有内点重新拟合
    if best_inlier_count >= min_samples:
        try:
            s_inliers = s[best_inliers]
            z_inliers = z[best_inliers]
            best_a, best_s0, best_c = fit_catenary(s_inliers, z_inliers, method='lm_robust')
        except:
            pass
    
    return best_a, best_s0, best_c, best_inliers


if __name__ == "__main__":
    # 测试代码
    import matplotlib.pyplot as plt
    
    # 创建测试数据：悬链线 + 噪声 + 异常值
    s_true = np.linspace(-20, 20, 50)
    a_true, s0_true, c_true = 5.0, 0.0, 10.0
    z_true = a_true * np.cosh((s_true - s0_true) / a_true) + c_true
    
    # 添加噪声
    noise = np.random.normal(0, 0.5, len(s_true))
    z_noisy = z_true + noise
    
    # 添加异常值
    outlier_indices = np.random.choice(len(s_true), 5, replace=False)
    z_noisy[outlier_indices] += np.random.normal(0, 3, 5)
    
    print("测试悬链线拟合")
    print(f"真实参数: a={a_true}, s0={s0_true}, c={c_true}")
    
    # 标准拟合
    a_std, s0_std, c_std = fit_catenary(s_true, z_noisy, method='lm_standard')
    print(f"标准拟合: a={a_std:.3f}, s0={s0_std:.3f}, c={c_std:.3f}")
    
    # 鲁棒拟合
    a_rob, s0_rob, c_rob = fit_catenary(s_true, z_noisy, method='lm_robust')
    print(f"鲁棒拟合: a={a_rob:.3f}, s0={s0_rob:.3f}, c={c_rob:.3f}")
    
    # RANSAC拟合
    a_ransac, s0_ransac, c_ransac, inliers = ransac_catenary_fit(s_true, z_noisy)
    print(f"RANSAC拟合: a={a_ransac:.3f}, s0={s0_ransac:.3f}, c={c_ransac:.3f}")
    print(f"内点数量: {np.sum(inliers)}/{len(s_true)}")
    
    # 计算拟合质量
    quality_std = compute_fitting_quality(s_true, z_noisy, a_std, s0_std, c_std)
    quality_rob = compute_fitting_quality(s_true, z_noisy, a_rob, s0_rob, c_rob)
    quality_ransac = compute_fitting_quality(s_true[inliers], z_noisy[inliers], a_ransac, s0_ransac, c_ransac)
    
    print(f"\n拟合质量:")
    print(f"标准拟合 R²: {quality_std['r_squared']:.3f}, RMSE: {quality_std['rmse']:.3f}")
    print(f"鲁棒拟合 R²: {quality_rob['r_squared']:.3f}, RMSE: {quality_rob['rmse']:.3f}")
    print(f"RANSAC拟合 R²: {quality_ransac['r_squared']:.3f}, RMSE: {quality_ransac['rmse']:.3f}")
    
    # 可视化结果
    plt.figure(figsize=(15, 5))
    
    # 原始数据和真实曲线
    plt.subplot(1, 3, 1)
    plt.scatter(s_true, z_noisy, c='blue', alpha=0.6, s=20, label='带噪声数据')
    plt.scatter(s_true[outlier_indices], z_noisy[outlier_indices], c='red', s=50, label='异常值')
    plt.plot(s_true, z_true, 'g-', linewidth=2, label='真实悬链线')
    plt.xlabel('s')
    plt.ylabel('z')
    plt.title('原始数据')
    plt.legend()
    plt.grid(True)
    
    # 拟合结果对比
    plt.subplot(1, 3, 2)
    plt.scatter(s_true, z_noisy, c='blue', alpha=0.3, s=10, label='数据')
    
    # 计算拟合曲线
    s_plot = np.linspace(np.min(s_true), np.max(s_true), 100)
    
    z_std = a_std * np.cosh((s_plot - s0_std) / a_std) + c_std
    z_rob = a_rob * np.cosh((s_plot - s0_rob) / a_rob) + c_rob
    z_ransac = a_ransac * np.cosh((s_plot - s0_ransac) / a_ransac) + c_ransac
    
    plt.plot(s_plot, z_std, 'r-', linewidth=2, label=f'标准拟合 (R²={quality_std["r_squared"]:.3f})')
    plt.plot(s_plot, z_rob, 'orange', linewidth=2, label=f'鲁棒拟合 (R²={quality_rob["r_squared"]:.3f})')
    plt.plot(s_plot, z_ransac, 'purple', linewidth=2, label=f'RANSAC拟合 (R²={quality_ransac["r_squared"]:.3f})')
    plt.plot(s_plot, a_true * np.cosh((s_plot - s0_true) / a_true) + c_true, 'g--', linewidth=2, label='真实曲线')
    
    plt.xlabel('s')
    plt.ylabel('z')
    plt.title('拟合结果对比')
    plt.legend()
    plt.grid(True)
    
    # RANSAC内点
    plt.subplot(1, 3, 3)
    plt.scatter(s_true[inliers], z_noisy[inliers], c='green', alpha=0.6, s=20, label='内点')
    plt.scatter(s_true[~inliers], z_noisy[~inliers], c='red', alpha=0.6, s=20, label='外点')
    plt.plot(s_plot, z_ransac, 'purple', linewidth=2, label='RANSAC拟合')
    plt.xlabel('s')
    plt.ylabel('z')
    plt.title('RANSAC结果')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
