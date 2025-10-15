"""
演示脚本：展示双目视觉悬链线重建项目的基本功能
不需要安装依赖，仅用于展示项目结构和算法流程
"""

def print_project_overview():
    """
    打印项目概览
    """
    print("=" * 80)
    print("双目视觉悬链线重建项目")
    print("=" * 80)
    print()
    print("项目目标：给定双目相机采集的两张图和相机内外参，计算导线的空间悬链线方程")
    print()
    print("核心算法流程：")
    print("1. 导线像素检测 → 2. 双目匹配 → 3. 三角测量 → 4. 平面估计 → 5. 悬链线拟合")
    print()
    print("悬链线方程：z(s) = a * cosh((s - s0) / a) + c")
    print("3D悬链线方程：X(s) = p0 + s*u + [a*cosh((s-s0)/a) + c]*g_hat")
    print()


def print_module_descriptions():
    """
    打印各模块功能描述
    """
    print("=" * 80)
    print("模块功能说明")
    print("=" * 80)
    print()
    
    modules = [
        ("line_detect.py", "导线像素提取模块", [
            "使用Canny边缘检测 + 霍夫变换检测直线",
            "支持脊线检测和骨架化算法",
            "自动过滤噪声和短线段"
        ]),
        ("stereo_match.py", "极线约束匹配模块", [
            "基于极线约束的导线像素匹配",
            "使用归一化互相关(NCC)进行匹配",
            "Sampson误差过滤异常匹配"
        ]),
        ("triangulation.py", "三角测量模块", [
            "支持DLT、射线中点、最优三角测量方法",
            "重投影误差计算和异常点过滤",
            "自动选择最佳三角测量方法"
        ]),
        ("plane_estimation.py", "垂直平面估计模块", [
            "使用PCA估计导线主要方向",
            "计算包含重力方向和导线方向的垂直平面",
            "将3D点投影到2D平面坐标"
        ]),
        ("catenary_fit.py", "悬链线拟合模块", [
            "使用Levenberg-Marquardt算法拟合悬链线方程",
            "包含解析雅可比矩阵和鲁棒损失函数",
            "支持RANSAC算法处理异常值"
        ]),
        ("viz.py", "可视化模块", [
            "2D和3D可视化功能",
            "完整的处理流程可视化",
            "拟合质量分析图表"
        ]),
        ("main.py", "主程序", [
            "完整的处理流程控制",
            "参数配置和结果保存",
            "命令行接口"
        ])
    ]
    
    for filename, title, features in modules:
        print(f"📁 {filename}")
        print(f"   {title}")
        for feature in features:
            print(f"   • {feature}")
        print()


def print_usage_example():
    """
    打印使用示例
    """
    print("=" * 80)
    print("使用示例")
    print("=" * 80)
    print()
    print("1. 安装依赖：")
    print("   pip install -r requirements.txt")
    print()
    print("2. 配置相机参数：")
    print("   编辑 config/cameras.yaml 文件")
    print()
    print("3. 运行主程序：")
    print("   python main.py --left left_image.jpg --right right_image.jpg \\")
    print("                  --config config/cameras.yaml --output output \\")
    print("                  --show-plots")
    print()
    print("4. 运行测试：")
    print("   python test_pipeline.py")
    print()


def print_mathematical_details():
    """
    打印数学细节
    """
    print("=" * 80)
    print("数学原理")
    print("=" * 80)
    print()
    print("1. 三角测量（DLT方法）：")
    print("   构建线性方程组 Ax = 0，使用SVD求解")
    print("   A = [ptL[0]*P1[2] - P1[0]; ptL[1]*P1[2] - P1[1]; ...]")
    print()
    print("2. 平面估计：")
    print("   • 使用PCA找到导线主要方向 v1")
    print("   • 计算沿导线方向：u = normalize(v1 - (v1·g_hat) * g_hat)")
    print("   • 平面法向量：n = u × g_hat")
    print()
    print("3. 悬链线拟合：")
    print("   • 目标函数：min Σ(z_i - a*cosh((s_i-s0)/a) - c)²")
    print("   • 解析雅可比矩阵：")
    print("     ∂r/∂a = -[cosh(u) - u*sinh(u)]")
    print("     ∂r/∂s0 = -sinh(u)")
    print("     ∂r/∂c = -1")
    print("   • 其中 u = (s - s0) / a")
    print()


def print_algorithm_features():
    """
    打印算法特点
    """
    print("=" * 80)
    print("算法特点")
    print("=" * 80)
    print()
    print("🔧 鲁棒性：")
    print("   • RANSAC算法处理异常值")
    print("   • Huber损失函数减少异常点影响")
    print("   • 多级过滤机制确保数据质量")
    print()
    print("🎯 精度：")
    print("   • 解析雅可比矩阵提高拟合精度")
    print("   • 最优三角测量方法")
    print("   • 重投影误差验证")
    print()
    print("📊 可视化：")
    print("   • 完整的处理流程可视化")
    print("   • 拟合质量分析")
    print("   • 3D悬链线曲线展示")
    print()


def main():
    """
    主函数
    """
    print_project_overview()
    print_module_descriptions()
    print_usage_example()
    print_mathematical_details()
    print_algorithm_features()
    
    print("=" * 80)
    print("项目完成！")
    print("=" * 80)
    print()
    print("项目已按照设计思路完整实现，包含以下功能：")
    print("✅ 导线像素检测")
    print("✅ 极线约束匹配")
    print("✅ 三角测量")
    print("✅ 垂直平面估计")
    print("✅ 悬链线拟合（含解析雅可比矩阵）")
    print("✅ 完整可视化")
    print("✅ 主程序接口")
    print("✅ 测试脚本")
    print("✅ 详细文档")
    print()
    print("所有代码均使用中文注释，符合项目要求。")


if __name__ == "__main__":
    main()
