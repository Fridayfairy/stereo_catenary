"""
测试系统中可用的中文字体
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

def list_available_fonts():
    """
    列出系统中可用的字体
    """
    print("=" * 60)
    print("系统中可用的字体")
    print("=" * 60)
    
    # 获取所有字体
    fonts = [f.name for f in fm.fontManager.ttflist]
    fonts = sorted(set(fonts))  # 去重并排序
    
    print(f"总共找到 {len(fonts)} 个字体:")
    
    # 查找可能支持中文的字体
    chinese_keywords = ['chinese', 'cjk', 'han', 'noto', 'source', 'sim', 'microsoft', 'yahei', 'wenquanyi']
    chinese_fonts = []
    
    for font in fonts:
        font_lower = font.lower()
        if any(keyword in font_lower for keyword in chinese_keywords):
            chinese_fonts.append(font)
    
    print(f"\n可能支持中文的字体 ({len(chinese_fonts)} 个):")
    for font in chinese_fonts:
        print(f"  - {font}")
    
    # 显示前20个字体
    print(f"\n前20个字体:")
    for i, font in enumerate(fonts[:20]):
        print(f"  {i+1:2d}. {font}")
    
    if len(fonts) > 20:
        print(f"  ... 还有 {len(fonts) - 20} 个字体")
    
    return chinese_fonts

def test_chinese_display():
    """
    测试中文字符显示
    """
    print("\n" + "=" * 60)
    print("测试中文字符显示")
    print("=" * 60)
    
    # 测试文本
    test_texts = [
        "导线检测结果",
        "双目匹配",
        "三角测量",
        "平面估计",
        "悬链线拟合",
        "拟合质量",
        "参数",
        "坐标"
    ]
    
    # 获取可能的中文字体
    chinese_fonts = list_available_fonts()
    
    if not chinese_fonts:
        print("未找到中文字体，尝试使用默认字体")
        chinese_fonts = ['DejaVu Sans']
    
    # 测试每个字体
    for font_name in chinese_fonts[:5]:  # 只测试前5个字体
        try:
            print(f"\n测试字体: {font_name}")
            
            # 设置字体
            plt.rcParams['font.sans-serif'] = [font_name]
            plt.rcParams['axes.unicode_minus'] = False
            
            # 创建测试图
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 绘制测试文本
            y_positions = np.linspace(0.9, 0.1, len(test_texts))
            for i, text in enumerate(test_texts):
                ax.text(0.1, y_positions[i], text, fontsize=14, 
                       transform=ax.transAxes, fontfamily='sans-serif')
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(f'中文字体测试 - {font_name}', fontsize=16)
            ax.axis('off')
            
            # 保存测试图片
            output_path = f"output/font_test_{font_name.replace(' ', '_')}.png"
            import os
            os.makedirs('output', exist_ok=True)
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"  测试图片已保存: {output_path}")
            
            plt.close(fig)
            
        except Exception as e:
            print(f"  字体 {font_name} 测试失败: {e}")

def create_fallback_solution():
    """
    创建备用解决方案
    """
    print("\n" + "=" * 60)
    print("创建备用解决方案")
    print("=" * 60)
    
    # 创建使用英文标签的测试图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 英文标签
    english_labels = [
        "Wire Detection Results",
        "Stereo Matching", 
        "Triangulation",
        "Plane Estimation",
        "Catenary Fitting",
        "Fitting Quality",
        "Parameters",
        "Coordinates"
    ]
    
    y_positions = np.linspace(0.9, 0.1, len(english_labels))
    for i, label in enumerate(english_labels):
        ax.text(0.1, y_positions[i], label, fontsize=14, 
               transform=ax.transAxes)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('English Labels Test', fontsize=16)
    ax.axis('off')
    
    # 保存英文版本
    output_path = "output/english_labels_test.png"
    import os
    os.makedirs('output', exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"英文标签测试图片已保存: {output_path}")
    
    plt.close(fig)

def main():
    """
    主函数
    """
    print("字体测试开始...")
    
    # 设置matplotlib后端
    matplotlib.use('Agg')
    
    # 列出可用字体
    chinese_fonts = list_available_fonts()
    
    # 测试中文字符显示
    test_chinese_display()
    
    # 创建备用解决方案
    create_fallback_solution()
    
    print("\n" + "=" * 60)
    print("字体测试完成")
    print("=" * 60)
    
    if chinese_fonts:
        print("✅ 找到中文字体，可以正常显示中文")
        print("建议使用以下字体:")
        for font in chinese_fonts[:3]:
            print(f"  - {font}")
    else:
        print("⚠️  未找到中文字体，建议:")
        print("1. 安装中文字体包: sudo apt-get install fonts-wqy-microhei")
        print("2. 或者使用英文标签")
        print("3. 或者手动指定字体文件路径")

if __name__ == "__main__":
    main()
