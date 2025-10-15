"""
测试可视化功能
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2

def test_matplotlib_backend():
    """
    测试matplotlib后端和显示功能
    """
    import matplotlib
    print("=" * 60)
    print("测试matplotlib可视化功能")
    print("=" * 60)
    
    # 检查当前后端
    print(f"当前matplotlib后端: {matplotlib.get_backend()}")
    
    # 检查可用的后端
    try:
        import matplotlib.backends.backend_tkagg
        print("TkAgg后端可用")
    except ImportError:
        print("TkAgg后端不可用")
    
    try:
        import matplotlib.backends.backend_qt5agg
        print("Qt5Agg后端可用")
    except ImportError:
        print("Qt5Agg后端不可用")
    
    # 尝试不同的后端
    backends_to_try = ['TkAgg', 'Qt5Agg', 'Agg']
    
    for backend in backends_to_try:
        try:
            print(f"\n尝试使用后端: {backend}")
            matplotlib.use(backend)
            plt.figure(figsize=(8, 6))
            plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
            plt.title(f'测试图表 - 后端: {backend}')
            plt.xlabel('X轴')
            plt.ylabel('Y轴')
            plt.grid(True)
            
            # 尝试显示图表
            try:
                plt.show(block=False)
                plt.pause(2)  # 暂停2秒
                plt.close()
                print(f"✅ 后端 {backend} 工作正常")
                return backend
            except Exception as e:
                print(f"❌ 后端 {backend} 显示失败: {e}")
                plt.close()
                
        except Exception as e:
            print(f"❌ 后端 {backend} 初始化失败: {e}")
    
    print("\n⚠️  所有交互式后端都失败，将使用非交互式后端")
    matplotlib.use('Agg')
    return 'Agg'

def create_simple_visualization():
    """
    创建简单的可视化测试
    """
    print("\n创建简单可视化测试...")
    
    # 创建测试数据
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 第一个子图
    ax1.plot(x, y1, 'b-', label='sin(x)')
    ax1.plot(x, y2, 'r-', label='cos(x)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('三角函数')
    ax1.legend()
    ax1.grid(True)
    
    # 第二个子图 - 散点图
    np.random.seed(42)
    x_scatter = np.random.randn(50)
    y_scatter = 2 * x_scatter + np.random.randn(50)
    ax2.scatter(x_scatter, y_scatter, alpha=0.6)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('散点图')
    ax2.grid(True)
    
    plt.tight_layout()
    
    # 尝试显示
    try:
        plt.show(block=False)
        plt.pause(3)
        print("✅ 图表显示成功")
        return True
    except Exception as e:
        print(f"❌ 图表显示失败: {e}")
        return False
    finally:
        plt.close()

def test_image_display():
    """
    测试图像显示功能
    """
    print("\n测试图像显示功能...")
    
    # 创建测试图像
    img = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
    
    # 添加一些几何形状
    cv2.rectangle(img, (50, 50), (150, 100), (255, 0, 0), -1)
    cv2.circle(img, (200, 150), 30, (0, 255, 0), -1)
    cv2.line(img, (0, 0), (300, 200), (0, 0, 255), 3)
    
    # 显示图像
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('测试图像')
    plt.axis('off')
    
    try:
        plt.show(block=False)
        plt.pause(2)
        print("✅ 图像显示成功")
        return True
    except Exception as e:
        print(f"❌ 图像显示失败: {e}")
        return False
    finally:
        plt.close()

def test_3d_plot():
    """
    测试3D绘图功能
    """
    print("\n测试3D绘图功能...")
    
    try:
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 创建3D数据
        x = np.linspace(-5, 5, 50)
        y = np.linspace(-5, 5, 50)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(np.sqrt(X**2 + Y**2))
        
        # 绘制3D表面
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D表面图')
        
        try:
            plt.show(block=False)
            plt.pause(3)
            print("✅ 3D图表显示成功")
            return True
        except Exception as e:
            print(f"❌ 3D图表显示失败: {e}")
            return False
        finally:
            plt.close()
            
    except ImportError as e:
        print(f"❌ 3D绘图模块导入失败: {e}")
        return False

def main():
    """
    主测试函数
    """
    print("开始可视化测试...")
    
    # 测试后端
    backend = test_matplotlib_backend()
    
    # 测试基本绘图
    plot_success = create_simple_visualization()
    
    # 测试图像显示
    image_success = test_image_display()
    
    # 测试3D绘图
    plot3d_success = test_3d_plot()
    
    print("\n" + "=" * 60)
    print("测试结果总结:")
    print("=" * 60)
    print(f"后端: {backend}")
    print(f"基本绘图: {'✅ 成功' if plot_success else '❌ 失败'}")
    print(f"图像显示: {'✅ 成功' if image_success else '❌ 失败'}")
    print(f"3D绘图: {'✅ 成功' if plot3d_success else '❌ 失败'}")
    
    if not (plot_success or image_success or plot3d_success):
        print("\n⚠️  可视化功能可能无法正常显示")
        print("建议:")
        print("1. 检查是否在图形界面环境中运行")
        print("2. 尝试安装图形界面支持: sudo apt-get install python3-tk")
        print("3. 使用 --save-plots 参数保存图表到文件")
        print("4. 在Jupyter notebook中运行")
    else:
        print("\n🎉 可视化功能正常！")

if __name__ == "__main__":
    main()
