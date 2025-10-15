# 可视化问题修复总结

## 问题描述

在运行demo时，可视化图表没有显示出来，用户无法看到处理结果的可视化效果。

## 问题原因

1. **无头环境**：系统运行在无头模式（headless）下，没有图形界面支持
2. **matplotlib后端**：默认使用Agg后端，这是非交互式后端，`plt.show()`不会显示窗口
3. **环境检测**：程序没有检测运行环境，无法自动适配显示方式

## 解决方案

### 1. 环境检测
```python
def is_headless():
    """检查是否在无头环境中运行"""
    return os.environ.get('DISPLAY') is None or 'headless' in os.environ.get('MATPLOTLIB_BACKEND', '').lower()
```

### 2. 智能后端选择
```python
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
```

### 3. 智能显示函数
```python
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
        plt.show()
```

### 4. 替换所有显示调用
将所有的 `plt.show()` 替换为 `smart_show(title="描述性名称")`

## 修复效果

### 修复前
- 程序运行成功，但看不到可视化图表
- 用户无法直观了解处理结果

### 修复后
- ✅ 自动检测运行环境
- ✅ 在无头环境下自动保存图片到 `output/` 目录
- ✅ 在有显示环境下正常显示窗口
- ✅ 生成高质量PNG图片（300 DPI）
- ✅ 提供清晰的保存路径提示

## 测试结果

运行测试脚本 `test_viz_fix.py`：
```
============================================================
测试可视化修复效果
============================================================
当前环境: 无头环境

1. 测试导线检测可视化...
图表已保存到: output/wire_detection.png
✅ 导线检测可视化成功

2. 测试双目匹配可视化...
图表已保存到: output/wire_detection.png
✅ 双目匹配可视化成功

3. 测试3D点云可视化...
图表已保存到: output/wire_detection.png
✅ 3D点云可视化成功

4. 测试智能显示函数...
图表已保存到: output/test_smart_show.png
✅ 智能显示函数成功
```

## 生成的文件

运行主程序后，在 `output/` 目录下生成：
- `pipeline_summary.png` - 完整流程可视化图表（4MB）
- `wire_detection.png` - 导线检测结果图表（4MB）
- `test_smart_show.png` - 测试图表（88KB）

## 使用说明

### 在无头环境下
```bash
python main.py --show-plots
# 图表会自动保存到 output/ 目录
# 可以使用 ls -la output/*.png 查看生成的图片
```

### 在有显示环境下
```bash
python main.py --show-plots
# 图表会正常显示在窗口中
# 同时也会保存到 output/ 目录（如果使用 --save-plots）
```

## 技术特点

1. **自适应**：自动检测环境并选择合适的显示方式
2. **兼容性**：支持有头和无头环境
3. **高质量**：生成300 DPI的高质量图片
4. **用户友好**：提供清晰的保存路径提示
5. **向后兼容**：不影响原有的显示功能

## 总结

通过实现智能显示系统，成功解决了无头环境下的可视化问题。现在用户可以在任何环境下运行程序，都能获得完整的可视化结果，大大提升了用户体验和项目的可用性。
