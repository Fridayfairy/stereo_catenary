# 双目视觉悬链线重建项目

## 项目简介

本项目实现了基于双目视觉的导线悬链线重建系统。给定双目相机采集的两张图像和相机内外参数，系统能够自动检测导线像素、进行双目匹配、三角测量，并最终拟合出导线的3D悬链线方程。

## 项目结构

```
stereo_catenary/
├── main.py                 # 主程序入口
├── line_detect.py          # 导线像素提取模块
├── stereo_match.py         # 极线约束匹配模块
├── triangulation.py        # 三角测量模块
├── plane_estimation.py     # 垂直平面估计模块
├── catenary_fit.py         # 悬链线拟合模块
├── viz.py                  # 可视化模块
├── config/
│   └── cameras.yaml        # 相机配置文件
├── requirements.txt        # 依赖包列表
└── README.md              # 项目说明文档
```

## 核心功能

### 1. 导线像素检测 (`line_detect.py`)
- 使用Canny边缘检测 + 霍夫变换检测直线
- 支持脊线检测和骨架化算法
- 自动过滤噪声和短线段

### 2. 双目匹配 (`stereo_match.py`)
- 基于极线约束的导线像素匹配
- 使用归一化互相关(NCC)进行匹配
- Sampson误差过滤异常匹配

### 3. 三角测量 (`triangulation.py`)
- 支持DLT、射线中点、最优三角测量方法
- 重投影误差计算和异常点过滤
- 自动选择最佳三角测量方法

### 4. 平面估计 (`plane_estimation.py`)
- 使用PCA估计导线主要方向
- 计算包含重力方向和导线方向的垂直平面
- 将3D点投影到2D平面坐标

### 5. 悬链线拟合 (`catenary_fit.py`)
- 使用Levenberg-Marquardt算法拟合悬链线方程
- 包含解析雅可比矩阵和鲁棒损失函数
- 支持RANSAC算法处理异常值

### 6. 可视化 (`viz.py`)
- 2D和3D可视化功能
- 完整的处理流程可视化
- 拟合质量分析图表

## 悬链线方程

系统拟合的悬链线方程为：

```
z(s) = a * cosh((s - s0) / a) + c
```

其中：
- `s`: 沿导线方向的坐标
- `z`: 重力方向的坐标
- `a`: 悬链线参数（与张力相关）
- `s0`: 悬链线中心位置
- `c`: 垂直偏移量

最终的3D悬链线方程为：

```
X(s) = p0 + s*u + [a*cosh((s-s0)/a) + c]*g_hat
```

其中：
- `p0`: 平面参考点
- `u`: 沿导线方向的单位向量
- `g_hat`: 重力方向单位向量

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 配置相机参数

编辑 `config/cameras.yaml` 文件，设置相机的内参和外参：

```yaml
# 左相机内参矩阵 K1 (3x3)
left_camera:
  intrinsic_matrix:
    - [1000.0, 0.0, 320.0]      # fx, 0, cx
    - [0.0, 1000.0, 240.0]      # 0, fy, cy  
    - [0.0, 0.0, 1.0]           # 0, 0, 1

# 右相机内参矩阵 K2 (3x3)  
right_camera:
  intrinsic_matrix:
    - [1000.0, 0.0, 320.0]      # fx, 0, cx
    - [0.0, 1000.0, 240.0]      # 0, fy, cy
    - [0.0, 0.0, 1.0]           # 0, 0, 1

# 相机外参
left_camera:
  rotation_matrix:
    - [1.0, 0.0, 0.0]           # 单位矩阵，左相机作为参考
    - [0.0, 1.0, 0.0]
    - [0.0, 0.0, 1.0]
  translation_vector: [0.0, 0.0, 0.0]  # 左相机位置作为原点

right_camera:
  rotation_matrix:
    - [1.0, 0.0, 0.0]           # 假设右相机与左相机平行
    - [0.0, 1.0, 0.0]
    - [0.0, 0.0, 1.0]
  translation_vector: [-100.0, 0.0, 0.0]  # 基线距离100mm

# 世界坐标系重力方向 (单位向量)
gravity_direction: [0.0, 0.0, -1.0]  # 重力向下
```

### 2. 运行主程序

```bash
python main.py --left left_image.jpg --right right_image.jpg --config config/cameras.yaml --output output --show-plots
```

参数说明：
- `--left`: 左图像路径
- `--right`: 右图像路径
- `--config`: 相机配置文件路径（默认：config/cameras.yaml）
- `--output`: 输出目录（默认：output）
- `--show-plots`: 显示可视化图表
- `--save-plots`: 保存可视化图表
- `--verbose`: 显示详细信息（默认开启）

### 3. 输出结果

程序会在输出目录中生成以下文件：
- `catenary_params.txt`: 悬链线参数和拟合质量
- `points_3d.txt`: 3D点坐标
- `projection_2d.txt`: 2D投影坐标
- `pipeline_summary.png`: 完整流程可视化图表（如果使用--save-plots）

## 算法特点

### 1. 鲁棒性
- 使用RANSAC算法处理异常值
- Huber损失函数减少异常点影响
- 多级过滤机制确保数据质量

### 2. 精度
- 解析雅可比矩阵提高拟合精度
- 最优三角测量方法
- 重投影误差验证

### 3. 可视化
- 完整的处理流程可视化
- 拟合质量分析
- 3D悬链线曲线展示

## 技术细节

### 数学原理

1. **三角测量**: 使用直接线性变换(DLT)将匹配点对转换为3D坐标
2. **平面估计**: 通过PCA找到导线主要方向，结合重力方向确定垂直平面
3. **悬链线拟合**: 使用Levenberg-Marquardt算法最小化重投影误差

### 关键算法

1. **导线检测**: Canny边缘检测 + 霍夫变换
2. **双目匹配**: 极线约束 + 归一化互相关
3. **异常值处理**: RANSAC + Sampson误差过滤
4. **参数优化**: 带解析雅可比的LM算法

## 注意事项

1. 确保相机标定精度，特别是基线距离和重力方向
2. 图像中导线应清晰可见，避免遮挡
3. 导线应基本在同一垂直平面内
4. 建议使用高分辨率图像以获得更好的检测效果

## 扩展功能

项目支持以下扩展：
- 多帧融合（bundle adjustment）
- 端点锚定（检测绝缘子等硬件）
- 置信区间估计
- 实时处理优化

## 许可证

本项目采用MIT许可证。
