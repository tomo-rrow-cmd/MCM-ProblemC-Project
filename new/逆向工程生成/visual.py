import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# ==========================================
# 1. 严谨学术风全局配置 (Copy This!)
# ==========================================
def set_pub_style():
    """
    配置 Matplotlib 为顶级期刊风格 (Nature/Science)
    """
    # 字体设置：优先使用 Times New Roman
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']

    # 字体大小：确保插入 Word 后依然清晰
    base_size = 14
    plt.rcParams['axes.labelsize'] = base_size + 2  # 轴标签略大
    plt.rcParams['axes.titlesize'] = base_size + 4  # 标题更大
    plt.rcParams['xtick.labelsize'] = base_size
    plt.rcParams['ytick.labelsize'] = base_size
    plt.rcParams['legend.fontsize'] = base_size - 1

    # 线条与布局
    plt.rcParams['axes.linewidth'] = 1.5      # 坐标轴线变粗
    plt.rcParams['grid.alpha'] = 0.3          # 网格极淡
    plt.rcParams['grid.linestyle'] = '--'     # 网格虚线
    plt.rcParams['lines.linewidth'] = 2.5     # 数据线变粗
    plt.rcParams['lines.markersize'] = 8      # 数据点变大

    # 配色方案 (Colorblind Friendly & Professional)
    # 蓝色：主数据 / 稳定
    # 红色：强调 / 异常 / 淘汰
    # 灰色：背景 / 对照
    # 黄色/紫色：辅助类别
    return ["#0C5DA5", "#C1272D", "#FFBF00", "#555555"]

# 应用配置
COLORS = set_pub_style()
COLOR_MAIN = COLORS[0]  # 深蓝
COLOR_ALERT = COLORS[1] # 砖红
COLOR_GRAY = COLORS[3]  # 深灰

# ==========================================
# 2. 示例：带误差带的时间序列图 (严谨+美观)
# 场景：展示模型预测结果及不确定性
# ==========================================
def plot_example_trajectory():
    # 生成模拟数据
    x = np.linspace(0, 10, 20)
    y_mean = np.sin(x) + x/2
    y_std = 0.2 + 0.1 * x  # 误差随时间变大（体现严谨性）

    plt.figure(figsize=(10, 6))

    # 1. 画主线 (Line)
    plt.plot(x, y_mean, color=COLOR_MAIN, label='Model Estimate', marker='o', markeredgecolor='white')

    # 2. 画置信区间 (Confidence Interval) -> 体现严谨性
    # 颜色使用主色的浅色版本，alpha=0.2
    plt.fill_between(x, y_mean - 1.96*y_std, y_mean + 1.96*y_std,
                     color=COLOR_MAIN, alpha=0.15, linewidth=0, label='95% Confidence Interval')

    # 3. 标注关键点 (Annotation) -> 体现分析深度
    max_idx = np.argmax(y_mean)
    plt.annotate(f'Peak Value: {y_mean[max_idx]:.2f}',
                 xy=(x[max_idx], y_mean[max_idx]),
                 xytext=(x[max_idx]-2, y_mean[max_idx]+1),
                 arrowprops=dict(facecolor=COLOR_ALERT, shrink=0.05),
                 fontsize=12, fontweight='bold', color=COLOR_ALERT)

    # 4. 装饰 (Decorations)
    plt.title("Figure 1: Temporal Dynamics with Uncertainty Quantification", fontweight='bold', pad=20)
    plt.xlabel("Time (Arbitrary Units)")
    plt.ylabel("Value ($y = f(x)$)")

    # 去掉上方和右方的边框 (Despine) -> 现代审美关键
    sns.despine()
    plt.grid(axis='y') # 只保留水平网格

    plt.legend(frameon=False, loc='upper left') # 图例无框
    plt.tight_layout()
    plt.savefig('academic_line_plot.png', dpi=300)
    print("✅ Generated: academic_line_plot.png")

# ==========================================
# 3. 示例：对比分析柱状图 (带统计意义)
# 场景：展示不同策略/组别的效果差异
# ==========================================
def plot_example_comparison():
    categories = ['Strategy A', 'Strategy B', 'Strategy C', 'Strategy D']
    means = [0.85, 0.62, 0.91, 0.45]
    errors = [0.05, 0.08, 0.04, 0.10] # 误差棒

    plt.figure(figsize=(8, 6))

    # 定义颜色：根据数值高低自动变色，或者重点突出某一项
    # 这里突出 Strategy C (最高分)
    bar_colors = [COLOR_GRAY, COLOR_GRAY, COLOR_MAIN, COLOR_ALERT]

    # 画柱状图，必须带 yerr (误差棒) -> 严谨性
    bars = plt.bar(categories, means, yerr=errors,
                   color=bar_colors, capsize=5, alpha=0.9, width=0.6)

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{height:.2f}', ha='center', va='bottom', fontweight='bold')

    plt.title("Figure 2: Performance Comparison across Strategies", fontweight='bold', pad=15)
    plt.ylabel("Accuracy Score")

    # 标注阈值线
    plt.axhline(y=0.8, color=COLOR_ALERT, linestyle='--', alpha=0.5)
    plt.text(3.6, 0.81, "Threshold (0.8)", color=COLOR_ALERT, fontsize=10)

    sns.despine(left=True) # 去掉左边框，只保留底部
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.yticks([]) # 去掉Y轴刻度，因为已经标了数字

    plt.tight_layout()
    plt.savefig('academic_bar_plot.png', dpi=300)
    print("✅ Generated: academic_bar_plot.png")

# ==========================================
# 4. 示例：相关性热力图 (极简风)
# 场景：展示变量间的关系
# ==========================================
def plot_example_heatmap():
    data = np.random.rand(6, 6)
    # 制造一些相关性结构
    for i in range(6): data[i,i] = 1.0

    plt.figure(figsize=(7, 6))

    # 使用冷暖色调：RdBu_r (红=正相关，蓝=负相关) 或 Viridis/Mako
    # square=True 保证格子是正方形
    sns.heatmap(data, cmap="mako", annot=True, fmt=".2f",
                linewidths=1, linecolor='white', # 格子之间留白
                cbar_kws={'label': 'Correlation Coefficient'})

    plt.title("Figure 3: Parameter Sensitivity Matrix", fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('academic_heatmap.png', dpi=300)
    print("✅ Generated: academic_heatmap.png")

if __name__ == "__main__":
    plot_example_trajectory()
    plot_example_comparison()
    plot_example_heatmap()