import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ================= 配置区域 =================
# 1. 输入文件名
INPUT_CSV = 'latent_estimates.csv'

# 2. 你想画哪一季哪一周？(在这里修改)
TARGET_SEASON = 10
TARGET_WEEK = 3


# ===========================================

def main():
    # 1. 读取数据
    if not os.path.exists(INPUT_CSV):
        print(f"❌ 错误：找不到文件 {INPUT_CSV}")
        print("请确保 CSV 文件和代码在同一个文件夹里！")
        return

    df = pd.read_csv(INPUT_CSV)
    print(f"✅ 成功读取数据，共 {len(df)} 行。")

    # 2. 筛选特定的一周
    subset = df[(df['season'] == TARGET_SEASON) & (df['week'] == TARGET_WEEK)].copy()

    if subset.empty:
        print(f"❌ 警告：Season {TARGET_SEASON} Week {TARGET_WEEK} 没有数据！请检查 CSV。")
        return

    # 3. 数据准备
    # 按粉丝得票均值排序
    subset = subset.sort_values('fan_vote_mean', ascending=True)

    # 计算误差棒长度 (matplotlib 需要的是距离均值的差值)
    # xerr 需要是 shape (2, N) -> [[left_errors], [right_errors]]
    xerr = [
        (subset['fan_vote_mean'] - subset['ci_low']).values,  # 左边长度
        (subset['ci_high'] - subset['fan_vote_mean']).values  # 右边长度
    ]

    # 4. 开始画图
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")  # 设置背景风格

    # 颜色逻辑：被淘汰的是红色，安全的是蓝色
    # 注意：确保 CSV 里有 'is_eliminated_this_week' 这一列，且是 True/False
    colors = ['#d62728' if x else '#1f77b4' for x in subset['is_eliminated_this_week']]

    # 画水平柱状图
    bars = plt.barh(
        y=subset['celebrity_name'],
        width=subset['fan_vote_mean'],
        xerr=xerr,  # 加上误差棒
        color=colors,
        capsize=5,  # 误差棒两头的小横线
        alpha=0.8  # 透明度
    )

    # 5. 图表装饰
    plt.xlabel('Estimated Fan Vote Share')
    plt.title(
        f'Q1 Results: True Fan Preferences (Season {TARGET_SEASON} Week {TARGET_WEEK})\nError Bars represent 95% Confidence Interval')

    # 手动加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#d62728', label='Eliminated (Actual Result)'),
        Patch(facecolor='#1f77b4', label='Safe')
    ]
    plt.legend(handles=legend_elements, loc='lower right')

    # 6. 保存图片 (关键一步！)
    output_filename = f'Q1_Analysis_S{TARGET_SEASON}_W{TARGET_WEEK}.png'
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)  # dpi=300 让图片更清晰
    print(f"🎉 图片已保存为: {output_filename}")
    print("快去文件夹里打开看看吧！")

    # 同时也展示一下
    plt.show()


if __name__ == "__main__":
    main()