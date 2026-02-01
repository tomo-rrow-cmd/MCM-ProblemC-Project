import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import beta, norm


# --- 1. 环境与样式配置 ---
def set_academic_style():
    # 采用更接近示例图的白色网格风格
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False


# --- 2. 数据加载与增强 ---
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    # 统一列名映射，防止找不到列
    column_mapping = {
        'estimated_fan_share': 'est_fan_share',
        'certainty_index': 'certainty_score'
    }
    df = df.rename(columns=column_mapping)

    df['season_stage'] = pd.Categorical(
        df['week'].apply(lambda w: 'Early' if w <= 4 else ('Mid' if w <= 8 else 'Final')),
        categories=['Early', 'Mid', 'Final'], ordered=True
    )
    # 状态分类：关键配色依据
    df['Status'] = df['is_eliminated_this_week'].map({
        1: 'Eliminated (Actual Result)',
        0: 'Safe'
    })
    return df


# --- 3. 核心绘图逻辑 ---
def generate_plots(df):
    # ==========================================================
    # 1. Figure 2: 横向带误差棒条形图 (模仿示例图风格)
    # ==========================================================
    plt.figure(figsize=(12, 6))
    # 选取特定赛季周次进行展示 (模仿示例中的 Season 10 Week 3)
    plot_df_f2 = df[df['season'] == df['season'].max()].sort_values('est_fan_share', ascending=False)

    # 模仿示例图配色：淘汰红，安全蓝
    color_map = {'Eliminated (Actual Result)': '#d65f5f', 'Safe': '#488fbf'}

    ax2 = sns.barplot(
        data=plot_df_f2, x='est_fan_share', y='celebrity_name',
        hue='Status', palette=color_map, dodge=False,
        capsize=.3, errorbar=('ci', 95), errcolor='black', errwidth=1.2
    )

    plt.title('Figure 2: True Fan Preferences (Results Analysis)\nError Bars represent 95% Confidence Interval',
              fontsize=13, pad=15)
    plt.xlabel('Estimated Fan Vote Share')
    plt.ylabel('')
    plt.legend(loc='lower right', frameon=True)
    plt.savefig('Figure_2_Standardized.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ==========================================================
    # 2. Figure 3: 精细化山脊图 (模仿示例图红蓝堆叠风格)
    # ==========================================================
    # 筛选展示选手 (通常选取当前周次的所有选手)
    ridge_df = df[df['season'] == df['season'].max()].copy()

    # 确保按得票率排序，使图形从上到下有层次感
    ridge_df = ridge_df.sort_values('est_fan_share', ascending=False)

    # 建立多层画布
    g = sns.FacetGrid(ridge_df, row="celebrity_name", hue="Status",
                      aspect=12, height=0.8, palette=color_map, sharex=True)

    def academic_ridge_plot(x, **kwargs):
        ax = plt.gca()
        mu, sigma = x.mean(), x.std() + 0.02  # 增加平滑度
        x_range = np.linspace(0, 0.45, 500)  # 匹配示例图 X 轴范围

        # 使用矩估计拟合 Beta 分布 (保证在 0-1 之间)
        if mu < 1.0:
            alpha_p = max(0.1, mu * (mu * (1 - mu) / sigma ** 2 - 1))
            beta_p = max(0.1, (1 - mu) * (mu * (1 - mu) / sigma ** 2 - 1))
            y = beta.pdf(x_range, alpha_p, beta_p)
        else:
            y = norm.pdf(x_range, mu, sigma)

        # 归一化高度，使其在 FacetGrid 中整齐
        y = y / y.max() if y.max() > 0 else y

        ax.fill_between(x_range, y, alpha=0.8, color=kwargs.get('color'))
        ax.plot(x_range, y, color=kwargs.get('color'), lw=1.5)
        ax.axhline(y=0, color=kwargs.get('color'), lw=2, clip_on=False)

    g.map(academic_ridge_plot, "est_fan_share")

    # 在基准线上方标注姓名 (模仿示例图)
    def label_contestant(x, color, label):
        ax = plt.gca()
        ax.text(0, 0.2, label, fontweight="bold", fontsize=11,
                color='black', ha="left", transform=ax.transAxes)

    g.map(label_contestant, "celebrity_name")

    # 样式微调：移除轴线，实现紧凑堆叠
    g.figure.subplots_adjust(hspace=-0.3)
    g.set_titles("")
    g.set(yticks=[], ylabel="", xlabel="Estimated Fan Vote Share (0.0 - 1.0)")
    g.despine(bottom=False, left=True)

    plt.suptitle(f'Figure 3: Uncertainty in Fan Preferences (Season {ridge_df["season"].iloc[0]})',
                 fontsize=14, y=1.02)
    plt.savefig('Figure_3_Standardized.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ==========================================================
    # 3. 2x2 面板保持原逻辑但应用统一色调
    # ==========================================================
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # C1. 热力图
    pivot_df = df.pivot_table(index='elim_priority', columns='week', values='certainty_score', aggfunc='mean')
    sns.heatmap(pivot_df, cmap='YlGnBu', annot=True, fmt=".1f", ax=axes[0, 0])
    axes[0, 0].set_title('Certainty Heatmap (Placement vs. Week)')

    # C2. 散点图
    sns.scatterplot(data=df, x='judge_share', y='est_fan_share',
                    hue='Status', palette=color_map, ax=axes[0, 1])
    axes[0, 1].plot([0, 0.5], [0, 0.5], 'r--', alpha=0.5)
    axes[0, 1].set_title('Logic Displacement Plot')

    # C3. 机制小提琴图
    sns.violinplot(data=df, x='Status', y='consistency_slack', palette=color_map, ax=axes[1, 0])
    axes[1, 0].set_title('Consistency Slack by Elimination Status')

    # C4. 箱线图
    sns.boxplot(data=df, x='season_stage', y='est_fan_share', palette='Pastel1', ax=axes[1, 1])
    axes[1, 1].set_title('Fan Share Dispersion (Matthew Effect)')

    plt.tight_layout()
    plt.savefig('Figure_Combined_Evidence_Standardized.png', dpi=300)


if __name__ == "__main__":
    set_academic_style()
    try:
        results_df = load_and_prepare_data('Q1_Model_Results.csv')
        generate_plots(results_df)
        print("✨ 风格对齐成功！生成的图片已保存在当前目录。")
    except Exception as e:
        print(f"❌ 运行失败: {e}")