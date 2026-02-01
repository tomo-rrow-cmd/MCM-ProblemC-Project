import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
from pathlib import Path

# 路径配置
DATA_PATH = Path("./outputs/q4_counterfactual_full_data.csv")
VIS_DIR = Path("./outputs/visualization")
VIS_DIR.mkdir(parents=True, exist_ok=True)


def plot_interception_heatmap(df):
    """1. 拦截热力图：展示模型触发拦截的活跃区域"""
    print("  -> 正在绘制：拦截热力图 (拦截强度版)")
    plt.figure(figsize=(12, 8))

    # 计算排名变化的绝对值
    df['rank_delta_abs'] = (df['arhm_rank'] - df['original_rank']).abs()

    # 过滤掉缺失值
    plot_data = df.dropna(subset=['cv_j', 'fan_vote_share_pct', 'rank_delta_abs'])

    # 提高灵敏度：使用 qcut (等频分箱) 确保每个格子都有数据支撑，或者保持 cut 增加 bins
    plot_data['cv_bin'] = pd.cut(plot_data['cv_j'], bins=6)
    plot_data['fan_bin'] = pd.cut(plot_data['fan_vote_share_pct'], bins=6)

    # 修改点：使用 'sum' 或 'max'。
    # 'sum' 代表该区域的拦截活跃度，'max' 代表该区域最极端的拦截案例
    pivot_table = plot_data.pivot_table(
        index='fan_bin', columns='cv_bin', values='rank_delta_abs',
        aggfunc='sum',  # 改为求和，突出拦截集中的区域
        observed=False
    )

    # 使用 Reds 预警色调，0 变动将显示为白色
    sns.heatmap(pivot_table, cmap='YlOrRd', annot=True, fmt=".0f",
                cbar_kws={'label': 'Cumulative Rank Displacement'})

    plt.title("ARHM Active Defense Zone: Where the Model Intervenes", fontsize=14)
    plt.xlabel("Judge Disagreement (CV_j) -> High CV triggers Damping", fontsize=12)
    plt.ylabel("Fan Support Share (%) -> High Fan needs Oversight", fontsize=12)

    plt.savefig(VIS_DIR / "fig_q4_interception_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_champion_radar(df):
    """2. 冠军成色雷达图：修复 Inhomogeneous shape 错误"""
    print("  -> 正在绘制：冠军成色雷达图")
    targets = {"Bobby Bones": 27, "Charli D'Amelio": 31}
    metrics = ['Professionalism', 'Popularity', 'ARHM_Stability', 'Fairness_Index']

    plt.figure(figsize=(9, 9))
    ax = plt.subplot(111, polar=True)
    colors = ['#E74C3C', '#2E86C1']  # 红色代表争议，蓝色代表技术英雄

    for i, (name, season) in enumerate(targets.items()):
        person_df = df[(df['celebrity_name'] == name) & (df['season'] == season)]
        if person_df.empty:
            print(f"     ⚠️ 找不到选手数据: {name} (S{season})")
            continue

        # 归一化计算维度分 (强制转换为纯 float 标量)
        prof = float(person_df['judge_raw'].mean() / df['judge_raw'].max())
        pop = float(person_df['fan_vote_share_pct'].mean() / df['fan_vote_share_pct'].max())

        # 稳定性：排名变动的标准差（倒序，越稳分越高）
        std_val = person_df['arhm_rank'].std()
        stab = float(1 - (std_val / 5)) if not np.isnan(std_val) else 1.0

        # 公平性：模拟排名与原始排名的偏差（倒序，偏差越小分越高）
        diff_val = (person_df['arhm_rank'] - person_df['original_rank']).abs().mean()
        fair = float(1 - (diff_val / 5))

        values = [prof, pop, stab, fair]
        values = [max(0.1, v) for v in values]  # 防止 0 值导致图形塌陷
        values += values[:1]  # 闭合

        angles = [n / float(len(metrics)) * 2 * np.pi for n in range(len(metrics))]
        angles += angles[:1]

        # 使用 np.array 显式转换，解决数据类型不一致导致的 ValueError
        ax.plot(np.array(angles), np.array(values), color=colors[i], linewidth=3, label=f"{name} (S{season})")
        ax.fill(np.array(angles), np.array(values), color=colors[i], alpha=0.2)

    plt.xticks(angles[:-1], metrics, fontsize=11)
    plt.title("Champion Pedigree: Controversial vs. Technical Leader", fontsize=15, pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.savefig(VIS_DIR / "fig_q4_champion_radar.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_rank_ridge(df):
    print("  -> 正在绘制：高对比度排名分布图")
    plt.figure(figsize=(10, 6))

    # 1. 准备数据并处理索引
    df_new = df[['arhm_rank']].dropna().copy()
    df_new['Type'] = 'ARHM Rules (New)'

    df_old = df[['original_rank']].dropna().copy()
    df_old.columns = ['arhm_rank']
    df_old['Type'] = 'Traditional (History)'

    # 2. 分开绘制以确保 zorder 生效
    # 先画红色（背景），较低的 zorder
    sns.kdeplot(data=df_old, x='arhm_rank', fill=True,
                color='#E74C3C', label='History (Red)',
                alpha=0.3, linewidth=2, zorder=1)

    # 后画蓝色（前景），较高的 zorder，且稍微加深颜色
    sns.kdeplot(data=df_new, x='arhm_rank', fill=True,
                color='#3498DB', label='ARHM (Blue)',
                alpha=0.6, linewidth=3, zorder=10)  # zorder=10 确保它在最上面

    # 3. 关键改进：在底部添加地毯图 (Rug Plot) 展示数据点的位移
    # 蓝色点（新）会叠在红色点（旧）上面
    sns.rugplot(data=df_old, x='arhm_rank', color='red', alpha=0.2, height=0.03)
    sns.rugplot(data=df_new, x='arhm_rank', color='blue', alpha=0.5, height=0.05)

    plt.title("System Stability: ARHM vs Traditional Distribution", fontsize=14)
    plt.xlabel("Rank Position (1=Champion)")
    plt.xlim(1, 12)
    plt.gca().invert_xaxis()
    plt.legend(loc='upper left')
    plt.grid(axis='y', linestyle=':', alpha=0.5)

    # 4. 保存
    plt.savefig(VIS_DIR / "fig_q4_rank_ridge.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_displacement_scatter(df):
    """ 替代山脊图：排名位移散点图"""
    print("  -> 正在绘制：排名位移散点图")
    plt.figure(figsize=(10, 6))

    # 计算差异
    df['diff'] = df['arhm_rank'] - df['original_rank']

    # 只绘制有变动的点
    changed = df[df['diff'] != 0]
    stable = df[df['diff'] == 0]

    plt.scatter(stable['original_rank'], stable['arhm_rank'], alpha=0.1, color='gray', label='Stable (No Change)')
    plt.scatter(changed['original_rank'], changed['arhm_rank'], alpha=0.8, color='red', s=100,
                edgecolor='black', label='Impacted (ARHM Active)')

    # 绘制恒等线
    plt.plot([1, 12], [1, 12], '--', color='blue', alpha=0.5)

    plt.xlabel("Original Rank")
    plt.ylabel("ARHM Rank (Gamma=4.8)")
    plt.title("Interception Scatter: Who got moved?", fontsize=14)
    plt.legend()
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.savefig(VIS_DIR / "fig_q4_rank_scatter.png", dpi=300)
    plt.close()


def main():
    if not DATA_PATH.exists():
        print(f"❌ 错误：找不到文件 {DATA_PATH}。请确保脚本 2 已正确运行并导出 CSV。")
        return

    df = pd.read_csv(DATA_PATH)
    print("🚀 启动高级可视化引擎...")

    plot_interception_heatmap(df)
    plot_champion_radar(df)
    plot_rank_ridge(df)
    plot_displacement_scatter(df)

    print(f"✨ 任务完成！请在 {VIS_DIR} 目录查看生成的 3 张图表。")


if __name__ == "__main__":
    main()