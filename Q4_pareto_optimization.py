import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr

# 路径配置
INPUT_FILE = Path("./outputs/q4_risk_monitor_features.csv")
# 修改为用户指定的可视化目录
VIS_DIR = Path("./outputs/visualization")
VIS_DIR.mkdir(parents=True, exist_ok=True)
DATA_OUT_DIR = Path("./outputs")


def run_arhm_engine(df, gamma, w_j=0.5, min_k=0.1):
    """
    ARHM 向量化评分引擎
    """
    temp_df = df.copy()

    # 1. 计算周维度的信号强度 (CV)
    group_stats = temp_df.groupby(['season', 'week'])['judge_raw'].agg(['mean', 'std'])
    temp_df = temp_df.merge(group_stats, on=['season', 'week'])
    temp_df['cv_j'] = (temp_df['std'] / temp_df['mean']).fillna(0)

    # 2. 计算自适应阻尼 k (带有数值稳定性保护)
    temp_df['dynamic_k'] = (temp_df['adaptive_k'] * np.exp(-gamma * temp_df['cv_j'])).clip(lower=min_k)

    # 3. 计算对数压缩后的粉丝得分 (log1p)
    temp_df['damped_f'] = np.log1p(temp_df['dynamic_k'] * (temp_df['fan_vote_share_pct'] / 100.0))

    # 4. 权重归一化与最终得分
    f_sum = temp_df.groupby(['season', 'week'])['damped_f'].transform('sum')
    temp_df['adj_fan_share'] = temp_df['damped_f'] / f_sum.replace(0, 1)

    p_j = temp_df['judge_raw'] / temp_df.groupby(['season', 'week'])['judge_raw'].transform('sum')
    temp_df['arhm_total_score'] = w_j * p_j + (1 - w_j) * temp_df['adj_fan_share']

    # 5. 排名计算
    temp_df['arhm_rank'] = temp_df.groupby(['season', 'week'])['arhm_total_score'].rank(ascending=False)

    return temp_df


def find_pareto_optimal():
    print("🚀 启动帕累托前沿寻优...")

    if not INPUT_FILE.exists():
        print(f"❌ 错误: 找不到特征文件 {INPUT_FILE}")
        return None

    df = pd.read_csv(INPUT_FILE).dropna(subset=['judge_raw', 'fan_vote_share_pct'])
    df['judge_rank_ref'] = df.groupby(['season', 'week'])['judge_raw'].rank(ascending=False)
    df['fan_rank_ref'] = df.groupby(['season', 'week'])['fan_vote_share_pct'].rank(ascending=False)

    gammas = np.linspace(0, 5, 51)
    tuning_logs = []

    for g in gammas:
        sim = run_arhm_engine(df, gamma=g)

        # 指标 1: TF - 技术保真度 (Spearman)
        tf_corr = sim['arhm_rank'].corr(sim['judge_rank_ref'], method='spearman')

        # 指标 2: FEI - 粉丝参与度 (Top 1 粉丝宠儿留在 Top 2 安全区的概率)
        fan_top1 = sim[sim['fan_rank_ref'] == 1]
        fei_score = (fan_top1['arhm_rank'] <= 2).mean() if not fan_top1.empty else 0

        tuning_logs.append({'gamma': g, 'TF': tf_corr, 'FEI': fei_score})

    tuning_df = pd.DataFrame(tuning_logs).fillna(0)

    # 寻找最优平衡点 (Geometric Mean 思想，平衡公平与人气)
    tuning_df['balance_score'] = np.sqrt(tuning_df['TF'] * tuning_df['FEI'])
    best_row = tuning_df.loc[tuning_df['balance_score'].idxmax()]
    best_gamma = best_row['gamma']

    # 保存日志
    tuning_df.to_csv(DATA_OUT_DIR / "q4_gamma_search_log.csv", index=False)

    # --- 绘图部分 ---
    plt.style.use('seaborn-v0_8-muted')  # 使用美观的样式
    plt.figure(figsize=(12, 7))

    plt.plot(tuning_df['gamma'], tuning_df['TF'], label='Technical Fidelity (Professionalism)',
             color='#2E86C1', linewidth=2.5, marker='o', markevery=5)
    plt.plot(tuning_df['gamma'], tuning_df['FEI'], label='Fan Engagement (Retention)',
             color='#28B463', linestyle='--', linewidth=2.5, marker='s', markevery=5)

    # 标注最优 Gamma
    plt.axvline(x=best_gamma, color='#C0392B', linestyle=':', linewidth=2,
                label=f'Pareto Optimal Gamma: {best_gamma:.2f}')

    # 添加注释
    plt.annotate(f'Optimal Balance\nTF: {best_row["TF"]:.3f}\nFEI: {best_row["FEI"]:.3f}',
                 xy=(best_gamma, best_row['balance_score']), xytext=(best_gamma + 0.5, 0.5),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1),
                 fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    plt.title("Q4: Pareto Frontier Search for ARHM Model Parameters", fontsize=16, fontweight='bold')
    plt.xlabel("Control Parameter Gamma (Adaptive Risk Hedging)", fontsize=13)
    plt.ylabel("Index Performance Score (0-1)", fontsize=13)
    plt.legend(loc='lower left', frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.6)

    # 保存到指定的可视化目录
    vis_path = VIS_DIR / "fig_q4_pareto_frontier.png"
    plt.savefig(vis_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"🎯 最优 Gamma: {best_gamma:.2f}")
    print(f"🖼️ 帕累托前沿图已存至: {vis_path}")

    return best_gamma


if __name__ == "__main__":
    best_gamma = find_pareto_optimal()