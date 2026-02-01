import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 配置与路径
INPUT_FILE = Path("./outputs/q4_risk_monitor_features.csv")
VIS_DIR = Path("./outputs/visualization")
VIS_DIR.mkdir(parents=True, exist_ok=True)
BEST_GAMMA = 4.8


def run_arhm_engine(df, gamma, w_j=0.5, min_k=0.1):
    """ARHM 评分引擎 (保持与脚本1一致)"""
    temp_df = df.copy()
    grp_stats = temp_df.groupby(['season', 'week'])['judge_raw'].agg(['mean', 'std'])
    temp_df = temp_df.merge(grp_stats, on=['season', 'week'])
    temp_df['cv_j'] = (temp_df['std'] / temp_df['mean']).fillna(0)
    temp_df['dynamic_k'] = (temp_df['adaptive_k'] * np.exp(-gamma * temp_df['cv_j'])).clip(lower=min_k)
    temp_df['damped_f'] = np.log1p(temp_df['dynamic_k'] * (temp_df['fan_vote_share_pct'] / 100.0))
    f_sum = temp_df.groupby(['season', 'week'])['damped_f'].transform('sum')
    temp_df['adj_fan_share'] = temp_df['damped_f'] / f_sum.replace(0, 1)
    p_j = temp_df['judge_raw'] / temp_df.groupby(['season', 'week'])['judge_raw'].transform('sum')
    temp_df['arhm_total_score'] = w_j * p_j + (1 - w_j) * temp_df['adj_fan_share']
    temp_df['arhm_rank'] = temp_df.groupby(['season', 'week'])['arhm_total_score'].rank(ascending=False)
    # 获取每周存活总人数，用于定义动态危险区
    temp_df['n_active'] = temp_df.groupby(['season', 'week'])['celebrity_name'].transform('count')
    return temp_df


def plot_dual_universe(original, simulated, celebrity_name, season_num, tag=""):
    """
    绘制平行宇宙对比图，增加动态危险区
    """
    old_data = original[
        (original['celebrity_name'] == celebrity_name) & (original['season'] == season_num)].sort_values('week')
    new_data = simulated[
        (simulated['celebrity_name'] == celebrity_name) & (simulated['season'] == season_num)].sort_values('week')

    if old_data.empty: return

    plt.figure(figsize=(10, 6))

    # 绘制动态危险区 (最后两名)
    weeks = old_data['week'].values
    n_active = new_data['n_active'].values
    plt.fill_between(weeks, n_active, n_active - 1.5, color='red', alpha=0.15, label='Danger Zone (Bottom 2)')

    # 绘制排名曲线
    plt.plot(weeks, old_data['arhm_rank'], 'o--', color='gray', label='Original (Linear 50/50)', alpha=0.6)
    plt.plot(weeks, new_data['arhm_rank'], 's-', color='#E74C3C' if tag == "Controversial" else '#2E86C1',
             linewidth=3, label=f'ARHM (Gamma={BEST_GAMMA})')

    plt.gca().invert_yaxis()
    plt.title(f"Counterfactual Trajectory: {celebrity_name} (S{season_num}) - {tag}", fontsize=14)
    plt.xlabel("Competition Week")
    plt.ylabel("Rank Position")
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.5)

    # 设置 Y 轴刻度为整数排名
    all_ranks = np.arange(1, int(n_active.max()) + 1)
    plt.yticks(all_ranks)

    save_path = VIS_DIR / f"q4_universe_{tag}_{celebrity_name.replace(' ', '_')}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    df = pd.read_csv(INPUT_FILE)

    # 模拟两套规则
    original_df = run_arhm_engine(df, gamma=0)  # 原始
    simulated_df = run_arhm_engine(df, gamma=BEST_GAMMA)  # 新规则

    # 1. 案例分析：Bobby Bones (反面教材)
    plot_dual_universe(original_df, simulated_df, "Bobby Bones", 27, tag="Controversial")

    # 我们将原始排名合并进去，方便后续直接读取对比
    simulated_df['original_rank'] = original_df['arhm_rank']
    simulated_df.to_csv(Path("./outputs/q4_counterfactual_full_data.csv"), index=False)
    print(f"💾 完整模拟数据已导出至: ./outputs/q4_counterfactual_full_data.csv")
    # ----------------------------------------------------

    # 2. 案例分析：Charli D'Amelio (正面教材)
    # 如果数据里有 Charli，证明规则不会误伤强者
    if "Charli D'Amelio" in df['celebrity_name'].values:
        plot_dual_universe(original_df, simulated_df, "Charli D'Amelio", 31, tag="Technical_Hero")

    # 3. 统计影响：谁受影响最大？
    diff_table = simulated_df.copy()
    diff_table['rank_drop'] = simulated_df['arhm_rank'] - original_df['arhm_rank']
    # 找到平均排名跌落最大的前 5 人
    impact_rank = diff_table.groupby('celebrity_name')['rank_drop'].mean().sort_values(ascending=False).head(5)
    print("\n🚨 受 ARHM 规则影响最严重的选手 (潜在拦截对象):")
    print(impact_rank)


if __name__ == "__main__":
    main()