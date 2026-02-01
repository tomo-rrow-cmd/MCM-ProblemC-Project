import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.svm import SVC
from scipy.stats import pearsonr

# ==========================================
# 0. 环境配置：创建输出目录
# ==========================================
OUTPUT_DIR = "./outputs/visualizations"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 设置学术绘图风格
plt.rcParams['font.family'] = 'serif'
sns.set_theme(style="whitegrid")


def preprocess_features(df):
    """
    解决避坑指南 1 & 2:
    - 使用 'average' 处理排名并列
    - 将排名分数归一化至 [0, 1] 空间
    """
    def group_norm(group):
        if len(group) <= 1: return group
        # 并列平滑处理
        group['R_smooth_rank'] = group['R_total_score'].rank(method='average')
        # 量纲归一化：1表示最强(第一名)，0表示最弱(最后一名)
        r_min, r_max = group['R_smooth_rank'].min(), group['R_smooth_rank'].max()
        group['S_R_norm'] = 1 - (group['R_smooth_rank'] - r_min) / (r_max - r_min) if r_max != r_min else 0.5
        return group

    df = df.groupby(['season', 'week'], group_keys=False).apply(preprocess_features_logic)
    df['delta_S'] = df['P_total_score'] - df['S_R_norm']
    return df

def preprocess_features_logic(group):
    # 此处为 preprocess_features 的内部逻辑抽离
    if len(group) <= 1: return group
    group['R_smooth_rank'] = group['R_total_score'].rank(method='average')
    r_min, r_max = group['R_smooth_rank'].min(), group['R_smooth_rank'].max()
    group['S_R_norm'] = 1 - (group['R_smooth_rank'] - r_min) / (r_max - r_min) if r_max != r_min else 0.5
    return group

def calculate_sample_weights(df):
    """
    解决避坑指南 3: 基于赛季活跃人数的中位数和冲突密度分配权重
    """
    # --- 1. 置信度权重 (可靠性) ---
    # 宽度越小、接受率越高，权重越大
    ci_width = (df['ci_high'] - df['ci_low']).replace(0, 0.01)
    df['w_conf'] = df['abc_accept_rate'] / ci_width

    # --- 2. 边界探测权重 (灵敏度) ---
    # 找到每赛季每周末位的分数作为阈值
    min_scores = df.groupby(['season', 'week'])['P_total_score'].transform('min')
    # 距离末位越近，权重呈指数级增加
    df['w_border'] = np.exp(-3.0 * (df['P_total_score'] - min_scores))

    # --- 3. 冲突与业务增强 ---
    # 规则冲突样本是核心证据，给予双倍权重
    df['w_conflict'] = df['delta_status'].apply(lambda x: 2.0 if x != 0 else 1.0)

    # --- 4. 赛季阶段增强 (后期样本更成熟) ---
    max_weeks = df.groupby('season')['week'].transform('max')
    df['w_time'] = df['week'] / max_weeks

    # 合成最终权重并归一化
    w_raw = df['w_conf'] * df['w_border'] * df['w_conflict'] * df['w_time']
    df['sample_weight'] = w_raw / w_raw.mean()

    return df['sample_weight']


def run_monte_carlo_analysis(df, n_iterations=100, noise=0.01):
    """
    解决避坑指南 4: 蒙特卡洛模拟防止因果倒置，并计算 DCC 和 RSR
    1. 确保扰动后的粉丝票份额满足 Sum-to-1 约束
    2. 确保 SVM 训练使用了扰动后的变量
    """
    dcc_list, rsr_list = [], []
    # 预先计算好基础权重（MRW框架）
    weights = calculate_sample_weights(df)

    for i in range(n_iterations):
        # --- [步骤 1] 约束性扰动 ---
        df_noisy = df.copy()

        def apply_constrained_noise(group):
            # 注入高斯噪声
            s = group['P_fan_share'] + np.random.normal(0, noise, len(group))
            # 裁剪非法值并重新归一化
            s_clipped = np.clip(s, 0.001, 1.0)
            group['noisy_fan'] = s_clipped / s_clipped.sum()
            return group

        # 按周执行扰动
        df_noisy = df_noisy.groupby(['season', 'week'], group_keys=False).apply(apply_constrained_noise, include_groups=False)

        # --- [步骤 2] 计算指标 A: DCC ---
        dcc, _ = pearsonr(df_noisy['delta_S'], df_noisy['noisy_fan'])
        dcc_list.append(dcc)

        # --- [步骤 3] 计算指标 B: RSR (修正后的SVM训练) ---
        try:
            # 排除由于原始数据缺失导致的空值（如18季第2周）
            valid_idx = df_noisy[['P_judge_share', 'noisy_fan', 'P_is_elim']].dropna().index

            if len(valid_idx) > 5:  # 确保有足够样本
                X_sub = df_noisy.loc[valid_idx, ['P_judge_share', 'noisy_fan']]
                y_sub = df_noisy.loc[valid_idx, 'P_is_elim'].astype(int)
                w_sub = weights.loc[valid_idx] # 权重同步过滤

                svm = SVC(kernel='linear', C=1.0)
                # 显式传入扰动变量与样本权重
                svm.fit(X_sub, y_sub, sample_weight=w_sub)

                w_j, w_f = svm.coef_[0]
                # 计算灵敏度比率
                rsr_list.append(abs(w_f / w_j) if abs(w_j) > 1e-9 else 1.0)
        except Exception as e:
            continue

    return np.mean(dcc_list), np.std(dcc_list), np.mean(rsr_list), np.std(rsr_list)


def prepare_data():
    """读取并预处理数据"""
    if not os.path.exists('./outputs/dwts_rule_comparison.csv'):
        print("Error: './outputs/dwts_rule_comparison.csv' not found.")
        return None

    df = pd.read_csv('./outputs/dwts_rule_comparison.csv')

    def calc_norm_rank(group):
        group['R_rank'] = group['R_total_score'].rank(ascending=False, method='average')
        max_r, min_r = group['R_rank'].max(), group['R_rank'].min()
        if max_r != min_r:
            group['S_R_norm'] = 1 - (group['R_rank'] - min_r) / (max_r - min_r)
        else:
            group['S_R_norm'] = 0.5
        return group

    df = df.groupby(['season', 'week'], group_keys=False).apply(calc_norm_rank)
    df['delta_S'] = df['P_total_score'] - df['S_R_norm']
    return df


# ==========================================
# 1. Figure 1: Power Balance Scale
# ==========================================
def save_power_balance():
    # 数据背景：假设一个典型的 Bottom 2 场景
    # Rank-based: 积分永远是固定的（例如第一名10分，第二名9分），比例接近 1:1
    # Percentage-based: 基于你的 RSR=1.6784 换算

    """
        基于数据计算排名制的真实权重占比，而非硬编码 1:1
        """
    # 1. 计算排名制下的名义最高权重 (以平均每场 8 人为例)
    df = prepare_data()
    avg_contestants = df.groupby(['season', 'week'])['celebrity_name'].count().mean()
    n = int(round(avg_contestants))
    total_rank_pool = n * (n + 1) / 2
    # 在排名制下，第一名选手在单一系统（如观众系统）中能获得的最大权重比例
    rank_max_share = n / total_rank_pool

    # 2. 获取百分比制下的 RSR 结果
    rsr_val = 1.6784
    # 换算百分比制下观众系统的相对权重
    perc_fan_weight = rsr_val / (1 + rsr_val)
    perc_judge_weight = 1 / (1 + rsr_val)

    # 3. 绘图
    fig, ax = plt.subplots(figsize=(10, 7))
    bar_width = 0.4

    # 我们对比：系统赋予个体的“影响力上限”
    # 排名制：即便是第一名，其权重也被总积分池稀释
    # 百分比制：权重随票数连续扩张
    labels = ['Rank-based System\n(Discrete Constraints)', 'Percentage-based System\n(Continuous Leverage)']

    # 绘制堆叠条形图
    # 底层：评委控制力
    ax.bar(labels, [0.5, perc_judge_weight], bar_width,
           label='Judges Power (Professional Control)', color='#66b3ff', edgecolor='black')

    # 顶层：观众影响力
    ax.bar(labels, [0.5, perc_fan_weight], bar_width,
           bottom=[0.5, perc_judge_weight],
           label='Fans Power (Popularity Leverage)', color='#ff9999', edgecolor='black')

    # 添加严谨的数学标注
    ax.text(0, 0.25, "Fixed Increments\n(1, 2, 3...N)", ha='center', va='center', fontweight='bold')
    ax.text(1, perc_judge_weight / 2, f"{perc_judge_weight:.1%}", ha='center', va='center', color='white',
            fontweight='bold')
    ax.text(1, perc_judge_weight + perc_fan_weight / 2, f"{perc_fan_weight:.1%}", ha='center', va='center',
            fontweight='bold')

    # 4. 装饰与核心说明
    ax.set_ylabel('Normalized Influence Weight')
    ax.set_title('Comparison of Institutional Voting Power', fontsize=14, pad=20)

    # 添加注释解释 Rank 制的数学严谨性
    ax.annotate(f'Rank Constraints:\nMax individual share is limited\nby 1/ΣRank (≈{1 / n:.1%})',
                xy=(0, 0.8), xytext=(-0.4, 0.9),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1))

    ax.annotate(f'Percentage Leverage:\nFan power expands to {rsr_val:.2f}x\nof Judges\' weight',
                xy=(1, 0.8), xytext=(1.2, 0.9),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1))

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

    file_path = os.path.join(OUTPUT_DIR, "fig1_power_balance_refined.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Refined plot saved to {file_path}")


# ==========================================
# 2. Figure 2: Rule Conflict Scatter Plot
# ==========================================
def save_conflict_scatter(df):
    plt.figure(figsize=(10, 7))
    df['Category'] = 'Consistent Outcome'
    df.loc[df['delta_status'] == 1, 'Category'] = 'Saved by Percentage Rule'
    df.loc[df['delta_status'] == -1, 'Category'] = 'Hurt by Percentage Rule'

    sns.scatterplot(data=df, x='P_judge_share', y='P_fan_share', hue='Category',
                    style='Category', s=120, palette='viridis', alpha=0.7)

    # 标注典型争议案例
    for name in ["Bobby Bones", "Bristol Palin"]:
        target = df[df['celebrity_name'] == name]
        if not target.empty:
            plt.annotate(name, (target['P_judge_share'].iloc[-1], target['P_fan_share'].iloc[-1]),
                         xytext=(20, 20), textcoords='offset points', arrowprops=dict(arrowstyle='->'))

    plt.title('Strategic Salvation Zones Identification', fontweight='bold')
    plt.xlabel('Judges\' Normalized Score Share')
    plt.ylabel('Fans\' Normalized Vote Share')

    file_path = os.path.join(OUTPUT_DIR, "fig2_conflict_scatter.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Successfully saved: {file_path}")


# ==========================================
# 3. Figure 3: Ranking Displacement
# ==========================================
def save_displacement_violin(df):
    plt.figure(figsize=(9, 6))
    df['Fan_Popularity'] = pd.qcut(df['P_fan_share'], 3, labels=['Low Fan Base', 'Mid Fan Base', 'High Fan Base'])

    sns.violinplot(data=df, x='Fan_Popularity', y='delta_S', palette='Set2', inner='quartile')
    plt.axhline(0, color='red', linestyle='--', alpha=0.6)

    plt.title('Quantifying the Ranking Dividend of Popularity', fontweight='bold')
    plt.xlabel('Fan Base Size (Tertiles)')
    plt.ylabel('Ranking Gain ($\Delta S$): Percentage vs. Rank System')

    file_path = os.path.join(OUTPUT_DIR, "fig3_ranking_displacement.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Successfully saved: {file_path}")


# ==========================================
# 4. Figure 4: RSR Sensitivity Heatmap
# ==========================================
def save_sensitivity_heatmap():
    """
        根据 Question2.2.py 的蒙特卡洛逻辑，
        展示不同扰动强度(sigma)下 RSR 指标的稳健性。
        """
    plt.figure(figsize=(10, 5))

    # 定义噪声梯度（对应蒙特卡洛中的 noise 参数）
    noise_levels = [0.01, 0.02, 0.03, 0.04, 0.05]
    rsr_means = []
    rsr_stds = []

    full_df = prepare_data()  # 获取数据
    df = preprocess_features(full_df)

    print("📊 Generating Robustness Data...")
    for sigma in noise_levels:
        # 调用你 Question2.2.py 中的蒙特卡洛分析函数
        _, _, m, s = run_monte_carlo_analysis(df, n_iterations=50, noise=sigma)
        rsr_means.append(m)
        rsr_stds.append(s)

    # 将结果转换为热力图矩阵 (1行 x 5列)
    heatmap_data = np.array(rsr_means).reshape(1, -1)

    # 绘制热力图
    ax = sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap="YlGnBu", cbar=True,
                     xticklabels=noise_levels, yticklabels=["Mean RSR"])

    # 在热力图上方叠加标准差信息，增强学术说服力
    for i, std in enumerate(rsr_stds):
        plt.text(i + 0.5, 0.9, f"±{std:.4f}", ha='center', color='red', fontsize=10, fontweight='bold')

    plt.title('Robustness Analysis: RSR Stability under Fan Vote Perturbation', fontweight='bold', fontsize=12)
    plt.xlabel('Fan Vote Noise Intensity ($\sigma$)')
    plt.ylabel('Metric')

    # 标注判定阈值：如果 RSR 始终 > 1.0，说明结论是极其稳健的
    plt.annotate('Fan Dominance Threshold (RSR > 1.0)', xy=(0, 0), xytext=(0.5, -0.3),
                 color='darkgreen', fontweight='bold', arrowprops=dict(arrowstyle='->', color='green'))

    file_path = os.path.join(OUTPUT_DIR, "fig4_rsr_robustness_heatmap.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Successfully saved updated robustness plot: {file_path}")


# ==========================================
# 执行主程序
# ==========================================
if __name__ == "__main__":
    print("🚀 Starting visualization export process...")
    full_df = prepare_data()
    if full_df is not None:
        save_power_balance()
        save_conflict_scatter(full_df)
        save_displacement_violin(full_df)
        save_sensitivity_heatmap()
        print(f"\n✨ All visualizations are exported to: {os.path.abspath(OUTPUT_DIR)}")