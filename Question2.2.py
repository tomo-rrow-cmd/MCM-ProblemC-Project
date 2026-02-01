import pandas as pd
import numpy as np
from sklearn.svm import SVC
from scipy.stats import pearsonr
import os

# ==========================================
# 模块 1: 数据预处理 (量纲归一化与并列平滑)
# ==========================================
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


# ==========================================
# 模块 2: 动态样本加权 (解决权重逻辑简单问题)
# ==========================================
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


# ==========================================
# 模块 3: 核心指标计算与敏感度建模
# ==========================================
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
        if df_noisy['delta_S'].std() > 1e-7 and df_noisy['noisy_fan'].std() > 1e-7:
            dcc, _ = pearsonr(df_noisy['delta_S'], df_noisy['noisy_fan'])
            if not np.isnan(dcc):
                dcc_list.append(dcc)

        # --- [步骤 3] 计算指标 B: RSR (修正后的SVM训练) ---
        try:
            # 排除由于原始数据缺失导致的空值（如18季第2周）
            valid_idx = df_noisy[['P_judge_share', 'noisy_fan', 'P_is_elim']].dropna().index

            if len(valid_idx) > 5:  # 确保有足够样本
                X_sub = df_noisy.loc[valid_idx, ['P_judge_share', 'noisy_fan']]
                y_sub = df_noisy.loc[valid_idx, 'P_is_elim'].astype(int)
                w_sub = weights.loc[valid_idx] # 权重同步过滤

                if len(y_sub.unique()) > 1:  # 确保样本包含 0 和 1 两种状态
                    svm = SVC(kernel='linear', C=1.0)
                    svm.fit(X_sub, y_sub, sample_weight=w_sub)

                w_j, w_f = svm.coef_[0]
                # 计算灵敏度比率
                rsr_list.append(abs(w_f / w_j) if abs(w_j) > 1e-9 else 1.0)
        except Exception as e:
            continue

    return np.mean(dcc_list), np.std(dcc_list), np.mean(rsr_list), np.std(rsr_list)


def calculate_fsr(df):
    """
    指标 3: 粉丝救赎率 (冲突样本归因)
    """
    salvaged = df[df['delta_status'] == 1]
    if salvaged.empty: return 0.0, 0.0
    val = salvaged['P_fan_share'].mean()
    intensity = val / df['P_fan_share'].mean()
    return val, intensity


# ==========================================
# 主函数 (Main Control Flow)
# ==========================================
def main():
    print("开始 Q2 深度建模分析...")

    # 1. 加载数据
    path = './outputs/dwts_rule_comparison.csv'
    if not os.path.exists(path):
        print("错误：未找到数据文件。")
        return
    df = pd.read_csv(path)

    # --- 【新增】合并 Q1 的可靠性指标 ---
    latent_file = './outputs/latent_estimates_filled.csv'
    if os.path.exists(latent_file):
        df_latent = pd.read_csv(latent_file)[
            ['season', 'week', 'celebrity_name', 'abc_accept_rate', 'ci_low', 'ci_high']]
        df = df.merge(df_latent, on=['season', 'week', 'celebrity_name'], how='left')
    # 填充缺失的可靠性指标（防止后续计算报错）
    df['abc_accept_rate'] = df['abc_accept_rate'].fillna(df['abc_accept_rate'].mean())
    df['ci_low'] = df['ci_low'].fillna(0)
    df['ci_high'] = df['ci_high'].fillna(1)

    # 2. 调用方法：预处理 (量纲归一化 & 并列平滑)
    df = preprocess_features(df)

    # 3. 调用方法：敏感度建模与求解 (DCC & RSR)
    dcc_m, dcc_s, rsr_m, rsr_s = run_monte_carlo_analysis(df, n_iterations=200)

    # 4. 调用方法：计算 FSR (粉丝救赎率)
    fsr_val, fsr_int = calculate_fsr(df)

    # 5. 打印最终量化成果
    print("\n" + "═" * 60)
    print(f"{'Dancing with the Stars Scoring System Analysis':^60}")
    print(f"{'Quantitative Evidence Report (Q2)':^60}")
    print("═" * 60)

    # 指标 1: DCC
    print(f"【指标 A】位移相关性 DCC (Displacement Correlation Coefficient)")
    print(f"   ➤ 均值(μ): {dcc_m:.4f} | 标准差(σ): {dcc_s:.4f}")
    dcc_desc = "强正相关，粉丝量直接决定规则红利" if dcc_m > 0.5 else "中等相关" if dcc_m > 0.2 else "低相关"
    print(f"   ➤ 状态解析: {dcc_desc}")

    # 指标 2: RSR
    print(f"\n【指标 B】权重灵敏度 RSR (Rule Sensitivity Ratio)")
    print(f"   ➤ 均值(μ): {rsr_m:.4f} | 标准差(σ): {rsr_s:.4f}")
    rsr_desc = "观众票控制权主导 (Fan Dominance)" if rsr_m > 1.0 else "评委打分控制权主导 (Judge Dominance)"
    print(f"   ➤ 状态解析: {rsr_desc}")

    # 指标 3: FSR
    print(f"\n【指标 C】粉丝救赎强度 FSR (Fan Salvage Rate)")
    print(f"   ➤ 冲突幸存者平均粉丝占比: {fsr_val:.2%}")
    print(f"   ➤ 相对救赎倍率: {fsr_int:.2f}x")
    print(f"   ➤ 状态解析: 百分比制对高人气选手的“特赦”能力是平均水平的 {fsr_int:.2f} 倍")

    print("-" * 60)

    # 6. 最终综合判定逻辑
    # 判定准则：如果 DCC 显著为正 且 RSR > 1，则判定为百分比制显著倾向观众
    is_fan_biased = dcc_m > 0.3 and rsr_m > 1.05

    print(f"【最终综合判定】")
    if is_fan_biased:
        bias_type = "Percentage-based (显著倾向观众票)"
        reason = "百分比制通过‘连续性权重’放大了粉丝数量优势，产生了显著的杠杆效应。"
    else:
        bias_type = "Rank-based (更倾向专业评委/平衡)"
        reason = "排名制通过‘阶梯式阶阶’平滑了粉丝数量的极端影响，维持了评委的专业控制力。"

    print(f"   🏆 结论: {bias_type}")
    print(f"   📝 依据: {reason}")
    print("═" * 60)


if __name__ == "__main__":
    main()