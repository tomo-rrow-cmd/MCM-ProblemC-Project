import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr

# 路径自动适配
DATA_PATH = Path("./outputs/q4_risk_monitor_features.csv")
OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def arhm_engine_v3(df, gamma, w_j=0.5, min_k=0.1):
    """
    ARHM 3.0: 向量化加速 + 异常鲁棒机制
    """
    df = df.copy()

    # 1. 计算变异系数 (CV) - 使用 transform 提升速度
    grp = df.groupby(['season', 'week'])['judge_raw']
    df['cv_j'] = grp.transform(lambda x: x.std() / x.mean() if x.mean() > 0 else 0).fillna(0)

    # 2. 动态阻尼 k 计算 (带有 min_k 保底)
    # 使用 np.clip 防止极端 gamma 导致权重归零
    df['dynamic_k'] = (df['adaptive_k'] * np.exp(-gamma * df['cv_j'])).clip(lower=min_k)

    # 3. 计算评委占比与粉丝占比 (Log-Damping)
    df['p_j'] = df['judge_raw'] / df.groupby(['season', 'week'])['judge_raw'].transform('sum')

    # 使用 log1p (log(1+x)) 提高数值稳定性
    df['damped_f'] = np.log1p(df['dynamic_k'] * (df['fan_vote_share_pct'] / 100.0))
    df['p_f_adj'] = df['damped_f'] / df.groupby(['season', 'week'])['damped_f'].transform('sum')

    # 4. 汇总得分与排名
    df['arhm_total_score'] = w_j * df['p_j'] + (1 - w_j) * df['p_f_adj']
    df['arhm_rank'] = df.groupby(['season', 'week'])['arhm_total_score'].rank(ascending=False)

    return df


def tune_and_test_robustness(df):
    """
    参数调优 + 鲁棒性校验
    """
    gamma_range = np.linspace(0, 5, 26)
    tuning_results = []

    # 预先计算评委排名用于对比
    df['judge_rank_ref'] = df.groupby(['season', 'week'])['judge_raw'].rank(ascending=False)

    for g in gamma_range:
        sim = arhm_engine_v3(df, gamma=g)

        # 指标 A: 技术保真度 (TF) - 排除 NaN 干扰
        valid_mask = sim['arhm_rank'].notna() & sim['judge_rank_ref'].notna()
        if valid_mask.any():
            tf_score = spearmanr(sim.loc[valid_mask, 'arhm_rank'],
                                 sim.loc[valid_mask, 'judge_rank_ref']).correlation
        else:
            tf_score = 0

        # 指标 B: 粉丝保留率 (FEI)
        # 逻辑：历史粉丝票前二名，在规则下是否依然留在前三？
        fan_top2 = sim.groupby(['season', 'week'])['fan_vote_share_pct'].rank(ascending=False) <= 2
        fei_score = (sim.loc[fan_top2, 'arhm_rank'] <= 3).mean()

        tuning_results.append({'gamma': g, 'TF': tf_score, 'FEI': fei_score})

    tuning_df = pd.DataFrame(tuning_results).fillna(0)  # 核心修复：填充所有 NaN

    # 平衡得分：寻找 TF 和 FEI 的几何平均最大点
    tuning_df['balance_score'] = np.sqrt(tuning_df['TF'] * tuning_df['FEI'])

    # 安全获取索引
    if tuning_df['balance_score'].max() > 0:
        best_idx = tuning_df['balance_score'].idxmax()
    else:
        best_idx = 0  # 退回默认 gamma=0

    return tuning_df.loc[best_idx, 'gamma'], tuning_df


def main():
    if not DATA_PATH.exists():
        print(f"❌ 找不到数据文件: {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH).dropna(subset=['judge_raw', 'fan_vote_share_pct'])

    # 1. 运行优化
    best_gamma, tuning_history = tune_and_test_robustness(df)
    print(f"✅ 最优参数确认: γ = {best_gamma:.2f}")

    # 2. 生成最终模拟结果
    final_sim = arhm_engine_v3(df, gamma=best_gamma)

    # 3. 鲁棒性检查 (Robustness Test): 计算排名变动标准差
    # 模拟 γ 波动 ±10% 时排名的变化幅度
    sim_plus = arhm_engine_v3(df, gamma=best_gamma * 1.1)
    rank_diff = np.abs(final_sim['arhm_rank'] - sim_plus['arhm_rank']).mean()
    print(f"🛡️ 规则鲁棒性测试: 排名波动度 = {rank_diff:.4f} (越小越稳健)")

    # 保存结果
    final_sim.to_csv(OUTPUT_DIR / "q4_final_arhm.csv", index=False)
    tuning_history.to_csv(OUTPUT_DIR / "q4_gamma_tuning.csv", index=False)

    # 绘图
    plt.figure(figsize=(10, 5))
    plt.plot(tuning_history['gamma'], tuning_history['TF'], 'b-', label='Technical Fidelity')
    plt.plot(tuning_history['gamma'], tuning_history['FEI'], 'g--', label='Fan Engagement')
    plt.axvline(best_gamma, color='r', label=f'Optimal Gamma: {best_gamma}')
    plt.title("Pareto Frontier & Robustness Analysis")
    plt.xlabel("Gamma (Sensitivity)")
    plt.ylabel("Index Score")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_DIR/"fig_q4_pareto_gamma.png")


if __name__ == "__main__":
    main()