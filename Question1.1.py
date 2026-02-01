import pandas as pd
import numpy as np
import time


def estimate_fan_votes_with_metrics(file_path, num_simulations=10000):
    print("🚀 启动量化评估模型，正在执行高性能模拟...")
    df = pd.read_csv(file_path)
    results = []
    start_time = time.time()

    # 按照赛季和周进行分组
    grouped = df.groupby(['season', 'week'])
    total_weeks = len(grouped)

    for (season, week), group in grouped:
        n = len(group)
        mechanism = group['mechanism'].iloc[0]
        true_priority = group['elim_priority'].values
        j_shares = group['judge_share'].values
        j_ranks = group['judge_rank'].values

        # 1. 向量化生成 Dirichlet 样本 (保持得票率之和为 1)
        fan_samples = np.random.dirichlet(np.ones(n), size=num_simulations)

        # 2. 向量化计算综合排名 (基于题目定义的两种机制)
        if mechanism == 'Percent':
            total_scores = j_shares + fan_samples
            calc_ranks = np.argsort(np.argsort(-total_scores, axis=1), axis=1) + 1
        else:
            # 排名法：裁判排名 + 粉丝排名 (票数多则排名小/靠前)
            fan_ranks = np.argsort(np.argsort(-fan_samples, axis=1), axis=1) + 1
            total_scores = j_ranks + fan_ranks
            calc_ranks = np.argsort(np.argsort(total_scores, axis=1), axis=1) + 1

        # 3. 验证一致性 (Consistency Check)
        # 比对每一行模拟出的排名顺序是否与真实 elim_priority 一致
        true_sort_order = np.argsort(true_priority)
        calc_sort_order = np.argsort(calc_ranks, axis=1)
        matches = np.all(calc_sort_order == true_sort_order, axis=1)

        valid_samples = fan_samples[matches]
        num_valid = len(valid_samples)

        # --- 指标结算 ---
        # 一致性指标：成功模拟出真实结果的概率 (P_valid)
        consistency_val = num_valid / num_simulations

        if num_valid > 0:
            means = valid_samples.mean(axis=0)
            stds = valid_samples.std(axis=0)
            # 确定性指标：变异系数的倒数 (Signal-to-Noise Ratio)
            # 只有当有效解空间越窄，std越小，确定性越高
            certainty_vals = np.clip(means / (stds + 1e-6), 0, 1000)
        else:
            # 兜底逻辑
            means = np.full(n, 1.0 / n)
            certainty_vals = np.zeros(n)

        # 封装
        for i, name in enumerate(group['celebrity_name']):
            results.append({
                'season': season,
                'week': week,
                'celebrity_name': name,
                'est_fan_share': means[i],
                'certainty_score': certainty_vals[i],
                'consistency_metric': consistency_val
            })

    end_time = time.time()
    print(f"✅ 计算完成！总耗时: {end_time - start_time:.2f}s")
    return pd.DataFrame(results)


# 执行并分析结果
if __name__ == "__main__":
    # 执行 15000 次模拟以获得更稳健的统计特性
    output_df = estimate_fan_votes_with_metrics('Cleaned_For_question1.csv', 15000)

    # 保存结果
    output_df.to_csv('Q1_Fan_Vote_Estimates_With_Metrics.csv', index=False)

    # --- 统计汇总（直接用于回答题目） ---
    print("\n" + "=" * 60)
    print("【题目要求量化指标 1：一致性 (Consistency)】")
    avg_consistency = output_df['consistency_metric'].mean()
    print(f"全赛季模型预测与淘汰名单的平均一致性概率: {avg_consistency:.4f}")
    print("注：该值代表模型在解空间中找到符合实际淘汰结果的‘逻辑通路’的能力。")

    print("\n【题目要求量化指标 2：确定性 (Certainty)】")
    certainty_stats = output_df['certainty_score'].describe()
    print(certainty_stats)

    # 验证确定性是否随周次/参赛者改变
    certainty_cv = output_df['certainty_score'].std() / output_df['certainty_score'].mean()
    print(f"\n确定性的变异系数 (CV): {certainty_cv:.4f}")
    if certainty_cv > 0.1:
        print("结论：确定性 CV 显著，证明确定性随参赛者和比赛周次动态波动，并非恒定。")
    print("=" * 60)