import pandas as pd
import numpy as np


def repair_season_18_week_2(file_path, output_path):
    # 1. 加载数据
    df = pd.read_csv(file_path)

    # 定义需要对齐的技术指标和基础数据
    # 基础数值：用于后续计算
    base_val_cols = ['judge_raw', 'judge_metric', 'fan_vote_mean']
    # 技术约束：直接线性平滑对齐
    tech_constraint_cols = ['abc_accept_rate', 'feasible_min', 'feasible_max', 'slack_min']

    all_target_cols = base_val_cols + tech_constraint_cols

    def process_group(group):
        # 仅针对第18季的选手进行处理
        if group['season'].iloc[0] != 18:
            return group

        group = group.sort_values('week')

        # 检查是否缺失第2周，或者第2周数据全为空
        if 2 in group['week'].values:
            # 执行线性插值
            # limit_direction='both' 确保如果第1周或最后一周缺失也能处理（虽本例不涉及）
            group[all_target_cols] = group[all_target_cols].interpolate(method='linear', limit_direction='both')

            

        return group

    # 2. 执行分组插值
    print("正在进行线性平滑填充...")
    df_filled = df.groupby('celebrity_name', group_keys=False).apply(process_group)

    # 3. 核心修正：针对第18季第2周重新归一化百分比份额
    # 因为单独对每个选手的 mean 进行插值后，总和可能不等于 100
    print("正在根据逆向工程逻辑重新校准份额...")
    s18w2_mask = (df_filled['season'] == 18) & (df_filled['week'] == 2)

    if s18w2_mask.any():
        # 提取第2周所有选手的平均投票值
        means = df_filled.loc[s18w2_mask, 'fan_vote_mean']
        # 归一化计算
        total_mean = means.sum()
        if total_mean > 0:
            df_filled.loc[s18w2_mask, 'fan_vote_share_pct'] = (means / total_mean) * 100

        # 重新填充置信区间 (CI)
        # 逻辑：取前后两周 CI 宽度的平均值，应用到当前的 mean 上
        group_gb = df_filled[df_filled['season'] == 18].groupby('celebrity_name')
        for name, group in group_gb:
            if 2 in group['week'].values:
                # 简单线性插值处理 ci_low 和 ci_high
                idx = df_filled[(df_filled['celebrity_name'] == name) & (df_filled['season'] == 18) & (
                            df_filled['week'] == 2)].index
                df_filled.loc[idx, ['ci_low', 'ci_high']] = group[['ci_low', 'ci_high']].interpolate().iloc[1:2].values

    # 4. 保存结果
    df_filled.to_csv(output_path, index=False)
    print(f"处理完成！文件已保存至: {output_path}")


# 调用函数
repair_season_18_week_2('./outputs/latent_estimates.csv', './outputs/latent_estimates_filled.csv')