import pandas as pd
import numpy as np
import os

# 1. 环境准备
# 脚本检测：从 Question2 阶段读取你刚才逆向工程生成的表格
input_path = './outputs/latent_estimates_filled.csv'
output_dir = './outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 2. 读取数据
df = pd.read_csv(input_path)

def process_dwts_logic(group):
    """
    更新后的计算逻辑：使用逆向工程得到的 share_pct
    """
    n_active = len(group)
    group['n_active'] = n_active

    # --- R (Rank-based) 路径计算 ---
    # 评委排名：分数越高排名越小(1st)
    group['R_judge_rank'] = group['judge_raw'].rank(ascending=False, method='min')

    # 粉丝排名：根据你预估的 fan_vote_mean 排序
    # 注意：fan_vote_mean 小的代表名次靠前，所以 ascending=True
    group['R_fan_rank'] = group['fan_vote_mean'].rank(ascending=True, method='min')

    group['R_total_score'] = group['R_judge_rank'] + group['R_fan_rank']

    # 排名制判定：排名之和最大的选手（即表现最差的）标记为 True
    max_r_score = group['R_total_score'].max()
    group['R_is_elim'] = group['R_total_score'] == max_r_score

    # --- P (Percentage-based) 路径计算 ---
    # 评委占比：(个人总分 / 该周所有人总分)
    group['P_judge_share'] = group['judge_raw'] / group['judge_raw'].sum()

    # 粉丝占比：直接使用你“逆向工程”得到的严谨数据
    # 将百分数转换为 0-1 的小数
    group['P_fan_share'] = group['fan_vote_share_pct'] / 100.0

    group['P_total_score'] = group['P_judge_share'] + group['P_fan_share']

    # 百分比制判定：总分（占比之和）最低的选手标记为 True
    min_p_score = group['P_total_score'].min()
    group['P_is_elim'] = group['P_total_score'] == min_p_score

    # --- 冲突分析 (Conflict Detection) ---
    # 逻辑：
    #  0: 两者结论一致（都觉得该走或都觉得该留）
    #  1: 排名制判定淘汰，但百分比制判定留（说明百分比制保护了此人）
    # -1: 百分比制判定淘汰，但排名制判定留（说明排名制保护了此人）
    group['delta_status'] = group['R_is_elim'].astype(int) - group['P_is_elim'].astype(int)

    return group


# 3. 执行分组计算
print("正在执行 Q2 核心模拟：对比排名制与百分比制的判定差异...")
# group_keys=False 保持索引结构
new_df = df.groupby(['season', 'week'], group_keys=False).apply(process_dwts_logic)

# 4. 格式化并输出
output_columns = [
    'season', 'week', 'celebrity_name', 'n_active', 'is_eliminated_this_week',
    'R_judge_rank', 'R_fan_rank', 'R_total_score', 'R_is_elim',
    'P_judge_share', 'P_fan_share', 'P_total_score', 'P_is_elim',
    'delta_status'
]

final_table = new_df[output_columns]
output_file = os.path.join(output_dir, 'dwts_rule_comparison.csv')
final_table.to_csv(output_file, index=False)

print("-" * 30)
print(f"处理完成！核心分析表已生成: {output_file}")
conflict_count = len(final_table[final_table['delta_status'] != 0])
print(f"共检测到 {conflict_count} 例规则冲突（命运翻转案例）。")