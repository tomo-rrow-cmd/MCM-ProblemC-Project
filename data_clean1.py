import pandas as pd
import numpy as np

def generate_cleaned_modeling_table(input_file):
    # 1. 加载原始数据
    df = pd.read_csv(input_file)

    # 2. 核心清洗：宽表转长表 (每一行代表：一名选手在某赛季某周的表现)
    rows = []
    # 遍历 34 个赛季的数据
    for _, celebrity in df.iterrows():
        # 遍历可能的 11 周
        for week in range(1, 12):
            # 动态获取评委得分列名
            j_cols = [f'week{week}_judge{i}_score' for i in range(1, 5)]
            # 提取得分
            scores = [celebrity.get(col) for col in j_cols if col in df.columns]

            # 计算总分（忽略 NA）
            valid_scores = [float(s) for s in scores if pd.notna(s) and str(s).strip() != 'N/A']
            total_judge_score = sum(valid_scores)
            judge_count = len(valid_scores)

            # 只有当选手在该周有分数且分数大于 0 时，才认为其参与了博弈
            if judge_count > 0 and total_judge_score > 0:
                rows.append({
                    'celebrity_name': celebrity['celebrity_name'],
                    'season': celebrity['season'],
                    'week': week,
                    'total_judge_score': total_judge_score,
                    'judge_count': judge_count,
                    'placement_final': celebrity['placement'],  # 全局最终排名
                    'results_raw': celebrity['results'],
                    'industry': celebrity['celebrity_industry'],
                    'age': celebrity['celebrity_age_during_season']
                })

    # 转换为 DataFrame
    clean_df = pd.DataFrame(rows)

    # 3. 机制标记 (Mechanism Tagging)
    # S1-S2, S28-S34 为 Rank (排名法); 其余为 Percent (百分比法)
    def assign_mechanism(s):
        if 1 <= s <= 2 or 28 <= s <= 34: return 'Rank'
        return 'Percent'

    clean_df['mechanism'] = clean_df['season'].apply(assign_mechanism)

    # 4. 构建“周内博弈”特征（处理双重淘汰的关键）
    # 按赛季和周分组，计算周内排名和占比
    def process_week_logic(group):
        # 裁判分占比 (用于 Percent 法)
        group['judge_share'] = group['total_judge_score'] / group['total_judge_score'].sum()

        # 裁判排名 (用于 Rank 法)
        group['judge_rank'] = group['total_judge_score'].rank(ascending=False, method='min')

        # 打破平局逻辑：利用 placement_final 建立严格的淘汰顺序
        # 最终排名越靠后（数值越大），综合实力排名越低
        # 我们创建一个 combined_sort_val，数值越小综合实力越强
        group['elim_priority'] = group['placement_final'].rank(ascending=True)
        return group

    clean_df = clean_df.groupby(['season', 'week']).apply(process_week_logic).reset_index(drop=True)

    # 5. 标记该周是否被淘汰
    # 如果 results 里的周数等于当前周数，则标记为 1
    def check_elim(row):
        res = str(row['results_raw'])
        if f'Week {row["week"]}' in res and 'Eliminated' in res:
            return 1
        return 0

    clean_df['is_eliminated_this_week'] = clean_df.apply(check_elim, axis=1)

    # 6. 保存新表格
    clean_df.to_csv('Cleaned_For_question1.csv', index=False)
    print("清洗完成！生成了新表格：Cleaned_For_question1.csv")
    return clean_df

# 执行
result_table = generate_cleaned_modeling_table('2026_MCM_Problem_C_Data.csv')