import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# --- 配置 ---
# 请确保这个路径指向你之前生成的 dwts_rule_comparison_final.csv
input_file = './outputs/dwts_rule_comparison.csv'
output_dir = './outputs/visualizations'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- 1. 读取数据 ---
df_results = pd.read_csv(input_file)

# --- 2. 选择一个或几个“典型冲突周”进行可视化 ---
# 筛选出有冲突的周 (delta_status != 0)
conflict_weeks = df_results[df_results['delta_status'] != 0][['season', 'week']].drop_duplicates()

if conflict_weeks.empty:
    print("没有检测到规则冲突的周，无法生成冲突冲积图。")
    print("请检查数据或尝试更宽泛的筛选条件。")
else:
    # 示例：选择第一个有冲突的周进行可视化
    # 你可以手动选择一个你认为最具代表性的赛季和周
    # sample_season = conflict_weeks.iloc[0]['season']
    # sample_week = conflict_weeks.iloc[0]['week']

    # 也可以选择 Bobby Bones 或 Jennie Garth 的相关周进行重点分析
    # sample_season = 27 # Bobby Bones 赛季
    # sample_week = 11   # Bobby Bones 夺冠周前的淘汰周，或其他有争议的周

    # 修改为自动选择检测到的第一个冲突周
    if not conflict_weeks.empty:
        sample_season = conflict_weeks.iloc[0]['season']
        sample_week = conflict_weeks.iloc[0]['week']
        print(f"Detected conflict in Season {sample_season}, Week {sample_week}. Generating plot...")
    else:
        # 如果实在没有冲突，再手动指定一个普通周查看逻辑
        sample_season = 1
        sample_week = 2

    # 获取特定周的数据
    week_data = df_results[(df_results['season'] == sample_season) &
                           (df_results['week'] == sample_week)].copy()

    if week_data.empty:
        print(f"警告：选择的赛季 {sample_season} 周 {sample_week} 没有数据或不是冲突周。")
        # 如果手动选择了，但没数据，就尝试第一个冲突周
        sample_season = conflict_weeks.iloc[0]['season']
        sample_week = conflict_weeks.iloc[0]['week']
        week_data = df_results[(df_results['season'] == sample_season) &
                               (df_results['week'] == sample_week)].copy()

    week_data = week_data.sort_values(by='R_total_score', ascending=True).reset_index(drop=True)

    print(f"\n正在为赛季 {sample_season} 周 {sample_week} 生成冲积图...")
    print("--------------------------------------------------")

    # --- 3. 准备绘图数据 ---
    # 定义淘汰状态的数值表示 (方便绘图)
    # 0: 晋级，1: 淘汰
    week_data['R_status'] = week_data['R_is_elim'].astype(int)
    week_data['P_status'] = week_data['P_is_elim'].astype(int)

    # 给选手排序，以便在图上清晰显示
    # 可以按 R_total_score 排序，让排名制的结果看起来有序
    # week_data = week_data.sort_values(by='R_total_score', ascending=True)

    # 计算淘汰者的“名次”
    elim_rank_r = week_data[week_data['R_is_elim']].index.tolist()
    elim_rank_p = week_data[week_data['P_is_elim']].index.tolist()

    # --- 4. 绘制模拟冲积图 ---
    fig, ax = plt.subplots(figsize=(10, len(week_data) * 0.8))  # 动态调整图高

    # 绘制左侧 (排名制结果)
    y_pos = np.arange(len(week_data))

    ax.barh(y_pos, 0.4, color='skyblue', align='center', label='Rank-based Result')
    # 标记排名制下的淘汰者
    elim_indices_r = week_data[week_data['R_is_elim']].index
    ax.barh(elim_indices_r, 0.4, color='red', align='center')

    # 绘制右侧 (百分比制结果)
    ax.barh(y_pos, 0.4, left=1.6, color='lightgreen', align='center', label='Percentage-based Result')
    # 标记百分比制下的淘汰者
    elim_indices_p = week_data[week_data['P_is_elim']].index
    ax.barh(elim_indices_p, 0.4, left=1.6, color='red', align='center')

    # 绘制连接线
    for idx, row in week_data.iterrows():
        # 如果状态发生冲突 (R_is_elim != P_is_elim)
        if row['delta_status'] != 0:
            # R_status = 1 (淘汰), P_status = 0 (晋级) -> 红色线 (R-elim to P-save)
            if row['R_is_elim'] and not row['P_is_elim']:
                line_color = 'purple'  # 排名制淘汰，百分比制挽救
                label_text = 'Rank Elim, Pct Save'
            # R_status = 0 (晋级), P_status = 1 (淘汰) -> 橙色线 (R-save to P-elim)
            elif not row['R_is_elim'] and row['P_is_elim']:
                line_color = 'orange'  # 排名制挽救，百分比制淘汰
                label_text = 'Rank Save, Pct Elim'
            else:
                line_color = 'gray'  # 其他冲突（理论上 delta_status 只会有 1, -1）

            ax.plot([0.4, 1.6], [idx, idx], color=line_color, linestyle='--', linewidth=2, alpha=0.7)

            # 添加文本标签以识别冲突的选手
            ax.text(1.0, idx, f"{row['celebrity_name']}",
                    verticalalignment='center', horizontalalignment='center',
                    fontsize=9, color='black', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

        # 即使没有冲突，也画一条淡色的连接线，表明选手在两种规则下都是相同的命运
        else:
            ax.plot([0.4, 1.6], [idx, idx], color='lightgray', linestyle='-', linewidth=0.5, alpha=0.5)

    # 设置标签和标题
    ax.set_yticks(y_pos)
    ax.set_yticklabels(week_data['celebrity_name'])
    ax.set_xlim(-0.1, 2.1)
    ax.set_xticks([0.2, 1.8])
    ax.set_xticklabels(['Rank-based Result', 'Percentage-based Result'], fontsize=12)
    ax.set_title(f'Contestant Fate Flow - Season {sample_season}, Week {sample_week}', fontsize=14)
    ax.invert_yaxis()  # 让排名第一的在最上面

    # 添加图例
    red_patch = mpatches.Patch(color='red', label='Eliminated (模拟)')
    purple_patch = mpatches.Patch(color='purple', label='Rule Conflict: R-Elim, Pct-Save')
    orange_patch = mpatches.Patch(color='orange', label='Rule Conflict: Pct-Elim, R-Save')
    lightgray_patch = mpatches.Patch(color='lightgray', label='Consistent Fate')

    ax.legend(handles=[red_patch, purple_patch, orange_patch, lightgray_patch],
              loc='upper right', bbox_to_anchor=(1.0, 1.0))

    ax.axis('off')  # 隐藏坐标轴边框

    # 保存图像
    output_image_path = os.path.join(output_dir, f'alluvial_s{sample_season}_w{sample_week}.png')
    plt.tight_layout()
    plt.savefig(output_image_path)
    plt.close()
    print(f"冲积图已保存至: {output_image_path}")