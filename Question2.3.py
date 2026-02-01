import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_case_study():
    # 1. 加载前期生成的数据
    input_path = './outputs/dwts_rule_comparison.csv'
    if not pd.io.common.file_exists(input_path):
        print("❌ 错误：请先运行 Question2.2.py 以生成对比数据文件。")
        return

    df = pd.read_csv(input_path)

    # 2. 定义目标明星及其所在赛季
    controversial_stars = {
        "Jerry Rice": 2,
        "Billy Ray Cyrus": 4,
        "Bristol Palin": 11,
        "Bobby Bones": 27
    }

    results = []

    print("🚀 启动争议选手轨迹模拟 (反事实分析)...")
    print("-" * 60)

    for name, season in controversial_stars.items():
        # 提取该选手在该赛季的所有周数据
        star_data = df[(df['celebrity_name'] == name) & (df['season'] == season)].sort_values('week')

        if star_data.empty:
            continue

        # 获取实际结局（最后一站）
        actual_last_week = star_data['week'].max()

        # 模拟轨迹：在另一种规则下，哪一周会先触发淘汰？
        # 情况 A：现实是百分比制，模拟排名制 (R_is_elim)
        # 情况 B：现实是排名制，模拟百分比制 (P_is_elim)
        # 注意：CSV中的 R_is_elim 和 P_is_elim 已经根据各自规则算好了该周谁是垫底

        r_elim_week = star_data[star_data['R_is_elim'] == True]['week'].min()
        p_elim_week = star_data[star_data['P_is_elim'] == True]['week'].min()

        results.append({
            "Celebrity": name,
            "Season": season,
            "Max_Week": actual_last_week,
            "Rank_Rule_Fate": f"Week {int(r_elim_week)}" if not np.isnan(r_elim_week) else "Survived/Final",
            "Percent_Rule_Fate": f"Week {int(p_elim_week)}" if not np.isnan(p_elim_week) else "Survived/Final"
        })

    # 3. 打印对比报告
    report_df = pd.DataFrame(results)
    print(report_df.to_string(index=False))
    print("-" * 60)

    # 4. 深度逻辑解读
    print("\n💡 模拟轨迹解读：")
    for _, row in report_df.iterrows():
        if row['Rank_Rule_Fate'] != row['Percent_Rule_Fate']:
            print(f"⚠️ {row['Celebrity']}: 规则敏感型选手。")
            print(f"   在排名制下命运为 [{row['Rank_Rule_Fate']}]，而在百分比制下为 [{row['Percent_Rule_Fate']}]。")
            print(f"   结论：评分方式的选择直接改变了该选手的职业生涯轨迹。")
        else:
            print(f"✅ {row['Celebrity']}: 规则稳健型选手。两种评分方式下存活轨迹基本一致。")


def plot_case_study_trajectories():
    df = pd.read_csv('./outputs/dwts_rule_comparison.csv')
    cases = {
        "Jerry Rice": (2, 8),
        "Billy Ray Cyrus": (4, 8),
        "Bristol Palin": (11, 10),
        "Bobby Bones": (27, 9)
    }

    # 增加 figsize 的高度，并使用 constrained_layout 自动优化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plt.subplots_adjust(top=0.9)  # 核心修改：手动为总标题预留顶部 10% 的空间

    axes = axes.flatten()

    for i, (name, (season, final_week)) in enumerate(cases.items()):
        star_data = df[(df['celebrity_name'] == name) & (df['season'] == season)].sort_values('week')
        weeks = star_data['week'].tolist()

        # 修正路径逻辑：一旦触发淘汰(True)，后续状态均为 0
        r_path, p_path = [], []
        r_alive, p_alive = 1, 1
        for _, row in star_data.iterrows():
            if row['R_is_elim']: r_alive = 0
            if row['P_is_elim']: p_alive = 0
            r_path.append(r_alive)
            p_path.append(p_alive)

        ax = axes[i]
        # 使用 step 函数绘制阶梯图
        ax.step(weeks, r_path, label='Rank-based', where='post', linestyle='--', color='#e74c3c', lw=2)
        ax.step(weeks, p_path, label='Percent-based', where='post', color='#3498db', lw=2)

        # 绘制历史真实结局点
        ax.scatter(final_week, 1, color='gold', s=120, edgecolors='black', label='Actual Outcome', zorder=5)

        ax.set_title(f"Case: {name} (Season {season})", pad=15, fontsize=12)
        ax.set_ylim(-0.1, 1.3)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Eliminated', 'Alive'])
        ax.grid(axis='x', linestyle=':', alpha=0.6)

        if i == 0: ax.legend(loc='lower left', fontsize=9)

    # 整体布局优化
    plt.tight_layout(rect=[0, 0, 1, 0.93])  # 设定子图区域，顶部留出 7% 给 Suptitle
    plt.suptitle("Survival Path Analysis: Counterfactual Simulation vs. Reality",
                 fontsize=16, fontweight='bold', y=0.97)

    plt.show()


if __name__ == "__main__":
    run_case_study()
    plot_case_study_trajectories()