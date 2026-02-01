"""
generate_supplementary.py
生成美赛论文所需的“硬核”验证数据和表格素材。
1. Confusion Matrix (验证模型对淘汰的解释力)
2. Shocking Eliminations (异常值案例分析)
3. Summary Statistics (模型整体性能表)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ================= 配置 =================
INPUT_FILE = "outputs/latent_estimates.csv"
OUT_DIR = Path("outputs/supplementary_materials")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# 统一学术风设置
def set_style():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12


set_style()


def load_data():
    path = Path(INPUT_FILE)
    if not path.exists(): path = Path("latent_estimates.csv")
    if not path.exists(): raise FileNotFoundError(f"Missing {INPUT_FILE}")
    return pd.read_csv(path)


# ================= 1. 生成混淆矩阵热力图 =================
def plot_hit_rate_heatmap(df):
    """
    逻辑：检查模型推断出的"粉丝投票倒数名次"是否涵盖了真实的"淘汰者"。
    如果模型推断某人是倒数第一，而他确实被淘汰了，说明模型很准。
    """
    print("📊 Generating Accuracy Heatmap...")

    # 只看有淘汰发生的周
    elim_weeks = df[df['is_eliminated_this_week'] == True].copy()

    # 辅助函数：计算该选手在当周的模型估算排名 (按 fan_vote_mean 从低到高排)
    # Rank 1 = 倒数第一 (票数最少)
    def get_estimated_bottom_rank(row, full_df):
        week_data = full_df[(full_df['season'] == row['season']) & (full_df['week'] == row['week'])]
        # 升序排列，票最少的排前面
        week_data = week_data.sort_values('fan_vote_mean', ascending=True).reset_index(drop=True)
        try:
            rank = week_data[week_data['celebrity_name'] == row['celebrity_name']].index[0] + 1
            return rank
        except:
            return np.nan

    # 这步稍微有点慢，要有耐心
    elim_weeks['estimated_bottom_rank'] = elim_weeks.apply(lambda r: get_estimated_bottom_rank(r, df), axis=1)

    # 统计：淘汰者通常处于模型预测的倒数第几名？
    # 我们希望大多数淘汰者都在 "Bottom 1" 或 "Bottom 2"
    rank_counts = elim_weeks['estimated_bottom_rank'].value_counts().sort_index()

    # 只取前5名 (倒数1-5名)
    rank_counts = rank_counts.head(5)
    total_eliminations = rank_counts.sum()
    percentages = rank_counts / total_eliminations

    # 画热力图 (1x5 的条状图)
    plt.figure(figsize=(8, 3))
    heatmap_data = pd.DataFrame(percentages).T
    heatmap_data.columns = [f"Bottom {int(i)}" for i in heatmap_data.columns]
    heatmap_data.index = ["Probability"]

    sns.heatmap(heatmap_data, annot=True, fmt=".1%", cmap="Blues", cbar=False, linewidths=1, linecolor='black')
    plt.title(
        "Model Validation: Where do actual eliminated contestants rank in our model?\n(Bottom 1 = Model predicted lowest votes)",
        fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "Fig_Validation_HitRate.png", dpi=300)
    plt.close()

    # 保存数据给论文写字用
    rank_counts.to_csv(OUT_DIR / "validation_hit_rate.csv")


# ================= 2. 生成“最大冤案”列表 (Shocking Eliminations) =================
def generate_shocking_table(df):
    """
    逻辑：找出【评委分排名很高】但【惨遭淘汰】的选手。
    这些是论文 Discussion 部分的最佳素材。
    """
    print("📝 Generating Shocking Eliminations Table...")

    # 必须有评委分
    valid = df.dropna(subset=['judge_metric', 'fan_vote_mean']).copy()

    # 计算每一周的评委排名 (Rank 1 = 最高分)
    valid['judge_rank_top'] = valid.groupby(['season', 'week'])['judge_metric'].rank(ascending=False, method='min')

    # 筛选：被淘汰的人
    eliminated = valid[valid['is_eliminated_this_week'] == True].copy()

    # 计算差异：评委排名高 (数值小) 但被淘汰了 -> 冤案
    # 我们找 Judge Rank <= 3 的被淘汰者 (评委眼里的前三名)
    shocking = eliminated[eliminated['judge_rank_top'] <= 4].sort_values('judge_rank_top')

    # 整理输出列
    output_cols = ['season', 'week', 'celebrity_name', 'judge_rank_top', 'fan_vote_mean', 'ci_low', 'ci_high']
    table = shocking[output_cols].head(15)  # 取前15个案例

    # 格式化一下
    table.columns = ['Season', 'Week', 'Contestant', 'Judge Rank (Top)', 'Est. Fan Vote', 'CI Lower', 'CI Upper']
    table.to_csv(OUT_DIR / "Table_Shocking_Eliminations.csv", index=False)
    print("   -> Saved to Table_Shocking_Eliminations.csv")


# ================= 3. 生成模型鲁棒性统计表 =================
def generate_summary_stats(df):
    """
    逻辑：生成一个LaTeX风格的统计表，展示模型在不同赛季的表现。
    """
    print("📝 Generating Summary Statistics...")

    df['ci_width'] = df['ci_high'] - df['ci_low']

    stats = df.groupby('season').agg({
        'fan_vote_mean': ['count', 'mean'],  # 样本量
        'ci_width': 'mean',  # 平均不确定性 (越小越好)
        'slack_min': 'mean'  # 平均误差 (越接近0越好)
    }).reset_index()

    # 扁平化列名
    stats.columns = ['Season', 'Observations', 'Avg Vote Share', 'Avg Uncertainty (CI Width)', 'Model Slack (Error)']

    # 选取代表性赛季 (比如 S1, S5, S10... S28)
    selected_seasons = [1, 5, 10, 15, 20, 25, 28]
    final_table = stats[stats['Season'].isin(selected_seasons)]

    final_table.to_csv(OUT_DIR / "Table_Model_Robustness.csv", index=False)
    print("   -> Saved to Table_Model_Robustness.csv")


def main():
    df = load_data()
    plot_hit_rate_heatmap(df)
    generate_shocking_table(df)
    generate_summary_stats(df)
    print(f"\n✅ All supplementary materials saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()