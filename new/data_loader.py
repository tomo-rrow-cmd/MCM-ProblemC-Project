#!/usr/bin/env python3
"""
data_loader_mcm_v7_fusion.py
Python 3.10+
【2026 MCM Problem C 数据清洗 - V7 完美融合版】

功能：
1. 内核：使用 V6 的高级逻辑（Rank=Min, Double Elimination Fix, Partner Quality）。
2. 输出：兼容旧版文件名 (cleaned_data.csv) + 自动生成体检报告和分布图。
3. 目的：让用户可以直接替换旧数据，而无需修改后续建模代码。
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("mcm_data_processing.log", mode="w", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# 1. 智能列名映射
# ---------------------------------------------------------
COLUMN_PATTERNS = {
    "celebrity_name": r"(celebrity|star).*name",
    "ballroom_partner": r"(ballroom|pro).*partner",
    "celebrity_industry": r"celebrity.*industry",
    "celebrity_homestate": r"celebrity.*homestate",
    "celebrity_homecountry": r"celebrity.*homecountry.*region",
    "celebrity_age": r"celebrity.*age.*season|age.*season",
    "season": r"^season$",
    "results": r"results?",
    "placement": r"placement",
    "judge_score": r"week(\d+)_judge(\d+)_score"
}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    new_cols = {}
    matched_standards = set()
    standard_keys = list(COLUMN_PATTERNS.keys())

    for col in df.columns:
        if re.match(COLUMN_PATTERNS["judge_score"], col, re.IGNORECASE):
            continue
        for std_key in standard_keys:
            if std_key == "judge_score": continue
            if std_key in matched_standards: continue
            if re.search(COLUMN_PATTERNS[std_key], col, re.IGNORECASE):
                new_cols[col] = std_key
                matched_standards.add(std_key)
                break

    required = ["celebrity_name", "season", "results"]
    missing = [k for k in required if k not in matched_standards and k not in df.columns]
    if missing:
        logger.error(f"严重错误：缺失核心列 {missing}")
        raise ValueError(f"Missing core columns: {missing}")

    return df.rename(columns=new_cols)


# ---------------------------------------------------------
# 2. 宽表转长表 (修复版：防止非US选手因缺失State被删除)
# ---------------------------------------------------------
def wide_to_long(df_wide: pd.DataFrame) -> pd.DataFrame:
    # 1. 找到分数列和ID列
    score_cols = [c for c in df_wide.columns if re.match(r"week(\d+)_judge(\d+)_score", c, re.IGNORECASE)]
    id_vars = [c for c in df_wide.columns if c not in score_cols]

    # 【关键修复步骤】
    # 在 pivot 之前，必须填充 ID 列的空值！
    # 否则 pivot_table 会把 homestate 为空的非美国人直接删掉
    for col in id_vars:
        # 如果是字符串类型的列，填 "Unknown"；如果是数字，填 -1
        if df_wide[col].dtype == 'object':
            df_wide[col] = df_wide[col].fillna("Unknown")
        else:
            df_wide[col] = df_wide[col].fillna(-1)

    # 2. Melt (宽变长)
    melted = df_wide.melt(
        id_vars=id_vars,
        value_vars=score_cols,
        var_name="temp_key",
        value_name="score"
    )

    # 3. 提取 Week 和 Judge
    pattern = re.compile(r"week(\d+)_judge(\d+)_score", re.IGNORECASE)
    extracted = melted["temp_key"].str.extract(pattern)
    melted["week"] = extracted[0].astype(int)
    melted["judge_idx"] = extracted[1].astype(int)
    melted["score"] = pd.to_numeric(melted["score"], errors='coerce')

    # 4. Pivot (透视回标准格式)
    pivot_index = [c for c in id_vars if c in melted.columns] + ["week"]

    # 去重
    melted = melted.drop_duplicates(subset=pivot_index + ["judge_idx"])

    pivoted = melted.pivot_table(
        index=pivot_index,
        columns="judge_idx",
        values="score",
        aggfunc="first"
    )

    pivoted.columns = [f"judge{c}_score" for c in pivoted.columns]
    df_long = pivoted.reset_index()

    # 5. 补齐评委列
    for j in range(1, 4):
        col = f"judge{j}_score"
        if col not in df_long.columns:
            df_long[col] = np.nan

    return df_long

# ---------------------------------------------------------
# 3. 基础特征与分数处理 (V6 核心逻辑)
# ---------------------------------------------------------
def process_results_and_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    judge_cols = [c for c in df.columns if c.startswith("judge") and c.endswith("_score")]

    # 1. 计算总分
    df["raw_total_score"] = df[judge_cols].sum(axis=1, min_count=1)
    df["judge_variance"] = df[judge_cols].std(axis=1)

    # 2. 活跃状态
    df["is_active"] = df["raw_total_score"].notna() & (df["raw_total_score"] > 0)

    # 3. 归一化
    df["row_judge_count"] = df[judge_cols].notna().sum(axis=1)
    max_judges_per_week = df.groupby(["season", "week"])["row_judge_count"].transform("max")
    df["week_valid_judge_count"] = max_judges_per_week  # 兼容旧报表
    df["max_possible_score"] = max_judges_per_week * 10

    df["normalized_score"] = np.where(
        (df["max_possible_score"] > 0) & df["is_active"],
        df["raw_total_score"] / df["max_possible_score"],
        np.nan
    )
    df["normalized_score"] = df["normalized_score"].clip(upper=1.0)

    # 4. 淘汰与排名
    df["is_eliminated"] = False
    df["elimination_week"] = np.nan
    df["final_rank"] = np.nan

    if "results" in df.columns:
        res = df["results"].astype(str).str.lower().str.strip()
        is_elim = res.str.contains("eliminated", na=False)
        df.loc[is_elim, "is_eliminated"] = True

        week_match = res.str.extract(r"week\s*(\d+)")
        df.loc[is_elim, "elimination_week"] = pd.to_numeric(week_match[0], errors='coerce')

        rank_match = res.str.extract(r"(\d+)(?:st|nd|rd|th)")
        df["final_rank"] = pd.to_numeric(rank_match[0], errors='coerce')

        if "placement" in df.columns:
            df["final_rank"] = pd.to_numeric(df["placement"], errors='coerce').fillna(df["final_rank"])

    # [V6 Fix] 双人淘汰排名对齐
    if "final_rank" in df.columns:
        eliminated_mask = df["is_eliminated"] & df["elimination_week"].notna()
        tie_fix = df[eliminated_mask].groupby(["season", "elimination_week"])["final_rank"].transform("min")
        df.loc[eliminated_mask, "final_rank"] = tie_fix
        logger.info("已执行双人淘汰排名对齐 (Double Elimination Fix)")

    df["is_eliminated_this_week"] = (df["is_eliminated"] & (df["week"] == df["elimination_week"]))
    return df


# ---------------------------------------------------------
# 4. 高级特征工程 (V6 逻辑)
# ---------------------------------------------------------
def extract_ultimate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Industry
    if "celebrity_industry" in df.columns:
        industry_map = {
            "Actor": "Entertainment", "Actress": "Entertainment", "Singer": "Music", "Rapper": "Music",
            "Musician": "Music", "Athlete": "Sports", "NFL": "Sports", "NBA": "Sports", "Olympian": "Sports",
            "Model": "Fashion", "TV Personality": "Media", "Host": "Media", "Journalist": "Media",
            "Comedian": "Entertainment", "Politician": "Politics", "Reality": "Reality TV",
            "Social Media": "Social Media", "Youtuber": "Social Media"
        }

        def map_ind(val):
            v = str(val).strip()
            for k, cat in industry_map.items():
                if k.lower() in v.lower(): return cat
            return "Other"

        df["industry_category"] = df["celebrity_industry"].apply(map_ind)
        for key_word in ["Athlete", "Actor", "Singer", "Model", "Reality"]:
            df[f"is_{key_word.lower()}"] = df["industry_category"].str.contains(
                key_word if key_word != "Actor" else "Entertainment", case=False
            ).astype(int)

    # Partner Quality
    if "ballroom_partner" in df.columns and "final_rank" in df.columns:
        partner_stats = df.groupby("ballroom_partner").agg({
            "final_rank": "mean",
            "season": "nunique"
        }).rename(columns={"final_rank": "partner_avg_rank", "season": "partner_seasons_count"})
        df = df.merge(partner_stats, on="ballroom_partner", how="left")
        df["partner_avg_rank"] = df["partner_avg_rank"].fillna(df["final_rank"].mean())

    # Trend
    df = df.sort_values(["season", "celebrity_name", "week"])
    df["weeks_survived"] = df.groupby(["season", "celebrity_name"]).cumcount() + 1
    df["score_improvement"] = df.groupby(["season", "celebrity_name"])["normalized_score"].diff().fillna(0)
    df["prev_avg_score"] = df.groupby(["season", "celebrity_name"])["normalized_score"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    df["score_consistency"] = df.groupby(["season", "celebrity_name"])["normalized_score"].transform(
        lambda x: x.shift(1).expanding().std().fillna(0)
    )

    # Z-score & Time
    df["score_z_score"] = df.groupby(["season", "week"])["normalized_score"].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )
    if "week" in df.columns and "season" in df.columns:
        active_counts = df[df["is_active"]].groupby(["season", "week"])["celebrity_name"].nunique()
        df = df.merge(active_counts.rename("contestants_remaining"), on=["season", "week"], how="left")
        max_weeks = df.groupby("season")["week"].transform("max")
        df["season_progress"] = df["week"] / max_weeks

    # Age & Region
    if "celebrity_age" in df.columns:
        df["celebrity_age"] = pd.to_numeric(df["celebrity_age"], errors='coerce')
        df["age_group"] = pd.cut(df["celebrity_age"], bins=[0, 20, 30, 40, 50, 60, 100],
                                 labels=["Under 20", "20-30", "30-40", "40-50", "50-60", "60+"])
    if "celebrity_homecountry" in df.columns:
        df["is_us_celebrity"] = df["celebrity_homecountry"].astype(str).str.strip().str.lower() == "united states"

    return df


# ---------------------------------------------------------
# 5. 投票方案 (V6: Rank=Min 修正)
# ---------------------------------------------------------
def analyze_voting_schemes(df: pd.DataFrame):
    df = df.copy()
    df["voting_scheme"] = np.where(
        df["season"].isin([1, 2] + list(range(28, 35))), "rank_based", "percentage_based"
    )
    # [V6 Update] method='min'
    df["judge_rank"] = df.groupby(["season", "week"])["raw_total_score"].rank(method="min", ascending=False)

    def calculate_percentage(x):
        total = x.sum()
        return x / total if total > 0 else 0

    df["judge_share"] = df.groupby(["season", "week"])["raw_total_score"].transform(calculate_percentage)

    df["judge_contribution"] = np.where(
        df["voting_scheme"] == "rank_based", -df["judge_rank"], df["judge_share"]
    )
    return df


# ---------------------------------------------------------
# 6. 争议提取
# ---------------------------------------------------------
def extract_controversies(df: pd.DataFrame, output_dir: Path):
    controversies = []
    for (season, week), group in df.groupby(["season", "week"]):
        eliminated = group[group["is_eliminated_this_week"]]
        active_survivors = group[~group["is_eliminated_this_week"] & group["is_active"]]
        if eliminated.empty or active_survivors.empty: continue

        min_survivor_score = active_survivors["raw_total_score"].min()
        for _, elim_row in eliminated.iterrows():
            elim_score = elim_row["raw_total_score"]
            if elim_score > min_survivor_score:
                lucky_ones = active_survivors[active_survivors["raw_total_score"] < elim_score]
                for _, lucky in lucky_ones.iterrows():
                    controversies.append({
                        "season": season, "week": week,
                        "victim": elim_row["celebrity_name"], "victim_score": elim_score,
                        "survivor": lucky["celebrity_name"], "survivor_score": lucky["raw_total_score"],
                        "score_gap": elim_score - lucky["raw_total_score"]
                    })
    if controversies:
        cdf = pd.DataFrame(controversies)
        cdf.to_csv(output_dir / "controversial_cases_analysis.csv", index=False)


# ---------------------------------------------------------
# 7. 报表与可视化 (移植自旧代码，用于兼容)
# ---------------------------------------------------------
def generate_reports(df_long: pd.DataFrame, df_wide: pd.DataFrame, out_dir: Path):
    # 1. Histogram
    hist_png = out_dir / "score_distribution_check.png"
    data = df_long.loc[df_long["is_active"], "normalized_score"].dropna()
    plt.figure(figsize=(8, 5), dpi=200)
    plt.hist(data, bins=30, color='skyblue', edgecolor='black')
    plt.title("Normalized Score Distribution ")
    plt.xlabel("Normalized Score (0-1)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(hist_png, dpi=300)
    plt.close()

    # 2. Text Report
    report_txt = out_dir / "data_verification_report.txt"
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write("=== MCM V6 DATA VERIFICATION REPORT ===\n")
        f.write(f"Total Rows: {len(df_long)}\n")
        f.write(f"Active Rows: {df_long['is_active'].sum()}\n")
        f.write(f"Seasons: {df_long['season'].nunique()}\n")
        if "judge_rank" in df_long.columns:
            f.write("Note: Judge Ranks calculated using method='min' (1224 ranking).\n")
        if "partner_avg_rank" in df_long.columns:
            f.write("Note: Partner Quality features successfully engineered.\n")

    logger.info(f"生成兼容报告: {report_txt}")
    logger.info(f"生成分布图: {hist_png}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="2026_MCM_Problem_C_Data.csv")
    parser.add_argument("--output_dir", default="mcm_data_v7_fusion")
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    out_path.mkdir(exist_ok=True, parents=True)

    try:
        try:
            df_raw = pd.read_csv(args.data_path, encoding='utf-8-sig')
        except UnicodeDecodeError:
            df_raw = pd.read_csv(args.data_path, encoding='latin1')

        # 核心清洗 (V6逻辑)
        df = standardize_columns(df_raw)
        df = wide_to_long(df)
        df = process_results_and_scores(df)
        df = extract_ultimate_features(df)
        df = analyze_voting_schemes(df)

        # 输出1: 争议分析
        extract_controversies(df, out_path)

        # 输出2: 兼容旧版名称的文件 (但内容是V6升级版!)
        #
        # 这样你不需要改Q1代码，直接替换文件即可
        df.to_csv(out_path / "cleaned_data.csv", index=False, encoding='utf-8-sig')
        logger.info(f"已生成升级版数据: {out_path / 'cleaned_data.csv'}")

        # 输出3: 建模专用版 (V6特色)
        model_cols = [
            "season", "week", "celebrity_name", "is_eliminated_this_week",
            "normalized_score", "score_z_score", "score_improvement", "prev_avg_score", "score_consistency",
            "judge_variance", "judge_rank", "judge_share", "judge_contribution", "voting_scheme",
            "industry_category", "is_athlete", "is_actor", "is_singer", "is_reality",
            "age_group", "is_us_celebrity",
            "partner_avg_rank", "partner_seasons_count",
            "weeks_survived", "contestants_remaining", "season_progress"
        ]
        final_cols = [c for c in model_cols if c in df.columns]
        df[final_cols].to_csv(out_path / "mcm_modeling_dataset.csv", index=False, encoding='utf-8-sig')

        # 输出4: 生成报告和图 (旧版功能)
        generate_reports(df, df_raw, out_path)

        logger.info("=== V7 Fusion Complete ===")
        logger.info("你可以直接用新的 cleaned_data.csv 替换旧文件，Q1代码可直接运行！")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()