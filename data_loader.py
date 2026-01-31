#!/usr/bin/env python3
"""
data_loader.py
Python 3.10+

目标：
1) 读取 2026_MCM_Problem_C_Data.csv（宽表），整理为长表（season × contestant × week）。
2) 按题目要求生成关键字段：is_active、week_valid_judge_count、max_possible_score、normalized_score。
3) 自动输出数据核验报告 data_verification_report.txt 与直方图 score_distribution_check.png。
4) 保存清洗后的长表 cleaned_data.csv。

使用示例：
python data_loader.py --data_path data/2026_MCM_Problem_C_Data.csv --output_dir outputs
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# 工具函数
# -----------------------------
WEEK_JUDGE_RE = re.compile(r"^week(\d+)_judge(\d+)_score$")


def find_score_columns(columns: list[str]) -> list[str]:
    """在原始宽表里找出所有 weekX_judgeY_score 列。"""
    return [c for c in columns if WEEK_JUDGE_RE.match(c)]


def wide_to_long(df_wide: pd.DataFrame, id_cols: list[str], score_cols: list[str]) -> pd.DataFrame:
    """
    宽表 -> 长表：
    - 先 melt 出 (week, judge, score)
    - 再 pivot 回 (judge1_score ... judge4_score)
    """
    melted = df_wide.melt(
        id_vars=id_cols,
        value_vars=score_cols,
        var_name="wk_j",
        value_name="score",
    )

    wk_j = melted["wk_j"].str.extract(r"^week(\d+)_judge(\d+)_score$")
    melted["week"] = wk_j[0].astype(int)
    melted["judge"] = wk_j[1].astype(int)

    # pivot：每行 = 选手-赛季-周，列 = judge1_score ... judge4_score
    pivoted = (
        melted.pivot_table(
            index=id_cols + ["week"],
            columns="judge",
            values="score",
            aggfunc="first",
        )
        .sort_index(axis=1)
    )

    # 重命名列
    pivoted.columns = [f"judge{int(j)}_score" for j in pivoted.columns]
    df_long = pivoted.reset_index()

    # 保证 judge1..judge4 都存在（即使某季从未出现 judge4，也补齐列）
    for j in [1, 2, 3, 4]:
        col = f"judge{j}_score"
        if col not in df_long.columns:
            df_long[col] = np.nan

    return df_long


def add_active_and_normalized_scores(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    核心清洗逻辑（全向量化，不用逐行循环）：
    1) 计算 raw_total_score = 四位评委分数之和（允许 NaN）
    2) 按题目定义生成 is_active：
       True 当且仅当 raw_total_score > 0 且 judge1_score 非 NaN
    3) 动态计算每个赛季每周的有效评委人数 week_valid_judge_count：
       对同一 season-week，在 is_active 行中统计“每行非 NaN 的评委列数”，取最大值（通常=3或4）
    4) max_possible_score = week_valid_judge_count * 10
    5) normalized_score = raw_total_score / max_possible_score（仅在 is_active==True 的行计算）
    """
    judge_cols = [f"judge{j}_score" for j in [1, 2, 3, 4]]

    # raw_total_score：注意 min_count=1，避免全 NaN 变成 0
    df_long["raw_total_score"] = df_long[judge_cols].sum(axis=1, min_count=1)

    # is_active：淘汰后通常会出现 0 或 NaN；按题意判定“仍在比赛”
    df_long["is_active"] = (df_long["raw_total_score"] > 0) & (~df_long["judge1_score"].isna())

    # 逐行统计该行有多少个评委给了分（非 NaN）
    df_long["row_valid_judge_count"] = df_long[judge_cols].notna().sum(axis=1)

    # 逐周逐季统计“该周实际有几位评委”：
    # 用 is_active 行来推断本周评委人数；取最大值更稳健（某些选手可能出现个别缺失）
    df_long["week_valid_judge_count"] = (
        df_long["row_valid_judge_count"]
        .where(df_long["is_active"])
        .groupby([df_long["season"], df_long["week"]])
        .transform("max")
    )

    # 理论最高分
    df_long["max_possible_score"] = df_long["week_valid_judge_count"] * 10

    # 归一化分数：仅对 active 行计算；其余置为 NaN
    df_long["normalized_score"] = np.where(
        df_long["is_active"] & df_long["max_possible_score"].notna(),
        df_long["raw_total_score"] / df_long["max_possible_score"],
        np.nan,
    )

    return df_long


def extract_elimination_week(results: pd.Series) -> pd.Series:
    """
    从 results 字段提取淘汰周数：
    - "Eliminated Week X" -> X
    - 其它（名次/Withdrew）-> NaN
    """
    extracted = results.astype(str).str.extract(r"Eliminated Week (\d+)")
    return pd.to_numeric(extracted[0], errors="coerce")


def verify_judges_save(df_long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    按用户定义验证“最低评委分未被淘汰”是否发生（潜在 Judges' Save）：

    对每个 season-week：
    - 在 is_active 行中找到当周最低 raw_total_score
    - 如果当周存在被淘汰选手（results = Eliminated Week {week}）
      但被淘汰选手的最低分 > 当周最低分，则说明：
      “最低分的人没有被淘汰，而更高分的人被淘汰” -> violation=True

    返回：
    - season_level: 每个 season 是否出现 violation（has_judges_save）
    - week_level: 具体在哪些周触发（便于报告）
    """
    df = df_long.copy()

    df["elimination_week"] = extract_elimination_week(df["results"])
    df["eliminated_this_week"] = df["elimination_week"].eq(df["week"])

    # 只在 active 行中比较评委分
    active = df[df["is_active"]].copy()

    # 当周最低评委分（active contestants）
    week_min_score = (
        active.groupby(["season", "week"])["raw_total_score"]
        .min()
        .rename("week_min_score")
        .reset_index()
    )

    # 当周被淘汰选手的最低评委分（通常只有1人；取 min 可兼容双淘汰）
    elim = active[active["eliminated_this_week"]].copy()
    elim_min_score = (
        elim.groupby(["season", "week"])["raw_total_score"]
        .min()
        .rename("eliminated_min_score")
        .reset_index()
    )

    # 合并并判定 violation
    week_check = week_min_score.merge(elim_min_score, on=["season", "week"], how="left")
    week_check["has_elimination_record"] = week_check["eliminated_min_score"].notna()

    # 只有在“确实有淘汰记录”的周才能检查
    eps = 1e-9
    week_check["violation_min_score_rule"] = (
        week_check["has_elimination_record"]
        & (week_check["eliminated_min_score"] > week_check["week_min_score"] + eps)
    )

    # season 级别汇总：任意一周 violation 即 True
    season_check = (
        week_check.groupby("season")["violation_min_score_rule"]
        .any()
        .rename("has_judges_save")
        .reset_index()
    )

    return season_check, week_check


def season_judge_panel_type(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    判定每个 season 是 3-judge / 4-judge / mixed：
    - 基于 active 行中的 week_valid_judge_count
    """
    active = df_long[df_long["is_active"] & df_long["week_valid_judge_count"].notna()].copy()

    panel = (
        active.groupby("season")["week_valid_judge_count"]
        .apply(lambda s: sorted(pd.unique(s.astype(int))))
        .rename("judge_count_set")
        .reset_index()
    )

    def classify(counts: list[int]) -> str:
        if counts == [3]:
            return "3-judge"
        if counts == [4]:
            return "4-judge"
        if len(counts) == 0:
            return "unknown"
        return "mixed"

    panel["judge_panel_type"] = panel["judge_count_set"].apply(classify)
    return panel


def missing_value_summary(df_wide: pd.DataFrame, score_cols: list[str], df_long: pd.DataFrame) -> dict:
    """
    生成缺失值/占位值的摘要（用于报告）：
    - 原始宽表：NaN 数量、0 数量（在 score_cols 内）
    - 长表：judge 列 NaN 数量；is_active True/False 数量
    """
    wide_scores = df_wide[score_cols]
    wide_nan = int(wide_scores.isna().sum().sum())
    wide_zero = int((wide_scores == 0).sum().sum())

    judge_cols = [f"judge{j}_score" for j in [1, 2, 3, 4]]
    long_nan_by_judge = df_long[judge_cols].isna().sum().to_dict()

    active_count = int(df_long["is_active"].sum())
    total_count = int(len(df_long))

    norm_active = df_long.loc[df_long["is_active"], "normalized_score"]
    out_of_range_hi = int((norm_active > 1).sum())
    out_of_range_lo = int((norm_active < 0).sum())

    return {
        "wide_nan_total": wide_nan,
        "wide_zero_total": wide_zero,
        "long_nan_by_judge": long_nan_by_judge,
        "active_rows": active_count,
        "total_rows": total_count,
        "normalized_min": float(norm_active.min()) if len(norm_active) else np.nan,
        "normalized_max": float(norm_active.max()) if len(norm_active) else np.nan,
        "normalized_out_of_range_hi": out_of_range_hi,
        "normalized_out_of_range_lo": out_of_range_lo,
    }


def write_verification_report(
    out_path: Path,
    panel_df: pd.DataFrame,
    season_check: pd.DataFrame,
    week_check: pd.DataFrame,
    mv: dict,
) -> None:
    """将所有核验结果写入 txt 报告。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # seasons by panel type
    panel_groups = panel_df.groupby("judge_panel_type")["season"].apply(list).to_dict()

    # judges' save seasons
    js_seasons = season_check.loc[season_check["has_judges_save"], "season"].tolist()

    # violation details (只列出触发的 season-week)
    viol = week_check[week_check["violation_min_score_rule"]].copy()
    viol = viol.sort_values(["season", "week"])

    lines: list[str] = []
    lines.append("DATA VERIFICATION REPORT\n")
    lines.append("=" * 80 + "\n")

    # Basic counts
    lines.append("1) Active Rows vs Total Rows\n")
    lines.append(f"- Active rows (is_active=True): {mv['active_rows']}\n")
    lines.append(f"- Total rows: {mv['total_rows']}\n")
    lines.append(f"- Active ratio: {mv['active_rows'] / mv['total_rows']:.3f}\n\n")

    # Missing values summary
    lines.append("2) Missing / Placeholder Values Summary\n")
    lines.append(f"- Wide table: total NaN in score cells = {mv['wide_nan_total']}\n")
    lines.append(f"- Wide table: total zeros in score cells = {mv['wide_zero_total']}\n")
    lines.append("- Long table: NaN counts by judge columns:\n")
    for k, v in mv["long_nan_by_judge"].items():
        lines.append(f"  * {k}: {int(v)}\n")
    lines.append("\n")

    # Normalized score range check
    lines.append("3) Normalized Score Range Check (expected: [0, 1])\n")
    lines.append(f"- min(normalized_score) among active rows: {mv['normalized_min']:.6f}\n")
    lines.append(f"- max(normalized_score) among active rows: {mv['normalized_max']:.6f}\n")
    lines.append(f"- count(normalized_score > 1): {mv['normalized_out_of_range_hi']}\n")
    lines.append(f"- count(normalized_score < 0): {mv['normalized_out_of_range_lo']}\n\n")

    # Panel type
    lines.append("4) Judge Panel Type by Season (3-judge / 4-judge / mixed)\n")
    for t in ["3-judge", "4-judge", "mixed", "unknown"]:
        seasons = panel_groups.get(t, [])
        lines.append(f"- {t}: {sorted(seasons)}\n")
    lines.append("\n")

    # Judges' save verification
    lines.append("5) Judges' Save Verification (min-score rule violation)\n")
    lines.append(
        "- Definition: If the lowest total_judge_score contestant was NOT eliminated while\n"
        "  - Note: This check compares elimination vs *judges' score only*. Since fan votes affect eliminations,\n"
        "  violations may appear even before S28. Interpret has_judges_save as \"min-judge-score not eliminated\".\n"
        "  someone with a higher score WAS eliminated in the same week, we flag a violation.\n"
    )
    lines.append(f"- Seasons with violation (has_judges_save=True): {sorted(js_seasons)}\n\n")

    if len(viol) > 0:
        lines.append("  Violation details (season, week, week_min_score, eliminated_min_score):\n")
        for r in viol.itertuples(index=False):
            lines.append(
                f"  * S{int(r.season)}, Week {int(r.week)}: "
                f"min={r.week_min_score:.2f}, eliminated_min={r.eliminated_min_score:.2f}\n"
            )
        lines.append("\n")
    else:
        lines.append("  No violation weeks detected based on available elimination records.\n\n")

    out_path.write_text("".join(lines), encoding="utf-8")


def save_histogram(df_long: pd.DataFrame, out_path: Path) -> None:
    """保存 normalized_score 的直方图（只画 is_active=True 的行）。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = df_long.loc[df_long["is_active"], "normalized_score"].dropna()
    if data.empty:
        # 如果没有数据，就输出空图并提示
        plt.figure(figsize=(8, 5), dpi=200)
        plt.title("Normalized Score Distribution (No active data)")
        plt.xlabel("Normalized Score (unitless)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        return

    plt.figure(figsize=(8, 5), dpi=200)
    plt.hist(data, bins=30)
    plt.title("Normalized Score Distribution (Expected Range: 0–1)")
    plt.xlabel("Normalized Score (unitless)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean MCM Problem C data and generate verification report.")
    p.add_argument(
        "--data_path",
        type=str,
        default="data/2026_MCM_Problem_C_Data.csv",
        help="Path to raw CSV file (placeholder default).",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save cleaned_data.csv, report, and plots.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # 1) 读取原始数据（宽表）
    # -----------------------------
    df_wide = pd.read_csv(data_path)

    # 识别 ID 列与分数列
    id_cols = [
        "celebrity_name",
        "ballroom_partner",
        "celebrity_industry",
        "celebrity_homestate",
        "celebrity_homecountry/region",
        "celebrity_age_during_season",
        "season",
        "results",
        "placement",
    ]
    missing_id = [c for c in id_cols if c not in df_wide.columns]
    if missing_id:
        raise ValueError(f"Missing expected id columns in CSV: {missing_id}")

    score_cols = find_score_columns(list(df_wide.columns))
    if not score_cols:
        raise ValueError("No score columns found. Expected pattern: weekX_judgeY_score")

    # -----------------------------
    # 2) 宽表 -> 长表
    # -----------------------------
    df_long = wide_to_long(df_wide=df_wide, id_cols=id_cols, score_cols=score_cols)

    # -----------------------------
    # 3) 生成 is_active + 动态最高分 + 归一化分数
    # -----------------------------
    df_long = add_active_and_normalized_scores(df_long)

    # -----------------------------
    # 4) Judges' Save（最低分未淘汰）核验
    # -----------------------------
    season_check, week_check = verify_judges_save(df_long)

    # -----------------------------
    # 5) 识别每季是 3评委 / 4评委 / 混合
    # -----------------------------
    panel_df = season_judge_panel_type(df_long)

    # -----------------------------
    # 6) 缺失值/占位值摘要
    # -----------------------------
    mv = missing_value_summary(df_wide=df_wide, score_cols=score_cols, df_long=df_long)

    # -----------------------------
    # 7) 保存清洗数据 + 报告 + 图
    # -----------------------------
    cleaned_csv = out_dir / "cleaned_data.csv"
    report_txt = out_dir / "data_verification_report.txt"
    hist_png = out_dir / "score_distribution_check.png"

    # 输出字段：包含题目要求字段 + 关键中间字段
    output_cols = [
        "season",
        "week",
        "celebrity_name",
        "ballroom_partner",
        "celebrity_industry",
        "celebrity_homestate",
        "celebrity_homecountry/region",
        "celebrity_age_during_season",
        "results",
        "placement",
        "judge1_score",
        "judge2_score",
        "judge3_score",
        "judge4_score",
        "raw_total_score",
        "is_active",
        "row_valid_judge_count",
        "week_valid_judge_count",
        "max_possible_score",
        "normalized_score",
    ]

    # 如果某些列不存在（极端情况），只保存存在的列
    output_cols = [c for c in output_cols if c in df_long.columns]
    df_long.to_csv(cleaned_csv, index=False, columns=output_cols)

    write_verification_report(
        out_path=report_txt,
        panel_df=panel_df,
        season_check=season_check,
        week_check=week_check,
        mv=mv,
    )

    save_histogram(df_long=df_long, out_path=hist_png)

    print(f"[OK] Saved: {cleaned_csv}")
    print(f"[OK] Saved: {report_txt}")
    print(f"[OK] Saved: {hist_png}")


if __name__ == "__main__":
    main()
