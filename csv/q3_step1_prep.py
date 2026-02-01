#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
q3_step1_prep_v3.py
Q3 Step1 — 自动补齐特征列 + 合并 latent + 归一化(可跳过缺失) + do_well 构造

Inputs (同目录；也会尝试 mcm_data_v7_fusion/ 下):
- mcm_modeling_dataset.csv
- latent_estimates.csv
- cleaned_data.csv   (用于补齐 ballroom/age/industry/placement)

Outputs (默认写到当前目录，可用 --out_dir 指定):
- q3_do_well_defs.csv
- q3_weekly_norm.csv
- q3_data_quality_summary.csv
- step1_audit.json

Run:
python q3_step1_prep_v3.py
python q3_step1_prep_v3.py --out_dir results/run_xxx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)

KEY_WEEK = ["season", "week", "celebrity_name"]
KEY_SEASON = ["season", "celebrity_name"]


def hard_fail(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(f"[HARD_FAIL] {msg}")


def find_input(filename: str) -> Path:
    p1 = Path(filename)
    p2 = Path("mcm_data_v7_fusion") / filename
    if p1.exists():
        return p1
    if p2.exists():
        return p2
    raise FileNotFoundError(f"Missing input: {filename} (checked ./{filename} and ./mcm_data_v7_fusion/{filename})")


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def season_level_lookup(cleaned: pd.DataFrame, source_col: str) -> pd.DataFrame:
    tmp = cleaned.loc[cleaned[source_col].notna(), KEY_SEASON + [source_col]].copy()
    tmp = tmp.drop_duplicates(KEY_SEASON, keep="first")
    return tmp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcm", type=str, default="mcm_modeling_dataset.csv")
    ap.add_argument("--latent", type=str, default="latent_estimates.csv")
    ap.add_argument("--cleaned", type=str, default="cleaned_data.csv")
    ap.add_argument("--out_dir", type=str, default=".")
    args = ap.parse_args()

    mcm_path = find_input(args.mcm)
    latent_path = find_input(args.latent)
    cleaned_path = find_input(args.cleaned)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mcm = pd.read_csv(mcm_path)
    latent = pd.read_csv(latent_path)
    cleaned = pd.read_csv(cleaned_path)

    # join keys
    for c in KEY_WEEK:
        hard_fail(c in mcm.columns, f"{mcm_path.name} missing join key: {c}")
        hard_fail(c in latent.columns, f"{latent_path.name} missing join key: {c}")

    # latent unique key
    dup = latent.duplicated(KEY_WEEK, keep=False)
    hard_fail(not dup.any(), f"{latent_path.name} has duplicate keys on {KEY_WEEK}; dup_rows={int(dup.sum())}")

    # --- 补齐 mcm 缺失的特征列（从 cleaned_data 按 season+celebrity 补） ---
    needed = {
        "ballroom_partner": ["ballroom_partner", "partner", "pro_dancer", "professional_dancer", "pro_partner", "partner_name"],
        "celebrity_age_during_season": ["celebrity_age_during_season", "celebrity_age", "age"],
        "celebrity_industry": ["celebrity_industry", "industry", "industry_category", "occupation", "celebrity_occupation"],
        "placement": ["placement", "final_placement", "final_rank", "overall_rank", "rank_final", "place"],
    }

    mapping = {"from_mcm": {}, "from_cleaned": {}, "unresolved": []}

    # from mcm
    for target, cands in needed.items():
        got = pick_col(mcm, cands)
        if got is not None:
            if got != target:
                mcm[target] = mcm[got]
            mapping["from_mcm"][target] = got

    # from cleaned (season-level)
    miss = [t for t in needed.keys() if t not in mcm.columns]
    if len(miss) > 0:
        hard_fail(all(k in cleaned.columns for k in KEY_SEASON), f"{cleaned_path.name} missing keys: {KEY_SEASON}")
        for target in miss:
            src = pick_col(cleaned, needed[target])
            if src is None:
                mapping["unresolved"].append(target)
                continue
            look = season_level_lookup(cleaned, src).rename(columns={src: target})
            mcm = mcm.merge(look, on=KEY_SEASON, how="left")
            mapping["from_cleaned"][target] = src

    still_missing = [t for t in needed.keys() if t not in mcm.columns]
    if len(still_missing) > 0:
        raise RuntimeError(
            "[HARD_FAIL] Required fields still missing.\n"
            + json.dumps(
                {
                    "still_missing": still_missing,
                    "candidates": needed,
                    "mcm_cols_preview": sorted(list(mcm.columns))[:120],
                    "cleaned_cols_preview": sorted(list(cleaned.columns))[:120],
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    # --- merge latent (LEFT JOIN) ---
    df = mcm.merge(latent, on=KEY_WEEK, how="left", suffixes=("", "_latent"))
    df["latent_missing_flag"] = df["fan_vote_mean"].isna().astype(int)

    # --- n_active: 缺失就用 mcm 自己算出来的周活跃人数补齐（不编造） ---
    # 参赛人数 = 同一 (season, week) 的 celebrity_name 去重计数
    week_n = df.groupby(["season", "week"])["celebrity_name"].transform("nunique")
    df["n_active_filled"] = pd.to_numeric(df["n_active"], errors="coerce").fillna(week_n)

    # rule_type 可能缺失：缺失行不做归一化，直接留空
    rule = df["rule_type"].astype(str)
    mask_percent = rule.eq("percent")
    mask_rank = rule.eq("rank")
    mask_known = mask_percent | mask_rank

    n_active = pd.to_numeric(df["n_active_filled"], errors="coerce")
    # n_active_filled 理论上不该 NA（除非 season/week/celebrity_name 自己也缺）
    hard_fail((~n_active.isna()).all(), "n_active_filled still contains NA; check season/week keys")

    def norm_series(x: pd.Series) -> pd.Series:
        x_num = pd.to_numeric(x, errors="coerce")
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[mask_percent] = x_num.loc[mask_percent]
        out.loc[mask_rank] = (n_active.loc[mask_rank] + 1 - x_num.loc[mask_rank]) / n_active.loc[mask_rank]
        # mask_unknown remains NA
        return out

    df["judge_norm"] = norm_series(df["judge_metric"])
    df["fan_norm"] = norm_series(df["fan_vote_mean"])

    # CI 同步归一化（unknown rule_type -> NA）
    ci_low_norm = norm_series(df["ci_low"])
    ci_high_norm = norm_series(df["ci_high"])
    df["fan_ci_low_norm"] = np.minimum(ci_low_norm, ci_high_norm)
    df["fan_ci_high_norm"] = np.maximum(ci_low_norm, ci_high_norm)

    df["fan_se"] = (df["fan_ci_high_norm"] - df["fan_ci_low_norm"]) / (2 * 1.96)
    df["fan_ci_invalid_flag"] = ((df["fan_se"].isna()) | (df["fan_se"] <= 0)).astype(int)
    df["fan_weight"] = np.where(df["fan_ci_invalid_flag"].eq(0), 1.0 / (df["fan_se"] ** 2), np.nan)

    # --- do_well：完全基于 latent 的淘汰周（不需要 mcm 有淘汰列） ---
    elim = latent.copy()
    elim["season"] = pd.to_numeric(elim["season"], errors="coerce")
    elim["week"] = pd.to_numeric(elim["week"], errors="coerce")
    elim["is_eliminated_this_week"] = pd.to_numeric(elim["is_eliminated_this_week"], errors="coerce")
    hard_fail((~elim["season"].isna()).all() and (~elim["week"].isna()).all(), "latent season/week not numeric")

    Tmax = elim.groupby("season")["week"].max().rename("Tmax_season").reset_index()
    first_elim = (
        elim.loc[elim["is_eliminated_this_week"].eq(1), ["season", "celebrity_name", "week"]]
        .groupby(["season", "celebrity_name"])["week"]
        .min()
        .rename("exit_week")
        .reset_index()
    )

    cs = (
        mcm[["season", "celebrity_name", "ballroom_partner", "celebrity_age_during_season", "celebrity_industry", "placement"]]
        .drop_duplicates(["season", "celebrity_name"])
        .copy()
    )
    cs["season"] = pd.to_numeric(cs["season"], errors="coerce")
    hard_fail((~cs["season"].isna()).all(), "mcm season not numeric")

    cs = cs.merge(Tmax, on="season", how="left").merge(first_elim, on=["season", "celebrity_name"], how="left")
    cs["exit_week"] = cs["exit_week"].fillna(cs["Tmax_season"])
    cs["exit_week"] = pd.to_numeric(cs["exit_week"], errors="coerce").astype("Int64")

    # do_well_main：Tmax_season 缺失就留 NA（不编）
    cs["do_well_main"] = np.where(
        (cs["Tmax_season"].notna()) & (cs["Tmax_season"] > 1) & (cs["exit_week"].notna()),
        (cs["exit_week"].astype(float) - 1) / (cs["Tmax_season"].astype(float) - 1),
        np.nan,
    )

    # do_well_alt_place：placement 缺失就 NA
    cs["placement"] = pd.to_numeric(cs["placement"], errors="coerce")
    max_place = cs.groupby("season")["placement"].max().rename("max_placement_season").reset_index()
    cs = cs.merge(max_place, on="season", how="left")
    cs["do_well_alt_place"] = 1 - (cs["placement"] - 1) / (cs["max_placement_season"] - 1)

    # --- data quality summary ---
    dq = df.copy()
    dq["rule_type_missing_flag"] = (~mask_known).astype(int)
    dq["reason"] = np.select(
        [
            dq["latent_missing_flag"].eq(1),
            dq["rule_type_missing_flag"].eq(1),
            dq["fan_ci_invalid_flag"].eq(1),
        ],
        [
            "latent_missing_flag==1 (no latent match)",
            "rule_type missing (skip normalization)",
            "fan_ci_invalid_flag==1 (fan_se<=0 or NA)",
        ],
        default="included",
    )
    dq_sum = (
        dq.loc[dq["reason"].ne("included"), ["season", "week", "reason"]]
        .groupby(["season", "week", "reason"], dropna=False)
        .size()
        .rename("n_rows_affected")
        .reset_index()
    )

    # --- write outputs (文件名就行) ---
    out_do = out_dir / "q3_do_well_defs.csv"
    out_weekly = out_dir / "q3_weekly_norm.csv"
    out_dq = out_dir / "q3_data_quality_summary.csv"
    out_audit = out_dir / "step1_audit.json"

    cs.to_csv(out_do, index=False)

    weekly_cols = KEY_WEEK + [
        "rule_type", "n_active", "n_active_filled",
        "judge_metric", "judge_norm",
        "fan_vote_mean", "fan_norm",
        "ci_low", "ci_high",
        "fan_ci_low_norm", "fan_ci_high_norm",
        "fan_se", "fan_weight",
        "latent_missing_flag", "fan_ci_invalid_flag",
        "ballroom_partner", "celebrity_age_during_season", "celebrity_industry",
    ]
    df[weekly_cols].to_csv(out_weekly, index=False)
    dq_sum.to_csv(out_dq, index=False)

    audit = {
        "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seed": SEED,
        "inputs": {
            "mcm_modeling_dataset": mcm_path.name,
            "latent_estimates": latent_path.name,
            "cleaned_data": cleaned_path.name,
        },
        "column_mapping": mapping,
        "counts": {
            "rows_mcm": int(len(mcm)),
            "rows_latent": int(len(latent)),
            "rows_merged": int(len(df)),
            "latent_missing_rows": int(df["latent_missing_flag"].sum()),
            "rule_type_missing_rows": int((~mask_known).sum()),
            "fan_ci_invalid_rows": int(df["fan_ci_invalid_flag"].sum()),
        },
        "outputs": [out_do.name, out_weekly.name, out_dq.name, out_audit.name],
    }
    out_audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[OK] outputs written:")
    print(" -", out_do.name)
    print(" -", out_weekly.name)
    print(" -", out_dq.name)
    print(" -", out_audit.name)


if __name__ == "__main__":
    main()
