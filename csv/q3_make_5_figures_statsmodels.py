#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm

SEED = 42
np.random.seed(SEED)

BASE = Path(__file__).resolve().parent

def f(name: str) -> Path:
    p = BASE / name
    if not p.exists():
        raise FileNotFoundError(f"Missing file (must be in same folder): {name}")
    return p

def hard_fail(msg: str) -> None:
    raise RuntimeError(f"[SCHEMA_HARD_FAIL] {msg}")

def set_style():
    matplotlib.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "Times New Roman",
    })

def top_bottom(df: pd.DataFrame, n=10):
    df = df.sort_values("effect")
    return pd.concat([df.head(n), df.tail(n)], axis=0).sort_values("effect")

def extract_effect_long(res, outcome: str, age_sd: float) -> pd.DataFrame:
    params = res.params
    se = res.bse
    ci = res.conf_int(alpha=0.05)
    ci.columns = ["ci_low", "ci_high"]

    t = pd.DataFrame({
        "term": params.index.astype(str),
        "effect": params.values,
        "se": se.values,
        "ci_low": ci["ci_low"].values,
        "ci_high": ci["ci_high"].values,
        "outcome": outcome,
    })

    term = t["term"]
    is_partner = term.str.startswith("C(ballroom_partner)[T.")
    is_ind = term.str.startswith("C(celebrity_industry)[T.")
    is_age = term.eq("celebrity_age_during_season")

    t["factor_type"] = np.select(
        [is_partner, is_ind, is_age],
        ["pro_dancer", "industry", "age_1sd"],
        default="other"
    )

    def strip_prefix(s: pd.Series, prefix: str) -> pd.Series:
        return s.str.replace(prefix, "", regex=False).str.rstrip("]")

    t["factor_level"] = np.select(
        [is_partner, is_ind, is_age],
        [
            strip_prefix(term, "C(ballroom_partner)[T."),
            strip_prefix(term, "C(celebrity_industry)[T."),
            "age_1sd",
        ],
        default=term
    )

    t["effect_type"] = np.where(is_age, "age_1sd_scaled", "raw_coef")

    # scale age into 1SD effect (effect, se, ci)
    if is_age.any():
        idx = t.index[is_age][0]
        t.loc[idx, ["effect", "se", "ci_low", "ci_high"]] = (
            t.loc[idx, ["effect", "se", "ci_low", "ci_high"]].astype(float) * float(age_sd)
        )

    return t

def most_common(series: pd.Series):
    s = series.dropna()
    if s.empty:
        return np.nan
    return s.mode().iloc[0]

def main():
    # ---------- inputs from Step1 ----------
    do_well = pd.read_csv(f("q3_do_well_defs.csv"))
    weekly = pd.read_csv(f("q3_weekly_norm_fixed.csv"))


    # ---------- required columns ----------
    need_do = ["season", "celebrity_name", "do_well_main"]
    for c in need_do:
        if c not in do_well.columns:
            hard_fail(f"q3_do_well_defs.csv missing: {c}")

    need_week = [
        "season", "week", "celebrity_name",
        "ballroom_partner", "celebrity_age_during_season", "celebrity_industry",
        "judge_norm", "fan_norm",
        "fan_se", "fan_weight",
        "latent_missing_flag", "fan_invalid_flag",
    ]
    miss = [c for c in need_week if c not in weekly.columns]
    if miss:
        hard_fail(f"q3_weekly_norm.csv missing: {miss}. (说明你 Step1 输出没带特征列，需要我给你补列版 Step1)")

    # ---------- season-celeb feature table (vectorized) ----------
    # 从 weekly 里抽季内固定特征（groupby first，无循环）
    feat = (weekly[["season","celebrity_name","ballroom_partner","celebrity_age_during_season","celebrity_industry"]]
            .sort_values(["season","celebrity_name"])
            .groupby(["season","celebrity_name"], as_index=False)
            .first())

    # age SD for 1SD effect (可比性要求)
    age_sd = float(pd.to_numeric(feat["celebrity_age_during_season"], errors="coerce").dropna().std(ddof=0))
    if not np.isfinite(age_sd) or age_sd <= 0:
        hard_fail("Invalid age SD computed from celebrity_age_during_season")

    # ---------- Model A dataset (season-celeb) ----------
    mA = (do_well[["season","celebrity_name","do_well_main"]]
          .merge(feat, on=["season","celebrity_name"], how="left"))
    if mA[["ballroom_partner","celebrity_age_during_season","celebrity_industry"]].isna().any().any():
        hard_fail("Model A missing IVs after merge (check Step1 feature fill).")

    # ---------- Model B dataset (weekly judge) ----------
    mB = weekly.copy()
    # ---------- Model C dataset (weekly fan, drop missing + invalid) ----------
    mC = weekly.loc[(weekly["latent_missing_flag"] == 0) & (weekly["fan_invalid_flag"] == 0)].copy()

    # ---------- fit models (statsmodels, auditable) ----------
    # OLS do_well_main
    formula_A = "do_well_main ~ C(ballroom_partner) + celebrity_age_during_season + C(celebrity_industry) + C(season)"
    resA = sm.OLS.from_formula(formula_A, data=mA).fit()

    # OLS judge_norm
    formula_B = "judge_norm ~ C(ballroom_partner) + celebrity_age_during_season + C(celebrity_industry) + C(season) + C(week)"
    resB = sm.OLS.from_formula(formula_B, data=mB).fit()

    # WLS fan_norm
    formula_C = "fan_norm ~ C(ballroom_partner) + celebrity_age_during_season + C(celebrity_industry) + C(season) + C(week)"
    resC = sm.WLS.from_formula(formula_C, data=mC, weights=mC["fan_weight"]).fit()

    # ---------- full effects long table (NO Others merge) ----------
    effA = extract_effect_long(resA, "do_well_main", age_sd)
    effB = extract_effect_long(resB, "judge_norm", age_sd)
    effC = extract_effect_long(resC, "fan_norm", age_sd)

    effects_all = pd.concat([effA, effB, effC], ignore_index=True)
    effects_all.to_csv(BASE / "q3_effects_standardized.csv", index=False)

    # ---------- compare judge vs fan (factor types only) ----------
    keep = effects_all.loc[effects_all["factor_type"].isin(["pro_dancer","industry","age_1sd"])].copy()

    j = keep.loc[keep["outcome"].eq("judge_norm")].rename(columns={
        "effect": "effect_judge", "ci_low": "ci_judge_low", "ci_high": "ci_judge_high"
    })[["factor_type","factor_level","effect_judge","ci_judge_low","ci_judge_high"]]

    fdf = keep.loc[keep["outcome"].eq("fan_norm")].rename(columns={
        "effect": "effect_fan", "ci_low": "ci_fan_low", "ci_high": "ci_fan_high"
    })[["factor_type","factor_level","effect_fan","ci_fan_low","ci_fan_high"]]

    comp = j.merge(fdf, on=["factor_type","factor_level"], how="inner")
    comp["diff_fan_minus_judge"] = comp["effect_fan"] - comp["effect_judge"]

    # diff CI: use SE inferred from CI width (not “Others”合并；这里只是差值不确定性近似)
    se_j = (comp["ci_judge_high"] - comp["ci_judge_low"]) / (2*1.96)
    se_f = (comp["ci_fan_high"] - comp["ci_fan_low"]) / (2*1.96)
    se_d = np.sqrt(se_j**2 + se_f**2)
    comp["diff_ci_low"] = comp["diff_fan_minus_judge"] - 1.96*se_d
    comp["diff_ci_high"] = comp["diff_fan_minus_judge"] + 1.96*se_d

    comp["direction_match"] = (np.sign(comp["effect_judge"]) == np.sign(comp["effect_fan"])).astype(int)
    comp.to_csv(BASE / "q3_compare_judge_vs_fan.csv", index=False)

    # ---------- age curve predictions (baseline) ----------
    age_min = float(pd.to_numeric(feat["celebrity_age_during_season"], errors="coerce").min())
    age_max = float(pd.to_numeric(feat["celebrity_age_during_season"], errors="coerce").max())
    age_grid = np.linspace(age_min, age_max, 60)

    base_partner = most_common(feat["ballroom_partner"])
    base_ind = most_common(feat["celebrity_industry"])
    base_season = most_common(mB["season"])
    base_week = most_common(mB["week"])

    pred_base = pd.DataFrame({
        "ballroom_partner": base_partner,
        "celebrity_industry": base_ind,
        "season": base_season,
        "week": base_week,
        "celebrity_age_during_season": age_grid,
    })

    pred_j = resB.get_prediction(pred_base).summary_frame(alpha=0.05)
    pred_f = resC.get_prediction(pred_base).summary_frame(alpha=0.05)

    age_pred = pd.DataFrame({
        "age": age_grid,
        "judge_mean": pred_j["mean"].values,
        "judge_ci_low": pred_j["mean_ci_lower"].values,
        "judge_ci_high": pred_j["mean_ci_upper"].values,
        "fan_mean": pred_f["mean"].values,
        "fan_ci_low": pred_f["mean_ci_lower"].values,
        "fan_ci_high": pred_f["mean_ci_upper"].values,
        "baseline_partner": base_partner,
        "baseline_industry": base_ind,
        "baseline_season": base_season,
        "baseline_week": base_week,
        "age_sd": age_sd,
    })
    age_pred.to_csv(BASE / "q3_age_curve_predictions.csv", index=False)

    # ---------- FIGURES ----------
    set_style()

    # FIG-01: Pro dancer -> do_well_main forest (Top/Bottom 10)
    pd_dw = effects_all.query("outcome=='do_well_main' and factor_type=='pro_dancer'")[["factor_level","effect","ci_low","ci_high"]].copy()
    if pd_dw.empty:
        hard_fail("No pro_dancer terms in do_well_main model; cannot plot FIG-01.")
    pd_dw = top_bottom(pd_dw, n=10).sort_values("effect")

    fig, ax = plt.subplots(figsize=(9, 7))
    y = np.arange(len(pd_dw))
    ax.hlines(y=y, xmin=pd_dw["ci_low"], xmax=pd_dw["ci_high"])
    ax.plot(pd_dw["effect"], y, marker="o", linestyle="None")
    ax.axvline(0, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(pd_dw["factor_level"])
    ax.set_xlabel("Effect on do_well_main (OLS coefficient, 95% CI)")
    ax.set_title("FIG-01 (Q3-T01/T03): Pro Dancer Effects on do_well_main (Top/Bottom 10)")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(BASE / "fig_q3_t01_pro_dancer_do_well.png", dpi=300)
    plt.close(fig)

    # FIG-02: Age curve judge vs fan
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(age_pred["age"], age_pred["judge_mean"], label="Judge (OLS)")
    ax.fill_between(age_pred["age"], age_pred["judge_ci_low"], age_pred["judge_ci_high"], alpha=0.2)
    ax.plot(age_pred["age"], age_pred["fan_mean"], label="Fan (WLS)")
    ax.fill_between(age_pred["age"], age_pred["fan_ci_low"], age_pred["fan_ci_high"], alpha=0.2)
    ax.set_xlabel("Celebrity age during season")
    ax.set_ylabel("Predicted normalized score")
    ax.set_title("FIG-02 (Q3-T02/T04): Age Effect Curve — Judge vs Fan (baseline controls)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(BASE / "fig_q3_t02_age_curve_judge_vs_fan.png", dpi=300)
    plt.close(fig)

    # FIG-03: Industry dumbbell (20 largest |fan-judge|)
    ind = comp.query("factor_type=='industry'").copy()
    if ind.empty:
        hard_fail("No industry terms in compare table; cannot plot FIG-03.")
    ind["abs_diff"] = ind["diff_fan_minus_judge"].abs()
    ind = ind.sort_values("abs_diff", ascending=False).head(20).sort_values("diff_fan_minus_judge")

    y = np.arange(len(ind))
    fig, ax = plt.subplots(figsize=(9, 7))
    for i in range(len(ind)):
        ax.plot([ind["effect_judge"].iloc[i], ind["effect_fan"].iloc[i]], [y[i], y[i]])
    ax.plot(ind["effect_judge"], y, marker="o", linestyle="None", label="Judge")
    ax.plot(ind["effect_fan"], y, marker="o", linestyle="None", label="Fan")
    ax.axvline(0, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(ind["factor_level"])
    ax.set_xlabel("Effect (coefficient, normalized outcome)")
    ax.set_title("FIG-03 (Q3-T02/T04): Industry Effects — Judge vs Fan (largest gaps)")
    ax.legend()
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(BASE / "fig_q3_t02_industry_dumbbell.png", dpi=300)
    plt.close(fig)

    # FIG-04: Consistency scatter (judge vs fan + 45°)
    allc = comp.copy()
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(allc["effect_judge"], allc["effect_fan"], alpha=0.7)
    lim = np.nanmax(np.abs(pd.concat([allc["effect_judge"], allc["effect_fan"]], axis=0)))
    if not np.isfinite(lim) or lim == 0:
        lim = 1.0
    ax.plot([-lim, lim], [-lim, lim], linestyle="--")
    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Judge effect (OLS)")
    ax.set_ylabel("Fan effect (WLS)")
    ax.set_title("FIG-04 (Q3-T04): Consistency — Judge vs Fan Effects")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(BASE / "fig_q3_t04_consistency_scatter.png", dpi=300)
    plt.close(fig)

    # FIG-05: Fan uncertainty histogram (kept vs all)
    fig, ax = plt.subplots(figsize=(9, 5))
    se_all = weekly["fan_se"]
    se_kept = weekly.loc[(weekly["latent_missing_flag"] == 0) & (weekly["fan_invalid_flag"] == 0), "fan_se"]
    ax.hist(se_all.dropna(), bins=40, alpha=0.5, label="All")
    ax.hist(se_kept.dropna(), bins=40, alpha=0.5, label="Used in WLS")
    ax.set_xlabel("fan_se (from normalized CI width)")
    ax.set_ylabel("Count")
    ax.set_title("FIG-05 (Innovation): Fan Uncertainty — All vs Used in WLS")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(BASE / "fig_q3_innovation_fan_uncertainty.png", dpi=300)
    plt.close(fig)

    print("[OK] wrote CSVs:")
    print(" - q3_effects_standardized.csv")
    print(" - q3_compare_judge_vs_fan.csv")
    print(" - q3_age_curve_predictions.csv")
    print("[OK] wrote figures:")
    print(" - fig_q3_t01_pro_dancer_do_well.png")
    print(" - fig_q3_t02_age_curve_judge_vs_fan.png")
    print(" - fig_q3_t02_industry_dumbbell.png")
    print(" - fig_q3_t04_consistency_scatter.png")
    print(" - fig_q3_innovation_fan_uncertainty.png")

if __name__ == "__main__":
    main()
