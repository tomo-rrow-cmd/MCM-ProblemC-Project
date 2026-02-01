#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
estimate_fan_votes.py

Dual-Model Hidden Fan Vote Estimation
- Option A (Deterministic):
    * Percent Rule (S3–S27): LP feasible bounds with slack epsilon via scipy.optimize.linprog(method="highs")
    * Rank Rule (S1–S2, S28–S34): ILP feasible bounds with slack delta via PuLP (CBC)
- Option B (Stochastic):
    * ABC (Approximate Bayesian Computation): 10,000 sims per week
    * Percent: Dirichlet(alpha=1) prior (uniform on simplex)
    * Rank: random permutation prior (baseline)
    * S28+: Bottom2 + probabilistic Judges' Save (sigmoid)
    * Post-processing: Maps latent ranks/votes to a normalized percentage share for visualization.

Usage:
    python estimate_fan_votes.py --data_path data/cleaned_data.csv --output_path outputs/latent_estimates.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from scipy.optimize import linprog
import pulp

# =========================
# Global Hyperparameters
# =========================
RANDOM_SEED = 42
N_SIMS = 10_000  # ABC simulations per week
BETA = 0.5  # sigmoid slope for Judges' Save (tunable)
EPS_TOL = 1e-9  # numerical stability
LP_METHOD = "highs"  # robust LP solver in SciPy

# =========================
# Regex for parsing results
# =========================
RESULT_WEEK_RE = re.compile(r"(Eliminated|Withdrew|Disqualified)\s+Week\s+(\d+)", re.IGNORECASE)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """数值稳定的 sigmoid。"""
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def classify_rule_type(season: int) -> str:
    """赛季规则分类：Percent (S3–S27) / Rank (S1–S2, S28–S34)."""
    if 3 <= season <= 27:
        return "percent"
    return "rank"


def parse_elimination_week_map(df_season: pd.DataFrame) -> pd.Series:
    """
    为一个 season 生成：每个 celebrity_name 的 elimination_week（整数或 NaN）
    """
    res_by_name = df_season.groupby("celebrity_name", sort=False)["results"].first()
    extracted = res_by_name.astype(str).str.extract(RESULT_WEEK_RE)
    elim_week = pd.to_numeric(extracted[1], errors="coerce")
    elim_week.name = "elimination_week"
    return elim_week


def get_actual_elims(elim_week_map: pd.Series, week: int) -> Set[str]:
    """返回该周真实淘汰的选手集合。"""
    names = elim_week_map.index[elim_week_map.eq(week)]
    return set(names.tolist())


def compute_judge_metrics(df_week: pd.DataFrame) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """
    计算当周评委相关量：
    - J: raw_total_score
    - judge_percent: J / sum(J)
    - judge_rank: 1..n (Using rank method='min' to align with V7 logic, allowing ties)
    """
    dfw = df_week[["celebrity_name", "raw_total_score"]].copy()
    dfw = dfw.sort_values("celebrity_name", kind="mergesort")

    contestants = dfw["celebrity_name"].to_numpy()
    J = dfw["raw_total_score"].to_numpy(dtype=float)
    name_to_idx = {name: i for i, name in enumerate(contestants)}

    J_sum = float(np.sum(J))
    if J_sum <= EPS_TOL:
        judge_percent = np.full_like(J, np.nan, dtype=float)
    else:
        judge_percent = J / J_sum

    # 使用 method='min' 允许并列排名 (e.g. 1, 1, 3)
    judge_rank = dfw["raw_total_score"].rank(method="min", ascending=False).to_numpy(dtype=float)

    return contestants, J, judge_percent, judge_rank, name_to_idx


def map_rank_to_percentage(mean_ranks: np.ndarray, n_active: int, k: float = 1.0) -> np.ndarray:
    """
    方案 B：指数衰减映射
    将排名期望值映射为粉丝得票百分比。
    k: 调节参数。k越大，头部效应（第一名与第二名的差距）越明显。
    """
    # 计算指数得分: Score_i = exp(-k * (Rank_i - 1) / (n_active - 1))
    if n_active <= 1:
        return np.array([1.0])

    # 这里的 mean_ranks 是 1..n 的数值
    scores = np.exp(-k * (mean_ranks - 1) / (n_active - 1))
    return scores / scores.sum()  # 归一化为百分比占比


# ==========================================================
# Option A - Percent Rule: LP feasible bounds
# ==========================================================
def solve_percent_feasible_bounds(
        contestants: np.ndarray,
        judge_percent: np.ndarray,
        actual_elims: Set[str],
) -> Tuple[np.ndarray, np.ndarray, float, bool]:
    n = len(contestants)
    feasible_min = np.full(n, np.nan, dtype=float)
    feasible_max = np.full(n, np.nan, dtype=float)

    k = len(actual_elims)
    if k == 0:
        return np.zeros(n), np.ones(n), 0.0, False

    elim_idx = np.array([i for i, name in enumerate(contestants) if name in actual_elims], dtype=int)
    if elim_idx.size == 0:
        return feasible_min, feasible_max, np.nan, True

    safe_idx = np.array([i for i in range(n) if i not in set(elim_idx.tolist())], dtype=int)
    if safe_idx.size == 0:
        return feasible_min, feasible_max, np.nan, True

    # 变量 x = [p_0..p_{n-1}, eps]
    A_eq = np.zeros((1, n + 1), dtype=float)
    A_eq[0, :n] = 1.0
    b_eq = np.array([1.0], dtype=float)

    bounds = [(0.0, 1.0)] * n + [(0.0, None)]

    # 约束：p_e - p_u - eps <= j_u - j_e
    e_rep = np.repeat(elim_idx, safe_idx.size)
    u_rep = np.tile(safe_idx, elim_idx.size)
    num_rows = e_rep.size

    A_ub = np.zeros((num_rows, n + 1), dtype=float)
    A_ub[np.arange(num_rows), e_rep] = 1.0
    A_ub[np.arange(num_rows), u_rep] = -1.0
    A_ub[:, -1] = -1.0

    b_ub = judge_percent[u_rep] - judge_percent[e_rep]

    # Stage 1: minimize eps
    c = np.zeros(n + 1, dtype=float)
    c[-1] = 1.0

    try:
        res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=LP_METHOD)
        if not res.success or res.x is None:
            return feasible_min, feasible_max, np.nan, True
        eps_star = float(max(0.0, res.x[-1]))
    except Exception:
        return feasible_min, feasible_max, np.nan, True

    is_noisy = eps_star > EPS_TOL

    # Stage 2: fix eps, solve min/max p_i
    bounds_fixed = [(0.0, 1.0)] * n + [(eps_star, eps_star)]

    for i in range(n):
        # min
        c_min = np.zeros(n + 1, dtype=float)
        c_min[i] = 1.0
        try:
            rmin = linprog(c=c_min, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds_fixed, method=LP_METHOD)
            if rmin.success: feasible_min[i] = float(rmin.x[i])
        except Exception:
            pass
        # max
        c_max = np.zeros(n + 1, dtype=float)
        c_max[i] = -1.0
        try:
            rmax = linprog(c=c_max, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds_fixed, method=LP_METHOD)
            if rmax.success: feasible_max[i] = float(rmax.x[i])
        except Exception:
            pass

    return feasible_min, feasible_max, eps_star, is_noisy


# ==========================================================
# Option A - Rank Rule: ILP feasible bounds
# ==========================================================
def _build_rank_ilp(
        contestants: np.ndarray,
        judge_rank: np.ndarray,
        mode: str,
        eliminated_names: Set[str],
        bottom2_partner: Optional[str] = None,
        delta_fixed: Optional[float] = None,
        objective: Optional[Tuple[str, int]] = None,
) -> Tuple[pulp.LpProblem, Dict, List, pulp.LpVariable]:
    n = len(contestants)
    I = list(range(n))
    R = list(range(1, n + 1))
    prob = pulp.LpProblem("RankFeasible", pulp.LpMinimize)
    x = {(i, r): pulp.LpVariable(f"x_{i}_{r}", cat="Binary") for i in I for r in R}

    if delta_fixed is None:
        delta = pulp.LpVariable("delta", lowBound=0.0)
    else:
        delta = pulp.LpVariable("delta", lowBound=delta_fixed, upBound=delta_fixed)

    for i in I: prob += pulp.lpSum(x[(i, r)] for r in R) == 1
    for r in R: prob += pulp.lpSum(x[(i, r)] for i in I) == 1

    fan_rank_expr = [pulp.lpSum(r * x[(i, r)] for r in R) for i in I]
    combined_expr = [float(judge_rank[i]) + fan_rank_expr[i] for i in I]
    elim_idx = [i for i, nm in enumerate(contestants) if nm in eliminated_names]

    if mode == "worst_eliminated":
        elim_set = set(elim_idx)
        safe_idx = [i for i in I if i not in elim_set]
        for u in safe_idx:
            for e in elim_set:
                prob += (combined_expr[u] - combined_expr[e]) <= delta

    elif mode == "bottom2":
        if len(elim_idx) != 1 or bottom2_partner is None:
            raise ValueError("bottom2 mode requires exactly one eliminated and a partner")
        e = elim_idx[0]
        b = int(np.where(contestants == bottom2_partner)[0][0])
        excluded = {e, b}
        for u in I:
            if u not in excluded:
                prob += (combined_expr[u] - combined_expr[e]) <= delta
                prob += (combined_expr[u] - combined_expr[b]) <= delta

    if objective is None:
        prob += delta
    else:
        kind, idx = objective
        if kind == "min_rank":
            prob += fan_rank_expr[idx]
        elif kind == "max_rank":
            prob += -fan_rank_expr[idx]

    return prob, x, fan_rank_expr, delta


def solve_rank_feasible_bounds(
        season: int,
        contestants: np.ndarray,
        judge_rank: np.ndarray,
        actual_elims: Set[str],
) -> Tuple[np.ndarray, np.ndarray, float, bool]:
    n = len(contestants)
    feasible_min = np.full(n, np.nan, dtype=float)
    feasible_max = np.full(n, np.nan, dtype=float)
    k = len(actual_elims)
    if k == 0:
        return np.ones(n), np.full(n, n, dtype=float), 0.0, False

    solver = pulp.PULP_CBC_CMD(msg=False)
    is_s28_plus = season >= 28

    delta_star = np.nan
    best_partner = None

    try:
        # Stage 1: Minimize Delta
        if is_s28_plus and k == 1:
            elim_name = next(iter(actual_elims))
            partners = [nm for nm in contestants.tolist() if nm != elim_name]
            best_delta = None
            for b in partners:
                prob, _, _, delta = _build_rank_ilp(
                    contestants, judge_rank, "bottom2", {elim_name}, b, None, None
                )
                if pulp.LpStatus[prob.solve(solver)] == "Optimal":
                    dval = float(pulp.value(delta))
                    if best_delta is None or dval < best_delta - EPS_TOL:
                        best_delta = dval
                        best_partner = b
            if best_delta is None: return feasible_min, feasible_max, np.nan, True
            delta_star = float(max(0.0, best_delta))
        else:
            prob, _, _, delta = _build_rank_ilp(
                contestants, judge_rank, "worst_eliminated", actual_elims, None, None, None
            )
            if pulp.LpStatus[prob.solve(solver)] != "Optimal":
                return feasible_min, feasible_max, np.nan, True
            delta_star = float(max(0.0, pulp.value(delta)))

        is_noisy = delta_star > EPS_TOL

        # Stage 2: Bounds
        # (Simplified logic: if Stage 1 worked, proceed with same mode)
        mode = "bottom2" if (is_s28_plus and k == 1 and best_partner) else "worst_eliminated"
        partner = best_partner if mode == "bottom2" else None

        for i in range(n):
            for kind in ["min_rank", "max_rank"]:
                prob, _, fan_expr, _ = _build_rank_ilp(
                    contestants, judge_rank, mode, actual_elims, partner, delta_star, (kind, i)
                )
                if pulp.LpStatus[prob.solve(solver)] == "Optimal":
                    val = float(pulp.value(fan_expr[i]))
                    if kind == "min_rank":
                        feasible_min[i] = val
                    else:
                        feasible_max[i] = val

    except Exception:
        return feasible_min, feasible_max, float(delta_star) if delta_star == delta_star else np.nan, True

    return feasible_min, feasible_max, float(delta_star), is_noisy


# ==========================================
# Option B - ABC Percent
# ==========================================
def abc_percent(
        rng: np.random.Generator,
        contestants: np.ndarray,
        judge_percent: np.ndarray,
        actual_elims: Set[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    n = len(contestants)
    k = len(actual_elims)
    p = rng.dirichlet(alpha=np.ones(n), size=N_SIMS)

    if k == 0:
        return p.mean(axis=0), np.quantile(p, 0.025, axis=0), np.quantile(p, 0.975, axis=0), 1.0

    actual_idx = np.array([i for i, nm in enumerate(contestants) if nm in actual_elims], dtype=int)
    if actual_idx.size != k: return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), 0.0

    S = p + judge_percent.reshape(1, -1)

    if k == 1:
        accept = np.argmin(S, axis=1) == actual_idx[0]
    else:
        bottomk = np.argpartition(S, kth=k - 1, axis=1)[:, :k]
        accept = np.all(np.sort(bottomk, axis=1) == np.sort(actual_idx).reshape(1, -1), axis=1)

    accepted = p[accept]
    accept_rate = float(np.mean(accept))
    if accepted.shape[0] == 0:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), accept_rate

    return accepted.mean(axis=0), np.quantile(accepted, 0.025, axis=0), np.quantile(accepted, 0.975,
                                                                                    axis=0), accept_rate


# ==========================================
# Option B - ABC Rank
# ==========================================
def abc_rank(
        rng: np.random.Generator,
        season: int,
        contestants: np.ndarray,
        judge_rank: np.ndarray,
        judge_raw: np.ndarray,
        actual_elims: Set[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    n = len(contestants)
    k = len(actual_elims)
    is_s28_plus = season >= 28

    rand = rng.random((N_SIMS, n))
    order = np.argsort(rand, axis=1)
    fan_rank = np.empty_like(order, dtype=int)
    fan_rank[np.arange(N_SIMS)[:, None], order] = np.arange(1, n + 1, dtype=int).reshape(1, -1)

    if k == 0:
        return fan_rank.mean(axis=0), np.quantile(fan_rank, 0.025, axis=0), np.quantile(fan_rank, 0.975, axis=0), 1.0

    actual_idx = np.array([i for i, nm in enumerate(contestants) if nm in actual_elims], dtype=int)
    if actual_idx.size != k: return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), 0.0

    combined = fan_rank + judge_rank.reshape(1, -1)

    if k >= 2 or (not is_s28_plus):
        if k == 1:
            accept = np.argmax(combined, axis=1) == actual_idx[0]
        else:
            worstk = np.argpartition(combined, kth=n - k, axis=1)[:, -k:]
            accept = np.all(np.sort(worstk, axis=1) == np.sort(actual_idx).reshape(1, -1), axis=1)
    else:
        # S28+ Single Elim: Bottom 2 + Save
        a = actual_idx[0]
        bottom2 = np.argpartition(combined, kth=n - 2, axis=1)[:, -2:]
        in_bottom2 = np.any(bottom2 == a, axis=1)
        other = np.where(bottom2[:, 0] == a, bottom2[:, 1], bottom2[:, 0])
        score_diff = (judge_raw[other] - judge_raw[a]).astype(float)
        P = sigmoid(BETA * score_diff)
        accept = in_bottom2 & (rng.random(N_SIMS) < P)

    accepted = fan_rank[accept]
    accept_rate = float(np.mean(accept))
    if accepted.shape[0] == 0:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), accept_rate

    return accepted.mean(axis=0), np.quantile(accepted, 0.025, axis=0), np.quantile(accepted, 0.975,
                                                                                    axis=0), accept_rate


# ==========================================
# Main Execution
# ==========================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Estimate hidden fan votes via Dual-Model (LP/ILP + ABC).")
    p.add_argument("--data_path", type=str, default="cleaned_data.csv", help="Path to cleaned_data.csv")
    p.add_argument("--output_path", type=str, default="outputs/latent_estimates.csv", help="Output CSV path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(RANDOM_SEED)

    # 1. Load Data
    df = pd.read_csv(data_path)
    required_cols = {"season", "week", "celebrity_name", "raw_total_score", "results", "is_active"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing cols: {required_cols - set(df.columns)}")

    df = df[df["is_active"] == True].copy()
    df["raw_total_score"] = pd.to_numeric(df["raw_total_score"], errors="coerce")

    seasons = sorted(df["season"].unique().tolist())
    out_rows: List[Dict] = []

    # 2. Iterate
    for s in tqdm(seasons, desc="Seasons"):
        rule_type = classify_rule_type(int(s))
        df_s = df[df["season"] == s].copy()
        elim_week_map = parse_elimination_week_map(df_s)
        weeks = sorted(df_s["week"].unique().tolist())

        for w in tqdm(weeks, desc=f"S{s} weeks", leave=False):
            df_sw = df_s[df_s["week"] == w].copy()
            contestants, J, judge_percent, judge_rank, name_to_idx = compute_judge_metrics(df_sw)

            n = len(contestants)
            if n < 2: continue

            actual_elims = get_actual_elims(elim_week_map, int(w))
            k = len(actual_elims)

            # --- A. Bounds ---
            if rule_type == "percent":
                if np.any(np.isnan(judge_percent)):
                    feasible_min, feasible_max = np.full(n, np.nan), np.full(n, np.nan)
                    slack_min, is_noisy = np.nan, True
                else:
                    feasible_min, feasible_max, slack_min, is_noisy = solve_percent_feasible_bounds(
                        contestants, judge_percent, actual_elims
                    )
                judge_metric = judge_percent
            else:
                feasible_min, feasible_max, slack_min, is_noisy = solve_rank_feasible_bounds(
                    int(s), contestants, judge_rank, actual_elims
                )
                judge_metric = judge_rank

            # --- B. ABC ---
            try:
                if rule_type == "percent":
                    fan_mean, ci_low, ci_high, accept_rate = abc_percent(rng, contestants, judge_percent, actual_elims)
                else:
                    fan_mean, ci_low, ci_high, accept_rate = abc_rank(rng, int(s), contestants, judge_rank, J,
                                                                      actual_elims)
            except Exception:
                fan_mean, ci_low, ci_high = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
                accept_rate = 0.0

            week_type = "no_elimination" if k == 0 else ("single_elimination" if k == 1 else f"multi_elimination_{k}")
            is_elim_vec = np.array([name in actual_elims for name in contestants], dtype=bool)

            for i, name in enumerate(contestants):
                out_rows.append({
                    "season": int(s),
                    "week": int(w),
                    "celebrity_name": name,
                    "rule_type": rule_type,
                    "week_type": week_type,
                    "n_active": int(n),
                    "judge_raw": float(J[i]),
                    "judge_metric": float(judge_metric[i]) if not np.isnan(judge_metric[i]) else np.nan,
                    "is_eliminated_this_week": bool(is_elim_vec[i]),
                    "fan_vote_mean": float(fan_mean[i]) if not np.isnan(fan_mean[i]) else np.nan,
                    "ci_low": float(ci_low[i]) if not np.isnan(ci_low[i]) else np.nan,
                    "ci_high": float(ci_high[i]) if not np.isnan(ci_high[i]) else np.nan,
                    "abc_accept_rate": float(accept_rate),
                    "feasible_min": float(feasible_min[i]) if not np.isnan(feasible_min[i]) else np.nan,
                    "feasible_max": float(feasible_max[i]) if not np.isnan(feasible_max[i]) else np.nan,
                    "slack_min": float(slack_min) if slack_min == slack_min else np.nan,
                    "is_noisy": bool(is_noisy) if slack_min == slack_min else True,
                })

    out_df = pd.DataFrame(out_rows)

    # 3. Post-Processing: Calculate fan_vote_share (Percentage Mapping)
    # =================================================================
    out_df["fan_vote_share"] = np.nan

    for (season, week), group in out_df.groupby(["season", "week"]):
        idx = group.index
        rule = group["rule_type"].iloc[0]
        n_active = len(group)

        if rule == "percent":
            # For percent rule, normalize the ABC means to sum strictly to 1
            shares = group["fan_vote_mean"].values
            total = np.nansum(shares)
            if total > 0:
                out_df.loc[idx, "fan_vote_share"] = shares / total
        else:
            # For rank rule, map latent rank to percentage
            mean_ranks = group["fan_vote_mean"].values
            if not np.all(np.isnan(mean_ranks)):
                out_df.loc[idx, "fan_vote_share"] = map_rank_to_percentage(mean_ranks, n_active, k=1.2)

    # Convert to 0-100 scale
    out_df["fan_vote_share_pct"] = out_df["fan_vote_share"] * 100

    # 4. Save
    cols = [
        "season", "week", "celebrity_name", "rule_type",
        "judge_raw", "judge_metric",
        "fan_vote_mean", "fan_vote_share_pct",  # Added this column
        "ci_low", "ci_high", "abc_accept_rate",
        "feasible_min", "feasible_max", "slack_min", "is_noisy",
        "week_type", "n_active", "is_eliminated_this_week",
    ]
    cols = [c for c in cols if c in out_df.columns]
    out_df = out_df[cols].sort_values(["season", "week", "celebrity_name"], kind="mergesort")

    out_df.to_csv(output_path, index=False)
    print(f"[OK] Saved latent estimates with vote shares to: {output_path}")


if __name__ == "__main__":
    main()