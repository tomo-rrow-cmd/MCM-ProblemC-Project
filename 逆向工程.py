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

Usage:
    python estimate_fan_votes.py --data_path data/cleaned_data.csv --output_path outputs/latent_estimates.csv

Notes (Beginner-friendly, Chinese):
    - 只使用 is_active == True 的行，因为淘汰后 0/NaN 会污染统计与约束
    - Q1 的“粉丝投票”是潜变量：我们通过“能复现历史淘汰结果”的条件来反推其分布/可行范围
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
N_SIMS = 10_000          # ABC simulations per week
BETA = 0.5               # sigmoid slope for Judges' Save (tunable)
EPS_TOL = 1e-9           # numerical stability
LP_METHOD = "highs"      # robust LP solver in SciPy


# =========================
# Regex for parsing results
# =========================
# 支持 Eliminated / Withdrew / Disqualified 三类（若数据里没有也不会影响）
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
    使用 results 字段抽取 "... Week k"
    """
    # 每个选手 results 在同一季通常是固定字符串，取 first 即可
    res_by_name = df_season.groupby("celebrity_name", sort=False)["results"].first()

    extracted = res_by_name.astype(str).str.extract(RESULT_WEEK_RE)
    # extracted[1] 是 week
    elim_week = pd.to_numeric(extracted[1], errors="coerce")
    elim_week.name = "elimination_week"
    return elim_week


def get_actual_elims(elim_week_map: pd.Series, week: int) -> Set[str]:
    """返回该周真实淘汰的选手集合（可能为空或多个）。"""
    names = elim_week_map.index[elim_week_map.eq(week)]
    return set(names.tolist())


def compute_judge_metrics(df_week: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """
    计算当周评委相关量：
    - contestants: np.array of names (length n)
    - J: raw_total_score aligned to contestants
    - judge_percent: J / sum(J)
    - judge_rank: 1..n（高分=rank 1；同分用名字字母序打破，确保唯一）
    - name_to_idx: mapping
    """
    # 保持一个稳定顺序：先按名字排序（只是为了可复现），后续 rank 再单独计算
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

    # judge_rank：按 (-J, name) 排序后赋值 1..n，保证唯一（避免 ties 导致 rank 不可用）
    order = np.lexsort((contestants, -J))  # primary: -J, secondary: name
    judge_rank = np.empty_like(order, dtype=int)
    judge_rank[order] = np.arange(1, len(contestants) + 1, dtype=int)

    return contestants, J, judge_percent, judge_rank.astype(float), name_to_idx


# ==========================================================
# Option A - Percent Rule: LP feasible bounds with slack eps
# ==========================================================
def solve_percent_feasible_bounds(
    contestants: np.ndarray,
    judge_percent: np.ndarray,
    actual_elims: Set[str],
) -> Tuple[np.ndarray, np.ndarray, float, bool]:
    """
    Option A (Percent): Two-stage LP
    Stage 1: minimize epsilon
    Stage 2: fix epsilon = epsilon_star, solve min/max p_i

    Returns:
        feasible_min: (n,) lower bounds for p
        feasible_max: (n,) upper bounds for p
        epsilon_star: minimized slack
        is_noisy: epsilon_star > 0
    """
    n = len(contestants)
    feasible_min = np.full(n, np.nan, dtype=float)
    feasible_max = np.full(n, np.nan, dtype=float)

    k = len(actual_elims)
    if k == 0:
        # 无淘汰：只有 simplex 约束 => p_i in [0,1]
        return np.zeros(n), np.ones(n), 0.0, False

    # 将淘汰集合转索引
    elim_idx = np.array([i for i, name in enumerate(contestants) if name in actual_elims], dtype=int)
    if elim_idx.size == 0:
        # 数据不一致：results 解析到的淘汰者不在当周 active 名单中
        return feasible_min, feasible_max, np.nan, True

    safe_idx = np.array([i for i in range(n) if i not in set(elim_idx.tolist())], dtype=int)
    if safe_idx.size == 0:
        # 全部淘汰（理论不会发生）
        return feasible_min, feasible_max, np.nan, True

    # 决策变量 x = [p_0..p_{n-1}, eps]
    # 等式：sum p = 1
    A_eq = np.zeros((1, n + 1), dtype=float)
    A_eq[0, :n] = 1.0
    b_eq = np.array([1.0], dtype=float)

    # bounds: p in [0,1], eps in [0, inf)
    bounds = [(0.0, 1.0)] * n + [(0.0, None)]

    # 构造不等式：p_e - p_u - eps <= j_u - j_e
    # 对所有 e in E, u in U
    e_rep = np.repeat(elim_idx, safe_idx.size)
    u_rep = np.tile(safe_idx, elim_idx.size)
    num_rows = e_rep.size

    A_ub = np.zeros((num_rows, n + 1), dtype=float)
    A_ub[np.arange(num_rows), e_rep] = 1.0
    A_ub[np.arange(num_rows), u_rep] = -1.0
    A_ub[:, -1] = -1.0

    b_ub = judge_percent[u_rep] - judge_percent[e_rep]

    # ---------- Stage 1: minimize eps ----------
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

    # ---------- Stage 2: fix eps=eps_star, solve min/max each p_i ----------
    bounds_fixed = [(0.0, 1.0)] * n + [(eps_star, eps_star)]

    for i in range(n):
        # min p_i
        c_min = np.zeros(n + 1, dtype=float)
        c_min[i] = 1.0
        try:
            rmin = linprog(c=c_min, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds_fixed, method=LP_METHOD)
            if rmin.success and rmin.x is not None:
                feasible_min[i] = float(rmin.x[i])
        except Exception:
            pass

        # max p_i == min (-p_i)
        c_max = np.zeros(n + 1, dtype=float)
        c_max[i] = -1.0
        try:
            rmax = linprog(c=c_max, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds_fixed, method=LP_METHOD)
            if rmax.success and rmax.x is not None:
                feasible_max[i] = float(rmax.x[i])
        except Exception:
            pass

    return feasible_min, feasible_max, eps_star, is_noisy


# ==========================================================
# Option A - Rank Rule: ILP feasible bounds with slack delta
# ==========================================================
def _build_rank_ilp(
    contestants: np.ndarray,
    judge_rank: np.ndarray,
    mode: str,
    eliminated_names: Set[str],
    bottom2_partner: Optional[str] = None,
    delta_fixed: Optional[float] = None,
    objective: Optional[Tuple[str, int]] = None,
) -> Tuple[pulp.LpProblem, Dict[Tuple[int, int], pulp.LpVariable], List[pulp.LpAffineExpression], pulp.LpVariable]:
    """
    Build a PuLP ILP:
      - x[i,r] binary assignment of fan ranks (r=1..n)
      - fan_rank_expr[i] = sum_r r*x[i,r]
      - combined_expr[i] = judge_rank[i] + fan_rank_expr[i]
      - delta >= 0 slack

    mode:
      - "worst_eliminated": for all safe u and eliminated e: combined_u - combined_e <= delta
      - "bottom2": enforce eliminated + partner are bottom2: for all u not in {elim, partner}:
            combined_u - combined_elim <= delta and combined_u - combined_partner <= delta
    objective:
      - None: minimize delta
      - ("min_rank", i): minimize fan_rank of contestant i
      - ("max_rank", i): maximize fan_rank of contestant i (implemented as minimize -fan_rank)
    """
    n = len(contestants)
    I = list(range(n))
    R = list(range(1, n + 1))

    prob = pulp.LpProblem("RankFeasible", pulp.LpMinimize)

    # Assignment variables
    x = {(i, r): pulp.LpVariable(f"x_{i}_{r}", cat="Binary") for i in I for r in R}

    # Delta slack
    if delta_fixed is None:
        delta = pulp.LpVariable("delta", lowBound=0.0)
    else:
        delta = pulp.LpVariable("delta", lowBound=delta_fixed, upBound=delta_fixed)

    # Permutation constraints
    for i in I:
        prob += pulp.lpSum(x[(i, r)] for r in R) == 1
    for r in R:
        prob += pulp.lpSum(x[(i, r)] for i in I) == 1

    fan_rank_expr = [pulp.lpSum(r * x[(i, r)] for r in R) for i in I]
    combined_expr = [float(judge_rank[i]) + fan_rank_expr[i] for i in I]

    elim_idx = [i for i, nm in enumerate(contestants) if nm in eliminated_names]
    if len(elim_idx) == 0:
        # nothing to constrain; leave it (caller handles no-elimination separately)
        pass

    if mode == "worst_eliminated":
        # enforce eliminated set are among worst (bottom k) vs safe set
        elim_set = set(elim_idx)
        safe_idx = [i for i in I if i not in elim_set]
        for u in safe_idx:
            for e in elim_set:
                prob += (combined_expr[u] - combined_expr[e]) <= delta

    elif mode == "bottom2":
        # Only meaningful when single eliminated
        if len(elim_idx) != 1 or bottom2_partner is None:
            raise ValueError("bottom2 mode requires exactly one eliminated and a partner name")

        e = elim_idx[0]
        b = int(np.where(contestants == bottom2_partner)[0][0])
        excluded = {e, b}
        for u in I:
            if u not in excluded:
                prob += (combined_expr[u] - combined_expr[e]) <= delta
                prob += (combined_expr[u] - combined_expr[b]) <= delta
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Objective
    if objective is None:
        prob += delta
    else:
        kind, idx = objective
        if kind == "min_rank":
            prob += fan_rank_expr[idx]
        elif kind == "max_rank":
            prob += -fan_rank_expr[idx]
        else:
            raise ValueError(f"Unknown objective kind: {kind}")

    return prob, x, fan_rank_expr, delta


def solve_rank_feasible_bounds(
    season: int,
    contestants: np.ndarray,
    judge_rank: np.ndarray,
    actual_elims: Set[str],
) -> Tuple[np.ndarray, np.ndarray, float, bool]:
    """
    Option A (Rank): Two-stage ILP with slack delta.
    - For S1–S2: "worst_eliminated"
    - For S28+: if single elimination, use "bottom2" by enumerating partner; pick minimal delta.

    Returns:
        feasible_min_rank, feasible_max_rank, delta_star, is_noisy
    """
    n = len(contestants)
    feasible_min = np.full(n, np.nan, dtype=float)
    feasible_max = np.full(n, np.nan, dtype=float)

    k = len(actual_elims)
    if k == 0:
        # 无淘汰：粉丝名次可为任意 1..n
        return np.ones(n), np.full(n, n, dtype=float), 0.0, False

    solver = pulp.PULP_CBC_CMD(msg=False)

    # Decide mode
    is_s28_plus = season >= 28

    # ---------- Stage 1: minimize delta ----------
    delta_star = np.nan
    best_partner = None

    try:
        if is_s28_plus and k == 1:
            # bottom2 + Judges Save: eliminated must be in bottom2, partner unknown -> enumerate
            elim_name = next(iter(actual_elims))
            partners = [nm for nm in contestants.tolist() if nm != elim_name]

            best_delta = None
            for b in partners:
                prob, _, _, delta = _build_rank_ilp(
                    contestants=contestants,
                    judge_rank=judge_rank,
                    mode="bottom2",
                    eliminated_names={elim_name},
                    bottom2_partner=b,
                    delta_fixed=None,
                    objective=None,
                )
                status = prob.solve(solver)
                if pulp.LpStatus[status] != "Optimal":
                    continue
                dval = float(pulp.value(delta))
                if best_delta is None or dval < best_delta - EPS_TOL:
                    best_delta = dval
                    best_partner = b

            if best_delta is None:
                return feasible_min, feasible_max, np.nan, True

            delta_star = float(max(0.0, best_delta))

        else:
            # S1–S2 or multi-elimination: worst eliminated (bottom k)
            prob, _, _, delta = _build_rank_ilp(
                contestants=contestants,
                judge_rank=judge_rank,
                mode="worst_eliminated",
                eliminated_names=actual_elims,
                bottom2_partner=None,
                delta_fixed=None,
                objective=None,
            )
            status = prob.solve(solver)
            if pulp.LpStatus[status] != "Optimal":
                return feasible_min, feasible_max, np.nan, True
            delta_star = float(max(0.0, pulp.value(delta)))

    except Exception:
        return feasible_min, feasible_max, np.nan, True

    is_noisy = delta_star > EPS_TOL

    # ---------- Stage 2: fix delta=delta_star, solve min/max fan rank for each contestant ----------
    try:
        if is_s28_plus and k == 1 and best_partner is not None:
            mode = "bottom2"
            elim_name = next(iter(actual_elims))
            for i in range(n):
                # min rank
                prob_min, _, fan_expr, _ = _build_rank_ilp(
                    contestants=contestants,
                    judge_rank=judge_rank,
                    mode=mode,
                    eliminated_names={elim_name},
                    bottom2_partner=best_partner,
                    delta_fixed=delta_star,
                    objective=("min_rank", i),
                )
                st = prob_min.solve(solver)
                if pulp.LpStatus[st] == "Optimal":
                    feasible_min[i] = float(pulp.value(fan_expr[i]))

                # max rank
                prob_max, _, fan_expr2, _ = _build_rank_ilp(
                    contestants=contestants,
                    judge_rank=judge_rank,
                    mode=mode,
                    eliminated_names={elim_name},
                    bottom2_partner=best_partner,
                    delta_fixed=delta_star,
                    objective=("max_rank", i),
                )
                st2 = prob_max.solve(solver)
                if pulp.LpStatus[st2] == "Optimal":
                    feasible_max[i] = float(pulp.value(fan_expr2[i]))

        else:
            # worst eliminated
            for i in range(n):
                prob_min, _, fan_expr, _ = _build_rank_ilp(
                    contestants=contestants,
                    judge_rank=judge_rank,
                    mode="worst_eliminated",
                    eliminated_names=actual_elims,
                    bottom2_partner=None,
                    delta_fixed=delta_star,
                    objective=("min_rank", i),
                )
                st = prob_min.solve(solver)
                if pulp.LpStatus[st] == "Optimal":
                    feasible_min[i] = float(pulp.value(fan_expr[i]))

                prob_max, _, fan_expr2, _ = _build_rank_ilp(
                    contestants=contestants,
                    judge_rank=judge_rank,
                    mode="worst_eliminated",
                    eliminated_names=actual_elims,
                    bottom2_partner=None,
                    delta_fixed=delta_star,
                    objective=("max_rank", i),
                )
                st2 = prob_max.solve(solver)
                if pulp.LpStatus[st2] == "Optimal":
                    feasible_max[i] = float(pulp.value(fan_expr2[i]))

    except Exception:
        # If bounds stage fails, still return delta_star
        return feasible_min, feasible_max, float(delta_star), True

    return feasible_min, feasible_max, float(delta_star), is_noisy


# ==========================================
# Option B - ABC Percent (Dirichlet prior)
# ==========================================
def abc_percent(
    rng: np.random.Generator,
    contestants: np.ndarray,
    judge_percent: np.ndarray,
    actual_elims: Set[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    ABC for Percent rule:
      - sample p ~ Dirichlet(1)
      - S = judge_percent + p
      - accept if simulated eliminated set == actual_elims

    Returns:
      mean (n,), ci_low (n,), ci_high (n,), accept_rate
    """
    n = len(contestants)
    k = len(actual_elims)

    # Sample from prior
    p = rng.dirichlet(alpha=np.ones(n), size=N_SIMS)  # (N_SIMS, n)

    # No elimination week: accept all
    if k == 0:
        mean = p.mean(axis=0)
        ci = np.quantile(p, [0.025, 0.975], axis=0)
        return mean, ci[0], ci[1], 1.0

    # Map actual elims to indices
    actual_idx = np.array([i for i, nm in enumerate(contestants) if nm in actual_elims], dtype=int)
    if actual_idx.size != k:
        # inconsistent; accept none
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), 0.0

    S = p + judge_percent.reshape(1, -1)

    if k == 1:
        sim_elim_idx = np.argmin(S, axis=1)
        accept = sim_elim_idx == actual_idx[0]
    else:
        bottomk = np.argpartition(S, kth=k - 1, axis=1)[:, :k]
        bottomk_sorted = np.sort(bottomk, axis=1)
        actual_sorted = np.sort(actual_idx)
        accept = np.all(bottomk_sorted == actual_sorted.reshape(1, -1), axis=1)

    accepted = p[accept]
    accept_rate = float(np.mean(accept))

    if accepted.shape[0] == 0:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), accept_rate

    mean = accepted.mean(axis=0)
    ci = np.quantile(accepted, [0.025, 0.975], axis=0)
    return mean, ci[0], ci[1], accept_rate


# ==========================================
# Option B - ABC Rank (random permutation)
# ==========================================
def abc_rank(
    rng: np.random.Generator,
    season: int,
    contestants: np.ndarray,
    judge_rank: np.ndarray,
    judge_raw: np.ndarray,
    actual_elims: Set[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    ABC for Rank rule:
      - sample random permutations as fan rank (uniform)
      - combined = judge_rank + fan_rank (lower better)
      - S1–S2: eliminated = argmax(combined)
      - S28+: bottom2 + Judges Save (sigmoid based on raw_total_score diff)
    """
    n = len(contestants)
    k = len(actual_elims)
    is_s28_plus = season >= 28

    # Sample random permutations (vectorized):
    # argsort of i.i.d uniform random -> uniform random permutation
    rand = rng.random((N_SIMS, n))
    order = np.argsort(rand, axis=1)  # best to worst order (position 0 = rank 1)
    fan_rank = np.empty_like(order, dtype=int)
    fan_rank[np.arange(N_SIMS)[:, None], order] = np.arange(1, n + 1, dtype=int).reshape(1, -1)

    # No elimination week: accept all
    if k == 0:
        mean = fan_rank.mean(axis=0)
        ci = np.quantile(fan_rank, [0.025, 0.975], axis=0)
        return mean, ci[0], ci[1], 1.0

    actual_idx = np.array([i for i, nm in enumerate(contestants) if nm in actual_elims], dtype=int)
    if actual_idx.size != k:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), 0.0

    combined = fan_rank + judge_rank.reshape(1, -1)

    # Multi-elimination: treat as bottom-k worst combined
    if k >= 2 or (not is_s28_plus):
        if k == 1:
            sim_elim_idx = np.argmax(combined, axis=1)
            accept = sim_elim_idx == actual_idx[0]
        else:
            worstk = np.argpartition(combined, kth=n - k, axis=1)[:, -k:]
            worstk_sorted = np.sort(worstk, axis=1)
            actual_sorted = np.sort(actual_idx)
            accept = np.all(worstk_sorted == actual_sorted.reshape(1, -1), axis=1)

        accepted = fan_rank[accept]
        accept_rate = float(np.mean(accept))

        if accepted.shape[0] == 0:
            return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), accept_rate

        mean = accepted.mean(axis=0)
        ci = np.quantile(accepted, [0.025, 0.975], axis=0)
        return mean, ci[0], ci[1], accept_rate

    # S28+ single elimination with Judges' Save:
    # Step 1: bottom2 by highest combined
    a = actual_idx[0]

    bottom2 = np.argpartition(combined, kth=n - 2, axis=1)[:, -2:]  # (N_SIMS, 2)
    in_bottom2 = np.any(bottom2 == a, axis=1)

    # other index in bottom2 (if actual in bottom2)
    other = np.where(bottom2[:, 0] == a, bottom2[:, 1], bottom2[:, 0])

    # Step 2: Judges' Save probability => accept iff (actual in bottom2) AND (U < P)
    # P(elim=actual | bottom2) = sigmoid(BETA * (J_other - J_actual))
    # if U < P => simulated elim = actual => accept
    score_diff = (judge_raw[other] - judge_raw[a]).astype(float)
    P = sigmoid(BETA * score_diff)
    U = rng.random(N_SIMS)
    accept = in_bottom2 & (U < P)

    accepted = fan_rank[accept]
    accept_rate = float(np.mean(accept))

    if accepted.shape[0] == 0:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), accept_rate

    mean = accepted.mean(axis=0)
    ci = np.quantile(accepted, [0.025, 0.975], axis=0)
    return mean, ci[0], ci[1], accept_rate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Estimate hidden fan votes via Dual-Model (LP/ILP + ABC).")
    p.add_argument("--data_path", type=str, default="cleaned_data.csv", help="Path to cleaned_data.csv")
    p.add_argument("--output_path", type=str, default="outputs/latent_estimates.csv", help="Output CSV path")
    return p.parse_args()


def map_rank_to_percentage(mean_ranks: np.ndarray, n_active: int, k: float = 1.0) -> np.ndarray:
    """
    方案 B：指数衰减映射
    将排名期望值映射为粉丝得票百分比。
    k: 调节参数。k越大，头部效应（第一名与第二名的差距）越明显。
    """
    # 计算指数得分: Score_i = exp(-k * (Rank_i - 1) / (n_active - 1))
    # 注意：如果 n_active 为 1，分母为 0，需特殊处理
    if n_active <= 1:
        return np.array([1.0])

    scores = np.exp(-k * (mean_ranks - 1) / (n_active - 1))
    return scores / scores.sum()  # 归一化为百分比占比


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(RANDOM_SEED)

    # -----------------------------
    # 1) Load and global filter
    # -----------------------------
    df = pd.read_csv(data_path)
    required_cols = {"season", "week", "celebrity_name", "raw_total_score", "results", "is_active"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df[df["is_active"] == True].copy()
    # 防御性处理：确保数值列是 float
    df["raw_total_score"] = pd.to_numeric(df["raw_total_score"], errors="coerce")

    seasons = sorted(df["season"].unique().tolist())

    out_rows: List[Dict] = []

    # -----------------------------
    # 2) Loop season -> week
    # -----------------------------
    for s in tqdm(seasons, desc="Seasons"):
        rule_type = classify_rule_type(int(s))
        df_s = df[df["season"] == s].copy()

        # 解析本 season 的每个选手淘汰周
        elim_week_map = parse_elimination_week_map(df_s)

        weeks = sorted(df_s["week"].unique().tolist())

        for w in tqdm(weeks, desc=f"S{s} weeks", leave=False):
            df_sw = df_s[df_s["week"] == w].copy()

            # contestants & judge metrics
            contestants, J, judge_percent, judge_rank, name_to_idx = compute_judge_metrics(df_sw)

            n = len(contestants)
            if n < 2:
                # 太小没意义
                continue

            # Actual eliminations in week w
            actual_elims = get_actual_elims(elim_week_map, int(w))
            k = len(actual_elims)

            # ------------- Option A (Deterministic) -------------
            if rule_type == "percent":
                if np.any(np.isnan(judge_percent)):
                    feasible_min = np.full(n, np.nan)
                    feasible_max = np.full(n, np.nan)
                    slack_min = np.nan
                    is_noisy = True
                else:
                    feasible_min, feasible_max, slack_min, is_noisy = solve_percent_feasible_bounds(
                        contestants=contestants,
                        judge_percent=judge_percent,
                        actual_elims=actual_elims,
                    )
                judge_metric = judge_percent  # percent

            else:
                # rank rule
                feasible_min, feasible_max, slack_min, is_noisy = solve_rank_feasible_bounds(
                    season=int(s),
                    contestants=contestants,
                    judge_rank=judge_rank,
                    actual_elims=actual_elims,
                )
                judge_metric = judge_rank  # rank

            # ------------- Option B (ABC) -------------
            try:
                if rule_type == "percent":
                    fan_mean, ci_low, ci_high, accept_rate = abc_percent(
                        rng=rng,
                        contestants=contestants,
                        judge_percent=judge_percent,
                        actual_elims=actual_elims,
                    )
                else:
                    fan_mean, ci_low, ci_high, accept_rate = abc_rank(
                        rng=rng,
                        season=int(s),
                        contestants=contestants,
                        judge_rank=judge_rank,
                        judge_raw=J,
                        actual_elims=actual_elims,
                    )
            except Exception:
                fan_mean = np.full(n, np.nan)
                ci_low = np.full(n, np.nan)
                ci_high = np.full(n, np.nan)
                accept_rate = 0.0

            # Week type string
            if k == 0:
                week_type = "no_elimination"
            elif k == 1:
                week_type = "single_elimination"
            else:
                week_type = f"multi_elimination_{k}"

            # ------------- Write output rows (one per contestant) -------------
            # is_eliminated_this_week: based on parsed results
            is_elim_vec = np.array([name in actual_elims for name in contestants], dtype=bool)

            for i, name in enumerate(contestants):
                out_rows.append(
                    {
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
                        "slack_min": float(slack_min) if slack_min == slack_min else np.nan,  # NaN-safe
                        "is_noisy": bool(is_noisy) if slack_min == slack_min else True,
                    }
                )

    out_df = pd.DataFrame(out_rows)

    # 1. 初始化新列
    out_df["fan_vote_share"] = np.nan

    # 2. 针对不同规则进行映射
    for (season, week), group in out_df.groupby(["season", "week"]):
        idx = group.index
        rule = group["rule_type"].iloc[0]
        n_active = len(group)

        if rule == "percent":
            # 对于百分比制，fan_vote_mean 本身就是占比，直接归一化确保 Sum to 1
            shares = group["fan_vote_mean"].values
            out_df.loc[idx, "fan_vote_share"] = shares / shares.sum()
        else:
            # 对于排名制，使用方案 B 进行转换
            mean_ranks = group["fan_vote_mean"].values
            out_df.loc[idx, "fan_vote_share"] = map_rank_to_percentage(mean_ranks, n_active, k=1.2)

    # 3. 将结果转换为百分数 (0-100) 以适配你的可视化需求
    out_df["fan_vote_share_pct"] = out_df["fan_vote_share"] * 100
    # 重要：保证列顺序符合要求
    cols = [
        "season", "week", "celebrity_name", "rule_type",
        "judge_raw", "judge_metric",
        "fan_vote_mean", "fan_vote_share_pct",
        "ci_low", "ci_high", "abc_accept_rate",
        "feasible_min", "feasible_max", "slack_min", "is_noisy",
        # optional helpful columns
        "week_type", "n_active", "is_eliminated_this_week",
    ]
    cols = [c for c in cols if c in out_df.columns]
    out_df = out_df[cols].sort_values(["season", "week", "celebrity_name"], kind="mergesort")

    out_df.to_csv(output_path, index=False)
    print(f"[OK] Saved latent estimates: {output_path}")


if __name__ == "__main__":
    main()