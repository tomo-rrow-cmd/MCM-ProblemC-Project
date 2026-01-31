"""
plot_Figure_formal.py (Python 3.10+)

Goal
----
Draw a ridge plot (one density curve per contestant) for a specific Season/Week
using ONLY the model output file `latent_estimates.csv`.

Key idea
--------
For each contestant we have:
- mean = fan_vote_mean
- 95% CI = [ci_low, ci_high]

We approximate a distribution that matches (mean, CI):
- Convert CI width to standard deviation:
    sigma ≈ (ci_high - ci_low) / 3.92   (because 95% ≈ ±1.96*sigma for Normal)

Then:
- If the week is 'percent' mode (values in [0,1]): try a Beta distribution when feasible;
  otherwise fall back to a Normal distribution.
- If the week is 'rank' mode: use Normal distribution.

Usage
-----
python plot_Figure_formal.py --data_path latent_estimates.csv --season 3 --week 1 --out_dir outputs

Output
------
outputs/Figure_Uncertainty_PERCENT_S3_W1.png   (or RANK if that week is rank-mode)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta, norm
import seaborn as sns


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments so you can run this easily in PyCharm/terminal."""
    p = argparse.ArgumentParser(description="Formal ridge plot for uncertainty (no iterrows).")
    p.add_argument(
        "--data_path",
        type=str,
        default="latent_estimates.csv",
        help="Path to latent_estimates.csv (placeholder default).",
    )
    p.add_argument("--season", type=int, default=3, help="Season index to plot.")
    p.add_argument("--week", type=int, default=1, help="Week index to plot.")
    p.add_argument("--x_points", type=int, default=500, help="Resolution (number of x grid points).")
    p.add_argument("--out_dir", type=str, default="outputs", help="Directory to save the figure.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Cannot find CSV: {data_path}. Put it next to this script or pass --data_path."
        )

    df = pd.read_csv(data_path)

    # --- Basic column validation (beginner-friendly) ---
    needed = {"season", "week", "celebrity_name", "fan_vote_mean", "ci_low", "ci_high", "is_eliminated_this_week"}
    missing = sorted(list(needed - set(df.columns)))
    if missing:
        raise ValueError(f"latent_estimates.csv missing columns: {missing}\nAvailable columns: {list(df.columns)}")

    # --- Filter one week ---
    sub = df[(df["season"] == args.season) & (df["week"] == args.week)].copy()
    if sub.empty:
        raise ValueError(f"No rows found for Season={args.season}, Week={args.week}. Check your CSV values.")

    # --- Detect mode: percent values are in [0,1], rank values are usually > 1 ---
    max_val = pd.to_numeric(sub["fan_vote_mean"], errors="coerce").max()
    mode = "rank" if (max_val is not None and max_val > 1.05) else "percent"

    # --- Sort for nicer reading ---
    if mode == "percent":
        sub = sub.sort_values("fan_vote_mean", ascending=False)
        # DWTS vote share is usually < 0.45, so we zoom into [0, 0.45] for readability
        x_axis = np.linspace(0.0, 0.45, args.x_points)
    else:
        sub = sub.sort_values("fan_vote_mean", ascending=True)
        x_axis = np.linspace(0.0, float(len(sub) + 2), args.x_points)

    # --- Convert columns to numeric arrays ---
    mean = pd.to_numeric(sub["fan_vote_mean"], errors="coerce").to_numpy(dtype=float)
    ci_low = pd.to_numeric(sub["ci_low"], errors="coerce").to_numpy(dtype=float)
    ci_high = pd.to_numeric(sub["ci_high"], errors="coerce").to_numpy(dtype=float)

    # sigma estimated from 95% CI width:
    # For Normal: 95% CI ≈ mean ± 1.96*sigma => width ≈ 3.92*sigma
    sigma = (ci_high - ci_low) / 3.92
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-4)  # defensive fallback

    # --- Vectorized density: compute all contestants' pdfs on the same x grid ---
    X = x_axis.reshape(1, -1)     # shape (1, m)
    M = mean.reshape(-1, 1)       # shape (n, 1)
    S = sigma.reshape(-1, 1)      # shape (n, 1)

    if mode == "percent":
        # Beta fit is only valid when variance < mean*(1-mean)
        var = sigma ** 2
        feasible = np.isfinite(mean) & np.isfinite(var) & (var > 0) & (var < mean * (1.0 - mean))

        # Method-of-moments Beta parameters:
        # nu = (mean*(1-mean)/var) - 1
        # alpha = mean*nu, beta = (1-mean)*nu
        nu = np.where(feasible, (mean * (1.0 - mean)) / var - 1.0, np.nan)
        alpha_param = mean * nu
        beta_param = (1.0 - mean) * nu

        # Beta pdf for feasible rows, otherwise Normal pdf fallback
        Y_beta = beta.pdf(X, alpha_param.reshape(-1, 1), beta_param.reshape(-1, 1))
        Y_norm = norm.pdf(X, M, S)
        Y = np.where(feasible.reshape(-1, 1), Y_beta, Y_norm)
    else:
        # Rank: use Normal pdf
        Y = norm.pdf(X, M, S)

    # --- Build long-form plotting dataframe WITHOUT iterrows ---
    names = sub["celebrity_name"].astype(str).to_numpy()
    is_elim = sub["is_eliminated_this_week"].astype(bool).to_numpy()
    n, m = len(names), len(x_axis)

    plot_df = pd.DataFrame(
        {
            "Name": np.repeat(names, m),
            "X_Value": np.tile(x_axis, n),
            "Density": Y.reshape(-1),
            "Is_Eliminated": np.repeat(is_elim, m),
        }
    )

    # --- Plot (publication-ready) ---
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    pal = {nm: ("#e74c3c" if el else "#3498db") for nm, el in zip(names, is_elim)}

    g = sns.FacetGrid(plot_df, row="Name", hue="Name", aspect=9, height=0.6, palette=pal)
    g.map(plt.fill_between, "X_Value", "Density", alpha=0.7)
    g.map(plt.plot, "X_Value", "Density", clip_on=False, color="white", lw=1.5)
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, 0.2, label, fontweight="bold", color="black", ha="left", va="center", transform=ax.transAxes)

    g.map(label, "X_Value")
    g.figure.subplots_adjust(hspace=-0.5)
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)

    if mode == "percent":
        plt.xlabel("Estimated Fan Vote Share (0.0 - 1.0)")
        plt.xlim(0.0, 0.45)
    else:
        plt.xlabel("Estimated Fan Rank (1 = Best)")
        plt.xlim(0.5, len(sub) + 1.5)

    plt.suptitle(f"Figure 3: Uncertainty in Fan Preferences (Season {args.season} Week {args.week})", y=0.98)

    out_name = out_dir / f"Figure_Uncertainty_{mode.upper()}_S{args.season}_W{args.week}.png"
    plt.savefig(out_name, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {out_name}")


if __name__ == "__main__":
    main()
