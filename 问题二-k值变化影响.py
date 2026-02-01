import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

# --- LaTeX Academic Style Config ---
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.titlesize": 16,
    "axes.labelsize": 12,
    "figure.figsize": (12, 8),
    "figure.dpi": 300
})


def map_rank_to_percentage(ranks, n_active, k):
    """
    Core Mapping Formula (Exponential Decay):
    P_i = exp(-k * (Rank_i - 1) / (n_active - 1)) / Sum(exp(...))
    """
    if n_active <= 1: return np.array([100.0])
    scores = np.exp(-k * (ranks - 1) / (n_active - 1))
    return (scores / scores.sum()) * 100


def run_comprehensive_analysis(csv_path="latent_estimates.csv"):
    """
    A unified execution of theoretical sweep and real-data verification.
    """
    # ---------------------------------------------------------
    # PART 1: REAL-DATA EMPIRICAL SWEEP (读取你的数据)
    # ---------------------------------------------------------
    if Path(csv_path).exists():
        print(f"Loading real data from {csv_path}...")
        df = pd.read_csv(csv_path)

        # 选取一个典型的复杂周次进行敏感性扫描（例如最近的赛季）
        target_s, target_w = 34, 9
        subset = df[(df['season'] == target_s) & (df['week'] == target_w)]

        if not subset.empty:
            n_active = subset['n_active'].iloc[0]
            names = subset['celebrity_name'].values
            latent_ranks = subset['fan_vote_mean'].values

            k_range = np.linspace(0.5, 4.0, 100)
            records = []
            for k in k_range:
                shares = map_rank_to_percentage(latent_ranks, n_active, k)
                for name, share in zip(names, shares):
                    records.append({"k": k, "Share (%)": share, "Candidate": name})

            plt.figure(figsize=(10, 6))
            sns.lineplot(data=pd.DataFrame(records), x="k", y="Share (%)", hue="Candidate", linewidth=2.5)
            plt.axvline(x=1.2, color='red', linestyle='--', label='Operational Baseline (k=1.2)')
            plt.title(
                f"Empirical Sensitivity Analysis: Season {target_s} Week {target_w}\n(Impact of k on Actual Contestant Shares)",
                fontweight='bold')
            plt.xlabel("Sensitivity Parameter $k$ (Contrast Control)")
            plt.ylabel("Projected Fan Vote Share (%)")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
            plt.tight_layout()
            plt.savefig("figure_empirical_k_sweep.png")
            plt.show()
    else:
        print(f"File {csv_path} not found. Please run the reverse engineering script first.")

    # ---------------------------------------------------------
    # PART 2: THEORETICAL PRECISION HEATMAP (无需外部数据)
    # ---------------------------------------------------------
    print("Generating High-Resolution Theoretical Heatmap...")
    k_range_h = np.linspace(0.1, 5.0, 25)
    ranks_h = np.arange(1, 9)  # Assume 8 players standard

    results = [map_rank_to_percentage(ranks_h, 8, k) for k in k_range_h]
    df_hm = pd.DataFrame(results, index=[f"{x:.1f}" for x in k_range_h],
                         columns=[f"Rank {int(r)}" for r in ranks_h])

    plt.figure(figsize=(14, 7))
    sns.heatmap(df_hm.T, cmap="magma", annot=True, fmt=".1f", linewidths=.3,
                cbar_kws={'label': 'Projected Vote Share (%)'})

    plt.title("Theoretic Heatmap: Competitive Gradient Under Parameter Sweep", size=18, pad=20, fontweight='bold')
    plt.xlabel("Sensitivity Coefficient ($k$)")
    plt.ylabel("Latent Rank Position")
    plt.tight_layout()
    plt.savefig("figure_theoretical_heatmap.png")
    plt.show()


if __name__ == "__main__":
    # 执行分析（如果 CSV 在当前目录或 outputs 目录下）
    possible_paths = ["outputs/latent_estimates.csv"]
    final_path = next((p for p in possible_paths if Path(p).exists()), "latent_estimates.csv")

    run_comprehensive_analysis(final_path)