#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)

KEY = ["season", "week", "celebrity_name"]


def hard_fail(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(f"[HARD_FAIL] {msg}")


BASE_DIR = Path(__file__).resolve().parent  # 脚本所在目录（你现在是 T3/csv）


def find_input(filename: str) -> Path:
    """
    绝不依赖运行时 working directory。
    搜索顺序：
    1) 脚本目录 / filename
    2) 脚本目录 / csv / filename
    3) 当前工作目录 / filename
    4) 当前工作目录 / csv / filename
    5) 脚本目录 / mcm_data_v7_fusion / filename
    6) 当前工作目录 / mcm_data_v7_fusion / filename
    """
    candidates = [
        BASE_DIR / filename,
        BASE_DIR / "csv" / filename,
        Path(filename),
        Path("csv") / filename,
        BASE_DIR / "mcm_data_v7_fusion" / filename,
        Path("mcm_data_v7_fusion") / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Missing input: {filename}\nSearched:\n- " +
        "\n- ".join([str(p).replace("\\", "/") for p in candidates])
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcm", type=str, default="mcm_modeling_dataset.csv")
    ap.add_argument("--latent", type=str, default="latent_estimates.csv")
    ap.add_argument("--out_dir", type=str, default=".")
    args = ap.parse_args()

    mcm_path = find_input(args.mcm)

    # prefer latent_estimates_filled.csv if exists
    latent_filled = (BASE_DIR / "latent_estimates_filled.csv")
    latent_filled2 = (BASE_DIR / "csv" / "latent_estimates_filled.csv")
    if latent_filled.exists():
        latent_path = latent_filled
    elif latent_filled2.exists():
        latent_path = latent_filled2
    else:
        latent_path = find_input(args.latent)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mcm = pd.read_csv(mcm_path)
    latent = pd.read_csv(latent_path)

    # minimal key checks
    for c in KEY:
        hard_fail(c in mcm.columns, f"{mcm_path.name} missing key: {c}")
        hard_fail(c in latent.columns, f"{latent_path.name} missing key: {c}")

    # latent unique key
    dup = latent.duplicated(KEY, keep=False)
    hard_fail(not dup.any(), f"{latent_path.name} has duplicate keys on {KEY}; dup_rows={int(dup.sum())}")

    # merge
    df = mcm.merge(latent, on=KEY, how="left", suffixes=("", "_latent"))
    df["latent_missing_flag"] = df["fan_vote_mean"].isna().astype(int)

    # n_active may be NA if no latent match -> keep NA, we just audit here
    audit = {
        "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "inputs": {"mcm": mcm_path.name, "latent": latent_path.name},
        "counts": {
            "rows_mcm": int(len(mcm)),
            "rows_latent": int(len(latent)),
            "rows_merged": int(len(df)),
            "latent_missing_rows": int(df["latent_missing_flag"].sum()),
        },
    }
    (out_dir / "step1_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    # just write the merge audit table (for debugging next steps)
    df[KEY + ["latent_missing_flag"]].to_csv(out_dir / "q3_merge_audit.csv", index=False)

    print("[OK] Found inputs:")
    print(" -", mcm_path)
    print(" -", latent_path)
    print("[OK] Wrote:")
    print(" - step1_audit.json")
    print(" - q3_merge_audit.csv")


if __name__ == "__main__":
    main()

