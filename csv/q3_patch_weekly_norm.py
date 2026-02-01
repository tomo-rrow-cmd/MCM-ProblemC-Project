#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent

def f(name: str) -> Path:
    p = BASE / name
    if not p.exists():
        raise FileNotFoundError(f"Missing file (same folder): {name}")
    return p

def hard_fail(msg: str):
    raise RuntimeError(f"[SCHEMA_HARD_FAIL] {msg}")

def main():
    w = pd.read_csv(f("q3_weekly_norm.csv"))

    # must have these to compute invalid flag properly
    need = ["fan_se", "latent_missing_flag"]
    miss = [c for c in need if c not in w.columns]
    if miss:
        hard_fail(f"q3_weekly_norm.csv missing required columns for patch: {miss}")

    # compute fan_weight if missing
    if "fan_weight" not in w.columns:
        w["fan_weight"] = 1.0 / (pd.to_numeric(w["fan_se"], errors="coerce") ** 2)

    fan_se = pd.to_numeric(w["fan_se"], errors="coerce")
    fan_weight = pd.to_numeric(w["fan_weight"], errors="coerce")

    invalid = fan_se.isna() | (fan_se <= 0) | (~np.isfinite(fan_weight)) | (fan_weight <= 0)
    # 只要 missing_flag=1 的行，本来 fan 模型就会 drop，这里也标 invalid=1 更直观
    if "latent_missing_flag" in w.columns:
        invalid = invalid | (pd.to_numeric(w["latent_missing_flag"], errors="coerce").fillna(1).astype(int) == 1)

    w["fan_invalid_flag"] = invalid.astype(int)

    w.to_csv(BASE / "q3_weekly_norm_fixed.csv", index=False)
    print("[OK] wrote: q3_weekly_norm_fixed.csv")
    print("Tip: use this file for plotting/modeling next.")

if __name__ == "__main__":
    main()
