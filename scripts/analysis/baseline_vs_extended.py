#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Baseline vs Extended model comparison.

Compare Ridge regression performance between:
  - Baseline: Classical features only (PG 8 + FILL 2 = 10 features)
  - Extended: All 19 features (Classical 10 + Novel 9)

Both models use Ridge (α=100) + 5-fold subject-wise CV + Permutation test.

Usage:
    python scripts/analysis/baseline_vs_extended.py \
        --xy_parquet artifacts/analysis/datasets/cejc_home2_hq1_XY_Conly_sonnet.parquet \
        --y_col Y_C \
        --out_dir artifacts/analysis/results/baseline_vs_extended
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# ── Feature sets ─────────────────────────────────────────────────────
CLASSICAL_FEATURES = [
    "PG_speech_ratio",
    "PG_pause_mean",
    "PG_pause_p50",
    "PG_pause_p90",
    "PG_resp_gap_mean",
    "PG_resp_gap_p50",
    "PG_resp_gap_p90",
    "PG_overlap_rate",       # 追加（CTRL→Classical復帰）
    "FILL_has_any",
    "FILL_rate_per_100chars",
]  # 10

NOVEL_FEATURES = [
    "IX_oirmarker_rate",
    "IX_oirmarker_after_question_rate",
    "IX_yesno_rate",
    "IX_yesno_after_question_rate",
    "IX_lex_overlap_mean",
    # IX_topic_drift_mean 削除（IX_lex_overlap_meanとの完全共線性）
    "RESP_NE_AIZUCHI_RATE",
    "RESP_NE_ENTROPY",
    "RESP_YO_ENTROPY",
    "PG_pause_variability",  # 追加（新規特徴量）
]  # 9

ALL_FEATURES = CLASSICAL_FEATURES + NOVEL_FEATURES  # 19


# ── Core functions (same logic as permutation_test_ridge_fixedalpha.py) ──
def pearsonr(a, b) -> float:
    """Pearson correlation coefficient (NaN-safe)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum()) * np.sqrt((b * b).sum())
    return float((a * b).sum() / den) if den != 0 else float("nan")


def cv_ridge_r(X, y, folds, seed, alpha):
    """5-fold CV Ridge regression, return mean Pearson r."""
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    rs = []
    for tr, te in kf.split(X):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]

        imp = SimpleImputer(strategy="median")
        Xtr = imp.fit_transform(Xtr)
        Xte = imp.transform(Xte)

        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xte = sc.transform(Xte)

        m = Ridge(alpha=alpha, random_state=seed)
        m.fit(Xtr, ytr)
        yh = m.predict(Xte)
        rs.append(pearsonr(yte, yh))
    return float(np.mean(rs))


def run_permutation_test(X, y, folds, seed, alpha, n_perm):
    """Run Ridge CV + permutation test. Return (r_obs, p_value)."""
    r_obs = cv_ridge_r(X, y, folds, seed, alpha)

    rng = np.random.default_rng(seed)
    r_perm = np.empty(n_perm, float)
    for i in range(n_perm):
        yp = rng.permutation(y)
        r_perm[i] = cv_ridge_r(X, yp, folds, seed, alpha)
        if (i + 1) % 500 == 0:
            print(f"    perm {i + 1}/{n_perm} done")

    p_val = (np.sum(np.abs(r_perm) >= abs(r_obs)) + 1.0) / (n_perm + 1.0)
    return r_obs, p_val


# ── Main ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Baseline (Classical 10) vs Extended (all 19) Ridge comparison"
    )
    ap.add_argument("--xy_parquet", required=True, help="XY dataset parquet path")
    ap.add_argument("--y_col", required=True, help="Target column name (e.g. Y_C)")
    ap.add_argument(
        "--exclude_cols",
        default="",
        help="Extra columns to exclude (comma-separated)",
    )
    ap.add_argument("--alpha", type=float, default=100.0, help="Ridge alpha")
    ap.add_argument("--cv_folds", type=int, default=5, help="CV folds")
    ap.add_argument("--n_perm", type=int, default=5000, help="Permutation rounds")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    args = ap.parse_args()

    # ── Load data ────────────────────────────────────────────────────
    df = pd.read_parquet(args.xy_parquet)
    df = df.replace([np.inf, -np.inf], np.nan)

    if args.y_col not in df.columns:
        raise SystemExit(f"y_col not found: {args.y_col}")

    y = df[args.y_col].astype(float).to_numpy()

    # Drop rows with NaN in y
    ok = ~np.isnan(y)
    if ok.sum() == 0:
        warnings.warn(f"All y values are NaN for {args.y_col}. Skipping.")
        return
    df_clean = df.loc[ok].copy()
    y = y[ok]

    # ── Validate Classical features ──────────────────────────────────
    missing_classical = [c for c in CLASSICAL_FEATURES if c not in df_clean.columns]
    if missing_classical:
        raise KeyError(
            f"Missing Classical feature columns: {missing_classical}"
        )

    # ── Validate Novel features (warning + partial execution) ────────
    missing_novel = [c for c in NOVEL_FEATURES if c not in df_clean.columns]
    run_extended = True
    if missing_novel:
        warnings.warn(
            f"Missing Novel feature columns: {missing_novel}. "
            "Extended model will be skipped; only baseline will run."
        )
        run_extended = False

    # ── Baseline: Classical features only ────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Baseline: {len(CLASSICAL_FEATURES)} Classical features")
    print(f"  y_col={args.y_col}, N={len(y)}")
    print(f"{'='*60}")

    X_baseline = (
        df_clean[CLASSICAL_FEATURES]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )
    r_baseline, p_baseline = run_permutation_test(
        X_baseline, y, args.cv_folds, args.seed, args.alpha, args.n_perm
    )
    print(f"  r_baseline = {r_baseline:.3f}, p = {p_baseline:.4f}")

    # ── Extended: All 18 features ────────────────────────────────────
    r_extended = float("nan")
    p_extended = float("nan")
    delta_r = float("nan")

    if run_extended:
        print(f"\n{'='*60}")
        print(f"  Extended: {len(ALL_FEATURES)} features (Classical + Novel)")
        print(f"{'='*60}")

        X_extended = (
            df_clean[ALL_FEATURES]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=float)
        )
        r_extended, p_extended = run_permutation_test(
            X_extended, y, args.cv_folds, args.seed, args.alpha, args.n_perm
        )
        delta_r = r_extended - r_baseline
        print(f"  r_extended = {r_extended:.3f}, p = {p_extended:.4f}")
        print(f"  Δr = {delta_r:+.3f}")

    # ── Derive trait and teacher from parquet filename ────────────────
    stem = Path(args.xy_parquet).stem  # e.g. cejc_home2_hq1_XY_Conly_sonnet
    trait = "unknown"
    teacher = "unknown"
    # Try to parse trait/teacher from filename pattern *_XY_{trait}only_{teacher}
    if "_XY_" in stem:
        after_xy = stem.split("_XY_")[1]  # e.g. Conly_sonnet
        parts = after_xy.split("_", 1)
        if len(parts) == 2:
            trait_part, teacher = parts
            trait = trait_part.replace("only", "")  # C

    # ── Write output TSV ─────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"baseline_vs_extended_{trait}_{teacher}.tsv"

    result = pd.DataFrame(
        [
            {
                "trait": trait,
                "teacher": teacher,
                "r_baseline": round(r_baseline, 4),
                "p_baseline": round(p_baseline, 4),
                "r_extended": round(r_extended, 4) if not np.isnan(r_extended) else float("nan"),
                "p_extended": round(p_extended, 4) if not np.isnan(p_extended) else float("nan"),
                "delta_r": round(delta_r, 4) if not np.isnan(delta_r) else float("nan"),
            }
        ]
    )
    result.to_csv(out_path, sep="\t", index=False)
    print(f"\nWrote {out_path}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
