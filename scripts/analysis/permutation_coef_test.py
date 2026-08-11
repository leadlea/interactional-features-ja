#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Permutation coefficient test for Ridge regression.

For each of the 19 features, test whether the observed Ridge regression
coefficient is significantly different from zero using a permutation test.

Unlike the CV-based permutation test (permutation_test_ridge_fixedalpha.py),
this script fits Ridge on the FULL dataset to obtain β_obs, then permutes y
n_perm times, each time fitting Ridge on the full data and recording
coefficients.  The p-value for each feature is:

    p = (count(|β_perm[i]| >= |β_obs[i]|) + 1) / (n_perm + 1)

Usage:
    python scripts/analysis/permutation_coef_test.py \
        --xy_parquet artifacts/analysis/datasets/cejc_home2_hq1_XY_Conly_sonnet.parquet \
        --y_col Y_C \
        --out_dir artifacts/analysis/results/permutation_coef
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
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


# ── Core testable function ───────────────────────────────────────────
def run_permutation_coef_test(
    df: pd.DataFrame,
    y_col: str,
    feature_cols: list[str],
    alpha: float = 100.0,
    n_perm: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """Run permutation test on Ridge regression coefficients.

    Fits Ridge on the full dataset to obtain observed coefficients (β_obs),
    then permutes y *n_perm* times, each time fitting Ridge on the full data
    and recording coefficients.  For each feature the p-value is:

        p = (count(|β_perm[i]| >= |β_obs[i]|) + 1) / (n_perm + 1)

    Args:
        df: DataFrame containing *feature_cols* and *y_col*.
        y_col: Target column name.
        feature_cols: List of feature column names.
        alpha: Ridge regularisation parameter.
        n_perm: Number of permutation rounds.
        seed: Random seed.

    Returns:
        DataFrame with columns: feature, coef_obs, p_value, significant.
        Empty DataFrame if all y values are NaN.

    Raises:
        KeyError: If any *feature_cols* are missing from *df*.
    """
    # ── Validate feature columns ─────────────────────────────────────
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")

    # ── Prepare y ────────────────────────────────────────────────────
    y = df[y_col].astype(float).to_numpy()
    ok = ~np.isnan(y)
    if ok.sum() == 0:
        warnings.warn(f"All y values are NaN for {y_col}. Skipping.")
        return pd.DataFrame()

    X_raw = (
        df.loc[ok, feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )
    y = y[ok]

    # ── Impute + Scale (full dataset) ────────────────────────────────
    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(X_raw)

    sc = StandardScaler()
    X = sc.fit_transform(X)

    # ── Observed coefficients ────────────────────────────────────────
    model = Ridge(alpha=alpha, random_state=seed)
    model.fit(X, y)
    beta_obs = model.coef_  # shape (n_features,)

    # ── Permutation loop ─────────────────────────────────────────────
    n_feat = len(feature_cols)
    count_ge = np.zeros(n_feat, dtype=int)

    rng = np.random.default_rng(seed)
    for i in range(n_perm):
        yp = rng.permutation(y)
        m = Ridge(alpha=alpha, random_state=seed)
        m.fit(X, yp)
        count_ge += (np.abs(m.coef_) >= np.abs(beta_obs)).astype(int)
        if (i + 1) % 500 == 0:
            print(f"    perm {i + 1}/{n_perm} done")

    p_values = (count_ge + 1.0) / (n_perm + 1.0)

    # ── Build result DataFrame ───────────────────────────────────────
    result = pd.DataFrame({
        "feature": feature_cols,
        "coef_obs": np.round(beta_obs, 6),
        "p_value": np.round(p_values, 4),
        "significant": p_values < 0.05,
    })
    return result


# ── Main ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Permutation test on Ridge regression coefficients "
        "(19 features, full-data fit)"
    )
    ap.add_argument("--xy_parquet", required=True, help="XY dataset parquet path")
    ap.add_argument("--y_col", required=True, help="Target column name (e.g. Y_C)")
    ap.add_argument("--alpha", type=float, default=100.0, help="Ridge alpha")
    ap.add_argument("--cv_folds", type=int, default=5, help="CV folds (unused, kept for CLI compat)")
    ap.add_argument("--n_perm", type=int, default=5000, help="Permutation rounds")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    args = ap.parse_args()

    # ── Load data ────────────────────────────────────────────────────
    df = pd.read_parquet(args.xy_parquet)
    df = df.replace([np.inf, -np.inf], np.nan)

    if args.y_col not in df.columns:
        raise SystemExit(f"y_col not found: {args.y_col}")

    # ── Validate feature columns ─────────────────────────────────────
    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")

    # ── Run permutation coefficient test ─────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Permutation coefficient test")
    print(f"  y_col={args.y_col}, N features={len(ALL_FEATURES)}")
    print(f"  alpha={args.alpha}, n_perm={args.n_perm}, seed={args.seed}")
    print(f"{'='*60}")

    result = run_permutation_coef_test(
        df=df,
        y_col=args.y_col,
        feature_cols=ALL_FEATURES,
        alpha=args.alpha,
        n_perm=args.n_perm,
        seed=args.seed,
    )

    if result.empty:
        print("No results produced (all y values NaN?).")
        return

    # ── Derive trait and teacher from parquet filename ────────────────
    stem = Path(args.xy_parquet).stem
    trait = "unknown"
    teacher = "unknown"
    if "_XY_" in stem:
        after_xy = stem.split("_XY_")[1]
        parts = after_xy.split("_", 1)
        if len(parts) == 2:
            trait_part, teacher = parts
            trait = trait_part.replace("only", "")

    # ── Write output TSV ─────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"permutation_coef_{trait}_{teacher}.tsv"

    result.to_csv(out_path, sep="\t", index=False)
    print(f"\nWrote {out_path}")
    print(result.to_string(index=False))

    n_sig = result["significant"].sum()
    print(f"\n  Significant features (p < 0.05): {n_sig}/{len(ALL_FEATURES)}")


if __name__ == "__main__":
    main()
