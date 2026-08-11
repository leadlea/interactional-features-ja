#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bootstrap variance analysis for Ridge regression coefficients.

For each of the 19 features (Classical 10 + Novel 9), compute the mean,
SD, and 95% CI of Ridge regression coefficients across 500 bootstrap
resamples (sampling N rows WITH replacement from the dataset).

For each bootstrap sample, fit Ridge(alpha=100) on the FULL bootstrap
sample (no CV needed), using SimpleImputer(median) + StandardScaler
before Ridge fit.  Record the 19 coefficients for each iteration.

After all iterations: compute mean, SD, 2.5th percentile, 97.5th
percentile for each feature.  ci_excludes_zero = True if ci_lower > 0
OR ci_upper < 0.

Usage:
    python scripts/analysis/bootstrap_variance.py \
        --xy_parquet artifacts/analysis/datasets/cejc_home2_hq1_XY_Conly_sonnet.parquet \
        --y_col Y_C \
        --out_dir artifacts/analysis/results/bootstrap_variance
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
def run_bootstrap_variance(
    df: pd.DataFrame,
    y_col: str,
    feature_cols: list[str],
    alpha: float = 100.0,
    n_boot: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Run bootstrap variance analysis on Ridge regression coefficients.

    For each bootstrap iteration, sample N rows WITH replacement from the
    dataset, fit Ridge(alpha) on the full bootstrap sample (with
    SimpleImputer + StandardScaler), and record the 19 coefficients.

    After all iterations: compute mean, SD, 2.5th/97.5th percentiles
    (95% CI) for each feature.

    Args:
        df: DataFrame containing *feature_cols* and *y_col*.
        y_col: Target column name.
        feature_cols: List of feature column names.
        alpha: Ridge regularisation parameter.
        n_boot: Number of bootstrap iterations.
        seed: Random seed.

    Returns:
        DataFrame with columns: feature, coef_mean, coef_sd, ci_lower,
        ci_upper, ci_excludes_zero.
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
    n_samples = len(y)
    n_feat = len(feature_cols)

    # ── Bootstrap loop ───────────────────────────────────────────────
    coef_matrix: list[np.ndarray] = []
    n_failed = 0

    rng = np.random.default_rng(seed)
    for i in range(n_boot):
        try:
            # Sample WITH replacement
            idx = rng.integers(0, n_samples, size=n_samples)
            X_boot = X_raw[idx]
            y_boot = y[idx]

            # Impute + Scale
            imp = SimpleImputer(strategy="median")
            X_boot = imp.fit_transform(X_boot)

            sc = StandardScaler()
            X_boot = sc.fit_transform(X_boot)

            # Fit Ridge on full bootstrap sample
            model = Ridge(alpha=alpha, random_state=seed)
            model.fit(X_boot, y_boot)
            coef_matrix.append(model.coef_.copy())
        except Exception as exc:
            n_failed += 1
            if n_failed <= 5:
                warnings.warn(
                    f"Bootstrap iteration {i + 1} failed: {exc}"
                )
            continue

        if (i + 1) % 100 == 0:
            print(f"    boot {i + 1}/{n_boot} done")

    n_valid = len(coef_matrix)
    if n_failed > 0:
        print(
            f"  Bootstrap: {n_valid} valid / {n_boot} total "
            f"({n_failed} failed)"
        )

    if n_valid < 100:
        warnings.warn(
            f"Only {n_valid} valid bootstrap iterations (< 100). "
            "Results may have low reliability."
        )

    if n_valid == 0:
        warnings.warn("No valid bootstrap iterations. Returning empty.")
        return pd.DataFrame()

    # ── Compute statistics ───────────────────────────────────────────
    coefs = np.array(coef_matrix)  # shape (n_valid, n_feat)

    coef_mean = np.mean(coefs, axis=0)
    coef_sd = np.std(coefs, axis=0, ddof=1) if n_valid > 1 else np.zeros(n_feat)
    ci_lower = np.percentile(coefs, 2.5, axis=0)
    ci_upper = np.percentile(coefs, 97.5, axis=0)

    # ci_excludes_zero: True if ci_lower > 0 OR ci_upper < 0
    ci_excludes_zero = (ci_lower > 0) | (ci_upper < 0)

    # ── Build result DataFrame ───────────────────────────────────────
    result = pd.DataFrame({
        "feature": feature_cols,
        "coef_mean": np.round(coef_mean, 6),
        "coef_sd": np.round(coef_sd, 6),
        "ci_lower": np.round(ci_lower, 6),
        "ci_upper": np.round(ci_upper, 6),
        "ci_excludes_zero": ci_excludes_zero,
    })
    return result


# ── Main ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Bootstrap variance analysis on Ridge regression "
        "coefficients (19 features, full-data fit per resample)"
    )
    ap.add_argument("--xy_parquet", required=True, help="XY dataset parquet path")
    ap.add_argument("--y_col", required=True, help="Target column name (e.g. Y_C)")
    ap.add_argument("--alpha", type=float, default=100.0, help="Ridge alpha")
    ap.add_argument("--n_boot", type=int, default=500, help="Bootstrap iterations")
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

    # ── Run bootstrap variance analysis ──────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Bootstrap variance analysis")
    print(f"  y_col={args.y_col}, N features={len(ALL_FEATURES)}")
    print(f"  alpha={args.alpha}, n_boot={args.n_boot}, seed={args.seed}")
    print(f"{'='*60}")

    result = run_bootstrap_variance(
        df=df,
        y_col=args.y_col,
        feature_cols=ALL_FEATURES,
        alpha=args.alpha,
        n_boot=args.n_boot,
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
    out_path = out_dir / f"bootstrap_variance_{trait}_{teacher}.tsv"

    result.to_csv(out_path, sep="\t", index=False)
    print(f"\nWrote {out_path}")
    print(result.to_string(index=False))

    n_excl = result["ci_excludes_zero"].sum()
    print(
        f"\n  Features with CI excluding zero: "
        f"{n_excl}/{len(ALL_FEATURES)}"
    )


if __name__ == "__main__":
    main()
