#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Confound variable control analysis.

Compare Ridge regression performance between:
  - Model A: 19 features only (Classical 10 + Novel 9)
  - Model B: 19 features + gender + age (21 explanatory variables)

Both models use Ridge (α=100) + 5-fold subject-wise CV + Permutation test.

Usage:
    python scripts/analysis/confound_analysis.py \
        --xy_parquet artifacts/analysis/datasets/cejc_home2_hq1_XY_Conly_sonnet.parquet \
        --y_col Y_C \
        --metadata_tsv artifacts/analysis/cejc_speaker_metadata.tsv \
        --out_dir artifacts/analysis/results/confound
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

# ── Feature sets (same as baseline_vs_extended.py) ───────────────────
CLASSICAL_FEATURES = [
    "PG_speech_ratio",
    "PG_pause_mean",
    "PG_pause_p50",
    "PG_pause_p90",
    "PG_resp_gap_mean",
    "PG_resp_gap_p50",
    "PG_resp_gap_p90",
    "PG_overlap_rate",
    "FILL_has_any",
    "FILL_rate_per_100chars",
]  # 10

NOVEL_FEATURES = [
    "IX_oirmarker_rate",
    "IX_oirmarker_after_question_rate",
    "IX_yesno_rate",
    "IX_yesno_after_question_rate",
    "IX_lex_overlap_mean",
    "RESP_NE_AIZUCHI_RATE",
    "RESP_NE_ENTROPY",
    "RESP_YO_ENTROPY",
    "PG_pause_variability",
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


# ── Metadata loading and confound preparation ────────────────────────
def load_and_join_metadata(
    df: pd.DataFrame, metadata_tsv: str
) -> tuple[pd.DataFrame, list[str]]:
    """Join XY data with metadata and prepare confound columns.

    Returns:
        (merged_df, confound_cols): merged DataFrame and list of confound
        column names that were successfully added.
    """
    meta_path = Path(metadata_tsv)
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Metadata TSV not found: {metadata_tsv}"
        )

    meta = pd.read_csv(meta_path, sep="\t")

    # Validate join keys
    for key in ("conversation_id", "speaker_id"):
        if key not in meta.columns:
            raise KeyError(f"Metadata TSV missing required column: {key}")

    # Inner join on conversation_id × speaker_id
    n_before = len(df)
    merged = df.merge(
        meta[["conversation_id", "speaker_id", "gender", "age"]].drop_duplicates(),
        on=["conversation_id", "speaker_id"],
        how="inner",
        suffixes=("", "_meta"),
    )
    n_after = len(merged)
    if n_after < n_before:
        n_excluded = n_before - n_after
        warnings.warn(
            f"Metadata join: {n_excluded}/{n_before} rows excluded "
            f"(no matching conversation_id × speaker_id in metadata)"
        )

    # Prepare confound columns
    confound_cols: list[str] = []

    # Gender → dummy variable (M=0, F=1)
    gender_col = "gender" if "gender" in merged.columns else None
    if gender_col is None or merged[gender_col].isna().all():
        warnings.warn(
            "Gender column missing or all NaN in metadata. "
            "Skipping gender as confound variable."
        )
    else:
        merged["confound_gender"] = (
            merged[gender_col].map({"M": 0, "F": 1}).astype(float)
        )
        n_gender_na = merged["confound_gender"].isna().sum()
        if n_gender_na > 0:
            warnings.warn(
                f"Gender: {n_gender_na} rows have unknown gender values "
                f"(not M/F). These will be imputed during CV."
            )
        confound_cols.append("confound_gender")

    # Age → numeric
    age_col = "age" if "age" in merged.columns else None
    if age_col is None or merged[age_col].isna().all():
        warnings.warn(
            "Age column missing or all NaN in metadata. "
            "Skipping age as confound variable."
        )
    else:
        merged["confound_age"] = pd.to_numeric(
            merged[age_col], errors="coerce"
        )
        n_age_na = merged["confound_age"].isna().sum()
        if n_age_na > 0:
            warnings.warn(
                f"Age: {n_age_na} rows have non-numeric age values. "
                f"These will be imputed during CV."
            )
        confound_cols.append("confound_age")

    if not confound_cols:
        warnings.warn(
            "No confound variables available (both gender and age missing). "
            "Only features-only model will be run."
        )

    return merged, confound_cols


# ── Main ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Confound variable control analysis: "
        "19 features only vs 19 features + gender + age"
    )
    ap.add_argument("--xy_parquet", required=True, help="XY dataset parquet path")
    ap.add_argument("--y_col", required=True, help="Target column name (e.g. Y_C)")
    ap.add_argument(
        "--metadata_tsv",
        default="artifacts/analysis/cejc_speaker_metadata.tsv",
        help="Path to speaker metadata TSV",
    )
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

    # ── Load XY data ─────────────────────────────────────────────────
    df = pd.read_parquet(args.xy_parquet)
    df = df.replace([np.inf, -np.inf], np.nan)

    if args.y_col not in df.columns:
        raise SystemExit(f"y_col not found: {args.y_col}")

    # ── Load and join metadata ───────────────────────────────────────
    merged, confound_cols = load_and_join_metadata(df, args.metadata_tsv)

    y = merged[args.y_col].astype(float).to_numpy()

    # Drop rows with NaN in y
    ok = ~np.isnan(y)
    if ok.sum() == 0:
        warnings.warn(f"All y values are NaN for {args.y_col}. Skipping.")
        return
    df_clean = merged.loc[ok].copy()
    y = y[ok]

    # ── Validate feature columns ─────────────────────────────────────
    missing_features = [c for c in ALL_FEATURES if c not in df_clean.columns]
    if missing_features:
        raise KeyError(f"Missing feature columns: {missing_features}")

    # ── Model A: 19 features only ────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Model A: {len(ALL_FEATURES)} features only")
    print(f"  y_col={args.y_col}, N={len(y)}")
    print(f"{'='*60}")

    X_features = (
        df_clean[ALL_FEATURES]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )
    r_features, p_features = run_permutation_test(
        X_features, y, args.cv_folds, args.seed, args.alpha, args.n_perm
    )
    print(f"  r_features_only = {r_features:.3f}, p = {p_features:.4f}")

    # ── Model B: 19 features + confounds ─────────────────────────────
    r_with_confounds = float("nan")
    p_with_confounds = float("nan")
    delta_r = float("nan")

    if confound_cols:
        cols_with_confounds = ALL_FEATURES + confound_cols
        print(f"\n{'='*60}")
        print(
            f"  Model B: {len(ALL_FEATURES)} features + "
            f"{len(confound_cols)} confounds ({', '.join(confound_cols)})"
        )
        print(f"  Total explanatory variables: {len(cols_with_confounds)}")
        print(f"{'='*60}")

        X_with_confounds = (
            df_clean[cols_with_confounds]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=float)
        )
        r_with_confounds, p_with_confounds = run_permutation_test(
            X_with_confounds, y, args.cv_folds, args.seed, args.alpha, args.n_perm
        )
        delta_r = r_with_confounds - r_features
        print(f"  r_with_confounds = {r_with_confounds:.3f}, p = {p_with_confounds:.4f}")
        print(f"  Δr = {delta_r:+.3f}")
    else:
        print("\n  [SKIP] Model B: no confound variables available.")

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
    out_path = out_dir / f"confound_{trait}_{teacher}.tsv"

    result = pd.DataFrame(
        [
            {
                "trait": trait,
                "teacher": teacher,
                "r_features_only": round(r_features, 4),
                "p_features_only": round(p_features, 4),
                "r_with_confounds": (
                    round(r_with_confounds, 4)
                    if not np.isnan(r_with_confounds)
                    else float("nan")
                ),
                "p_with_confounds": (
                    round(p_with_confounds, 4)
                    if not np.isnan(p_with_confounds)
                    else float("nan")
                ),
                "delta_r": (
                    round(delta_r, 4)
                    if not np.isnan(delta_r)
                    else float("nan")
                ),
            }
        ]
    )
    result.to_csv(out_path, sep="\t", index=False)
    print(f"\nWrote {out_path}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
