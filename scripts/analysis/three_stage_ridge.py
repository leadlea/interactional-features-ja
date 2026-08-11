#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Three-stage Ridge regression comparison.

Compare Ridge regression performance across three stages:
  - Stage 1: Demographics only (gender + age = 2 variables)
  - Stage 2: Demographics + Classical features (2 + 10 = 12 variables)
  - Stage 3: Demographics + Classical + Novel features (2 + 10 + 9 = 21 variables)

Each stage uses Ridge (α=100) + 5-fold subject-wise CV + Permutation test.
Δr between adjacent stages is computed.

Reuses cv_ridge_r / run_permutation_test logic from confound_analysis.py
and baseline_vs_extended.py.

Usage:
    python scripts/analysis/three_stage_ridge.py \
        --xy_parquet artifacts/analysis/datasets/cejc_home2_hq1_XY_Conly_sonnet.parquet \
        --y_col Y_C \
        --metadata_tsv artifacts/analysis/cejc_speaker_metadata.tsv \
        --out_dir artifacts/analysis/results/three_stage
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
DEMOGRAPHICS = ["confound_gender", "confound_age"]  # 2

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


# ── Core functions (reused from confound_analysis.py / baseline_vs_extended.py) ──
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
        raise FileNotFoundError(f"Metadata TSV not found: {metadata_tsv}")

    meta = pd.read_csv(meta_path, sep="\t")

    # Validate join keys
    for key in ("conversation_id", "speaker_id"):
        if key not in meta.columns:
            raise KeyError(f"Metadata TSV missing required column: {key}")

    # Inner join on conversation_id × speaker_id
    n_before = len(df)
    merged = df.merge(
        meta[["conversation_id", "speaker_id", "cejc_person_id", "gender", "age"]].drop_duplicates(),
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
            "Only demographics-free stages will be run."
        )

    return merged, confound_cols


# ── Three-stage analysis core ────────────────────────────────────────
def run_three_stage_analysis(
    df: pd.DataFrame,
    y_col: str,
    alpha: float = 100.0,
    cv_folds: int = 5,
    n_perm: int = 5000,
    seed: int = 42,
    trait: str = "unknown",
    teacher: str = "unknown",
) -> pd.DataFrame:
    """Run three-stage Ridge regression analysis.

    Args:
        df: DataFrame with confound columns (confound_gender, confound_age),
            Classical features, Novel features, and y_col.
        y_col: Target column name.
        alpha: Ridge alpha.
        cv_folds: Number of CV folds.
        n_perm: Number of permutation rounds.
        seed: Random seed.
        trait: Big5 trait name (for output).
        teacher: LLM teacher name (for output).

    Returns:
        DataFrame with columns: trait, teacher, stage, n_features,
        feature_set, r_obs, p_value, delta_r_from_prev
    """
    y = df[y_col].astype(float).to_numpy()

    # Drop rows with NaN in y
    ok = ~np.isnan(y)
    if ok.sum() == 0:
        warnings.warn(f"All y values are NaN for {y_col}. Skipping.")
        return pd.DataFrame()
    df_clean = df.loc[ok].copy()
    y = y[ok]

    results: list[dict] = []

    # ── Determine available demographics ─────────────────────────────
    avail_demo = [c for c in DEMOGRAPHICS if c in df_clean.columns]
    if not avail_demo:
        warnings.warn(
            "No demographic variables available. "
            "Stage 1 will be skipped."
        )
    else:
        # ── Stage 1: Demographics only ───────────────────────────────
        print(f"\n{'='*60}")
        print(f"  Stage 1: Demographics only ({len(avail_demo)} variables)")
        print(f"  Variables: {avail_demo}")
        print(f"  y_col={y_col}, N={len(y)}")
        print(f"{'='*60}")

        X1 = (
            df_clean[avail_demo]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=float)
        )
        r1, p1 = run_permutation_test(X1, y, cv_folds, seed, alpha, n_perm)
        print(f"  r_stage1 = {r1:.3f}, p = {p1:.4f}")

        results.append({
            "trait": trait,
            "teacher": teacher,
            "stage": 1,
            "n_features": len(avail_demo),
            "feature_set": "demographics",
            "r_obs": round(r1, 4),
            "p_value": round(p1, 4),
            "delta_r_from_prev": float("nan"),
        })

    # ── Validate Classical features ──────────────────────────────────
    missing_classical = [c for c in CLASSICAL_FEATURES if c not in df_clean.columns]
    if missing_classical:
        raise KeyError(f"Missing Classical feature columns: {missing_classical}")

    # ── Stage 2: Demographics + Classical ────────────────────────────
    stage2_cols = avail_demo + CLASSICAL_FEATURES
    print(f"\n{'='*60}")
    print(f"  Stage 2: Demographics + Classical ({len(stage2_cols)} variables)")
    print(f"{'='*60}")

    X2 = (
        df_clean[stage2_cols]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )
    r2, p2 = run_permutation_test(X2, y, cv_folds, seed, alpha, n_perm)
    print(f"  r_stage2 = {r2:.3f}, p = {p2:.4f}")

    delta_r_12 = float("nan")
    if results:  # Stage 1 was run
        delta_r_12 = r2 - results[0]["r_obs"]
        print(f"  Δr (Stage 1→2) = {delta_r_12:+.4f}")

    results.append({
        "trait": trait,
        "teacher": teacher,
        "stage": 2,
        "n_features": len(stage2_cols),
        "feature_set": "demographics+classical",
        "r_obs": round(r2, 4),
        "p_value": round(p2, 4),
        "delta_r_from_prev": round(delta_r_12, 4) if not np.isnan(delta_r_12) else float("nan"),
    })

    # ── Validate Novel features ──────────────────────────────────────
    missing_novel = [c for c in NOVEL_FEATURES if c not in df_clean.columns]
    if missing_novel:
        warnings.warn(
            f"Missing Novel feature columns: {missing_novel}. "
            "Stage 3 will be skipped."
        )
        return pd.DataFrame(results)

    # ── Stage 3: Demographics + Classical + Novel ────────────────────
    stage3_cols = avail_demo + CLASSICAL_FEATURES + NOVEL_FEATURES
    print(f"\n{'='*60}")
    print(f"  Stage 3: Demographics + Classical + Novel ({len(stage3_cols)} variables)")
    print(f"{'='*60}")

    X3 = (
        df_clean[stage3_cols]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )
    r3, p3 = run_permutation_test(X3, y, cv_folds, seed, alpha, n_perm)
    print(f"  r_stage3 = {r3:.3f}, p = {p3:.4f}")

    delta_r_23 = r3 - r2
    print(f"  Δr (Stage 2→3) = {delta_r_23:+.4f}")

    results.append({
        "trait": trait,
        "teacher": teacher,
        "stage": 3,
        "n_features": len(stage3_cols),
        "feature_set": "demographics+classical+novel",
        "r_obs": round(r3, 4),
        "p_value": round(p3, 4),
        "delta_r_from_prev": round(delta_r_23, 4),
    })

    return pd.DataFrame(results)


# ── Main ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Three-stage Ridge regression comparison: "
        "Demographics → +Classical → +Novel"
    )
    ap.add_argument("--xy_parquet", required=True, help="XY dataset parquet path")
    ap.add_argument("--y_col", required=True, help="Target column name (e.g. Y_C)")
    ap.add_argument(
        "--metadata_tsv",
        default="artifacts/analysis/cejc_speaker_metadata.tsv",
        help="Path to speaker metadata TSV",
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

    # ── Run three-stage analysis ─────────────────────────────────────
    result = run_three_stage_analysis(
        df=merged,
        y_col=args.y_col,
        alpha=args.alpha,
        cv_folds=args.cv_folds,
        n_perm=args.n_perm,
        seed=args.seed,
        trait=trait,
        teacher=teacher,
    )

    if result.empty:
        print("No results produced (all y values NaN?).")
        return

    # ── Write output TSV ─────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"three_stage_{trait}_{teacher}.tsv"

    result.to_csv(out_path, sep="\t", index=False)
    print(f"\nWrote {out_path}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
