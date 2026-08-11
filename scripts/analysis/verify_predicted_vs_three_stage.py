#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify numerical discrepancy between Table 6 (three-stage Ridge Stage 3)
and Figure (predicted vs observed scatter plot).

Context
-------
3段階Ridge Stage3 の r と予測値vs観測値散布図の r は同じ解析の別表示に見えるが，
値が一致しない。コードリードの結果、以下の2軸で差異が確認された:
  1. 特徴量セット:
     - three_stage Stage3: demographics (2) + classical (10) + novel (9) = 21
     - predicted_vs_observed: classical (10) + novel (9) = 19
  2. r_obs の集約方法:
     - three_stage: fold平均 (5 foldの r を平均)
     - predicted_vs_observed: OOF連結 (全120件のy_pred連結で Pearson r を1回)

本スクリプトは 5 trait × 4 条件 (2特徴量セット × 2集約方法) の r_obs を
計算し、どの軸が数値ずれの主因かを明らかにする。

計算コスト: Ridge 5-fold CV × 20回 → 数秒で完了（permutation なし）

Usage:
    python scripts/analysis/verify_predicted_vs_three_stage.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


# ── Feature sets (canonical, must match three_stage_ridge.py) ────────
DEMOGRAPHICS = ["confound_gender", "confound_age"]

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
]

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
]

INTERACTION_FEATURES = CLASSICAL_FEATURES + NOVEL_FEATURES  # 19
ALL_21_FEATURES = DEMOGRAPHICS + INTERACTION_FEATURES        # 21


def pearsonr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation (NaN-safe)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum()) * np.sqrt((b * b).sum())
    return float((a * b).sum() / den) if den != 0 else float("nan")


def cv_ridge_two_metrics(
    X: np.ndarray,
    y: np.ndarray,
    folds: int = 5,
    seed: int = 42,
    alpha: float = 100.0,
) -> tuple[float, float]:
    """Run 5-fold CV Ridge and return both r-aggregation methods.

    Returns
    -------
    (r_fold_mean, r_oof_concat):
        r_fold_mean: mean of per-fold Pearson r (three_stage_ridge.py method)
        r_oof_concat: Pearson r on concatenated OOF predictions
                      (collect_oof_predictions method)
    """
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    per_fold_r: list[float] = []
    y_pred_oof = np.full_like(y, np.nan, dtype=float)

    for tr, te in kf.split(X):
        imp = SimpleImputer(strategy="median")
        Xtr = imp.fit_transform(X[tr])
        Xte = imp.transform(X[te])

        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xte = sc.transform(Xte)

        m = Ridge(alpha=alpha, random_state=seed)
        m.fit(Xtr, y[tr])
        yh = m.predict(Xte)

        per_fold_r.append(pearsonr(y[te], yh))
        y_pred_oof[te] = yh

    r_fold_mean = float(np.mean(per_fold_r))

    # Drop any NaN (shouldn't happen) for the OOF-concat metric
    valid = ~np.isnan(y_pred_oof)
    r_oof_concat = pearsonr(y[valid], y_pred_oof[valid])
    return r_fold_mean, r_oof_concat


def load_and_join_metadata(
    xy_df: pd.DataFrame, metadata_path: Path
) -> pd.DataFrame:
    """Join XY data with speaker metadata, add confound columns."""
    meta = pd.read_csv(metadata_path, sep="\t")
    merged = xy_df.merge(
        meta[["conversation_id", "speaker_id", "gender", "age"]].drop_duplicates(),
        on=["conversation_id", "speaker_id"],
        how="inner",
        suffixes=("", "_meta"),
    )
    merged["confound_gender"] = (
        merged["gender"].map({"M": 0, "F": 1}).astype(float)
    )
    merged["confound_age"] = pd.to_numeric(merged["age"], errors="coerce")
    return merged


def read_published_stage3_r(
    results_dir: Path, trait: str, teacher: str = "ensemble"
) -> float | None:
    """Read published Stage 3 r_obs from three_stage_*_ensemble.tsv."""
    p = results_dir / f"three_stage_{trait}_{teacher}.tsv"
    if not p.exists():
        return None
    df = pd.read_csv(p, sep="\t")
    s3 = df[df["stage"] == 3]
    if s3.empty:
        return None
    return float(s3.iloc[0]["r_obs"])


def run_one_trait(
    trait: str,
    datasets_dir: Path,
    metadata_path: Path,
    results_dir: Path,
    alpha: float = 100.0,
    cv_folds: int = 5,
    seed: int = 42,
) -> list[dict]:
    """Run 4 conditions for a single trait and return rows for the comparison TSV."""
    xy_path = datasets_dir / f"cejc_home2_hq1_XY_{trait}only_ensemble.parquet"
    if not xy_path.exists():
        print(f"  [skip] {xy_path} not found")
        return []

    df = pd.read_parquet(xy_path).replace([np.inf, -np.inf], np.nan)
    merged = load_and_join_metadata(df, metadata_path)

    y_col = f"Y_{trait}"
    y = merged[y_col].astype(float).to_numpy()
    ok = ~np.isnan(y)
    merged = merged.loc[ok].copy()
    y = y[ok]
    n = len(y)

    published_stage3_r = read_published_stage3_r(results_dir, trait)

    rows: list[dict] = []

    # ── Condition 1: 21 features (demo + classical + novel) ──────────
    X21 = (
        merged[ALL_21_FEATURES]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )
    r_fold_21, r_oof_21 = cv_ridge_two_metrics(
        X21, y, folds=cv_folds, seed=seed, alpha=alpha
    )

    # ── Condition 2: 19 features (classical + novel, no demo) ────────
    X19 = (
        merged[INTERACTION_FEATURES]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )
    r_fold_19, r_oof_19 = cv_ridge_two_metrics(
        X19, y, folds=cv_folds, seed=seed, alpha=alpha
    )

    rows.append({
        "trait": trait,
        "N": n,
        "n_features": 21,
        "feature_set": "demo+classical+novel",
        "aggregation": "fold_mean",
        "r_obs": round(r_fold_21, 4),
        "matches_published_stage3": (
            abs(r_fold_21 - published_stage3_r) < 1e-3
            if published_stage3_r is not None else None
        ),
        "published_stage3_r": published_stage3_r,
        "method_label": "three_stage Stage 3 (Table 6)",
    })
    rows.append({
        "trait": trait,
        "N": n,
        "n_features": 21,
        "feature_set": "demo+classical+novel",
        "aggregation": "oof_concat",
        "r_obs": round(r_oof_21, 4),
        "matches_published_stage3": None,
        "published_stage3_r": published_stage3_r,
        "method_label": "Stage3 features, OOF-concat",
    })
    rows.append({
        "trait": trait,
        "N": n,
        "n_features": 19,
        "feature_set": "classical+novel",
        "aggregation": "fold_mean",
        "r_obs": round(r_fold_19, 4),
        "matches_published_stage3": None,
        "published_stage3_r": published_stage3_r,
        "method_label": "19-features, fold-mean",
    })
    rows.append({
        "trait": trait,
        "N": n,
        "n_features": 19,
        "feature_set": "classical+novel",
        "aggregation": "oof_concat",
        "r_obs": round(r_oof_19, 4),
        "matches_published_stage3": None,
        "published_stage3_r": published_stage3_r,
        "method_label": "predicted_vs_observed figure",
    })

    return rows


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Diagnose numerical discrepancy between Table 6 (three-stage "
            "Stage 3) and predicted_vs_observed scatter figure."
        )
    )
    ap.add_argument(
        "--datasets_dir",
        default="artifacts/analysis/datasets",
        help="Directory containing XY parquet files",
    )
    ap.add_argument(
        "--metadata_tsv",
        default="artifacts/analysis/cejc_speaker_metadata.tsv",
        help="Path to speaker metadata TSV",
    )
    ap.add_argument(
        "--results_dir",
        default="artifacts/analysis/results",
        help="Directory containing three_stage_*_ensemble.tsv",
    )
    ap.add_argument(
        "--out_dir",
        default="artifacts/analysis/results/consistency_check",
        help="Output directory for comparison TSV",
    )
    ap.add_argument("--alpha", type=float, default=100.0)
    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    datasets_dir = Path(args.datasets_dir)
    metadata_path = Path(args.metadata_tsv)
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Predicted-vs-Observed vs Three-Stage Stage 3 Consistency Check")
    print("  (Ensemble Big5, Ridge α=100, 5-fold CV, seed=42)")
    print("=" * 70)

    all_rows: list[dict] = []
    for trait in ["O", "C", "E", "A", "N"]:
        print(f"\n[Trait {trait}]")
        rows = run_one_trait(
            trait=trait,
            datasets_dir=datasets_dir,
            metadata_path=metadata_path,
            results_dir=results_dir,
            alpha=args.alpha,
            cv_folds=args.cv_folds,
            seed=args.seed,
        )
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out_tsv = out_dir / "predicted_vs_three_stage_comparison.tsv"
    df.to_csv(out_tsv, sep="\t", index=False)

    print("\n" + "=" * 70)
    print("  Comparison table (r_obs across 4 conditions × 5 traits)")
    print("=" * 70)
    # Wide-format summary for readability
    wide = df.pivot_table(
        index="trait",
        columns=["n_features", "aggregation"],
        values="r_obs",
    ).round(4)
    print("\n" + wide.to_string())

    # Published Stage 3 r_obs from TSV files
    print("\n" + "-" * 70)
    print("  Published Stage 3 r_obs (from three_stage_*_ensemble.tsv):")
    print("-" * 70)
    pub = df[df["aggregation"] == "fold_mean"][
        df["n_features"] == 21
    ][["trait", "r_obs", "published_stage3_r", "matches_published_stage3"]]
    print(pub.to_string(index=False))

    print(f"\nWrote {out_tsv}")

    # ── Diagnosis ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Diagnosis: Which axis dominates the discrepancy?")
    print("=" * 70)
    diag_rows = []
    for trait in ["O", "C", "E", "A", "N"]:
        sub = df[df["trait"] == trait].set_index(["n_features", "aggregation"])
        r_ts = sub.loc[(21, "fold_mean"), "r_obs"]         # Table 6 (Stage 3)
        r_fig = sub.loc[(19, "oof_concat"), "r_obs"]       # predicted_vs_observed figure
        # holding feature-set constant at 21 features
        delta_agg_21 = sub.loc[(21, "oof_concat"), "r_obs"] - r_ts
        # holding aggregation constant at fold_mean
        delta_feat_fm = sub.loc[(19, "fold_mean"), "r_obs"] - r_ts
        # total
        delta_total = r_fig - r_ts
        diag_rows.append({
            "trait": trait,
            "r_table6_stage3 (21f, fold_mean)": round(r_ts, 4),
            "r_figure (19f, oof_concat)": round(r_fig, 4),
            "total_delta": round(delta_total, 4),
            "delta_from_aggregation_only (21f)": round(delta_agg_21, 4),
            "delta_from_features_only (fold_mean)": round(delta_feat_fm, 4),
        })
    diag_df = pd.DataFrame(diag_rows)
    print("\n" + diag_df.to_string(index=False))

    diag_tsv = out_dir / "predicted_vs_three_stage_diagnosis.tsv"
    diag_df.to_csv(diag_tsv, sep="\t", index=False)
    print(f"\nWrote {diag_tsv}")


if __name__ == "__main__":
    main()
