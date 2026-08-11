#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ensemble Big5 Permutation Test.

Compute ensemble Big5 scores by averaging trait_score across 4 LLM teachers,
then run Ridge + Permutation test for all 5 traits (O, C, E, A, N).

Usage:
    python scripts/analysis/ensemble_permutation.py \
        --items_dir artifacts/big5/llm_scores \
        --features_parquet artifacts/analysis/features_min/features_cejc_home2_hq1.parquet \
        --out_dir artifacts/analysis/results/ensemble_perm
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

# ── Constants ────────────────────────────────────────────────────────
TRAITS = ["O", "C", "E", "A", "N"]
TEACHERS = ["sonnet4", "qwen3-235b", "deepseek-v3", "gpt-oss-120b"]

# Columns always excluded from features (control variables + collinear).
# PG_overlap_rate is intentionally NOT listed here — it is a Classical
# explanatory variable (restored from CTRL per yamashita-feedback-v4).
DEFAULT_EXCLUDE: set[str] = {
    "conversation_id",
    "speaker_id",
    # Collinear with IX_lex_overlap_mean (r = -1.00)
    "IX_topic_drift_mean",
    # Control / denominator columns
    "n_pairs_total",
    "n_pairs_after_NE",
    "n_pairs_after_YO",
    "IX_n_pairs",
    "IX_n_pairs_after_question",
    "PG_total_time",
    "PG_resp_overlap_rate",
    "FILL_text_len",
    "FILL_cnt_total",
    "FILL_cnt_eto",
    "FILL_cnt_e",
    "FILL_cnt_ano",
}


# ── Core functions (same logic as permutation_test_ridge_fixedalpha.py) ──
def pearsonr(a, b) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum()) * np.sqrt((b * b).sum())
    return float((a * b).sum() / den) if den != 0 else float("nan")


def cv_ridge_r(X, y, folds, seed, alpha):
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


# ── Data loading ─────────────────────────────────────────────────────
def load_trait_scores(items_dir: str, trait: str, teachers: list[str]) -> pd.DataFrame:
    """Load trait scores from all available teachers and average across them.

    Returns a DataFrame with columns: conversation_id, speaker_id, trait_score
    (averaged across teachers).
    """
    dfs = []
    for teacher in teachers:
        dir_name = f"dataset=cejc_home2_hq1_v1__items={trait}24__teacher={teacher}"
        merged_dir = Path(items_dir) / dir_name / "teacher_merged"
        parquet_path = merged_dir / f"trait_scores_{trait}_merged.parquet"

        if not parquet_path.exists():
            warnings.warn(
                f"Missing parquet for teacher={teacher}, trait={trait}: {parquet_path}"
            )
            continue

        df = pd.read_parquet(parquet_path)
        # Expect columns: conversation_id, speaker_id, trait, model_id, trait_score
        df = df[["conversation_id", "speaker_id", "trait_score"]].copy()
        df = df.rename(columns={"trait_score": f"score_{teacher}"})
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(
            f"No teacher parquets found for trait={trait} in {items_dir}"
        )

    n_teachers = len(dfs)
    if n_teachers < len(teachers):
        warnings.warn(
            f"trait={trait}: only {n_teachers}/{len(teachers)} teachers available"
        )

    # Merge all teachers on (conversation_id, speaker_id)
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on=["conversation_id", "speaker_id"], how="inner")

    # Average trait_score across teachers
    score_cols = [c for c in merged.columns if c.startswith("score_")]
    merged["trait_score"] = merged[score_cols].mean(axis=1)

    return merged[["conversation_id", "speaker_id", "trait_score"]].copy()


# ── Holm-Bonferroni correction ───────────────────────────────────────
def holm_correction(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni法による多重比較補正。

    Args:
        p_values: 補正前p値のリスト（長さm）

    Returns:
        補正後p値のリスト（元の順序を保持）

    Raises:
        ValueError: p値が [0, 1] 範囲外の場合
    """
    import math

    m = len(p_values)
    if m == 0:
        return []

    # NaN以外のp値を検証: [0, 1] 範囲外ならValueError
    for i, p in enumerate(p_values):
        if math.isnan(p):
            continue
        if p < 0.0 or p > 1.0:
            raise ValueError(
                f"p_values[{i}] = {p} is outside [0, 1]"
            )

    # 非NaN値のみ補正対象とする
    non_nan = [(i, p) for i, p in enumerate(p_values) if not math.isnan(p)]

    corrected = [float("nan")] * m

    if not non_nan:
        return corrected

    # 非NaN p値を昇順にソート（インデックスを保持）
    m_eff = len(non_nan)
    indexed = sorted(non_nan, key=lambda x: x[1])

    cummax = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted = p * (m_eff - rank)
        # 累積最大値を取る（単調性保証）
        cummax = max(cummax, adjusted)
        # 1.0でクリップ
        corrected[orig_idx] = min(cummax, 1.0)

    return corrected


# ── Main ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Ensemble Big5 Permutation Test (4-teacher average)"
    )
    ap.add_argument(
        "--items_dir",
        default="artifacts/big5/llm_scores",
        help="Root dir containing dataset=...items=...teacher=... folders",
    )
    ap.add_argument(
        "--features_parquet",
        default="artifacts/analysis/features_min/features_cejc_home2_hq1.parquet",
        help="Path to features parquet",
    )
    ap.add_argument("--exclude_cols", default="", help="Extra columns to exclude (comma-separated)")
    ap.add_argument("--alpha", type=float, default=100.0, help="Ridge alpha")
    ap.add_argument("--cv_folds", type=int, default=5, help="CV folds")
    ap.add_argument("--n_perm", type=int, default=5000, help="Permutation rounds")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    args = ap.parse_args()

    # Load features
    feat_df = pd.read_parquet(args.features_parquet)
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan)

    # Columns to exclude from features: default controls + CLI extras
    excl = set(DEFAULT_EXCLUDE)
    excl |= set(c.strip() for c in args.exclude_cols.split(",") if c.strip())

    summary_rows = []

    for trait in TRAITS:
        print(f"\n{'='*60}")
        print(f"  Trait: {trait}")
        print(f"{'='*60}")

        # 1. Load and average trait scores across teachers
        scores_df = load_trait_scores(args.items_dir, trait, TEACHERS)
        print(f"  Ensemble scores: {len(scores_df)} rows")

        # 2. Merge with features
        merged = scores_df.merge(feat_df, on=["conversation_id", "speaker_id"], how="inner")
        if len(merged) < len(scores_df):
            warnings.warn(
                f"trait={trait}: join reduced rows from {len(scores_df)} to {len(merged)}"
            )
        print(f"  After join with features: {len(merged)} rows")

        # 3. Prepare X, y
        y_col = "trait_score"
        y = merged[y_col].astype(float).to_numpy()

        feat_cols = [
            c for c in merged.columns if c not in excl and c != y_col
        ]
        # Validate feature columns exist
        missing = [c for c in feat_cols if c not in merged.columns]
        if missing:
            raise KeyError(f"Missing feature columns: {missing}")

        X = merged[feat_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

        # Drop rows with NaN in y
        ok = ~np.isnan(y)
        X = X[ok]
        y = y[ok]
        print(f"  Features: {X.shape[1]} cols, {X.shape[0]} samples")

        # 4. Observed r
        r_obs = cv_ridge_r(X, y, args.cv_folds, args.seed, args.alpha)
        print(f"  r_obs = {r_obs:.3f}")

        # 5. Permutation test
        rng = np.random.default_rng(args.seed)
        r_perm = np.empty(args.n_perm, float)
        for i in range(args.n_perm):
            yp = rng.permutation(y)
            r_perm[i] = cv_ridge_r(X, yp, args.cv_folds, args.seed, args.alpha)
            if (i + 1) % 500 == 0:
                print(f"    perm {i+1}/{args.n_perm} done")

        p_val = (np.sum(np.abs(r_perm) >= abs(r_obs)) + 1.0) / (args.n_perm + 1.0)
        print(f"  p(|r|) = {p_val:.4f}  (n_perm={args.n_perm})")

        # 6. Write permutation.log
        trait_dir = Path(args.out_dir) / f"ensemble_{trait}"
        trait_dir.mkdir(parents=True, exist_ok=True)
        log_path = trait_dir / "permutation.log"
        with open(log_path, "w") as f:
            f.write(f"alpha={args.alpha}\n")
            f.write(f"r_obs={r_obs:.3f}\n")
            f.write(f"p(|r|)={p_val:.4f}  (n_perm={args.n_perm})\n")
        print(f"  Wrote {log_path}")

        summary_rows.append({"trait": trait, "r_obs": round(r_obs, 3), "p_value": round(p_val, 4)})

    # 7. Build summary_df and apply Holm correction
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summary_rows)

    # Apply Holm-Bonferroni correction across 5 traits
    p_corrected = holm_correction(summary_df["p_value"].tolist())
    summary_df["p_corrected"] = [round(pc, 4) for pc in p_corrected]

    summary_path = out_dir / "ensemble_summary.tsv"
    summary_df.to_csv(summary_path, sep="\t", index=False)
    print(f"\nWrote {summary_path}")
    print(summary_df.to_string(index=False))

    # 8. Per-teacher permutation tests with Holm correction
    print(f"\n{'='*60}")
    print("  Per-teacher permutation tests")
    print(f"{'='*60}")

    for teacher in TEACHERS:
        print(f"\n--- Teacher: {teacher} ---")
        teacher_rows: list[dict] = []

        for trait in TRAITS:
            # Load single-teacher trait scores
            dir_name = f"dataset=cejc_home2_hq1_v1__items={trait}24__teacher={teacher}"
            merged_dir = Path(args.items_dir) / dir_name / "teacher_merged"
            parquet_path = merged_dir / f"trait_scores_{trait}_merged.parquet"

            if not parquet_path.exists():
                warnings.warn(
                    f"Missing parquet for teacher={teacher}, trait={trait}: {parquet_path}"
                )
                teacher_rows.append({
                    "trait": trait, "r_obs": float("nan"),
                    "p_value": float("nan"), "p_corrected": float("nan"),
                })
                continue

            scores_df = pd.read_parquet(parquet_path)
            scores_df = scores_df[["conversation_id", "speaker_id", "trait_score"]].copy()

            merged = scores_df.merge(feat_df, on=["conversation_id", "speaker_id"], how="inner")
            y = merged["trait_score"].astype(float).to_numpy()
            feat_cols = [c for c in merged.columns if c not in excl and c != "trait_score"]
            X = merged[feat_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

            ok = ~np.isnan(y)
            X, y = X[ok], y[ok]

            r_obs = cv_ridge_r(X, y, args.cv_folds, args.seed, args.alpha)

            rng = np.random.default_rng(args.seed)
            r_perm = np.empty(args.n_perm, float)
            for i in range(args.n_perm):
                yp = rng.permutation(y)
                r_perm[i] = cv_ridge_r(X, yp, args.cv_folds, args.seed, args.alpha)

            p_val = (np.sum(np.abs(r_perm) >= abs(r_obs)) + 1.0) / (args.n_perm + 1.0)
            print(f"  {trait}: r_obs={r_obs:.3f}, p={p_val:.4f}")

            # Write per-teacher permutation.log
            teacher_trait_dir = Path(args.out_dir) / f"{teacher}_{trait}"
            teacher_trait_dir.mkdir(parents=True, exist_ok=True)
            log_path = teacher_trait_dir / "permutation.log"
            with open(log_path, "w") as f:
                f.write(f"teacher={teacher}\n")
                f.write(f"alpha={args.alpha}\n")
                f.write(f"r_obs={r_obs:.3f}\n")
                f.write(f"p(|r|)={p_val:.4f}  (n_perm={args.n_perm})\n")

            teacher_rows.append({
                "trait": trait, "r_obs": round(r_obs, 3), "p_value": round(p_val, 4),
            })

        # Apply Holm correction per teacher (across 5 traits)
        teacher_df = pd.DataFrame(teacher_rows)
        p_corr = holm_correction(teacher_df["p_value"].tolist())
        teacher_df["p_corrected"] = [round(pc, 4) for pc in p_corr]

        teacher_summary_path = out_dir / f"teacher_{teacher}_summary.tsv"
        teacher_df.to_csv(teacher_summary_path, sep="\t", index=False)
        print(f"  Wrote {teacher_summary_path}")
        print(teacher_df.to_string(index=False))


if __name__ == "__main__":
    main()
