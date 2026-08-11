#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Confound analysis with GroupKFold (subject-wise split).

Same as confound_analysis.py but uses GroupKFold(cejc_person_id).
Runs all trait x teacher combinations.

Usage:
    python scripts/analysis/confound_analysis_groupkfold.py \
        --datasets_dir artifacts/analysis/datasets \
        --metadata_tsv artifacts/analysis/cejc_speaker_metadata.tsv \
        --out_tsv artifacts/analysis/results/confound_groupkfold_all.tsv \
        --n_perm 1000
"""
from __future__ import annotations

import argparse
import glob
import re
import time

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ALL_FEATURES = [
    "PG_speech_ratio", "PG_pause_mean", "PG_pause_p50", "PG_pause_p90",
    "PG_resp_gap_mean", "PG_resp_gap_p50", "PG_resp_gap_p90", "PG_overlap_rate",
    "FILL_has_any", "FILL_rate_per_100chars",
    "IX_oirmarker_rate", "IX_oirmarker_after_question_rate",
    "IX_yesno_rate", "IX_yesno_after_question_rate", "IX_lex_overlap_mean",
    "RESP_NE_AIZUCHI_RATE", "RESP_NE_ENTROPY", "RESP_YO_ENTROPY",
    "PG_pause_variability",
]


def pearsonr(a, b) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum()) * np.sqrt((b * b).sum())
    return float((a * b).sum() / den) if den != 0 else float("nan")


def cv_ridge_r(X, y, splits, seed, alpha, groups):
    kf = GroupKFold(n_splits=splits)
    rs = []
    for tr, te in kf.split(X, y, groups):
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


def run_perm(X, y, splits, seed, alpha, n_perm, groups):
    r_obs = cv_ridge_r(X, y, splits, seed, alpha, groups)
    rng = np.random.default_rng(seed)
    r_perm = np.empty(n_perm, float)
    for i in range(n_perm):
        yp = rng.permutation(y)
        r_perm[i] = cv_ridge_r(X, yp, splits, seed, alpha, groups)
    p = (np.sum(np.abs(r_perm) >= abs(r_obs)) + 1.0) / (n_perm + 1.0)
    return r_obs, p


def parse_trait_teacher(path: str):
    stem = path.split("/")[-1].replace(".parquet", "")
    m = re.search(r"_XY_(\w+)only_(.+)$", stem)
    if m:
        return m.group(1), m.group(2)
    return "?", "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets_dir", default="artifacts/analysis/datasets")
    ap.add_argument("--metadata_tsv", default="artifacts/analysis/cejc_speaker_metadata.tsv")
    ap.add_argument("--out_tsv", default="artifacts/analysis/results/confound_groupkfold_all.tsv")
    ap.add_argument("--alpha", type=float, default=100.0)
    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--n_perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    meta = pd.read_csv(args.metadata_tsv, sep="\t")
    files = sorted(glob.glob(f"{args.datasets_dir}/cejc_home2_hq1_XY_*only_*.parquet"))
    files = [f for f in files if "ensemble" not in f and "driftv2" not in f]

    # Only C trait for now (main result)
    c_files = [f for f in files if "_Conly_" in f]
    print(f"Found {len(c_files)} C datasets")

    rows = []
    for i, fpath in enumerate(c_files):
        trait, teacher = parse_trait_teacher(fpath)
        y_col = f"Y_{trait}"

        df = pd.read_parquet(fpath).replace([np.inf, -np.inf], np.nan)
        merged = df.merge(
            meta[["conversation_id", "speaker_id", "cejc_person_id", "gender", "age"]],
            on=["conversation_id", "speaker_id"],
            how="left",
        )

        y = merged[y_col].astype(float).to_numpy()
        available_features = [c for c in ALL_FEATURES if c in merged.columns]
        X_feat = merged[available_features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        groups = merged["cejc_person_id"].to_numpy()

        # Confound columns
        merged["confound_gender"] = merged["gender"].map({"M": 0, "F": 1}).astype(float)
        merged["confound_age"] = pd.to_numeric(merged["age"], errors="coerce")
        confound_cols = ["confound_gender", "confound_age"]
        X_conf = merged[available_features + confound_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

        print(f"[{i+1}/{len(c_files)}] {trait} × {teacher} (N={len(y)}, feat={X_feat.shape[1]}, conf={X_conf.shape[1]})")

        t0 = time.time()
        r_feat, p_feat = run_perm(X_feat, y, args.cv_folds, args.seed, args.alpha, args.n_perm, groups)
        r_conf, p_conf = run_perm(X_conf, y, args.cv_folds, args.seed, args.alpha, args.n_perm, groups)
        elapsed = time.time() - t0

        delta_r = r_conf - r_feat
        print(f"  feat: r={r_feat:.4f} p={p_feat:.4f}")
        print(f"  conf: r={r_conf:.4f} p={p_conf:.4f}  Δr={delta_r:+.4f}  ({elapsed:.1f}s)")

        rows.append({
            "trait": trait,
            "teacher": teacher,
            "r_features_only": round(r_feat, 4),
            "p_features_only": round(p_feat, 4),
            "r_with_confounds": round(r_conf, 4),
            "p_with_confounds": round(p_conf, 4),
            "delta_r": round(delta_r, 4),
        })

    result = pd.DataFrame(rows)
    from pathlib import Path
    Path(args.out_tsv).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out_tsv, sep="\t", index=False)
    print(f"\nWrote {args.out_tsv}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
