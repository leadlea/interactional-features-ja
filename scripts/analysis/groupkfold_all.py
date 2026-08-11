#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run GroupKFold (subject-wise) permutation test for all trait × teacher.

Outputs a single TSV comparing KFold vs GroupKFold results.

Usage:
    python scripts/analysis/groupkfold_all.py \
        --datasets_dir artifacts/analysis/datasets \
        --metadata_tsv artifacts/analysis/cejc_speaker_metadata.tsv \
        --out_tsv artifacts/analysis/results/groupkfold_vs_kfold_all.tsv \
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
from sklearn.model_selection import GroupKFold, KFold
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


def cv_ridge_r(X, y, splits, seed, alpha, groups=None):
    if groups is not None:
        kf = GroupKFold(n_splits=splits)
        split_iter = kf.split(X, y, groups)
    else:
        kf = KFold(n_splits=splits, shuffle=True, random_state=seed)
        split_iter = kf.split(X)

    rs = []
    for tr, te in split_iter:
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


def run_perm(X, y, splits, seed, alpha, n_perm, groups=None):
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
    ap.add_argument("--out_tsv", default="artifacts/analysis/results/groupkfold_vs_kfold_all.tsv")
    ap.add_argument("--alpha", type=float, default=100.0)
    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--n_perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    meta = pd.read_csv(args.metadata_tsv, sep="\t")
    files = sorted(glob.glob(f"{args.datasets_dir}/cejc_home2_hq1_XY_*only_*.parquet"))
    files = [f for f in files if "ensemble" not in f and "driftv2" not in f]

    print(f"Found {len(files)} datasets")
    print(f"n_perm={args.n_perm}, alpha={args.alpha}, folds={args.cv_folds}")
    print()

    rows = []
    for i, fpath in enumerate(files):
        trait, teacher = parse_trait_teacher(fpath)
        y_col = f"Y_{trait}"

        df = pd.read_parquet(fpath).replace([np.inf, -np.inf], np.nan)
        if y_col not in df.columns:
            print(f"  SKIP {trait}/{teacher}: {y_col} not found")
            continue

        merged = df.merge(
            meta[["conversation_id", "speaker_id", "cejc_person_id"]],
            on=["conversation_id", "speaker_id"],
            how="left",
        )

        y = merged[y_col].astype(float).to_numpy()
        available_features = [c for c in ALL_FEATURES if c in merged.columns]
        X = merged[available_features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        groups = merged["cejc_person_id"].to_numpy()

        print(f"[{i+1}/{len(files)}] {trait} × {teacher} (N={len(y)})")

        t0 = time.time()
        r_kf, p_kf = run_perm(X, y, args.cv_folds, args.seed, args.alpha, args.n_perm, groups=None)
        r_gkf, p_gkf = run_perm(X, y, args.cv_folds, args.seed, args.alpha, args.n_perm, groups=groups)
        elapsed = time.time() - t0

        delta_r = r_gkf - r_kf
        sig_kf = "**" if p_kf < 0.05 else "n.s."
        sig_gkf = "**" if p_gkf < 0.05 else "n.s."

        print(f"  KFold: r={r_kf:.4f} p={p_kf:.4f} {sig_kf}")
        print(f"  Group: r={r_gkf:.4f} p={p_gkf:.4f} {sig_gkf}  Δr={delta_r:+.4f}  ({elapsed:.1f}s)")

        rows.append({
            "trait": trait,
            "teacher": teacher,
            "r_kfold": round(r_kf, 4),
            "p_kfold": round(p_kf, 4),
            "r_groupkfold": round(r_gkf, 4),
            "p_groupkfold": round(p_gkf, 4),
            "delta_r": round(delta_r, 4),
            "sig_kfold": sig_kf,
            "sig_groupkfold": sig_gkf,
        })

    result = pd.DataFrame(rows)
    from pathlib import Path
    Path(args.out_tsv).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out_tsv, sep="\t", index=False)
    print(f"\nWrote {args.out_tsv}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
