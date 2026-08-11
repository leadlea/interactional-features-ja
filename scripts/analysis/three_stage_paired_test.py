#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Three-stage Ridge: paired error test (Stage 2 vs Stage 3).

目的:
  3.4.3節をΔR²/誤差ベースに切り替えるにあたり，Novel特徴量追加(Stage 2->3)
  による予測誤差の減少が偶然を超えるかを，被験者ごとのOOF二乗誤差の
  ペア比較で検定する．

手法:
  - three_stage_metrics_diag.py と同一CV設定（GroupKFold subject-wise, groups=cejc_person_id）で
    Stage2/Stage3のOOF予測を取得（同一話者が訓練・検証foldに跨らない）
  - 各レコードの二乗誤差 SE2, SE3 を算出（ペア）
  - 改善量 d_i = SE2_i - SE3_i （正なら Stage3 が誤差を減らした）
  - paired bootstrap（5000回, seed固定）で mean(d) の95%CIと両側p
    p = 2 * min(P(boot<=0), P(boot>=0)) をクリップ
  - Wilcoxon 符号付順位検定（両側）も併記（分布非依存の頑健チェック）
  - 参考: ΔRMSE = RMSE3 - RMSE2 （負なら改善）

Usage:
    python -m scripts.analysis.three_stage_paired_test \
        --teacher ensemble \
        --out_dir artifacts/analysis/results/three_stage_metrics
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from scripts.analysis.three_stage_ridge import (
    CLASSICAL_FEATURES,
    NOVEL_FEATURES,
    DEMOGRAPHICS,
    load_and_join_metadata,
)
from scripts.analysis.three_stage_metrics_diag import cv_oof_predictions

TRAITS = ["O", "C", "E", "A", "N"]


def paired_bootstrap(d: np.ndarray, n_boot: int, seed: int):
    """mean(d) の paired bootstrap 95%CI と両側p を返す."""
    rng = np.random.default_rng(seed)
    n = len(d)
    boot = np.empty(n_boot, float)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[i] = d[idx].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p_left = (np.sum(boot <= 0) + 1.0) / (n_boot + 1.0)
    p_right = (np.sum(boot >= 0) + 1.0) / (n_boot + 1.0)
    p_two = min(1.0, 2.0 * min(p_left, p_right))
    return float(d.mean()), float(lo), float(hi), float(p_two)


def run_one_trait(trait, teacher, metadata_tsv, alpha, folds, seed, n_boot):
    parquet = Path(
        f"artifacts/analysis/datasets/cejc_home2_hq1_XY_{trait}only_{teacher}.parquet"
    )
    df = pd.read_parquet(parquet).replace([np.inf, -np.inf], np.nan)
    y_col = f"Y_{trait}"
    merged, _ = load_and_join_metadata(df, metadata_tsv)
    y = merged[y_col].astype(float).to_numpy()
    ok = ~np.isnan(y)
    merged = merged.loc[ok].copy()
    y = y[ok]
    groups = merged["cejc_person_id"].to_numpy()  # subject-wise split (GroupKFold)

    avail_demo = [c for c in DEMOGRAPHICS if c in merged.columns]
    cols2 = avail_demo + CLASSICAL_FEATURES
    cols3 = avail_demo + CLASSICAL_FEATURES + NOVEL_FEATURES

    X2 = merged[cols2].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    X3 = merged[cols3].apply(pd.to_numeric, errors="coerce").to_numpy(float)

    yt2, yp2, _ = cv_oof_predictions(X2, y, folds, seed, alpha, groups=groups)
    yt3, yp3, _ = cv_oof_predictions(X3, y, folds, seed, alpha, groups=groups)
    # yt2 と yt3 は同一（同じy, 同じgroups, 同じGroupKFold分割）なのでペア対応が保証される
    assert np.allclose(yt2, yt3)

    se2 = (yp2 - yt2) ** 2
    se3 = (yp3 - yt3) ** 2
    d = se2 - se3  # 正 = Stage3が誤差を減らした

    mean_d, lo, hi, p_boot = paired_bootstrap(d, n_boot, seed)

    # Wilcoxon（差が全てゼロでない前提; ゼロ差はwilcoxonがzero_method既定で処理）
    try:
        w_stat, w_p = wilcoxon(se2, se3, alternative="two-sided")
        w_p = float(w_p)
    except ValueError:
        w_p = float("nan")

    rmse2 = float(np.sqrt(np.mean(se2)))
    rmse3 = float(np.sqrt(np.mean(se3)))

    return {
        "trait": trait,
        "teacher": teacher,
        "N": int(len(y)),
        "RMSE_stage2": round(rmse2, 4),
        "RMSE_stage3": round(rmse3, 4),
        "dRMSE_2to3": round(rmse3 - rmse2, 4),       # 負=改善
        "mean_dSE_2to3": round(mean_d, 5),            # 正=改善
        "boot_ci_lo": round(lo, 5),
        "boot_ci_hi": round(hi, 5),
        "p_boot": round(p_boot, 4),
        "p_wilcoxon": round(w_p, 4) if not np.isnan(w_p) else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="ensemble")
    ap.add_argument("--metadata_tsv",
                    default="artifacts/analysis/cejc_speaker_metadata.tsv")
    ap.add_argument("--alpha", type=float, default=100.0)
    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--out_dir",
                    default="artifacts/analysis/results/three_stage_metrics")
    args = ap.parse_args()

    rows = [
        run_one_trait(t, args.teacher, args.metadata_tsv, args.alpha,
                      args.cv_folds, args.seed, args.n_boot)
        for t in TRAITS
    ]
    out = pd.DataFrame(rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"three_stage_paired_test_{args.teacher}.tsv"
    out.to_csv(out_path, sep="\t", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print(f"\nWrote {out_path}\n")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
