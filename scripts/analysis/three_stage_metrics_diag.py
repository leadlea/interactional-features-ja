#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Three-stage Ridge: alternative metric diagnostic (r / R2 / RMSE / MAE).

目的:
  3段階Ridge比較の主指標を fold平均 Pearson r 以外の集計
  （増分妥当性 ΔR² / 予測誤差 RMSE・MAE）でも算出する．
  本文の表・図（tab:three_stage / fig:three_stage）は，この出力の
  OOF-pooled R² / RMSE を主指標として作成される．

方針:
  - 特徴量セットとメタデータ結合は three_stage_ridge.py を再利用（同一定義を保証）
  - CV設定は本文と同一の subject-wise split（Ridge alpha=100, GroupKFold groups=cejc_person_id,
    5-fold, median imputation + StandardScaler）。同一話者が訓練・検証foldに跨らない
  - 集計は OOF-pooled（全foldのout-of-fold予測を連結してから1回計算）
    → fold平均より安定で，予測値vs観測値の散布図(本文の図)と整合する集計
  - 各 stage で pooled の r / R² / RMSE / MAE を算出，stage間Δを併記
  - 参考として fold平均 r（既存図と同じ集計）も併記

注意:
  これは「見栄えのいい指標を採用する」ためではなく，どの集計が
  データと整合しかつ査読耐性があるかを判断するための材料収集である．
  採否は数表を見てから別途決める．

Usage:
    python scripts/analysis/three_stage_metrics_diag.py \
        --teacher ensemble \
        --out_dir artifacts/analysis/results/three_stage_metrics
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler

from scripts.analysis.three_stage_ridge import (
    CLASSICAL_FEATURES,
    NOVEL_FEATURES,
    DEMOGRAPHICS,
    load_and_join_metadata,
)

TRAITS = ["O", "C", "E", "A", "N"]


def _pearsonr(a, b) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum()) * np.sqrt((b * b).sum())
    return float((a * b).sum() / den) if den != 0 else float("nan")


def cv_oof_predictions(X, y, folds, seed, alpha, groups=None):
    """同一CV設定でOOF予測を連結して返す． fold平均rも併せて返す．

    groups（cejc_person_id）が与えられた場合は GroupKFold（subject-wise split）で
    同一話者が訓練・検証foldに跨らないようにする（本文の subject-wise CV と整合）。
    groups=None の場合は従来どおり通常 KFold（shuffle, seed）。
    """
    if groups is not None:
        kf = GroupKFold(n_splits=folds)
        split_iter = kf.split(X, y, groups)
    else:
        kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
        split_iter = kf.split(X)
    y_true_pool = np.empty_like(y, dtype=float)
    y_pred_pool = np.empty_like(y, dtype=float)
    fold_rs = []
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

        y_true_pool[te] = yte
        y_pred_pool[te] = yh
        fold_rs.append(_pearsonr(yte, yh))
    return y_true_pool, y_pred_pool, float(np.mean(fold_rs))


def metrics_from_oof(y_true, y_pred) -> dict:
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    r = _pearsonr(y_true, y_pred)
    # OOF R² (1 - SS_res/SS_tot), SS_tot は観測値の全体平均基準
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else float("nan")
    return {"r_oof": r, "R2_oof": r2, "RMSE": rmse, "MAE": mae}


def run_one_trait(trait, teacher, metadata_tsv, alpha, folds, seed):
    parquet = Path(
        f"artifacts/analysis/datasets/cejc_home2_hq1_XY_{trait}only_{teacher}.parquet"
    )
    if not parquet.exists():
        raise FileNotFoundError(parquet)
    df = pd.read_parquet(parquet)
    df = df.replace([np.inf, -np.inf], np.nan)
    y_col = f"Y_{trait}"
    if y_col not in df.columns:
        raise SystemExit(f"y_col not found: {y_col} in {parquet}")

    merged, _ = load_and_join_metadata(df, metadata_tsv)
    y = merged[y_col].astype(float).to_numpy()
    ok = ~np.isnan(y)
    merged = merged.loc[ok].copy()
    y = y[ok]
    groups = merged["cejc_person_id"].to_numpy()  # subject-wise split (GroupKFold)

    avail_demo = [c for c in DEMOGRAPHICS if c in merged.columns]
    stage_sets = {
        1: avail_demo,
        2: avail_demo + CLASSICAL_FEATURES,
        3: avail_demo + CLASSICAL_FEATURES + NOVEL_FEATURES,
    }

    rows = []
    prev = {}
    for stage in (1, 2, 3):
        cols = stage_sets[stage]
        X = merged[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        yt, yp, fold_mean_r = cv_oof_predictions(X, y, folds, seed, alpha, groups=groups)
        m = metrics_from_oof(yt, yp)
        row = {
            "trait": trait,
            "teacher": teacher,
            "stage": stage,
            "n_features": len(cols),
            "N": int(len(y)),
            "r_foldmean": round(fold_mean_r, 4),
            "r_oof": round(m["r_oof"], 4),
            "R2_oof": round(m["R2_oof"], 4),
            "RMSE": round(m["RMSE"], 4),
            "MAE": round(m["MAE"], 4),
        }
        # stage間Δ
        if prev:
            row["dR2_from_prev"] = round(m["R2_oof"] - prev["R2_oof"], 4)
            row["dRMSE_from_prev"] = round(m["RMSE"] - prev["RMSE"], 4)
            row["dMAE_from_prev"] = round(m["MAE"] - prev["MAE"], 4)
            row["dr_oof_from_prev"] = round(m["r_oof"] - prev["r_oof"], 4)
        else:
            row["dR2_from_prev"] = float("nan")
            row["dRMSE_from_prev"] = float("nan")
            row["dMAE_from_prev"] = float("nan")
            row["dr_oof_from_prev"] = float("nan")
        prev = m
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="ensemble")
    ap.add_argument(
        "--metadata_tsv",
        default="artifacts/analysis/cejc_speaker_metadata.tsv",
    )
    ap.add_argument("--alpha", type=float, default=100.0)
    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out_dir", default="artifacts/analysis/results/three_stage_metrics"
    )
    args = ap.parse_args()

    all_rows = []
    for trait in TRAITS:
        all_rows.extend(
            run_one_trait(
                trait, args.teacher, args.metadata_tsv,
                args.alpha, args.cv_folds, args.seed,
            )
        )

    out = pd.DataFrame(all_rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"three_stage_metrics_{args.teacher}.tsv"
    out.to_csv(out_path, sep="\t", index=False)

    # 見やすい要約: ΔはStage3行（Stage2->3 = Novel追加効果）に出る
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print(f"\nWrote {out_path}\n")
    cols_show = [
        "trait", "stage", "n_features",
        "r_foldmean", "r_oof", "R2_oof", "RMSE", "MAE",
        "dr_oof_from_prev", "dR2_from_prev", "dRMSE_from_prev", "dMAE_from_prev",
    ]
    print(out[cols_show].to_string(index=False))

    print("\n=== Novel追加効果サマリ (Stage 2->3) ===")
    s3 = out[out["stage"] == 3][
        ["trait", "dr_oof_from_prev", "dR2_from_prev", "dRMSE_from_prev", "dMAE_from_prev"]
    ]
    print(s3.to_string(index=False))


if __name__ == "__main__":
    main()
