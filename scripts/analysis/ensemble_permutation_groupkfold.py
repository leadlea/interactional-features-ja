#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Permutation test against the virtual Big5 scores under a subject-wise split.

Why this exists
---------------
The headline permutation test was previously computed by
``scripts/analysis/ensemble_permutation.py``, whose ``cv_ridge_r`` uses
``KFold(shuffle=True)``. That contradicts the cross-validation design the
manuscript describes: GroupKFold over ``cejc_person_id`` (a subject-wise split).
Because 59.2% of the records come from speakers who appear in more than one
conversation, a plain KFold lets the same speaker straddle the training and
validation folds, which leaks speaker-specific feature patterns.

The staged ridge analysis was moved to GroupKFold earlier; this script brings the
headline test in line with it. The main analysis now rests on this test alone, so
the split has to match what the manuscript claims.

For transparency both designs are computed and written side by side into the same
TSV, following the pattern of ``scripts/analysis/groupkfold_all.py``.

The predictor set is fixed at the 19 features (Classical 10 + Novel 9) so that it
is identical to ``permutation_coef_test.py``, ``bootstrap_variance.py`` and
``confound_analysis_groupkfold.py``.

Usage
-----
    python scripts/analysis/ensemble_permutation_groupkfold.py \
        --datasets_dir artifacts/analysis/datasets \
        --metadata_tsv artifacts/analysis/cejc_speaker_metadata.tsv \
        --out_tsv artifacts/analysis/results/ensemble_perm_groupkfold/ \
                  ensemble_summary_groupkfold.tsv \
        --n_perm 5000 --alpha 100 --cv_folds 5 --seed 42

    # sensitivity to the regularisation parameter (GroupKFold only, all five traits)
    python scripts/analysis/ensemble_permutation_groupkfold.py \
        --alpha_sweep 10,50,100,200,500 --n_perm 5000
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler

TRAITS = ["O", "C", "E", "A", "N"]

# 19 features = Classical 10 + Novel 9. Matches the feature definition table.
ALL_FEATURES = [
    # Classical (10)
    "PG_speech_ratio", "PG_pause_mean", "PG_pause_p50", "PG_pause_p90",
    "PG_resp_gap_mean", "PG_resp_gap_p50", "PG_resp_gap_p90", "PG_overlap_rate",
    "FILL_has_any", "FILL_rate_per_100chars",
    # Novel (9)
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


def cv_ridge_r(X, y, splits, seed, alpha, groups=None, return_oof=False):
    """Return the mean per-fold Pearson r (the aggregation used elsewhere here).

    With ``return_oof=True`` the function returns
    ``(fold-averaged r, out-of-fold prediction vector)`` instead, which is what the
    pooled metrics (pooled r, R^2, RMSE) are computed from.

    Ridge shrinks the predictions toward the mean, so a correlation alone is a weak
    summary of fit and the RMSE should be reported next to it. The fold-averaged r
    correlates within each fold and then averages, so it does not notice offsets
    between folds; the pooled metrics do.
    """
    if groups is not None:
        split_iter = GroupKFold(n_splits=splits).split(X, y, groups)
    else:
        split_iter = KFold(n_splits=splits, shuffle=True, random_state=seed).split(X)
    rs = []
    oof = np.full(len(y), np.nan) if return_oof else None
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
        pred = m.predict(Xte)
        rs.append(pearsonr(yte, pred))
        if return_oof:
            oof[te] = pred
    r_foldmean = float(np.mean(rs))
    if return_oof:
        return r_foldmean, oof
    return r_foldmean


def pooled_metrics(y, oof) -> dict[str, float]:
    """Metrics computed from the concatenated out-of-fold predictions.

    - ``r_oof``: Pearson r over all records pooled together
    - ``R2_oof``: 1 - SSE/SST (the coefficient of determination; can be negative)
    - ``RMSE`` / ``MAE``: the size of the prediction error; read alongside SD(y)
    - ``sd_y``: the reference needed to interpret the RMSE
    """
    y = np.asarray(y, float)
    oof = np.asarray(oof, float)
    ok = ~np.isnan(oof)
    y, oof = y[ok], oof[ok]
    sse = float(np.sum((y - oof) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return {
        "r_oof": pearsonr(y, oof),
        "R2_oof": 1.0 - sse / sst if sst > 0 else float("nan"),
        "RMSE": float(np.sqrt(sse / len(y))),
        "MAE": float(np.mean(np.abs(y - oof))),
        "sd_y": float(y.std(ddof=1)),
    }


def run_perm(X, y, splits, seed, alpha, n_perm, groups=None):
    r_obs = cv_ridge_r(X, y, splits, seed, alpha, groups)
    rng = np.random.default_rng(seed)
    r_perm = np.empty(n_perm, float)
    for i in range(n_perm):
        r_perm[i] = cv_ridge_r(X, rng.permutation(y), splits, seed, alpha, groups)
    p = (np.sum(np.abs(r_perm) >= abs(r_obs)) + 1.0) / (n_perm + 1.0)
    return r_obs, float(p)


def holm_correction(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni correction, preserving the input order."""
    m = len(p_values)
    if m == 0:
        return []
    for i, p in enumerate(p_values):
        if math.isnan(p):
            continue
        if p < 0.0 or p > 1.0:
            raise ValueError(f"p_values[{i}] = {p} is outside [0, 1]")
    non_nan = [(i, p) for i, p in enumerate(p_values) if not math.isnan(p)]
    corrected = [float("nan")] * m
    if not non_nan:
        return corrected
    m_eff = len(non_nan)
    cummax = 0.0
    for rank, (orig_idx, p) in enumerate(sorted(non_nan, key=lambda x: x[1])):
        cummax = max(cummax, p * (m_eff - rank))
        corrected[orig_idx] = min(cummax, 1.0)
    return corrected


CONFOUND_COLS = ["confound_gender", "confound_age"]


def load_trait_data(datasets_dir: str, meta: pd.DataFrame, trait: str,
                    include_confounds: bool = False):
    """Return (X, y, groups, feature_names) for one trait; rows with NaN y dropped.

    With ``include_confounds=True`` the design matrix has 21 columns: the 19
    interactional features plus speaker sex (M=0 / F=1) and age. The coding matches
    ``confound_analysis_groupkfold.py``. This confound-controlled design is the one
    reported as the headline result: rather than comparing two models to check for
    confounding, sex and age are entered as predictors and the resulting model is
    reported directly.
    """
    fpath = Path(datasets_dir) / f"cejc_home2_hq1_XY_{trait}only_ensemble.parquet"
    if not fpath.exists():
        raise SystemExit(f"dataset not found: {fpath}")
    df = pd.read_parquet(fpath).replace([np.inf, -np.inf], np.nan)

    meta_cols = ["conversation_id", "speaker_id", "cejc_person_id"]
    if include_confounds:
        meta_cols += ["gender", "age"]
    merged = df.merge(meta[meta_cols], on=["conversation_id", "speaker_id"], how="left")

    missing = [c for c in ALL_FEATURES if c not in merged.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")
    if merged["cejc_person_id"].isna().any():
        raise SystemExit("some records could not be joined to a cejc_person_id")

    feature_names = list(ALL_FEATURES)
    if include_confounds:
        merged["confound_gender"] = merged["gender"].map({"M": 0, "F": 1}).astype(float)
        merged["confound_age"] = pd.to_numeric(merged["age"], errors="coerce")
        feature_names = feature_names + CONFOUND_COLS

    y = merged[f"Y_{trait}"].astype(float).to_numpy()
    X = merged[feature_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    groups = merged["cejc_person_id"].to_numpy()
    ok = ~np.isnan(y)
    return X[ok], y[ok], groups[ok], feature_names


def run_alpha_sweep(args, meta: pd.DataFrame, alphas: list[float]) -> None:
    """Sensitivity to the regularisation parameter (GroupKFold, all five traits)."""
    rows = []
    for trait in TRAITS:
        X, y, groups, _ = load_trait_data(args.datasets_dir, meta, trait,
                                          args.include_confounds)
        for a in alphas:
            t0 = time.time()
            r_obs, p = run_perm(X, y, args.cv_folds, args.seed, a,
                                args.n_perm, groups=groups)
            print(f"[{trait}] alpha={a:g}: r={r_obs:.4f} p={p:.4f} "
                  f"({time.time() - t0:.1f}s)")
            rows.append({"trait": trait, "alpha": a,
                         "r_obs": round(r_obs, 4), "p_value": round(p, 4)})
    result = pd.DataFrame(rows)
    # Holm-correct within each alpha condition (five parallel tests per condition)
    holm = []
    for a in alphas:
        sub = result[result["alpha"] == a]
        corr = holm_correction(sub["p_value"].tolist())
        holm.extend(zip(sub.index.tolist(), [round(v, 4) for v in corr]))
    result["p_holm"] = pd.Series(dict(holm))
    out_path = Path(args.alpha_sweep_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, sep="\t", index=False)
    print(f"\nWrote {out_path}")
    print(result.to_string(index=False))


def main():
    ap = argparse.ArgumentParser(
        description="Ensemble Big5 permutation test with subject-wise (GroupKFold) CV"
    )
    ap.add_argument("--datasets_dir", default="artifacts/analysis/datasets")
    ap.add_argument("--metadata_tsv",
                    default="artifacts/analysis/cejc_speaker_metadata.tsv")
    ap.add_argument(
        "--out_tsv",
        default="artifacts/analysis/results/ensemble_perm_groupkfold/"
                "ensemble_summary_groupkfold.tsv",
    )
    ap.add_argument("--alpha", type=float, default=100.0)
    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--n_perm", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--alpha_sweep", default="",
        help="comma-separated alphas; runs the alpha sensitivity analysis "
             "(GroupKFold only) and writes it to --alpha_sweep_tsv",
    )
    ap.add_argument(
        "--alpha_sweep_tsv",
        default="artifacts/analysis/results/ensemble_perm_groupkfold/"
                "sensitivity_alpha_groupkfold.tsv",
    )
    ap.add_argument(
        "--include_confounds", action="store_true",
        help="run with 21 predictors: the 19 features plus speaker sex and age "
             "(the confound-controlled design). This is the headline result.",
    )
    args = ap.parse_args()

    meta = pd.read_csv(args.metadata_tsv, sep="\t")

    if args.alpha_sweep:
        alphas = [float(a) for a in args.alpha_sweep.split(",") if a.strip()]
        run_alpha_sweep(args, meta, alphas)
        return

    rows = []
    for trait in TRAITS:
        X, y, groups, _ = load_trait_data(args.datasets_dir, meta, trait,
                                          args.include_confounds)
        print(f"[{trait}] N={len(y)}, features={X.shape[1]}, "
              f"speakers={len(np.unique(groups))}")
        t0 = time.time()
        r_kf, p_kf = run_perm(X, y, args.cv_folds, args.seed, args.alpha,
                              args.n_perm, groups=None)
        r_gkf, p_gkf = run_perm(X, y, args.cv_folds, args.seed, args.alpha,
                                args.n_perm, groups=groups)
        # The pooled metrics are not used for inference (the test statistic is the
        # fold-averaged r); they are computed so they can be reported alongside it.
        _, oof = cv_ridge_r(X, y, args.cv_folds, args.seed, args.alpha,
                            groups=groups, return_oof=True)
        pm = pooled_metrics(y, oof)
        print(f"  KFold      : r={r_kf:.4f} p={p_kf:.4f}")
        print(f"  GroupKFold : r={r_gkf:.4f} p={p_gkf:.4f} "
              f"(dr={r_gkf - r_kf:+.4f}, {time.time() - t0:.1f}s)")
        print(f"  pooled     : r_oof={pm['r_oof']:.4f} R2={pm['R2_oof']:+.4f} "
              f"RMSE={pm['RMSE']:.4f} (SD(y)={pm['sd_y']:.4f})")
        rows.append({
            "trait": trait,
            "n": int(len(y)),
            "n_features": int(X.shape[1]),
            "r_kfold": round(r_kf, 4),
            "p_kfold": round(p_kf, 4),
            "r_groupkfold": round(r_gkf, 4),
            "p_groupkfold": round(p_gkf, 4),
            "delta_r": round(r_gkf - r_kf, 4),
            "r_oof": round(pm["r_oof"], 4),
            "R2_oof": round(pm["R2_oof"], 4),
            "RMSE": round(pm["RMSE"], 4),
            "MAE": round(pm["MAE"], 4),
            "sd_y": round(pm["sd_y"], 4),
        })

    result = pd.DataFrame(rows)
    result["p_kfold_holm"] = [
        round(v, 4) for v in holm_correction(result["p_kfold"].tolist())
    ]
    result["p_groupkfold_holm"] = [
        round(v, 4) for v in holm_correction(result["p_groupkfold"].tolist())
    ]
    result["sig_groupkfold_holm"] = result["p_groupkfold_holm"] < 0.05

    out_path = Path(args.out_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, sep="\t", index=False)
    print(f"\nWrote {out_path}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
