#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Estimate what magnitude of association this design is able to detect.

Why this exists
---------------
The submission affirmation for Behavior Research Methods asks that "the manuscript
includes appropriate measures of variability, effect size, and (when relevant)
statistical power", and the Statistical Guidelines add that "the Method section should
make clear what criteria were used to determine the sample size".

The N=120 in this study is every record that passed the eligibility filter (adjacency
pairs >= 80, characters >= 2,000, post-question pairs >= 10); the sample size was not
fixed by an a priori power calculation. The guidelines allow for this:

  "If there is no smallest effect size of interest for an a priori power analysis,
   then authors can report the effect size at which the design and test have 50% power"

This script measures that effect size.

What it does
------------
It holds the observed design matrix X (its real correlation structure) and the real
speaker grouping fixed, generates the outcome synthetically, and runs the same pipeline
as the main analysis (median imputation -> standardisation -> Ridge(alpha=100) ->
5-fold GroupKFold).

1. **Null distribution**: draw y independent of X ``n_null`` times and collect the
   fold-averaged Pearson r. This is how large r gets when there is no association. The
   two-sided 5% point becomes the critical value ``r_crit``. Because the permutation
   test shuffles only the outcome, this is the critical value of that test.
2. **Power**: sweep the population multiple correlation rho over a grid, generate
   y = X b + e with Var(Xb)/Var(y) = rho^2 ``n_sim`` times, and count how often r
   exceeds ``r_crit``. The direction of b is redrawn every time so that no particular
   direction is favoured.
3. Linearly interpolate the rho at which power reaches 50% and 80%.

Holm correction
---------------
The manuscript applies a Holm correction across the five dimensions. The least
favourable dimension is effectively judged at close to alpha/5, so both alpha = 0.05 and
alpha = 0.01 are reported: the former for a single dimension tested on its own, the
latter as a guide for the least favourable position among five.

Note
----
This is not a post hoc power analysis. Back-calculating power from an observed effect
size is explicitly rejected by the guidelines ("not post hoc power"). What is computed
here is the sensitivity of the design, which does not depend on the observed values.

Usage
-----
    python scripts/analysis/power_analysis_design.py
    python scripts/analysis/power_analysis_design.py --include_confounds
    python scripts/analysis/power_analysis_design.py --n_null 4000 --n_sim 800
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# The same 19 features as the manuscript, in the same order as gen_coef_all5.py and
# confound_analysis_groupkfold.py.
ALL_FEATURES = [
    "PG_speech_ratio", "PG_pause_mean", "PG_pause_p50", "PG_pause_p90",
    "PG_resp_gap_mean", "PG_resp_gap_p50", "PG_resp_gap_p90", "PG_overlap_rate",
    "FILL_has_any", "FILL_rate_per_100chars",
    "IX_oirmarker_rate", "IX_oirmarker_after_question_rate",
    "IX_yesno_rate", "IX_yesno_after_question_rate", "IX_lex_overlap_mean",
    "RESP_NE_AIZUCHI_RATE", "RESP_NE_ENTROPY", "RESP_YO_ENTROPY",
    "PG_pause_variability",
]


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    den = np.sqrt((a * a).sum()) * np.sqrt((b * b).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")


def make_pipeline(alpha: float) -> Pipeline:
    """The same estimator and preprocessing as the manuscript.

    The imputer and the scaler are fitted on the training fold only.
    """
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])


def fold_mean_r(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                alpha: float, n_splits: int) -> float:
    """Mean per-fold Pearson r; the same aggregation as the reported r_obs."""
    gkf = GroupKFold(n_splits=n_splits)
    rs = []
    for tr, te in gkf.split(X, y, groups):
        pipe = make_pipeline(alpha).fit(X[tr], y[tr])
        pred = pipe.predict(X[te])
        r = pearson(y[te], pred)
        if np.isfinite(r):
            rs.append(r)
    return float(np.mean(rs)) if rs else float("nan")


def load_design(datasets_dir: Path, metadata_tsv: Path, trait: str,
                include_confounds: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Take X and the speaker groups from the real data; the outcome is not used.

    With ``include_confounds=True`` sex and age are added, giving 21 predictors. The
    headline result uses that design, so the power has to be estimated for it as well:
    adding predictors changes the spread of the null distribution.
    """
    pq = datasets_dir / f"cejc_home2_hq1_XY_{trait}only_ensemble.parquet"
    df = pd.read_parquet(pq).replace([np.inf, -np.inf], np.nan)
    meta = pd.read_csv(metadata_tsv, sep="\t")
    meta_cols = ["conversation_id", "speaker_id", "cejc_person_id"]
    if include_confounds:
        meta_cols += ["gender", "age"]
    merged = df.merge(meta[meta_cols], on=["conversation_id", "speaker_id"], how="left")
    if merged["cejc_person_id"].isna().any():
        n = int(merged["cejc_person_id"].isna().sum())
        raise SystemExit(f"{n} record(s) have no speaker id; check the metadata.")

    cols = list(ALL_FEATURES)
    if include_confounds:
        merged["confound_gender"] = merged["gender"].map({"M": 0, "F": 1}).astype(float)
        merged["confound_age"] = pd.to_numeric(merged["age"], errors="coerce")
        cols += ["confound_gender", "confound_age"]

    X = merged[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    groups = merged["cejc_person_id"].to_numpy()
    return X, groups


def null_distribution(X: np.ndarray, groups: np.ndarray, alpha: float,
                      n_splits: int, n_null: int,
                      rng: np.random.Generator) -> np.ndarray:
    """Distribution of the fold-averaged r under a y independent of X."""
    n = X.shape[0]
    out = np.empty(n_null)
    for i in range(n_null):
        y = rng.standard_normal(n)
        out[i] = fold_mean_r(X, y, groups, alpha, n_splits)
    return out


def simulate_r(X: np.ndarray, groups: np.ndarray, rho: float,
               alpha: float, n_splits: int, n_sim: int,
               rng: np.random.Generator) -> np.ndarray:
    """Return n_sim fold-averaged r values observed at population correlation rho.

    y = Xz b + e, with b drawn in a random direction and the noise variance set so that
    Var(Xz b) / Var(y) = rho^2. Xz is X standardised column-wise, with missing values
    filled by the column median.

    The critical value is applied to the returned draws afterwards. That way several
    alpha levels can reuse the same random draws, so comparisons between levels are not
    contaminated by simulation noise.
    """
    Xz = X.copy()
    for j in range(Xz.shape[1]):
        col = Xz[:, j]
        med = np.nanmedian(col)
        col = np.where(np.isnan(col), med, col)
        sd = col.std()
        Xz[:, j] = (col - col.mean()) / sd if sd > 0 else 0.0

    n, p = Xz.shape
    rs = []
    for _ in range(n_sim):
        b = rng.standard_normal(p)
        signal = Xz @ b
        sd_sig = signal.std()
        if sd_sig == 0:
            continue
        signal = signal / sd_sig
        # From rho^2 = Var(signal) / (Var(signal) + Var(noise)).
        noise_sd = np.sqrt((1.0 - rho ** 2) / rho ** 2) if rho > 0 else np.inf
        y = signal + rng.standard_normal(n) * noise_sd
        rs.append(fold_mean_r(X, y, groups, alpha, n_splits))
    return np.asarray(rs, float)


def interpolate_rho(rhos: list[float], powers: list[float],
                    target: float) -> float | None:
    """Linearly interpolate the rho at which power crosses the target."""
    for i in range(1, len(rhos)):
        p0, p1 = powers[i - 1], powers[i]
        if (p0 - target) * (p1 - target) <= 0 and p1 != p0:
            w = (target - p0) / (p1 - p0)
            return rhos[i - 1] + w * (rhos[i] - rhos[i - 1])
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets_dir", default="artifacts/analysis/datasets")
    ap.add_argument("--metadata_tsv",
                    default="artifacts/analysis/cejc_speaker_metadata.tsv")
    ap.add_argument("--trait", default="C",
                    help="which trait's dataset to borrow the design from "
                         "(X is the same for every trait)")
    ap.add_argument("--alpha", type=float, default=100.0)
    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--n_null", type=int, default=3000)
    ap.add_argument("--n_sim", type=int, default=500)
    ap.add_argument("--rho_grid",
                    default="0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.60")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include_confounds", action="store_true",
                    help="estimate for 21 predictors, adding sex and age "
                         "(the design used for the headline result)")
    ap.add_argument("--out_json", default="",
                    help="defaults to power_analysis_design.json, or "
                         "power_analysis_design_modelB.json with --include_confounds")
    args = ap.parse_args()
    if not args.out_json:
        # Pick the default path from the design so that a Model B run cannot silently
        # overwrite the Model A numbers.
        suffix = "_modelB" if args.include_confounds else ""
        args.out_json = (
            f"artifacts/analysis/results/power_analysis_design{suffix}.json"
        )

    datasets_dir = REPO_ROOT / args.datasets_dir
    metadata_tsv = REPO_ROOT / args.metadata_tsv
    X, groups = load_design(datasets_dir, metadata_tsv, args.trait,
                            args.include_confounds)
    n, p = X.shape
    n_speakers = len(np.unique(groups))
    print(f"design: N={n} records / {p} predictors / {n_speakers} speakers / "
          f"{args.cv_folds}-fold GroupKFold / Ridge alpha={args.alpha}")

    rng = np.random.default_rng(args.seed)

    print(f"\nbuilding the null distribution ({args.n_null} iterations)...")
    null = null_distribution(X, groups, args.alpha, args.cv_folds, args.n_null, rng)
    null = null[np.isfinite(null)]
    # The test is two-sided, so the critical value comes from the distribution of |r|.
    crit = {
        "0.05": float(np.quantile(np.abs(null), 0.95)),
        "0.01": float(np.quantile(np.abs(null), 0.99)),
    }
    print(f"  null r: mean={null.mean():+.4f}, SD={null.std():.4f}, "
          f"|r| 95th={crit['0.05']:.4f}, 99th={crit['0.01']:.4f}")

    rho_grid = [float(x) for x in args.rho_grid.split(",")]

    # Simulate once per rho and apply both alpha levels to the same draws.
    print(f"\nsimulating r at each rho ({args.n_sim} iterations)...")
    draws: dict[float, np.ndarray] = {}
    med_rs: list[float] = []
    for rho in rho_grid:
        rs = simulate_r(X, groups, rho, args.alpha, args.cv_folds, args.n_sim, rng)
        draws[rho] = rs
        med_rs.append(float(np.nanmedian(rs)))
        print(f"  rho={rho:.2f}  median observed r={med_rs[-1]:+.3f}")

    results: dict[str, dict] = {}
    for level, r_crit in crit.items():
        powers = [float(np.mean(draws[rho][np.isfinite(draws[rho])] > r_crit))
                  for rho in rho_grid]
        rho50 = interpolate_rho(rho_grid, powers, 0.50)
        rho80 = interpolate_rho(rho_grid, powers, 0.80)
        results[level] = {
            "r_crit": r_crit,
            "rho_grid": rho_grid,
            "power": powers,
            "median_r_obs": med_rs,
            "rho_at_power_50": rho50,
            "rho_at_power_80": rho80,
        }
        print(f"\npower (two-sided alpha={level}, critical value r>{r_crit:.4f})")
        for rho, pw in zip(rho_grid, powers):
            print(f"  rho={rho:.2f}  power={pw:5.1%}")
        s50 = f"{rho50:.3f}" if rho50 else "outside the grid"
        s80 = f"{rho80:.3f}" if rho80 else "outside the grid"
        print(f"  -> 50% power: rho={s50} / 80% power: rho={s80}")

    payload = {
        "design": {
            "n_records": n, "n_predictors": p, "n_speakers": n_speakers,
            "cv": f"{args.cv_folds}-fold GroupKFold (cejc_person_id)",
            "ridge_alpha": args.alpha, "seed": args.seed,
            "n_null": int(len(null)), "n_sim": args.n_sim,
        },
        "null": {"mean": float(null.mean()), "sd": float(null.std())},
        "levels": results,
    }
    out = REPO_ROOT / args.out_json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
