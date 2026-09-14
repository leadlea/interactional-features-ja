#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Recompute the three baseline conditions under a subject-wise split.

Why this exists
---------------
The baseline validation (three conditions) was previously computed with
``ensemble_permutation.py``, which uses a plain KFold. Once the headline analysis
was unified to GroupKFold (a subject-wise split), Condition 1 no longer matched
the main result. A $\Delta r$ comparison across conditions is only meaningful if
all three share the same CV design, so all three are recomputed here with
GroupKFold.

Conditions
----------
- Condition 1 (full text):      ``...__teacher={teacher}``
- Condition 2 (summary only):   ``...__teacher={teacher}__condition=summary``
- Condition 3 (random text):    ``...__teacher={teacher}__condition=random``

For each condition the four models' trait scores are averaged, and ridge
regression on the 19 interactional features ($\alpha=100$) is evaluated with a
5-fold GroupKFold over ``cejc_person_id``, a permutation test (5,000 by default)
and a Holm correction across the five traits.

Usage
-----
    PYTHONPATH=scripts/analysis python \
        scripts/analysis/baseline_conditions_groupkfold.py --n_perm 5000
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ensemble_permutation_groupkfold import (  # noqa: E402
    ALL_FEATURES, TRAITS, holm_correction, run_perm,
)

TEACHERS = ["sonnet4", "qwen3-235b", "deepseek-v3", "gpt-oss-120b"]
CONDITIONS = {
    "cond1_text": "",
    "cond2_summary": "__condition=summary",
    "cond3_random": "__condition=random",
}


def load_condition_scores(items_dir: Path, trait: str, suffix: str) -> pd.DataFrame:
    """Average the four models' trait scores for one condition and one trait."""
    frames = []
    for teacher in TEACHERS:
        d = items_dir / (
            f"dataset=cejc_home2_hq1_v1__items={trait}24__teacher={teacher}{suffix}"
        )
        # The main condition stores scores under teacher_merged/; conditions 2 and 3
        # write trait_scores.parquet directly in the run directory.
        candidates = [
            d / "teacher_merged" / f"trait_scores_{trait}_merged.parquet",
            d / "trait_scores.parquet",
        ]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            raise FileNotFoundError(f"trait_scores not found under {d}")
        df = pd.read_parquet(path)[["conversation_id", "speaker_id", "trait_score"]]
        frames.append(df.rename(columns={"trait_score": f"score_{teacher}"}))
    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on=["conversation_id", "speaker_id"], how="inner")
    score_cols = [c for c in merged.columns if c.startswith("score_")]
    merged["y"] = merged[score_cols].mean(axis=1)
    return merged[["conversation_id", "speaker_id", "y"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items_dir", default="artifacts/big5/llm_scores")
    ap.add_argument(
        "--features_parquet",
        default="artifacts/analysis/features_min/features_cejc_home2_hq1.parquet",
    )
    ap.add_argument("--metadata_tsv",
                    default="artifacts/analysis/cejc_speaker_metadata.tsv")
    ap.add_argument(
        "--out_tsv",
        default="artifacts/analysis/results/baseline_validation/"
                "baseline_conditions_groupkfold.tsv",
    )
    ap.add_argument("--alpha", type=float, default=100.0)
    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--n_perm", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--include_confounds", action="store_true",
        help="run with 21 predictors, adding speaker sex and age (the design used "
             "for the headline result). Condition 1 has to match the headline "
             "result, otherwise the delta r across conditions is not comparable.",
    )
    args = ap.parse_args()

    feat = pd.read_parquet(args.features_parquet).replace([np.inf, -np.inf], np.nan)
    meta = pd.read_csv(args.metadata_tsv, sep="\t")

    meta_cols = ["conversation_id", "speaker_id", "cejc_person_id"]
    feature_cols = list(ALL_FEATURES)
    if args.include_confounds:
        meta_cols += ["gender", "age"]
        feature_cols += ["confound_gender", "confound_age"]

    rows = []
    for cond_name, suffix in CONDITIONS.items():
        for trait in TRAITS:
            scores = load_condition_scores(Path(args.items_dir), trait, suffix)
            merged = (
                scores.merge(feat, on=["conversation_id", "speaker_id"], how="inner")
                      .merge(meta[meta_cols],
                             on=["conversation_id", "speaker_id"], how="left")
            )
            if args.include_confounds:
                merged["confound_gender"] = (
                    merged["gender"].map({"M": 0, "F": 1}).astype(float))
                merged["confound_age"] = pd.to_numeric(merged["age"], errors="coerce")
            missing = [c for c in feature_cols if c not in merged.columns]
            if missing:
                raise KeyError(f"Missing feature columns: {missing}")
            y = merged["y"].astype(float).to_numpy()
            X = merged[feature_cols].apply(pd.to_numeric, errors="coerce") \
                                    .to_numpy(dtype=float)
            groups = merged["cejc_person_id"].to_numpy()
            ok = ~np.isnan(y)
            X, y, groups = X[ok], y[ok], groups[ok]

            t0 = time.time()
            r_obs, p = run_perm(X, y, args.cv_folds, args.seed, args.alpha,
                                args.n_perm, groups=groups)
            print(f"[{cond_name}][{trait}] N={len(y)} r={r_obs:.4f} p={p:.4f} "
                  f"({time.time() - t0:.1f}s)")
            rows.append({"condition": cond_name, "trait": trait, "n": int(len(y)),
                         "r_obs": round(r_obs, 4), "p_value": round(p, 4)})

    result = pd.DataFrame(rows)
    # Holm-correct across the five traits within each condition
    parts = []
    for cond, sub in result.groupby("condition", sort=False):
        sub = sub.copy()
        sub["p_holm"] = [round(v, 4) for v in holm_correction(sub["p_value"].tolist())]
        parts.append(sub)
    result = pd.concat(parts, ignore_index=True)

    # delta r relative to Condition 1 (Condition 1 minus the condition in question)
    base = result[result["condition"] == "cond1_text"].set_index("trait")["r_obs"]
    result["delta_r_vs_cond1"] = [
        round(float(base[t]) - float(r), 4)
        for t, r in zip(result["trait"], result["r_obs"])
    ]

    out_path = Path(args.out_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, sep="\t", index=False)
    print(f"\nWrote {out_path}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
