#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the XY datasets for the confound-controlled design (19 features + sex + age).

Why this exists
---------------
The headline result is reported from an "interactional features + sex + age" model.
Rather than comparing two models to check for confounding, sex and age are entered as
predictors and that model is reported directly.

The coefficient-level analyses (``permutation_coef_test.py`` and
``bootstrap_variance.py``) take an ``--xy_parquet`` and select predictors by column
name, so once a dataset carrying the sex and age columns exists, both scripts only
need the extra column names.

The existing Model A datasets (``cejc_home2_hq1_XY_*``) are left untouched. Output goes
to separate files (``cejc_home2_hq1_XYB_*``) so the 19-predictor results remain exactly
reproducible; the appendix uses them for the cross-validation design comparison.

The coding matches ``confound_analysis_groupkfold.py``: sex is M=0 / F=1 and age is
numeric.

Usage
-----
    python scripts/analysis/build_modelb_datasets.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

TRAITS = ["O", "C", "E", "A", "N"]
CONFOUND_COLS = ["confound_gender", "confound_age"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets_dir", default="artifacts/analysis/datasets")
    ap.add_argument("--metadata_tsv",
                    default="artifacts/analysis/cejc_speaker_metadata.tsv")
    ap.add_argument("--teacher", default="ensemble")
    args = ap.parse_args()

    datasets_dir = REPO_ROOT / args.datasets_dir
    meta = pd.read_csv(REPO_ROOT / args.metadata_tsv, sep="\t")

    need = {"conversation_id", "speaker_id", "gender", "age"}
    if not need.issubset(meta.columns):
        raise SystemExit(
            f"metadata is missing columns: {sorted(need - set(meta.columns))}"
        )

    for trait in TRAITS:
        src = datasets_dir / f"cejc_home2_hq1_XY_{trait}only_{args.teacher}.parquet"
        dst = datasets_dir / f"cejc_home2_hq1_XYB_{trait}only_{args.teacher}.parquet"
        if not src.exists():
            raise SystemExit(f"source dataset not found: {src}")

        df = pd.read_parquet(src)
        n_before = len(df)
        merged = df.merge(
            meta[["conversation_id", "speaker_id", "gender", "age"]],
            on=["conversation_id", "speaker_id"], how="left",
        )
        if len(merged) != n_before:
            raise SystemExit(
                f"{trait}: the merge changed the row count ({n_before} -> {len(merged)});"
                " the metadata may contain duplicate keys"
            )

        merged["confound_gender"] = merged["gender"].map({"M": 0, "F": 1}).astype(float)
        merged["confound_age"] = pd.to_numeric(merged["age"], errors="coerce")

        n_bad = int(merged[CONFOUND_COLS].isna().any(axis=1).sum())
        if n_bad:
            print(f"  [warning] {trait}: {n_bad} row(s) have a missing sex or age"
                  " (handled by the within-fold median imputation)")

        # Keep the original columns and drop the raw gender/age, retaining only the
        # numeric encodings.
        out = merged.drop(columns=["gender", "age"])
        out.to_parquet(dst, index=False)
        print(f"  {dst.name}: {len(out)} rows / {len(out.columns)} columns"
              f" (added {'+'.join(CONFOUND_COLS)})")

    print("\n[OK] confound-controlled datasets written")
    print("  run the coefficient-level analyses with --include_confounds:")
    print("    bash scripts/analysis/run_coef_bootstrap_all5_modelb.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
