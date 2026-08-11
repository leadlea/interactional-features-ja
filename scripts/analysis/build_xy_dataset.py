#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Join the interaction features (X) with a trait score (Y) into one analysis table.

X comes from extract_interaction_features_min.py, Y from the merged model scores.
The join is an inner join on (conversation_id, speaker_id): a record survives
only if both a feature vector and a trait score exist for it. The row count of
the result is the N reported for the analysis, so the script prints the loss at
each side of the join rather than letting it pass unnoticed.

Usage
-----
    python scripts/analysis/build_xy_dataset.py \
      --features_parquet artifacts/analysis/features_min/features_cejc_home2_hq1.parquet \
      --scores_parquet "artifacts/big5/llm_scores/dataset=.../teacher_merged/trait_scores_C_merged.parquet" \
      --trait C \
      --out_parquet artifacts/analysis/datasets/cejc_home2_hq1_XY_Conly_sonnet.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

KEYS = ["conversation_id", "speaker_id"]
SCORE_COL_CANDIDATES = ("trait_score", "score", "value")


def pick_score_column(df: pd.DataFrame) -> str:
    for c in SCORE_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise SystemExit(
        f"[ERROR] no score column found. tried={SCORE_COL_CANDIDATES} "
        f"available={list(df.columns)}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--features_parquet", required=True)
    ap.add_argument("--scores_parquet", required=True)
    ap.add_argument("--trait", required=True, choices=["O", "C", "E", "A", "N"])
    ap.add_argument("--out_parquet", required=True)
    args = ap.parse_args()

    y_col = f"Y_{args.trait}"

    x = pd.read_parquet(args.features_parquet)
    y = pd.read_parquet(args.scores_parquet)
    score_col = pick_score_column(y)

    # Keep only the requested trait when the score table holds several.
    if "trait" in y.columns:
        y = y[y["trait"].astype(str).str.strip() == args.trait]
        if y.empty:
            raise SystemExit(f"[ERROR] no rows for trait={args.trait} in {args.scores_parquet}")

    y = y[KEYS + [score_col]].rename(columns={score_col: y_col})
    for k in KEYS:
        x[k] = x[k].astype(str)
        y[k] = y[k].astype(str)

    df = x.merge(y, on=KEYS, how="inner")
    print(f"[1/2] X={len(x)} rows, Y={len(y)} rows -> joined={len(df)} rows")
    dropped_x = len(x) - len(df)
    dropped_y = len(y) - len(df)
    if dropped_x or dropped_y:
        print(f"      unmatched: {dropped_x} from X, {dropped_y} from Y")
    if df.empty:
        raise SystemExit("[ERROR] the join produced 0 rows; check the key values")

    out = Path(args.out_parquet)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"[2/2] OK: {out}  shape={df.shape}  y_col={y_col}")


if __name__ == "__main__":
    main()
