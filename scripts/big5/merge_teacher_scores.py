#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Concatenate per-shard model scores into a single trait-score table.

Scoring writes one parquet per (shard, model). This step globs them back
together and checks that every record was scored exactly once, which is the
failure mode that a resumable shard runner can silently produce (a shard skipped
because a stale output file existed).

Usage
-----
    python scripts/big5/merge_teacher_scores.py \
      --scores_root "artifacts/big5/llm_scores/dataset=cejc_home2_hq1_v1__items=C24__teacher=sonnet4" \
      --out_parquet "artifacts/big5/llm_scores/dataset=.../teacher_merged/trait_scores_C_merged.parquet"

Exits non-zero if any (conversation_id, speaker_id, trait) appears more than
once, unless --allow_duplicates is given.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

KEYS = ["conversation_id", "speaker_id"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--scores_root", required=True)
    ap.add_argument("--out_parquet", required=True)
    ap.add_argument(
        "--glob",
        default="shard=*/model=*/trait_scores.parquet",
        help="pattern relative to --scores_root",
    )
    ap.add_argument("--expected_records", type=int, default=0,
                    help="if >0, fail unless this many unique records are present")
    ap.add_argument("--allow_duplicates", action="store_true")
    args = ap.parse_args()

    root = Path(args.scores_root)
    parts = sorted(root.glob(args.glob))
    if not parts:
        raise SystemExit(f"[ERROR] no score files matched {root}/{args.glob}")

    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    print(f"[1/2] merged {len(parts)} shard files -> {len(df)} rows")

    traits = sorted(df["trait"].astype(str).unique().tolist())
    n_records = df[KEYS].drop_duplicates().shape[0]
    print(f"      traits={traits}  unique_records={n_records}")

    dup_keys = KEYS + (["trait"] if "trait" in df.columns else [])
    dups = df[df.duplicated(subset=dup_keys, keep=False)]
    if not dups.empty:
        msg = (
            f"[{'WARN' if args.allow_duplicates else 'ERROR'}] "
            f"{len(dups)} duplicated rows on {dup_keys}"
        )
        print(msg)
        if not args.allow_duplicates:
            print(dups[dup_keys].drop_duplicates().head(10).to_string(index=False))
            raise SystemExit(1)

    if args.expected_records and n_records != args.expected_records:
        raise SystemExit(
            f"[ERROR] expected {args.expected_records} unique records, got {n_records}"
        )

    out = Path(args.out_parquet)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"[2/2] OK: {out}")


if __name__ == "__main__":
    main()
