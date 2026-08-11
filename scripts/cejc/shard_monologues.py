#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3 of the pipeline: split the pinned monologue table into fixed-size shards.

Scoring runs against the Bedrock API and can fail part-way through (throttling,
transient errors). Sharding makes the scoring step resumable: the runner skips a
shard whose output already exists, so a rerun costs only the missing shards.

Shards are cut on row order, not at random, so the split is deterministic and a
rerun reproduces byte-identical shard boundaries.

Usage
-----
    python scripts/cejc/shard_monologues.py \
      --monologues_parquet artifacts/cejc/monologues_cejc_home2_hq1_v1.parquet \
      --out_dir artifacts/cejc/shards_home2_hq1 \
      --shard_size 10
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--monologues_parquet", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--shard_size", type=int, default=10)
    ap.add_argument(
        "--prefix",
        default=None,
        help="shard filename prefix (default: stem of --monologues_parquet)",
    )
    args = ap.parse_args()

    src = Path(args.monologues_parquet)
    df = pd.read_parquet(src)
    if df.empty:
        raise SystemExit(f"[ERROR] {src} contains no rows")
    if args.shard_size < 1:
        raise SystemExit("[ERROR] --shard_size must be >= 1")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or src.stem

    n_shards = math.ceil(len(df) / args.shard_size)
    for i in range(n_shards):
        chunk = df.iloc[i * args.shard_size : (i + 1) * args.shard_size]
        chunk.to_parquet(out_dir / f"{prefix}_shard{i:03d}.parquet", index=False)

    print(
        f"OK: {out_dir}  rows={len(df)}  shards={n_shards}  shard_size={args.shard_size}"
    )


if __name__ == "__main__":
    main()
