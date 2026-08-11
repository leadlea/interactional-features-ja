#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split the IPIP-NEO-120 item file into per-trait subsets (24 items each).

Each trait is scored in its own request so that one model call answers 24 items
rather than 120. This keeps every response inside the output token budget and
makes a failed trait cheap to re-run, at the cost of the model not seeing the
other traits' items in the same context.

Usage
-----
    python scripts/big5/subset_items_by_trait.py \
      --items_csv artifacts/big5/items_ipipneo120_ja.csv \
      --out_dir artifacts/big5 \
      --traits O C E A N
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

TRAIT_COL_CANDIDATES = ("trait", "Trait", "TRAIT")


def pick_trait_column(df: pd.DataFrame) -> str:
    for c in TRAIT_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise SystemExit(
        f"[ERROR] no trait column found. tried={TRAIT_COL_CANDIDATES} "
        f"available={list(df.columns)}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--items_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--traits", nargs="+", default=["O", "C", "E", "A", "N"])
    ap.add_argument(
        "--expected_per_trait",
        type=int,
        default=24,
        help="warn if a subset does not have this many items (0 disables the check)",
    )
    args = ap.parse_args()

    items = pd.read_csv(args.items_csv)
    trait_col = pick_trait_column(items)
    labels = items[trait_col].astype(str).str.strip()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.items_csv).stem

    for t in args.traits:
        subset = items[labels == t]
        if subset.empty:
            raise SystemExit(f"[ERROR] no items found for trait={t}")
        if args.expected_per_trait and len(subset) != args.expected_per_trait:
            print(
                f"[WARN] trait={t}: {len(subset)} items "
                f"(expected {args.expected_per_trait})"
            )
        out = out_dir / f"{stem}_{t}{len(subset)}.csv"
        subset.to_csv(out, index=False)
        print(f"OK: {out}  items={len(subset)}")


if __name__ == "__main__":
    main()
