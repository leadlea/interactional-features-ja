#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2 of the pipeline: build pseudo-monologues for the HQ1 target pairs.

For every (conversation_id, speaker_id) in the target-pair table, the speaker's
utterances are concatenated in chronological order into a single text block.
These blocks are the input the language models see when they answer the
IPIP-NEO-120 items; the interaction features are computed from the utterance
table separately, so the two views never share a preprocessing step.

The output is written twice: once under `--out_parquet` and once under a
version-pinned copy `--out_pinned_parquet`, accompanied by a sha256 file. All
downstream steps read the pinned copy, so a rerun of this script cannot
silently change what was scored.

Usage
-----
    python scripts/cejc/build_monologues_hq1.py \
      --utterances_parquet artifacts/_tmp_utt/cejc_utterances/part-00000.parquet \
      --target_pairs_parquet artifacts/analysis/target_pairs/cejc_home2_hq1_pairs.parquet \
      --out_parquet artifacts/cejc/monologues_cejc_home2_hq1.parquet \
      --out_pinned_parquet artifacts/cejc/monologues_cejc_home2_hq1_v1.parquet \
      --out_sha256 artifacts/cejc/monologues_cejc_home2_hq1_v1.sha256.txt \
      --out_preview_tsv artifacts/cejc/monologues_cejc_home2_hq1_preview.tsv
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

KEYS = ["conversation_id", "speaker_id"]


def char_len_no_space(texts) -> int:
    return int(sum(len(str(t).replace(" ", "").replace("\u3000", "")) for t in texts))


def build_monologues(
    utterances: pd.DataFrame, target_pairs: pd.DataFrame
) -> pd.DataFrame:
    u = utterances.copy()
    for k in KEYS:
        u[k] = u[k].astype(str)
    u["text"] = u["text"].fillna("").astype(str)

    pairs = target_pairs[KEYS].drop_duplicates().copy()
    for k in KEYS:
        pairs[k] = pairs[k].astype(str)

    uu = u.merge(pairs, on=KEYS, how="inner")
    if uu.empty:
        raise SystemExit(
            "[ERROR] no utterances matched the target pairs; check the key dtypes"
        )

    # Stable chronological order within each (conversation, speaker).
    uu = uu.sort_values(
        ["conversation_id", "speaker_id", "start_time", "end_time"], kind="mergesort"
    )
    return uu.groupby(KEYS, as_index=False).agg(
        n_utt=("text", "count"),
        n_chars=("text", char_len_no_space),
        text=("text", lambda s: "\n".join(s.tolist()).strip()),
    )


def pin_with_sha256(src: Path, pinned: Path, sha_path: Path) -> str:
    pinned.parent.mkdir(parents=True, exist_ok=True)
    pinned.write_bytes(src.read_bytes())
    digest = hashlib.sha256(pinned.read_bytes()).hexdigest()
    sha_path.parent.mkdir(parents=True, exist_ok=True)
    sha_path.write_text(f"sha256 {digest}\n", encoding="utf-8")
    return digest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--utterances_parquet", required=True)
    ap.add_argument("--target_pairs_parquet", required=True)
    ap.add_argument("--out_parquet", required=True)
    ap.add_argument("--out_pinned_parquet", default=None,
                    help="version-pinned copy read by downstream steps")
    ap.add_argument("--out_sha256", default=None,
                    help="sha256 of the pinned copy")
    ap.add_argument("--out_preview_tsv", default=None)
    ap.add_argument("--preview_rows", type=int, default=200)
    args = ap.parse_args()

    utterances = pd.read_parquet(args.utterances_parquet)
    target_pairs = pd.read_parquet(args.target_pairs_parquet)

    mono = build_monologues(utterances, target_pairs)
    out = Path(args.out_parquet)
    out.parent.mkdir(parents=True, exist_ok=True)
    mono.to_parquet(out, index=False)
    print(f"OK: {out}  rows={len(mono)}")

    if args.out_pinned_parquet and args.out_sha256:
        digest = pin_with_sha256(out, Path(args.out_pinned_parquet), Path(args.out_sha256))
        print(f"OK: {args.out_pinned_parquet}  sha256={digest[:12]}...")
    elif args.out_pinned_parquet or args.out_sha256:
        raise SystemExit(
            "[ERROR] --out_pinned_parquet and --out_sha256 must be given together"
        )

    if args.out_preview_tsv:
        prev = mono.copy()
        prev["head200"] = (
            prev["text"].str.replace("\n", " ", regex=False).str.slice(0, 200)
        )
        path = Path(args.out_preview_tsv)
        path.parent.mkdir(parents=True, exist_ok=True)
        prev.sort_values("n_chars", ascending=False).head(args.preview_rows)[
            ["conversation_id", "speaker_id", "n_chars", "n_utt", "head200"]
        ].to_csv(path, sep="\t", index=False)
        print(f"OK: {path}")


if __name__ == "__main__":
    main()
