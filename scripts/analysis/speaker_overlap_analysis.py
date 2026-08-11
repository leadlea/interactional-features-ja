#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Speaker overlap analysis for CEJC speaker metadata.

Analyse cejc_person_id duplicates in cejc_speaker_metadata.tsv and report:
  - Total records, unique speakers, duplicate speakers, duplicate records
  - Gender breakdown (F / M)
  - Per-duplicate-speaker detail (person_id, n_records, conversation_ids, gender)

Usage:
    python scripts/analysis/speaker_overlap_analysis.py \
        --metadata_tsv artifacts/analysis/cejc_speaker_metadata.tsv \
        --out artifacts/analysis/results
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd


# ── Core analysis ────────────────────────────────────────────────────
def compute_overlap_stats(df: pd.DataFrame) -> dict:
    """Compute speaker overlap statistics from metadata DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least ``cejc_person_id`` and ``conversation_id``.
        ``gender`` is optional (used for gender breakdown).

    Returns
    -------
    dict with keys:
        summary : dict  – aggregate counts
        detail  : pd.DataFrame – per-duplicate-speaker rows
    """
    if "cejc_person_id" not in df.columns:
        available = ", ".join(df.columns.tolist())
        raise KeyError(
            f"Column 'cejc_person_id' not found. Available columns: {available}"
        )
    if "conversation_id" not in df.columns:
        available = ", ".join(df.columns.tolist())
        raise KeyError(
            f"Column 'conversation_id' not found. Available columns: {available}"
        )

    total_records = len(df)

    # Group by cejc_person_id
    grouped = df.groupby("cejc_person_id")
    counts = grouped.size().rename("n_records")
    unique_speakers = len(counts)
    duplicate_mask = counts >= 2
    duplicate_speakers = int(duplicate_mask.sum())
    duplicate_records = int(counts[duplicate_mask].sum())

    # Gender breakdown (per unique speaker — take first occurrence)
    has_gender = "gender" in df.columns
    gender_f = 0
    gender_m = 0
    if has_gender:
        first_gender = grouped["gender"].first()
        gender_f = int((first_gender == "F").sum())
        gender_m = int((first_gender == "M").sum())
    else:
        warnings.warn(
            "Column 'gender' not found in metadata. "
            "Gender breakdown will be skipped.",
            stacklevel=2,
        )

    summary = {
        "total_records": total_records,
        "unique_speakers": unique_speakers,
        "duplicate_speakers": duplicate_speakers,
        "duplicate_records": duplicate_records,
    }
    if has_gender:
        summary["gender_F"] = gender_f
        summary["gender_M"] = gender_m

    # Detail: duplicate speakers only
    dup_ids = counts[duplicate_mask].index
    detail_rows = []
    for pid in dup_ids:
        sub = df[df["cejc_person_id"] == pid]
        conv_ids = ",".join(sorted(sub["conversation_id"].unique()))
        gender_val = sub["gender"].iloc[0] if has_gender else ""
        detail_rows.append(
            {
                "cejc_person_id": pid,
                "n_records": int(counts[pid]),
                "conversation_ids": conv_ids,
                "gender": gender_val,
            }
        )

    detail = pd.DataFrame(detail_rows)
    if not detail.empty:
        detail = detail.sort_values("n_records", ascending=False).reset_index(drop=True)

    return {"summary": summary, "detail": detail}


# ── Main ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Speaker overlap analysis for CEJC speaker metadata"
    )
    ap.add_argument(
        "--metadata_tsv",
        required=True,
        help="Path to cejc_speaker_metadata.tsv",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Output directory for speaker_overlap_report.tsv",
    )
    args = ap.parse_args()

    # ── Load metadata ────────────────────────────────────────────────
    tsv_path = Path(args.metadata_tsv)
    if not tsv_path.exists():
        raise FileNotFoundError(f"Metadata TSV not found: {tsv_path}")

    df = pd.read_csv(tsv_path, sep="\t")

    # ── Run analysis ─────────────────────────────────────────────────
    result = compute_overlap_stats(df)
    summary = result["summary"]
    detail = result["detail"]

    # ── Print summary to stdout ──────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Speaker Overlap Analysis")
    print(f"{'='*60}")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"{'='*60}\n")

    # ── Write detail TSV ─────────────────────────────────────────────
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "speaker_overlap_report.tsv"
    detail.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote {out_path}")
    if not detail.empty:
        print(detail.to_string(index=False))
    else:
        print("  (No duplicate speakers found)")


if __name__ == "__main__":
    main()
