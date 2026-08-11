#!/usr/bin/env python3
"""Reorganize condition-specific LLM score directories for ensemble_permutation.py.

score_big5_bedrock.py outputs trait_scores.parquet into directories like:
    {scores_dir}/dataset=cejc_home2_hq1_v1__items={trait}24__teacher={teacher}__condition={cond}/
        trait_scores.parquet

ensemble_permutation.py's load_trait_scores() expects:
    {items_dir}/dataset=cejc_home2_hq1_v1__items={trait}24__teacher={teacher}/
        teacher_merged/trait_scores_{trait}_merged.parquet

This script bridges the gap by reading from the condition-specific directories
and writing to the ensemble-compatible structure.

Usage:
    python scripts/baseline/prepare_ensemble_dirs.py
    python scripts/baseline/prepare_ensemble_dirs.py --scores_dir artifacts/big5/llm_scores
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd

TEACHERS = ["sonnet4", "qwen3-235b", "deepseek-v3", "gpt-oss-120b"]
TRAITS = ["O", "C", "E", "A", "N"]
CONDITIONS = {"summary", "random"}


def _source_dir(scores_dir: Path, trait: str, teacher: str, condition: str) -> Path:
    """Return the condition-specific source directory path."""
    name = (
        f"dataset=cejc_home2_hq1_v1__items={trait}24"
        f"__teacher={teacher}__condition={condition}"
    )
    return scores_dir / name


def _dest_path(out_dir: Path, trait: str, teacher: str) -> Path:
    """Return the ensemble-compatible destination parquet path."""
    name = f"dataset=cejc_home2_hq1_v1__items={trait}24__teacher={teacher}"
    return out_dir / name / "teacher_merged" / f"trait_scores_{trait}_merged.parquet"


def prepare_one(
    scores_dir: Path,
    out_dir: Path,
    condition: str,
    trait: str,
    teacher: str,
) -> Path | None:
    """Read trait_scores from a condition-specific dir and write to ensemble layout.

    Returns the destination path on success, or None if the source is missing.
    """
    src_dir = _source_dir(scores_dir, trait, teacher, condition)
    src_parquet = src_dir / "trait_scores.parquet"

    if not src_parquet.exists():
        warnings.warn(
            f"Missing source: {src_parquet} "
            f"(condition={condition}, trait={trait}, teacher={teacher})"
        )
        return None

    df = pd.read_parquet(src_parquet)

    # Validate expected columns
    expected_cols = {"conversation_id", "speaker_id", "trait", "model_id", "trait_score"}
    missing = expected_cols - set(df.columns)
    if missing:
        warnings.warn(f"Missing columns {missing} in {src_parquet}")
        return None

    # Filter to the target trait (safety check — should already be single-trait)
    if df["trait"].nunique() > 1:
        warnings.warn(
            f"Multiple traits found in {src_parquet}, filtering to {trait}"
        )
    df = df[df["trait"] == trait].copy()

    if df.empty:
        warnings.warn(f"No rows for trait={trait} in {src_parquet}")
        return None

    dest = _dest_path(out_dir, trait, teacher)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)

    return dest


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Prepare ensemble-compatible directory structure from condition-specific LLM scores."
    )
    ap.add_argument(
        "--scores_dir",
        type=str,
        default="artifacts/big5/llm_scores",
        help="Root dir with condition-specific LLM scores",
    )
    ap.add_argument(
        "--out_summary_dir",
        type=str,
        default="artifacts/big5/llm_scores_summary",
        help="Output dir for summary condition (ensemble-compatible)",
    )
    ap.add_argument(
        "--out_random_dir",
        type=str,
        default="artifacts/big5/llm_scores_random",
        help="Output dir for random condition (ensemble-compatible)",
    )
    args = ap.parse_args()

    scores_dir = Path(args.scores_dir)
    out_map = {
        "summary": Path(args.out_summary_dir),
        "random": Path(args.out_random_dir),
    }

    total = 0
    ok = 0
    missing = []

    for condition, out_dir in out_map.items():
        for trait in TRAITS:
            for teacher in TEACHERS:
                total += 1
                dest = prepare_one(scores_dir, out_dir, condition, trait, teacher)
                if dest is not None:
                    ok += 1
                    print(f"  ✓ {condition}/{trait}/{teacher} → {dest}")
                else:
                    missing.append(f"{condition}/{trait}/{teacher}")

    print(f"\n{'='*60}")
    print(f"Done: {ok}/{total} files written")
    if missing:
        print(f"Missing ({len(missing)}):")
        for m in missing:
            print(f"  ✗ {m}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
