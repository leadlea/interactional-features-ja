#!/usr/bin/env python3
"""Reorganize dose-response LLM score directories for ensemble_permutation.py.

score_big5_bedrock_v2.py outputs trait_scores.parquet into directories like:
    {scores_dir}/dataset=cejc_home2_hq1_v1__items={trait}24__teacher={teacher}__dose={feature}_x{level}/
        trait_scores.parquet

ensemble_permutation.py's load_trait_scores() expects:
    {items_dir}/dataset=cejc_home2_hq1_v1__items={trait}24__teacher={teacher}/
        teacher_merged/trait_scores_{trait}_merged.parquet

This script bridges the gap by reading from the dose-specific directories
and writing to the ensemble-compatible structure, one items_dir per dose level.

Usage:
    python scripts/dose_response/prepare_ensemble_dirs.py \
        --scores_dir artifacts/big5/llm_scores \
        --dose_levels 0,3 \
        --target_feature FILL \
        --out_dir artifacts/big5

    # Result:
    #   artifacts/big5/llm_scores_dose_FILL_x0/
    #     dataset=cejc_home2_hq1_v1__items=C24__teacher=sonnet4/
    #       teacher_merged/trait_scores_C_merged.parquet
    #   artifacts/big5/llm_scores_dose_FILL_x3/
    #     ...
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd

TEACHERS = ["sonnet4", "qwen3-235b", "deepseek-v3", "gpt-oss-120b"]
TRAITS = ["O", "C", "E", "A", "N"]


def _source_dir(
    scores_dir: Path, trait: str, teacher: str, feature: str, level: int
) -> Path:
    """Return the dose-specific source directory path."""
    name = (
        f"dataset=cejc_home2_hq1_v1__items={trait}24"
        f"__teacher={teacher}__dose={feature}_x{level}"
    )
    return scores_dir / name


def _dest_path(out_dir: Path, trait: str, teacher: str) -> Path:
    """Return the ensemble-compatible destination parquet path."""
    name = f"dataset=cejc_home2_hq1_v1__items={trait}24__teacher={teacher}"
    return out_dir / name / "teacher_merged" / f"trait_scores_{trait}_merged.parquet"


def prepare_one(
    scores_dir: Path,
    out_dir: Path,
    feature: str,
    level: int,
    trait: str,
    teacher: str,
) -> Path | None:
    """Read trait_scores from a dose-specific dir and write to ensemble layout.

    Returns the destination path on success, or None if the source is missing.
    """
    src_dir = _source_dir(scores_dir, trait, teacher, feature, level)
    src_parquet = src_dir / "trait_scores.parquet"

    if not src_parquet.exists():
        warnings.warn(
            f"Missing source: {src_parquet} "
            f"(feature={feature}, level={level}, trait={trait}, teacher={teacher})"
        )
        return None

    df = pd.read_parquet(src_parquet)

    # Validate expected columns
    expected_cols = {"conversation_id", "speaker_id", "trait", "model_id", "trait_score"}
    missing = expected_cols - set(df.columns)
    if missing:
        warnings.warn(f"Missing columns {missing} in {src_parquet}")
        return None

    # Filter to the target trait (safety check)
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
        description=(
            "Prepare ensemble-compatible directory structure "
            "from dose-response LLM scores."
        )
    )
    ap.add_argument(
        "--scores_dir",
        type=str,
        default="artifacts/big5/llm_scores",
        help="Root dir with dose-specific LLM scores",
    )
    ap.add_argument(
        "--dose_levels",
        type=str,
        default="0,3",
        help="Comma-separated dose levels to process (default: 0,3; ×1 reuses existing)",
    )
    ap.add_argument(
        "--target_feature",
        type=str,
        required=True,
        help="Target feature name (FILL, YESNO, OIR)",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="artifacts/big5",
        help="Parent output dir; creates llm_scores_dose_{feature}_x{level}/ under it",
    )
    args = ap.parse_args()

    scores_dir = Path(args.scores_dir)
    out_base = Path(args.out_dir)
    feature = args.target_feature
    levels = [int(x.strip()) for x in args.dose_levels.split(",")]

    total = 0
    ok = 0
    missing_list: list[str] = []

    for level in levels:
        dose_out_dir = out_base / f"llm_scores_dose_{feature}_x{level}"
        print(f"\n--- {feature} ×{level} → {dose_out_dir} ---")

        for trait in TRAITS:
            for teacher in TEACHERS:
                total += 1
                dest = prepare_one(
                    scores_dir, dose_out_dir, feature, level, trait, teacher
                )
                if dest is not None:
                    ok += 1
                    print(f"  ✓ {trait}/{teacher} → {dest}")
                else:
                    missing_list.append(f"{feature}_x{level}/{trait}/{teacher}")

    print(f"\n{'='*60}")
    print(f"Done: {ok}/{total} files written")
    if missing_list:
        print(f"Missing ({len(missing_list)}):")
        for m in missing_list:
            print(f"  ✗ {m}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
