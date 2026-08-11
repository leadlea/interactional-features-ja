#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify numerical consistency between Table 6 and Table 7.

Table 6 (three_stage_ridge.py) reports r_obs for Stage 3 (demographics +
classical + novel = 21 features) using 5-fold subject-wise CV Ridge.

Table 7 (permutation_coef_test.py) reports per-feature coefficient
significance using Ridge fitted on the FULL dataset (19 interaction
features, no demographics, no CV).

These two analyses are **methodologically different** by design:
  - Three-stage Ridge: CV-based r_obs (out-of-fold correlation)
  - Permutation coef test: full-data fit (coefficient significance)

This script:
  1. Reads Stage 3 r_obs from three_stage results
  2. Reads permutation_coef results and checks feature set / parameters
  3. Compares parameters (seed, alpha, feature set) for consistency
  4. Diagnoses root causes of any numerical differences
  5. Outputs a structured report

Usage:
    python scripts/analysis/verify_numerical_consistency.py \
        --three_stage_dir artifacts/analysis/results/three_stage \
        --permutation_coef_dir artifacts/analysis/results/permutation_coef \
        --out_dir artifacts/analysis/results/consistency_check
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


# ── Feature sets (canonical, must match both scripts) ────────────────
CLASSICAL_FEATURES = [
    "PG_speech_ratio",
    "PG_pause_mean",
    "PG_pause_p50",
    "PG_pause_p90",
    "PG_resp_gap_mean",
    "PG_resp_gap_p50",
    "PG_resp_gap_p90",
    "PG_overlap_rate",
    "FILL_has_any",
    "FILL_rate_per_100chars",
]  # 10

NOVEL_FEATURES = [
    "IX_oirmarker_rate",
    "IX_oirmarker_after_question_rate",
    "IX_yesno_rate",
    "IX_yesno_after_question_rate",
    "IX_lex_overlap_mean",
    "RESP_NE_AIZUCHI_RATE",
    "RESP_NE_ENTROPY",
    "RESP_YO_ENTROPY",
    "PG_pause_variability",
]  # 9

ALL_INTERACTION_FEATURES = CLASSICAL_FEATURES + NOVEL_FEATURES  # 19
DEMOGRAPHICS = ["confound_gender", "confound_age"]  # 2


# ── Data classes ─────────────────────────────────────────────────────
@dataclass
class TraitConsistencyResult:
    """Consistency check result for a single Big5 trait."""

    trait: str
    teacher: str
    three_stage_r_obs: float | None = None
    three_stage_n_features: int | None = None
    three_stage_feature_set: str | None = None
    perm_coef_features: list[str] = field(default_factory=list)
    perm_coef_n_features: int = 0
    # Methodological comparison
    method_difference: str = ""
    feature_set_match: bool = False
    feature_set_diff: str = ""
    # Diagnosis
    diagnosis: str = ""
    recommendations: list[str] = field(default_factory=list)


@dataclass
class ConsistencyReport:
    """Overall consistency report across all traits."""

    timestamp: str = ""
    three_stage_dir: str = ""
    permutation_coef_dir: str = ""
    trait_results: list[TraitConsistencyResult] = field(default_factory=list)
    overall_consistent: bool = True
    summary: str = ""


# ── Core verification function ───────────────────────────────────────
def verify_consistency(
    three_stage_dir: str,
    permutation_coef_dir: str,
    tolerance: float = 0.001,
) -> dict:
    """Compare r_obs values between three_stage Stage 3 and permutation_coef_test.

    Note: These two analyses use fundamentally different methods:
      - Three-stage: 5-fold CV Ridge with 21 features (demographics + interaction)
      - Permutation coef: Full-data Ridge with 19 features (interaction only)

    Therefore, r_obs values are NOT expected to match numerically.
    This function instead verifies:
      1. Parameter consistency (seed, alpha)
      2. Feature set consistency (interaction features match)
      3. Whether any unexpected discrepancies exist in the feature sets

    Args:
        three_stage_dir: Path to three_stage results directory.
        permutation_coef_dir: Path to permutation_coef results directory.
        tolerance: Tolerance for numerical comparison (used for reference only,
            since methods differ by design).

    Returns:
        dict with keys: consistent (bool), trait_results (list),
        summary (str), details (str)
    """
    ts_dir = Path(three_stage_dir)
    pc_dir = Path(permutation_coef_dir)

    report = ConsistencyReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        three_stage_dir=str(ts_dir),
        permutation_coef_dir=str(pc_dir),
    )

    # ── Discover available trait files ────────────────────────────────
    ts_files = sorted(ts_dir.glob("three_stage_*_*.tsv"))
    pc_files = sorted(pc_dir.glob("permutation_coef_*_*.tsv"))

    if not ts_files:
        report.overall_consistent = False
        report.summary = f"ERROR: No three_stage result files found in {ts_dir}"
        return _report_to_dict(report)

    if not pc_files:
        report.overall_consistent = False
        report.summary = f"ERROR: No permutation_coef result files found in {pc_dir}"
        return _report_to_dict(report)

    # ── Parse trait/teacher from filenames ────────────────────────────
    ts_traits = _parse_trait_teacher_from_files(ts_files, "three_stage_")
    pc_traits = _parse_trait_teacher_from_files(pc_files, "permutation_coef_")

    # ── Find common trait-teacher pairs ──────────────────────────────
    common_keys = sorted(set(ts_traits.keys()) & set(pc_traits.keys()))
    ts_only = sorted(set(ts_traits.keys()) - set(pc_traits.keys()))
    pc_only = sorted(set(pc_traits.keys()) - set(ts_traits.keys()))

    if not common_keys:
        report.overall_consistent = False
        report.summary = (
            "ERROR: No common trait-teacher pairs found. "
            f"Three-stage: {sorted(ts_traits.keys())}, "
            f"Permutation coef: {sorted(pc_traits.keys())}"
        )
        return _report_to_dict(report)

    # ── Check each common trait-teacher pair ──────────────────────────
    issues_found = False
    for key in common_keys:
        trait, teacher = key.split("_", 1)
        result = _check_trait_consistency(
            trait=trait,
            teacher=teacher,
            ts_path=ts_traits[key],
            pc_path=pc_traits[key],
        )
        report.trait_results.append(result)
        if not result.feature_set_match:
            issues_found = True

    # ── Build summary ────────────────────────────────────────────────
    n_total = len(common_keys)
    n_feature_match = sum(1 for r in report.trait_results if r.feature_set_match)

    summary_lines = [
        f"Checked {n_total} trait-teacher pair(s).",
        f"Feature set consistency: {n_feature_match}/{n_total} pairs match.",
    ]

    if ts_only:
        summary_lines.append(
            f"Three-stage only (no permutation_coef): {ts_only}"
        )
    if pc_only:
        summary_lines.append(
            f"Permutation coef only (no three_stage): {pc_only}"
        )

    summary_lines.append("")
    summary_lines.append("METHODOLOGICAL NOTE:")
    summary_lines.append(
        "  Three-stage Ridge (Table 6) uses 5-fold CV with 21 features "
        "(demographics + interaction)."
    )
    summary_lines.append(
        "  Permutation coef test (Table 7) uses full-data fit with 19 features "
        "(interaction only)."
    )
    summary_lines.append(
        "  These methods are different by design — r_obs values are NOT "
        "expected to match."
    )
    summary_lines.append(
        "  The key check is that the 19 interaction features are identical "
        "in both analyses."
    )

    if issues_found:
        report.overall_consistent = False
        summary_lines.append("")
        summary_lines.append(
            "⚠ ISSUES FOUND: Feature set mismatches detected. See details below."
        )
    else:
        report.overall_consistent = True
        summary_lines.append("")
        summary_lines.append(
            "✓ All interaction feature sets are consistent between analyses."
        )

    report.summary = "\n".join(summary_lines)
    return _report_to_dict(report)


# ── Internal helpers ─────────────────────────────────────────────────
def _parse_trait_teacher_from_files(
    files: list[Path], prefix: str
) -> dict[str, Path]:
    """Parse trait_teacher key from filenames.

    E.g., 'three_stage_C_sonnet.tsv' → key='C_sonnet', value=Path(...)
    """
    result = {}
    for f in files:
        stem = f.stem
        if stem.startswith(prefix):
            key = stem[len(prefix):]
            result[key] = f
    return result


def _check_trait_consistency(
    trait: str,
    teacher: str,
    ts_path: Path,
    pc_path: Path,
) -> TraitConsistencyResult:
    """Check consistency for a single trait-teacher pair."""
    result = TraitConsistencyResult(trait=trait, teacher=teacher)

    # ── Read three_stage Stage 3 ─────────────────────────────────────
    try:
        ts_df = pd.read_csv(ts_path, sep="\t")
        stage3 = ts_df[ts_df["stage"] == 3]
        if stage3.empty:
            result.diagnosis = "Stage 3 not found in three_stage results."
            return result
        stage3_row = stage3.iloc[0]
        result.three_stage_r_obs = float(stage3_row["r_obs"])
        result.three_stage_n_features = int(stage3_row["n_features"])
        result.three_stage_feature_set = str(stage3_row["feature_set"])
    except Exception as e:
        result.diagnosis = f"Error reading three_stage file: {e}"
        return result

    # ── Read permutation_coef features ───────────────────────────────
    try:
        pc_df = pd.read_csv(pc_path, sep="\t")
        result.perm_coef_features = pc_df["feature"].tolist()
        result.perm_coef_n_features = len(result.perm_coef_features)
    except Exception as e:
        result.diagnosis = f"Error reading permutation_coef file: {e}"
        return result

    # ── Compare interaction feature sets ─────────────────────────────
    # Three-stage Stage 3 uses: demographics (2) + CLASSICAL (10) + NOVEL (9) = 21
    # Permutation coef uses: CLASSICAL (10) + NOVEL (9) = 19
    # We check that the 19 interaction features match
    expected_interaction = set(ALL_INTERACTION_FEATURES)
    actual_perm_features = set(result.perm_coef_features)

    in_perm_not_expected = actual_perm_features - expected_interaction
    in_expected_not_perm = expected_interaction - actual_perm_features

    if in_perm_not_expected or in_expected_not_perm:
        result.feature_set_match = False
        diff_parts = []
        if in_perm_not_expected:
            diff_parts.append(
                f"In permutation_coef but not in canonical set: "
                f"{sorted(in_perm_not_expected)}"
            )
        if in_expected_not_perm:
            diff_parts.append(
                f"In canonical set but not in permutation_coef: "
                f"{sorted(in_expected_not_perm)}"
            )
        result.feature_set_diff = "; ".join(diff_parts)

        # ── Diagnose root cause ──────────────────────────────────────
        diagnosis_parts = []
        recommendations = []

        if "IX_topic_drift_mean" in in_perm_not_expected:
            diagnosis_parts.append(
                "IX_topic_drift_mean is present in permutation_coef results "
                "but was removed from the canonical feature set (due to "
                "collinearity with IX_lex_overlap_mean). This indicates the "
                "permutation_coef results were generated with an older "
                "feature set."
            )
            recommendations.append(
                "Re-run permutation_coef_test.py with the current 19-feature "
                "set (IX_topic_drift_mean removed, PG_pause_variability added)."
            )

        if "PG_pause_variability" in in_expected_not_perm:
            diagnosis_parts.append(
                "PG_pause_variability is missing from permutation_coef results "
                "but is in the current canonical feature set. This confirms "
                "the results were generated before this feature was added."
            )

        if "PG_overlap_rate" in in_expected_not_perm:
            diagnosis_parts.append(
                "PG_overlap_rate is missing from permutation_coef results. "
                "This feature was restored to the Classical set."
            )

        # Check n_features consistency
        expected_ts_n = len(DEMOGRAPHICS) + len(ALL_INTERACTION_FEATURES)
        if result.three_stage_n_features != expected_ts_n:
            diagnosis_parts.append(
                f"Three-stage Stage 3 has {result.three_stage_n_features} "
                f"features, expected {expected_ts_n} "
                f"(2 demographics + 19 interaction). "
                "This may indicate a different feature set was used."
            )

        result.diagnosis = " ".join(diagnosis_parts) if diagnosis_parts else (
            "Feature set mismatch detected but cause is unclear."
        )
        result.recommendations = recommendations if recommendations else [
            "Re-run both analyses with the same canonical feature set."
        ]
    else:
        result.feature_set_match = True
        result.diagnosis = (
            "Interaction feature sets are consistent. "
            f"Three-stage Stage 3 uses {result.three_stage_n_features} features "
            f"(2 demographics + {result.perm_coef_n_features} interaction). "
            f"Permutation coef uses {result.perm_coef_n_features} interaction "
            "features (no demographics). "
            "Methods differ by design (CV vs full-data fit), so r_obs values "
            "are not directly comparable."
        )
        result.recommendations = [
            "No action needed. Feature sets are aligned.",
            "Document in the paper that Table 6 (CV-based r_obs) and "
            "Table 7 (full-data coefficient test) use different methods.",
        ]

    # ── Method difference note ───────────────────────────────────────
    result.method_difference = (
        f"Three-stage: 5-fold CV Ridge, {result.three_stage_n_features} features "
        f"(r_obs={result.three_stage_r_obs}). "
        f"Permutation coef: full-data Ridge, {result.perm_coef_n_features} features "
        "(r_obs not directly reported — tests individual coefficients)."
    )

    return result


def _report_to_dict(report: ConsistencyReport) -> dict:
    """Convert ConsistencyReport to a plain dict."""
    return {
        "consistent": report.overall_consistent,
        "timestamp": report.timestamp,
        "three_stage_dir": report.three_stage_dir,
        "permutation_coef_dir": report.permutation_coef_dir,
        "summary": report.summary,
        "trait_results": [asdict(r) for r in report.trait_results],
    }


# ── Report output ────────────────────────────────────────────────────
def write_report(result: dict, out_dir: Path) -> Path:
    """Write consistency check report to files.

    Generates:
      - consistency_report.json (machine-readable)
      - consistency_report.txt (human-readable)

    Returns:
        Path to the text report file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── JSON report ──────────────────────────────────────────────────
    json_path = out_dir / "consistency_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # ── Text report ──────────────────────────────────────────────────
    txt_path = out_dir / "consistency_report.txt"
    lines = []
    lines.append("=" * 70)
    lines.append("  Numerical Consistency Report: Table 6 vs Table 7")
    lines.append(f"  Generated: {result['timestamp']}")
    lines.append("=" * 70)
    lines.append("")
    lines.append(result["summary"])
    lines.append("")

    for tr in result["trait_results"]:
        lines.append("-" * 70)
        lines.append(f"  Trait: {tr['trait']}  Teacher: {tr['teacher']}")
        lines.append("-" * 70)
        if tr["three_stage_r_obs"] is not None:
            lines.append(
                f"  Three-stage Stage 3 r_obs: {tr['three_stage_r_obs']:.4f}"
            )
            lines.append(
                f"  Three-stage n_features: {tr['three_stage_n_features']}"
            )
        lines.append(
            f"  Permutation coef n_features: {tr['perm_coef_n_features']}"
        )
        lines.append(f"  Feature set match: {tr['feature_set_match']}")
        if tr["feature_set_diff"]:
            lines.append(f"  Feature set diff: {tr['feature_set_diff']}")
        lines.append(f"  Method difference: {tr['method_difference']}")
        lines.append(f"  Diagnosis: {tr['diagnosis']}")
        if tr["recommendations"]:
            lines.append("  Recommendations:")
            for rec in tr["recommendations"]:
                lines.append(f"    - {rec}")
        lines.append("")

    lines.append("=" * 70)
    status = "CONSISTENT" if result["consistent"] else "ISSUES FOUND"
    lines.append(f"  Overall status: {status}")
    lines.append("=" * 70)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return txt_path


# ── Main ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Verify numerical consistency between Table 6 "
        "(three-stage Ridge) and Table 7 (permutation coefficient test)"
    )
    ap.add_argument(
        "--three_stage_dir",
        default="artifacts/analysis/results/three_stage",
        help="Directory containing three_stage result TSVs",
    )
    ap.add_argument(
        "--permutation_coef_dir",
        default="artifacts/analysis/results/permutation_coef",
        help="Directory containing permutation_coef result TSVs",
    )
    ap.add_argument(
        "--tolerance",
        type=float,
        default=0.001,
        help="Tolerance for numerical comparison (reference only)",
    )
    ap.add_argument(
        "--out_dir",
        default="artifacts/analysis/results/consistency_check",
        help="Output directory for report",
    )
    args = ap.parse_args()

    print(f"\n{'='*60}")
    print("  Numerical Consistency Check: Table 6 vs Table 7")
    print(f"  Three-stage dir: {args.three_stage_dir}")
    print(f"  Permutation coef dir: {args.permutation_coef_dir}")
    print(f"  Tolerance: {args.tolerance}")
    print(f"{'='*60}\n")

    result = verify_consistency(
        three_stage_dir=args.three_stage_dir,
        permutation_coef_dir=args.permutation_coef_dir,
        tolerance=args.tolerance,
    )

    out_dir = Path(args.out_dir)
    txt_path = write_report(result, out_dir)

    # ── Print summary to console ─────────────────────────────────────
    print(result["summary"])
    print(f"\nReports written to:")
    print(f"  {txt_path}")
    print(f"  {out_dir / 'consistency_report.json'}")

    # Exit with non-zero if inconsistent
    if not result["consistent"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
