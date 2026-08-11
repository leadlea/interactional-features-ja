#!/usr/bin/env python3
"""
再現性検証スクリプト: 論文記載値と既存結果ファイルの一致を検証する。

出力: artifacts/analysis/results/reproducibility_check.tsv
Usage: python scripts/analysis/verify_reproducibility.py
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np

# ---------------------------------------------------------------------------
# Paths (all relative to repo root)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

RESULTS_DIR = REPO_ROOT / "artifacts" / "analysis" / "results"
BOOTSTRAP_DIR = RESULTS_DIR / "bootstrap"
TEACHER_CORR_DIR = REPO_ROOT / "reports" / "model_agreement"
OUTPUT_TSV = RESULTS_DIR / "reproducibility_check.tsv"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
class CheckResult(NamedTuple):
    check_item: str
    expected: str
    actual: str
    match: bool
    diff: str


# ---------------------------------------------------------------------------
# Expected values (hardcoded from paper)
# ---------------------------------------------------------------------------
PERMUTATION_EXPECTED: list[dict] = [
    {"trait": "C", "teacher": "sonnet", "r_obs": 0.434, "p_value": 0.0008},
    {"trait": "C", "teacher": "qwen3-235b", "r_obs": 0.390, "p_value": 0.0010},
    {"trait": "C", "teacher": "gpt-oss-120b", "r_obs": 0.447, "p_value": 0.0008},
    {"trait": "C", "teacher": "deepseek-v3", "r_obs": 0.205, "p_value": 0.1130},
]

TEACHER_AGREEMENT_EXPECTED: list[dict] = [
    {"trait": "C", "mean_r": 0.699},
    {"trait": "A", "mean_r": 0.435},
]

BOOTSTRAP_TOP3_EXPECTED: list[str] = [
    "FILL_has_any",
    "IX_oirmarker_after_question_rate",
    "PG_speech_ratio",
]


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------
def parse_permutation_log(path: Path) -> dict[str, float] | None:
    """Parse permutation.log and return {'r_obs': float, 'p_value': float}.

    Returns None if file not found, or raises ValueError on parse failure.
    """
    if not path.exists():
        return None

    text = path.read_text(encoding="utf-8")
    r_match = re.search(r"r_obs=([\d.]+)", text)
    p_match = re.search(r"p\(\|r\|\)=([\d.]+)", text)

    if r_match is None or p_match is None:
        raise ValueError(f"Failed to parse permutation.log: {path}")

    return {
        "r_obs": float(r_match.group(1)),
        "p_value": float(p_match.group(1)),
    }


def parse_teacher_corr_tsv(path: Path) -> float | None:
    """Read a 4x4 teacher correlation TSV and return mean of upper-triangle
    off-diagonal values.

    Returns None if file not found, or raises ValueError on parse failure.
    """
    if not path.exists():
        return None

    try:
        rows: list[list[float]] = []
        with open(path, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader)  # skip header row
            for row in reader:
                # first column is row label, rest are numeric
                rows.append([float(v) for v in row[1:]])

        n = len(rows)
        off_diag: list[float] = []
        for i in range(n):
            for j in range(i + 1, n):
                off_diag.append(rows[i][j])

        if not off_diag:
            raise ValueError("No off-diagonal values found")

        return float(np.mean(off_diag))
    except (ValueError, IndexError, StopIteration) as exc:
        raise ValueError(f"Failed to parse teacher_corr TSV: {path}") from exc


def parse_bootstrap_top3(path: Path) -> list[str] | None:
    """Read bootstrap_summary.tsv and return top-3 features by topk_rate
    descending.

    Returns None if file not found, or raises ValueError on parse failure.
    """
    if not path.exists():
        return None

    try:
        features: list[tuple[str, float]] = []
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                features.append((row["feature"], float(row["topk_rate"])))

        features.sort(key=lambda x: x[1], reverse=True)
        return [f[0] for f in features[:3]]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Failed to parse bootstrap_summary.tsv: {path}") from exc


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------
def compare_numeric(
    expected: float, actual: float, precision: int
) -> tuple[bool, str]:
    """Compare two floats rounded to the same precision.

    Returns (match, diff_str).
    """
    exp_rounded = round(expected, precision)
    act_rounded = round(actual, precision)
    match = exp_rounded == act_rounded
    diff = abs(expected - actual)
    return match, f"{diff:.10f}"


def compare_string_list(
    expected: list[str], actual: list[str]
) -> tuple[bool, str]:
    """Compare two lists of strings for equality."""
    match = expected == actual
    diff = "" if match else "MISMATCH"
    return match, diff


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------
def check_permutation_results() -> list[CheckResult]:
    """Verify permutation test results for C trait across 4 teachers."""
    results: list[CheckResult] = []

    for entry in PERMUTATION_EXPECTED:
        trait = entry["trait"]
        teacher = entry["teacher"]
        exp_r = entry["r_obs"]
        exp_p = entry["p_value"]

        dir_name = f"cejc_home2_hq1_{trait}only_{teacher}_controls_excluded"
        log_path = RESULTS_DIR / dir_name / "permutation.log"

        try:
            parsed = parse_permutation_log(log_path)
        except ValueError:
            parsed = "PARSE_ERROR"

        # r_obs check
        item_r = f"permutation_r_obs_{trait}_{teacher}"
        if parsed is None:
            results.append(CheckResult(
                item_r, str(exp_r), "FILE_NOT_FOUND", False, "FILE_NOT_FOUND"
            ))
        elif parsed == "PARSE_ERROR":
            results.append(CheckResult(
                item_r, str(exp_r), "PARSE_ERROR", False, "PARSE_ERROR"
            ))
        else:
            r_precision = len(str(exp_r).split(".")[-1])
            match, diff = compare_numeric(exp_r, parsed["r_obs"], r_precision)
            results.append(CheckResult(
                item_r, str(exp_r), str(round(parsed["r_obs"], r_precision)),
                match, diff
            ))

        # p_value check
        item_p = f"permutation_p_{trait}_{teacher}"
        if parsed is None:
            results.append(CheckResult(
                item_p, str(exp_p), "FILE_NOT_FOUND", False, "FILE_NOT_FOUND"
            ))
        elif parsed == "PARSE_ERROR":
            results.append(CheckResult(
                item_p, str(exp_p), "PARSE_ERROR", False, "PARSE_ERROR"
            ))
        else:
            p_precision = len(str(exp_p).split(".")[-1])
            match, diff = compare_numeric(exp_p, parsed["p_value"], p_precision)
            results.append(CheckResult(
                item_p, str(exp_p), str(round(parsed["p_value"], p_precision)),
                match, diff
            ))

    return results


def check_teacher_agreement() -> list[CheckResult]:
    """Verify teacher agreement mean r values."""
    results: list[CheckResult] = []

    for entry in TEACHER_AGREEMENT_EXPECTED:
        trait = entry["trait"]
        exp_mean_r = entry["mean_r"]
        item = f"teacher_agreement_mean_r_{trait}"

        tsv_path = TEACHER_CORR_DIR / f"teacher_corr_{trait}.tsv"

        try:
            actual_mean_r = parse_teacher_corr_tsv(tsv_path)
        except ValueError:
            actual_mean_r = "PARSE_ERROR"

        if actual_mean_r is None:
            results.append(CheckResult(
                item, str(exp_mean_r), "FILE_NOT_FOUND", False, "FILE_NOT_FOUND"
            ))
        elif actual_mean_r == "PARSE_ERROR":
            results.append(CheckResult(
                item, str(exp_mean_r), "PARSE_ERROR", False, "PARSE_ERROR"
            ))
        else:
            precision = len(str(exp_mean_r).split(".")[-1])
            match, diff = compare_numeric(exp_mean_r, actual_mean_r, precision)
            results.append(CheckResult(
                item, str(exp_mean_r),
                str(round(actual_mean_r, precision)),
                match, diff
            ))

    return results


def check_bootstrap_top3() -> list[CheckResult]:
    """Verify bootstrap top-3 features for C/sonnet."""
    item = "bootstrap_top3_C_sonnet"
    tsv_path = (
        BOOTSTRAP_DIR
        / "cejc_home2_hq1_Conly_sonnet_controls_excluded"
        / "bootstrap_summary.tsv"
    )

    try:
        actual_top3 = parse_bootstrap_top3(tsv_path)
    except ValueError:
        actual_top3 = "PARSE_ERROR"

    if actual_top3 is None:
        return [CheckResult(
            item,
            ",".join(BOOTSTRAP_TOP3_EXPECTED),
            "FILE_NOT_FOUND",
            False,
            "FILE_NOT_FOUND",
        )]
    elif actual_top3 == "PARSE_ERROR":
        return [CheckResult(
            item,
            ",".join(BOOTSTRAP_TOP3_EXPECTED),
            "PARSE_ERROR",
            False,
            "PARSE_ERROR",
        )]
    else:
        match, diff = compare_string_list(BOOTSTRAP_TOP3_EXPECTED, actual_top3)
        return [CheckResult(
            item,
            ",".join(BOOTSTRAP_TOP3_EXPECTED),
            ",".join(actual_top3),
            match,
            diff,
        )]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def write_tsv(results: list[CheckResult], path: Path) -> None:
    """Write check results to TSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["check_item", "expected", "actual", "match", "diff"])
        for r in results:
            writer.writerow([r.check_item, r.expected, r.actual, r.match, r.diff])


def print_summary(results: list[CheckResult]) -> None:
    """Print pass/fail summary to stdout."""
    total = len(results)
    passed = sum(1 for r in results if r.match)
    failed = total - passed

    print(f"\n{'='*60}")
    print(f"Reproducibility Check Summary")
    print(f"{'='*60}")
    print(f"  Total checks : {total}")
    print(f"  PASS         : {passed}")
    print(f"  FAIL         : {failed}")
    print(f"{'='*60}")

    if failed > 0:
        print("\nFailed checks:")
        for r in results:
            if not r.match:
                print(f"  - {r.check_item}: expected={r.expected}, "
                      f"actual={r.actual}, diff={r.diff}")
    else:
        print("\nAll checks passed!")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    all_results: list[CheckResult] = []

    # Task 4.1: Permutation results
    all_results.extend(check_permutation_results())

    # Task 4.2: Teacher agreement + Bootstrap Top3
    all_results.extend(check_teacher_agreement())
    all_results.extend(check_bootstrap_top3())

    # Task 4.3: Write TSV output
    write_tsv(all_results, OUTPUT_TSV)
    print(f"Results written to: {OUTPUT_TSV}")

    # Print summary
    print_summary(all_results)


if __name__ == "__main__":
    main()
