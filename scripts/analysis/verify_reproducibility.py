#!/usr/bin/env python3
"""Check the values reported in the manuscript against the result files.

Each expected value below is transcribed from the manuscript. The script reads the
corresponding result file, compares at the precision the manuscript prints, and
writes one row per checked value.

Output: artifacts/analysis/results/reproducibility_check.tsv
Usage:  python scripts/analysis/verify_reproducibility.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np

# ---------------------------------------------------------------------------
# Paths (all relative to repo root)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

RESULTS_DIR = REPO_ROOT / "artifacts" / "analysis" / "results"
TEACHER_CORR_DIR = REPO_ROOT / "reports" / "model_agreement"
# The reported model has 21 predictors (the 19 features plus speaker sex and age), so
# every path below points at the *_modelB / *_modelb outputs. The 19-predictor
# directories still exist for the appendix comparisons; reading them here would check
# the manuscript against numbers it does not print.
MAIN_RESULT_TSV = (
    RESULTS_DIR / "ensemble_perm_groupkfold_modelB" / "ensemble_summary_modelB.tsv"
)
COEF_ALL5_DIR = RESULTS_DIR / "coef_bootstrap_all5_modelb"
POWER_JSON = RESULTS_DIR / "power_analysis_design_modelB.json"
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
# Expected values (transcribed from the manuscript)
# ---------------------------------------------------------------------------

# Main result: permutation test on the model-ensemble scores under the
# subject-wise split, 21 predictors, five dimensions, Holm-corrected across
# dimensions. The manuscript's headline table prints the fold-averaged r, the pooled
# out-of-fold r, R^2 and RMSE side by side, because ridge shrinkage makes the
# correlation alone a weak summary of fit; all four are checked.
# Read from MAIN_RESULT_TSV, columns r_groupkfold / p_groupkfold_holm / r_oof /
# R2_oof / RMSE.
MAIN_RESULT_EXPECTED: list[dict] = [
    {"trait": "O", "r_obs": 0.401, "p_holm": 0.0030,
     "r_oof": 0.403, "R2_oof": 0.154, "RMSE": 0.311},
    {"trait": "C", "r_obs": 0.447, "p_holm": 0.0020,
     "r_oof": 0.426, "R2_oof": 0.182, "RMSE": 0.382},
    {"trait": "E", "r_obs": 0.254, "p_holm": 0.0472,
     "r_oof": 0.269, "R2_oof": 0.072, "RMSE": 0.330},
    {"trait": "A", "r_obs": 0.517, "p_holm": 0.0020,
     "r_oof": 0.366, "R2_oof": 0.126, "RMSE": 0.240},
    {"trait": "N", "r_obs": 0.406, "p_holm": 0.0030,
     "r_oof": 0.294, "R2_oof": 0.079, "RMSE": 0.349},
]

# Appendix, CV-design comparison: the same quantity under a plain KFold. The
# appendix tabulates these alongside the subject-wise values, so they have to keep
# reproducing even though they are not the reported result.
# Read from MAIN_RESULT_TSV, column r_kfold.
KFOLD_EXPECTED: list[dict] = [
    {"trait": "O", "r_kfold": 0.481},
    {"trait": "C", "r_kfold": 0.469},
    {"trait": "E", "r_kfold": 0.237},
    {"trait": "A", "r_kfold": 0.538},
    {"trait": "N", "r_kfold": 0.311},
]

TEACHER_AGREEMENT_EXPECTED: list[dict] = [
    {"trait": "C", "mean_r": 0.699},
    {"trait": "A", "mean_r": 0.435},
]

# Dually supported features per dimension: permutation p < 0.05 AND a bootstrap 95%
# interval excluding zero. The manuscript names these individually, so the set is
# worth checking rather than only the count. Order does not matter; the check
# compares sorted lists. Read from COEF_ALL5_DIR.
#
# Only the 19 proposed features are listed. Sex and age are in the design matrix and
# can meet the same criterion (age for O and C, sex for A), but the manuscript lists
# them separately at the foot of the table because they are not part of the proposed
# instrument, so the parser drops them.
DUALLY_SUPPORTED_EXPECTED: dict[str, list[str]] = {
    "O": [
        "FILL_rate_per_100chars",
        "PG_overlap_rate",
        "PG_pause_mean",
        "PG_pause_p90",
        "PG_speech_ratio",
    ],
    "C": [
        "FILL_has_any",
        "IX_oirmarker_after_question_rate",
        "IX_yesno_rate",
        "PG_speech_ratio",
    ],
    "E": [
        "PG_pause_mean",
        "PG_pause_p50",
        "PG_pause_p90",
    ],
    "A": [
        "IX_yesno_rate",
        "PG_pause_mean",
        "PG_pause_p90",
        "RESP_NE_ENTROPY",
    ],
    "N": [
        "IX_oirmarker_after_question_rate",
        "IX_yesno_rate",
    ],
}

# The 19 proposed features. Used to exclude the two control variables from the
# dually-supported counts, matching how the manuscript reports them.
PROPOSED_FEATURES: frozenset[str] = frozenset({
    "PG_speech_ratio", "PG_pause_mean", "PG_pause_p50", "PG_pause_p90",
    "PG_resp_gap_mean", "PG_resp_gap_p50", "PG_resp_gap_p90",
    "PG_overlap_rate", "PG_pause_variability",
    "FILL_has_any", "FILL_rate_per_100chars",
    "IX_oirmarker_rate", "IX_oirmarker_after_question_rate",
    "IX_yesno_rate", "IX_yesno_after_question_rate", "IX_lex_overlap_mean",
    "RESP_NE_AIZUCHI_RATE", "RESP_NE_ENTROPY", "RESP_YO_ENTROPY",
})

# Appendix, decision-rule comparison: how many of the 19 features each rule
# identifies. Disclosed because requiring both procedures is stricter than either
# alone, and the count changes with the rule. Read from COEF_ALL5_DIR.
CRITERION_COUNTS_EXPECTED: dict[str, dict[str, int]] = {
    "O": {"perm_only": 5, "boot_only": 7, "both": 5},
    "C": {"perm_only": 4, "boot_only": 5, "both": 4},
    "E": {"perm_only": 3, "boot_only": 5, "both": 3},
    "A": {"perm_only": 4, "boot_only": 8, "both": 4},
    "N": {"perm_only": 4, "boot_only": 3, "both": 2},
}

# Appendix, statistical power of the design. Not post hoc power: the outcome is
# generated synthetically, so these do not depend on the observed effect sizes.
# Read from POWER_JSON.
POWER_EXPECTED: dict[str, float] = {
    "null_sd": 0.128,
    "r_crit_05": 0.250,
    "r_crit_01": 0.322,
    "rho_at_power_50_05": 0.349,
    "rho_at_power_80_05": 0.438,
    "rho_at_power_50_01": 0.419,
    "rho_at_power_80_01": 0.509,
}


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------
def parse_main_result_tsv(path: Path) -> dict[str, dict[str, float]] | None:
    """Read the summary TSV written by ``ensemble_permutation_groupkfold.py``.

    Returns ``{trait: {"r_obs", "p_holm", "r_oof", "R2_oof", "RMSE", "r_kfold"}}``,
    where ``r_obs`` / ``p_holm`` are the subject-wise (GroupKFold) values, ``r_oof`` /
    ``R2_oof`` / ``RMSE`` are computed on the concatenated out-of-fold predictions,
    and ``r_kfold`` is the plain-KFold value the same script records for comparison.
    Returns None if the file is absent; raises ValueError if the expected columns are
    missing.
    """
    if not path.exists():
        return None

    try:
        out: dict[str, dict[str, float]] = {}
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                out[row["trait"]] = {
                    "r_obs": float(row["r_groupkfold"]),
                    "p_holm": float(row["p_groupkfold_holm"]),
                    "r_oof": float(row["r_oof"]),
                    "R2_oof": float(row["R2_oof"]),
                    "RMSE": float(row["RMSE"]),
                    "r_kfold": float(row["r_kfold"]),
                }
        if not out:
            raise ValueError("no rows")
        return out
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Failed to parse main-result TSV: {path}") from exc


def _parse_coef_criteria(
    results_dir: Path, trait: str
) -> tuple[set[str], set[str]] | None:
    """Return (features with permutation p < 0.05, features whose 95% CI excludes 0).

    Both sets are restricted to the 19 proposed features. The design matrix also
    carries sex and age, and they can meet the same criteria, but the manuscript
    reports them separately at the foot of the coefficient table because they are
    not part of the proposed instrument.

    Returns None if either input file is absent; raises ValueError on a malformed
    file.
    """
    perm_path = results_dir / f"permutation_coef_{trait}_ensemble.tsv"
    boot_path = results_dir / f"bootstrap_variance_{trait}_ensemble.tsv"
    if not perm_path.exists() or not boot_path.exists():
        return None

    try:
        sig_perm: set[str] = set()
        with open(perm_path, encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                if row["feature"] not in PROPOSED_FEATURES:
                    continue
                if float(row["p_value"]) < 0.05:
                    sig_perm.add(row["feature"])

        excl_zero: set[str] = set()
        with open(boot_path, encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                if row["feature"] not in PROPOSED_FEATURES:
                    continue
                if row["ci_excludes_zero"].strip().lower() == "true":
                    excl_zero.add(row["feature"])
    except (KeyError, ValueError) as exc:
        raise ValueError(
            f"Failed to parse coefficient results for {trait}: {results_dir}"
        ) from exc

    return sig_perm, excl_zero


def parse_dually_supported_features(results_dir: Path, trait: str) -> list[str] | None:
    """Return the dually supported features for one trait, sorted.

    A feature is dually supported when the coefficient permutation test gives
    p < 0.05 *and* the bootstrap 95% interval excludes zero. Returns None if either
    input file is absent; raises ValueError on a malformed file.
    """
    parsed = _parse_coef_criteria(results_dir, trait)
    if parsed is None:
        return None
    sig_perm, excl_zero = parsed
    return sorted(sig_perm & excl_zero)


def parse_criterion_counts(results_dir: Path, trait: str) -> dict[str, int] | None:
    """Return how many features each decision rule identifies, for one trait.

    Keys are ``perm_only`` (permutation p < 0.05), ``boot_only`` (95% CI excluding
    zero) and ``both``. The first two are the counts under each rule taken alone,
    not the counts exclusive to that rule, which is how the appendix tabulates them.
    """
    parsed = _parse_coef_criteria(results_dir, trait)
    if parsed is None:
        return None
    sig_perm, excl_zero = parsed
    return {
        "perm_only": len(sig_perm),
        "boot_only": len(excl_zero),
        "both": len(sig_perm & excl_zero),
    }


def parse_power_json(path: Path) -> dict[str, float] | None:
    """Read the power-analysis JSON and flatten the values the appendix quotes.

    Returns None if the file is absent; raises ValueError if it does not carry the
    21-predictor design or the expected keys.
    """
    if not path.exists():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        n_pred = payload["design"]["n_predictors"]
        if n_pred != 21:
            raise ValueError(
                f"{n_pred} predictors; the reported model has 21, so the JSON must "
                "come from a run with --include_confounds"
            )
        lv05 = payload["levels"]["0.05"]
        lv01 = payload["levels"]["0.01"]
        return {
            "null_sd": float(payload["null"]["sd"]),
            "r_crit_05": float(lv05["r_crit"]),
            "r_crit_01": float(lv01["r_crit"]),
            "rho_at_power_50_05": float(lv05["rho_at_power_50"]),
            "rho_at_power_80_05": float(lv05["rho_at_power_80"]),
            "rho_at_power_50_01": float(lv01["rho_at_power_50"]),
            "rho_at_power_80_01": float(lv01["rho_at_power_80"]),
        }
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Failed to parse power JSON: {path}") from exc


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


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------
def compare_numeric(
    expected: float, actual: float, precision: int
) -> tuple[bool, str]:
    """Compare a value as printed in the manuscript against a stored value.

    ``expected`` is the manuscript string parsed as a float, so it carries only
    the digits the manuscript prints. A match is therefore anything that agrees to
    within half a unit of the last printed place, rather than exact equality after
    re-rounding.

    Re-rounding is not sufficient here because some result files store an already
    rounded value: a table cell printed as 0.432 can sit behind a stored 0.4315,
    and rounding that to three places gives 0.431. The discrepancy is double
    rounding, not disagreement. The half-unit rule still catches any difference
    large enough to change a printed digit.

    Returns (match, diff_str).
    """
    diff = abs(expected - actual)
    tolerance = 0.5 * 10 ** (-precision)
    match = round(expected, precision) == round(actual, precision) or (
        diff <= tolerance + 1e-12
    )
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
def _check_summary_values(
    expected: list[dict], fields: tuple[tuple[str, str], ...]
) -> list[CheckResult]:
    """Compare per-trait values in MAIN_RESULT_TSV against expected values.

    ``fields`` is a tuple of ``(row_key, item_name)`` pairs: ``row_key`` selects
    both the expected entry and the parsed value, and ``item_name`` is used to
    build the check label.
    """
    results: list[CheckResult] = []

    try:
        parsed = parse_main_result_tsv(MAIN_RESULT_TSV)
    except ValueError:
        parsed, status = None, "PARSE_ERROR"
    else:
        status = None if parsed is not None else "FILE_NOT_FOUND"

    for entry in expected:
        trait = entry["trait"]
        for row_key, item_name in fields:
            item = f"{item_name}_{trait}"
            exp = entry[row_key]
            if status is not None:
                results.append(CheckResult(item, str(exp), status, False, status))
                continue
            row = parsed.get(trait)
            if row is None:
                results.append(CheckResult(
                    item, str(exp), "TRAIT_NOT_FOUND", False, "TRAIT_NOT_FOUND"
                ))
                continue
            precision = len(str(exp).split(".")[-1])
            match, diff = compare_numeric(exp, row[row_key], precision)
            results.append(CheckResult(
                item, str(exp), str(round(row[row_key], precision)), match, diff
            ))

    return results


def check_main_result() -> list[CheckResult]:
    """Verify every value the headline table prints, for all five dimensions."""
    return _check_summary_values(
        MAIN_RESULT_EXPECTED,
        (
            ("r_obs", "main_r_fold"),
            ("p_holm", "main_p_corrected"),
            ("r_oof", "main_r_oof"),
            ("R2_oof", "main_R2_oof"),
            ("RMSE", "main_RMSE"),
        ),
    )


def check_kfold_comparison() -> list[CheckResult]:
    """Verify the plain-KFold r the appendix reports alongside the main result."""
    return _check_summary_values(
        KFOLD_EXPECTED, (("r_kfold", "cv_comparison_r_kfold"),)
    )


def check_dually_supported_features() -> list[CheckResult]:
    """Verify the dually supported feature set named in the manuscript, per dimension."""
    results: list[CheckResult] = []

    for trait, expected in DUALLY_SUPPORTED_EXPECTED.items():
        item = f"dually_supported_features_{trait}"
        exp_str = ",".join(expected)

        try:
            actual = parse_dually_supported_features(COEF_ALL5_DIR, trait)
        except ValueError:
            results.append(CheckResult(
                item, exp_str, "PARSE_ERROR", False, "PARSE_ERROR"
            ))
            continue

        if actual is None:
            results.append(CheckResult(
                item, exp_str, "FILE_NOT_FOUND", False, "FILE_NOT_FOUND"
            ))
            continue

        match, diff = compare_string_list(sorted(expected), actual)
        results.append(CheckResult(item, exp_str, ",".join(actual), match, diff))

    return results


def check_criterion_counts() -> list[CheckResult]:
    """Verify the appendix table comparing the three decision rules."""
    results: list[CheckResult] = []

    for trait, expected in CRITERION_COUNTS_EXPECTED.items():
        try:
            actual = parse_criterion_counts(COEF_ALL5_DIR, trait)
        except ValueError:
            actual, status = None, "PARSE_ERROR"
        else:
            status = None if actual is not None else "FILE_NOT_FOUND"

        for rule, exp in expected.items():
            item = f"criterion_count_{rule}_{trait}"
            if status is not None:
                results.append(CheckResult(item, str(exp), status, False, status))
                continue
            got = actual[rule]
            results.append(CheckResult(
                item, str(exp), str(got), exp == got,
                "" if exp == got else "MISMATCH",
            ))

    return results


def check_power_design() -> list[CheckResult]:
    """Verify the a priori power values the appendix quotes in prose."""
    results: list[CheckResult] = []

    try:
        actual = parse_power_json(POWER_JSON)
    except ValueError:
        actual, status = None, "PARSE_ERROR"
    else:
        status = None if actual is not None else "FILE_NOT_FOUND"

    for key, exp in POWER_EXPECTED.items():
        item = f"power_{key}"
        if status is not None:
            results.append(CheckResult(item, str(exp), status, False, status))
            continue
        precision = len(str(exp).split(".")[-1])
        match, diff = compare_numeric(exp, actual[key], precision)
        results.append(CheckResult(
            item, str(exp), str(round(actual[key], precision)), match, diff
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

    # main text: five dimensions, subject-wise split, 21 predictors
    all_results.extend(check_main_result())
    all_results.extend(check_dually_supported_features())

    # appendix: decision rules, CV-design comparison, power, between-model agreement
    all_results.extend(check_criterion_counts())
    all_results.extend(check_kfold_comparison())
    all_results.extend(check_power_design())
    all_results.extend(check_teacher_agreement())

    write_tsv(all_results, OUTPUT_TSV)
    print(f"Results written to: {OUTPUT_TSV}")

    # Print summary
    print_summary(all_results)


if __name__ == "__main__":
    main()
