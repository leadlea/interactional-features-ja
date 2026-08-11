#!/usr/bin/env python3
"""Dose-Response analysis report generator.

Reads ensemble_summary.tsv for each dose level (×0, ×1, ×3), compares
C scores across dose levels, performs Wilcoxon signed-rank tests, and
generates dose-response plots.

Usage:
    python scripts/dose_response/dose_response_report.py \
        --results_dir artifacts/analysis/results/dose_response \
        --baseline_tsv artifacts/analysis/results/ensemble_perm_v4/ensemble_summary.tsv \
        --target_feature FILL \
        --control_feature OIR \
        --out_dir artifacts/dose_response

Output:
    - dose_response_report.tsv: comparison table
    - dose_response_plot_{feature}.png: dose-response plot
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# ── Pure functions ──────────────────────────────────────────────────


def judge_dose_response(
    scores_by_dose: dict[int, float],
    coef_sign: str,
    alpha: float = 0.05,
) -> dict:
    """Judge dose-response relationship from dose-level scores.

    Args:
        scores_by_dose: {dose_level: mean_C_score} dict
            e.g. {0: 0.2, 1: 0.4, 3: 0.5}
        coef_sign: Expected sign of regression coefficient ("+" or "-")
        alpha: Significance level (unused in monotonicity check, kept for API)

    Returns:
        {monotonic: bool, trend_direction: str, judgment: str}
    """
    if coef_sign not in ("+", "-"):
        raise ValueError(f"coef_sign must be '+' or '-', got: {coef_sign!r}")

    sorted_doses = sorted(scores_by_dose.keys())
    values = [scores_by_dose[d] for d in sorted_doses]

    # Check monotonicity
    if coef_sign == "+":
        monotonic = all(a <= b for a, b in zip(values, values[1:]))
        trend_direction = "increasing"
    else:
        monotonic = all(a >= b for a, b in zip(values, values[1:]))
        trend_direction = "decreasing"

    if monotonic:
        judgment = f"dose-response relationship detected ({trend_direction})"
    else:
        judgment = f"no monotonic {trend_direction} trend"

    return {
        "monotonic": monotonic,
        "trend_direction": trend_direction,
        "judgment": judgment,
    }


def judge_control_condition(
    target_delta_c: dict[int, float],
    control_delta_c: dict[int, float],
    threshold: float = 0.05,
) -> dict:
    """Judge specificity by comparing target vs control Delta_C.

    Args:
        target_delta_c: {dose_level: delta_C_vs_baseline} for target feature
        control_delta_c: {dose_level: delta_C_vs_baseline} for control feature
        threshold: Minimum difference to declare specificity

    Returns:
        {specific: bool, target_max_delta: float,
         control_max_delta: float, judgment: str}
    """
    target_max = max(abs(v) for v in target_delta_c.values()) if target_delta_c else 0.0
    control_max = max(abs(v) for v in control_delta_c.values()) if control_delta_c else 0.0

    specific = (target_max - control_max) > threshold

    if specific:
        judgment = (
            f"specific: target |Δr|={target_max:.3f} > "
            f"control |Δr|={control_max:.3f} + {threshold}"
        )
    else:
        judgment = (
            f"not specific: target |Δr|={target_max:.3f} ≤ "
            f"control |Δr|={control_max:.3f} + {threshold}"
        )

    return {
        "specific": specific,
        "target_max_delta": target_max,
        "control_max_delta": control_max,
        "judgment": judgment,
    }


# ── I/O helpers ─────────────────────────────────────────────────────


def _load_ensemble_summary(path: Path) -> pd.DataFrame:
    """Load ensemble_summary.tsv with expected columns."""
    df = pd.read_csv(path, sep="\t")
    expected = {"trait", "r_obs", "p_value", "p_corrected"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")
    return df


def _load_dose_summaries(
    results_dir: Path,
    feature: str,
    baseline_tsv: Path,
    dose_levels: list[int] = [0, 1, 3],
) -> dict[int, pd.DataFrame]:
    """Load ensemble_summary.tsv for each dose level.

    ×1 uses the baseline_tsv (existing results).
    ×0, ×3 use results_dir/dose_{feature}_x{level}/ensemble_summary.tsv.
    """
    summaries: dict[int, pd.DataFrame] = {}

    for level in dose_levels:
        if level == 1:
            if baseline_tsv.exists():
                summaries[1] = _load_ensemble_summary(baseline_tsv)
            else:
                warnings.warn(f"Baseline TSV not found: {baseline_tsv}")
        else:
            path = results_dir / f"dose_{feature}_x{level}" / "ensemble_summary.tsv"
            if path.exists():
                summaries[level] = _load_ensemble_summary(path)
            else:
                warnings.warn(f"Dose summary not found: {path}")

    return summaries


# ── Report generation ───────────────────────────────────────────────


def build_report_df(
    summaries: dict[int, pd.DataFrame],
    feature: str,
    baseline_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build dose_response_report.tsv from per-dose ensemble summaries."""
    rows = []

    # Get baseline r_obs per trait for delta calculation
    baseline_r: dict[str, float] = {}
    if baseline_df is not None:
        for _, row in baseline_df.iterrows():
            baseline_r[row["trait"]] = row["r_obs"]

    for level in sorted(summaries.keys()):
        df = summaries[level]
        for _, row in df.iterrows():
            trait = row["trait"]
            r_obs = row["r_obs"]
            p_value = row["p_value"]
            p_corrected = row["p_corrected"]
            delta_r = r_obs - baseline_r.get(trait, 0.0)

            rows.append({
                "target_feature": feature,
                "dose_level": level,
                "trait": trait,
                "r_obs": round(r_obs, 4),
                "p_value": round(p_value, 4),
                "p_corrected": round(p_corrected, 4),
                "delta_r_vs_baseline": round(delta_r, 4),
                "wilcoxon_p": float("nan"),  # filled below
            })

    report_df = pd.DataFrame(rows)

    # Wilcoxon signed-rank test: compare each dose level vs baseline (×1)
    if baseline_df is not None and 1 in summaries:
        baseline_scores = baseline_df.set_index("trait")["r_obs"]

        for level in sorted(summaries.keys()):
            if level == 1:
                continue
            dose_df = summaries[level]
            dose_scores = dose_df.set_index("trait")["r_obs"]

            # Align traits
            common_traits = sorted(
                set(baseline_scores.index) & set(dose_scores.index)
            )
            if len(common_traits) < 3:
                warnings.warn(
                    f"Too few common traits ({len(common_traits)}) for "
                    f"Wilcoxon test at dose={level}"
                )
                continue

            x = np.array([baseline_scores[t] for t in common_traits])
            y = np.array([dose_scores[t] for t in common_traits])

            try:
                stat_result = stats.wilcoxon(x, y, alternative="two-sided")
                w_p = stat_result.pvalue
            except ValueError:
                # e.g., all differences are zero
                w_p = 1.0

            # Fill wilcoxon_p for this dose level
            mask = report_df["dose_level"] == level
            report_df.loc[mask, "wilcoxon_p"] = round(w_p, 4)

    return report_df


def generate_dose_response_plot(
    report_df: pd.DataFrame,
    feature: str,
    out_path: Path,
) -> None:
    """Generate dose-response plot: dose level (x) vs r_obs (y) per trait."""
    fig, ax = plt.subplots(figsize=(8, 5))

    traits = report_df["trait"].unique()
    dose_levels = sorted(report_df["dose_level"].unique())
    colors = {"O": "#1f77b4", "C": "#d62728", "E": "#2ca02c", "A": "#ff7f0e", "N": "#9467bd"}
    markers = {"O": "o", "C": "s", "E": "^", "A": "D", "N": "v"}

    for trait in traits:
        tdf = report_df[report_df["trait"] == trait].sort_values("dose_level")
        ax.plot(
            tdf["dose_level"],
            tdf["r_obs"],
            marker=markers.get(trait, "o"),
            color=colors.get(trait, "gray"),
            label=trait,
            linewidth=2,
            markersize=8,
        )

    ax.set_xlabel("Dose Level (×)", fontsize=12)
    ax.set_ylabel("r_obs (CV Ridge Pearson r)", fontsize=12)
    ax.set_title(f"Dose-Response: {feature}", fontsize=14)
    ax.set_xticks(dose_levels)
    ax.set_xticklabels([f"×{d}" for d in dose_levels])
    ax.legend(title="Trait", loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out_path}")


def generate_target_vs_control_plot(
    target_report: pd.DataFrame,
    control_report: pd.DataFrame,
    target_feature: str,
    control_feature: str,
    out_path: Path,
) -> None:
    """Generate target vs control comparison plot for trait C."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Focus on C trait
    for label, df, color, marker in [
        (f"{target_feature} (target)", target_report, "#d62728", "s"),
        (f"{control_feature} (control)", control_report, "#7f7f7f", "o"),
    ]:
        c_df = df[df["trait"] == "C"].sort_values("dose_level")
        if c_df.empty:
            continue
        ax.plot(
            c_df["dose_level"],
            c_df["delta_r_vs_baseline"],
            marker=marker,
            color=color,
            label=label,
            linewidth=2,
            markersize=8,
        )

    ax.axhline(y=0, color="black", linestyle="--", alpha=0.3)
    ax.set_xlabel("Dose Level (×)", fontsize=12)
    ax.set_ylabel("Δr (vs ×1 baseline)", fontsize=12)
    ax.set_title("Target vs Control: Δr for Conscientiousness (C)", fontsize=14)

    dose_levels = sorted(target_report["dose_level"].unique())
    ax.set_xticks(dose_levels)
    ax.set_xticklabels([f"×{d}" for d in dose_levels])
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out_path}")


# ── Main ────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate dose-response analysis report."
    )
    ap.add_argument(
        "--results_dir",
        type=str,
        default="artifacts/analysis/results/dose_response",
        help="Directory containing dose_{feature}_x{level}/ ensemble results",
    )
    ap.add_argument(
        "--baseline_tsv",
        type=str,
        default="artifacts/analysis/results/ensemble_perm_v4/ensemble_summary.tsv",
        help="Path to ×1 baseline ensemble_summary.tsv",
    )
    ap.add_argument(
        "--target_feature",
        type=str,
        required=True,
        help="Target feature name (FILL, YESNO, OIR)",
    )
    ap.add_argument(
        "--control_feature",
        type=str,
        default="OIR",
        help="Control feature name (default: OIR)",
    )
    ap.add_argument(
        "--verification_tsv",
        type=str,
        default="",
        help="Path to feature_verification.tsv (optional, for reference)",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="artifacts/dose_response",
        help="Output directory for report and plots",
    )
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    baseline_tsv = Path(args.baseline_tsv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target = args.target_feature
    control = args.control_feature

    print(f"Dose-Response Report: {target}")
    print(f"  Results dir:  {results_dir}")
    print(f"  Baseline:     {baseline_tsv}")
    print(f"  Control:      {control}")
    print(f"  Output:       {out_dir}")
    print()

    # ── Load target feature summaries ──
    target_summaries = _load_dose_summaries(
        results_dir, target, baseline_tsv
    )
    if not target_summaries:
        print("ERROR: No ensemble summaries found for target feature.")
        return

    baseline_df = target_summaries.get(1)

    # ── Build target report ──
    target_report = build_report_df(target_summaries, target, baseline_df)

    # ── Load control feature summaries ──
    control_summaries = _load_dose_summaries(
        results_dir, control, baseline_tsv
    )
    control_report = None
    if control_summaries:
        control_report = build_report_df(control_summaries, control, baseline_df)

    # ── Dose-response judgment (C trait) ──
    c_scores_target: dict[int, float] = {}
    for level, df in target_summaries.items():
        c_row = df[df["trait"] == "C"]
        if not c_row.empty:
            c_scores_target[level] = float(c_row["r_obs"].iloc[0])

    if c_scores_target:
        # Assume positive coefficient for FILL (more fillers → higher C)
        # This is a simplification; in practice, check bootstrap results
        dr_result = judge_dose_response(c_scores_target, "+")
        print(f"  Dose-response (C, {target}): {dr_result['judgment']}")
        print(f"    Monotonic: {dr_result['monotonic']}")
        print(f"    Scores: {c_scores_target}")

    # ── Control condition judgment ──
    if control_report is not None and baseline_df is not None:
        target_delta: dict[int, float] = {}
        control_delta: dict[int, float] = {}

        for _, row in target_report[target_report["trait"] == "C"].iterrows():
            target_delta[row["dose_level"]] = row["delta_r_vs_baseline"]

        for _, row in control_report[control_report["trait"] == "C"].iterrows():
            control_delta[row["dose_level"]] = row["delta_r_vs_baseline"]

        if target_delta and control_delta:
            ctrl_result = judge_control_condition(target_delta, control_delta)
            print(f"\n  Control condition: {ctrl_result['judgment']}")

    # ── Combine and write report TSV ──
    all_reports = [target_report]
    if control_report is not None:
        all_reports.append(control_report)
    combined = pd.concat(all_reports, ignore_index=True)

    report_path = out_dir / "dose_response_report.tsv"
    combined.to_csv(report_path, sep="\t", index=False)
    print(f"\n  Wrote {report_path}")

    # ── Generate plots ──
    # Target feature dose-response plot
    generate_dose_response_plot(
        target_report, target, out_dir / f"dose_response_plot_{target}.png"
    )

    # Control feature dose-response plot (if available)
    if control_report is not None:
        generate_dose_response_plot(
            control_report, control, out_dir / f"dose_response_plot_{control}.png"
        )

    # Target vs Control comparison plot
    if control_report is not None:
        generate_target_vs_control_plot(
            target_report,
            control_report,
            target,
            control,
            out_dir / f"dose_response_target_vs_control_{target}.png",
        )

    # ── Print verification reference ──
    if args.verification_tsv and Path(args.verification_tsv).exists():
        print(f"\n  Feature verification: {args.verification_tsv}")
        vdf = pd.read_csv(args.verification_tsv, sep="\t")
        target_rows = vdf[vdf["is_target"] == True]
        if not target_rows.empty:
            print(target_rows.to_string(index=False))

    print(f"\n✓ Dose-response report complete for {target}")


if __name__ == "__main__":
    main()
