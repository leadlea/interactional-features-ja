#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Sensitivity tables (alpha, CV design) under subject-wise CV.

Inputs
------
- ``artifacts/analysis/results/ensemble_perm_groupkfold/sensitivity_alpha_groupkfold.tsv``
  from ``ensemble_permutation_groupkfold.py --alpha_sweep 10,50,100,200,500 --n_perm 5000``
- ``artifacts/analysis/results/ensemble_perm_groupkfold/ensemble_summary_groupkfold.tsv``
  from ``ensemble_permutation_groupkfold.py --n_perm 5000``

Outputs (``reports/paper_figs_v2/``)
------------------------------------
- ``tab_sensitivity_alpha.tex``  r_obs and Holm-corrected p for alpha x 5 dimensions
- ``tab_cv_sensitivity.tex``     KFold versus GroupKFold comparison
- ``tab_permutation_all.tex``    per-model permutation results, reordered to O,C,E,A,N

An ``_en`` twin of each table is written as well, for the English manuscript.

Usage
-----
    python scripts/paper_figs/gen_sensitivity_tables.py
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

TRAITS = ["O", "C", "E", "A", "N"]
MAIN_ALPHA = 100.0


def holm_correction(p_values: list[float]) -> list[float]:
    m = len(p_values)
    non_nan = [(i, p) for i, p in enumerate(p_values) if not math.isnan(p)]
    corrected = [float("nan")] * m
    if not non_nan:
        return corrected
    m_eff = len(non_nan)
    cummax = 0.0
    for rank, (orig_idx, p) in enumerate(sorted(non_nan, key=lambda x: x[1])):
        cummax = max(cummax, p * (m_eff - rank))
        corrected[orig_idx] = min(cummax, 1.0)
    return corrected


def gen_tab_sensitivity_alpha(sweep: pd.DataFrame, out_dir: Path,
                              lang: str = "ja") -> None:
    """Alpha sensitivity table; lang="en" also writes an English-header twin."""
    L = {"ja": {"metric": "指標"}, "en": {"metric": "Metric"}}[lang]
    alphas = sorted(sweep["alpha"].unique())
    rows: list[str] = []
    for a in alphas:
        sub = sweep[sweep["alpha"] == a].set_index("trait")
        label = f"{a:g}"
        if a == MAIN_ALPHA:
            label = f"{label}$^{{\\dagger}}$"
        r_cells = [f"{sub.loc[t, 'r_obs']:.3f}" for t in TRAITS]
        rows.append(f"{label} & $r_{{\\mathrm{{obs}}}}$ & " + " & ".join(r_cells) + r" \\")
        p_cells = []
        for t in TRAITS:
            p = float(sub.loc[t, "p_holm"])
            s = f"{p:.4f}"
            if p < 0.05:
                s = f"{s}$^{{*}}$"
            p_cells.append(s)
        rows.append(r" & $p_{\mathrm{Holm}}$ & " + " & ".join(p_cells) + r" \\")
        if a != alphas[-1]:
            rows.append(r"\addlinespace")

    latex = (
        "\\begin{tabular}{llrrrrr}\n"
        "\\toprule\n"
        f"$\\alpha$ & {L['metric']} & O & C & E & A & N \\\\\n"
        "\\midrule\n"
        + "\n".join(rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )
    suffix = "" if lang == "ja" else "_en"
    path = out_dir / f"tab_sensitivity_alpha{suffix}.tex"
    path.write_text(latex, encoding="utf-8")
    print(f"wrote {path}")


def gen_tab_cv_sensitivity(summary: pd.DataFrame, out_dir: Path,
                           lang: str = "ja") -> None:
    """CV-design comparison table; lang="en" also writes an English-header twin."""
    L = {
        "ja": {"dim": "次元", "gkf": "GroupKFold（本文採用）",
               "kf": "通常KFold（参考）"},
        "en": {"dim": "Dimension", "gkf": "GroupKFold (adopted)",
               "kf": "Plain KFold (reference)"},
    }[lang]
    rows = []
    for trait in TRAITS:
        r = summary.loc[trait]
        gk_p = float(r["p_groupkfold_holm"])
        kf_p = float(r["p_kfold_holm"])
        gk = f"{gk_p:.4f}" + ("$^{*}$" if gk_p < 0.05 else "")
        kf = f"{kf_p:.4f}" + ("$^{*}$" if kf_p < 0.05 else "")
        rows.append(
            f"{trait} & {r['r_groupkfold']:.3f} & {gk} & "
            f"{r['r_kfold']:.3f} & {kf} & {r['delta_r']:+.3f} \\\\"
        )
    latex = (
        "\\begin{tabular}{lrrrrr}\n"
        "\\toprule\n"
        f" & \\multicolumn{{2}}{{c}}{{{L['gkf']}}} & "
        f"\\multicolumn{{2}}{{c}}{{{L['kf']}}} & \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
        f"{L['dim']} & $r_{{\\mathrm{{obs}}}}$ & $p_{{\\mathrm{{Holm}}}}$ & "
        f"$r_{{\\mathrm{{obs}}}}$ & $p_{{\\mathrm{{Holm}}}}$ & $\\Delta r$ \\\\\n"
        "\\midrule\n"
        + "\n".join(rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )
    suffix = "" if lang == "ja" else "_en"
    path = out_dir / f"tab_cv_sensitivity{suffix}.tex"
    path.write_text(latex, encoding="utf-8")
    print(f"wrote {path}")


MODEL_ORDER = [
    ("sonnet", "Claude Sonnet 4"),
    ("qwen3-235b", "Qwen3-235B"),
    ("gpt-oss-120b", "GPT-OSS-120B"),
    ("deepseek-v3", "DeepSeek-V3"),
]


def gen_tab_permutation_all(gkf_tsv: Path, out_dir: Path,
                            lang: str = "ja") -> None:
    """Per-model permutation table, reordered to O,C,E,A,N (GroupKFold values)."""
    L = {"ja": {"dim": "次元"}, "en": {"dim": "Dimension"}}[lang]
    df = pd.read_csv(gkf_tsv, sep="\t")
    rows = []
    for trait in TRAITS:
        cells = []
        for teacher, _ in MODEL_ORDER:
            sub = df[(df["trait"] == trait) & (df["teacher"] == teacher)]
            if sub.empty:
                cells.append("---")
                continue
            r = float(sub.iloc[0]["r_groupkfold"])
            p = float(sub.iloc[0]["p_groupkfold"])
            s = f"{r:.3f} ({p:.4f})"
            if p < 0.05:
                s = f"{s}$^{{*}}$"
            cells.append(s)
        rows.append(f"{trait} & " + " & ".join(cells) + r" \\")
    header = " & ".join(label for _, label in MODEL_ORDER)
    latex = (
        "\\begin{tabular}{lcccc}\n"
        "\\toprule\n"
        f"{L['dim']} & {header} \\\\\n"
        "\\midrule\n"
        + "\n".join(rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )
    suffix = "" if lang == "ja" else "_en"
    path = out_dir / f"tab_permutation_all{suffix}.tex"
    path.write_text(latex, encoding="utf-8")
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    base = "artifacts/analysis/results/ensemble_perm_groupkfold"
    ap.add_argument("--sweep_tsv", default=f"{base}/sensitivity_alpha_groupkfold.tsv")
    ap.add_argument("--summary_tsv", default=f"{base}/ensemble_summary_groupkfold.tsv")
    ap.add_argument(
        "--per_model_tsv",
        default="artifacts/analysis/results/groupkfold_vs_kfold_all_nperm5000.tsv",
    )
    ap.add_argument("--out_dir", default="reports/paper_figs_v2")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sweep = pd.read_csv(args.sweep_tsv, sep="\t")
    # Recompute the Holm correction per alpha condition if p_holm is absent
    if "p_holm" not in sweep.columns or sweep["p_holm"].isna().any():
        parts = []
        for a, sub in sweep.groupby("alpha"):
            sub = sub.copy()
            sub["p_holm"] = [round(v, 4) for v in holm_correction(sub["p_value"].tolist())]
            parts.append(sub)
        sweep = pd.concat(parts, ignore_index=True)

    summary = pd.read_csv(args.summary_tsv, sep="\t").set_index("trait")

    for lang in ("ja", "en"):
        gen_tab_sensitivity_alpha(sweep, out_dir, lang=lang)
        gen_tab_cv_sensitivity(summary, out_dir, lang=lang)
        gen_tab_permutation_all(Path(args.per_model_tsv), out_dir, lang=lang)


if __name__ == "__main__":
    main()
