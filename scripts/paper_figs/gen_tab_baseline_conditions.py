#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Comparison table for the three baseline conditions (subject-wise CV).

Input:  ``artifacts/analysis/results/baseline_validation/baseline_conditions_groupkfold.tsv``
        from ``scripts/analysis/baseline_conditions_groupkfold.py --n_perm 5000``
Output: ``reports/paper_figs_v2/tab_baseline_conditions.tex`` (plus an ``_en`` twin)

The verdict column is based on the delta r relative to Condition 1, with a threshold
of 0.1 (the same rule as ``scripts/baseline/compare_conditions.py``).

Usage:
    python scripts/paper_figs/gen_tab_baseline_conditions.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

TRAITS = ["O", "C", "E", "A", "N"]
LABELS = {
    "ja": {
        "cond1_text": "条件1（テキストあり）",
        "cond2_summary": "条件2（要約のみ）",
        "cond3_random": "条件3（ランダムテキスト）",
        "dim": "次元", "delta": "$\\Delta r$", "verdict": "判定",
        "sup": "条件1が優位", "eq": "条件1と同等以上",
    },
    "en": {
        "cond1_text": "Condition 1 (full text)",
        "cond2_summary": "Condition 2 (summary statistics only)",
        "cond3_random": "Condition 3 (random text)",
        "dim": "Dimension", "delta": "$\\Delta r$", "verdict": "Verdict",
        "sup": "Condition 1 higher", "eq": "Condition 1 not higher",
    },
}
THRESHOLD = 0.1


def build(df: pd.DataFrame, out_dir: Path, lang: str) -> None:
    """Three-condition comparison table; lang="en" writes the English twin."""
    L = LABELS[lang]
    idx = df.set_index(["condition", "trait"])

    rows: list[str] = []
    for cond in ["cond1_text", "cond2_summary", "cond3_random"]:
        rows.append(f"\\multicolumn{{6}}{{l}}{{{L[cond]}}} \\\\")
        for trait in TRAITS:
            r = idx.loc[(cond, trait)]
            p_holm = float(r["p_holm"])
            p_str = f"{p_holm:.4f}" + ("$^{*}$" if p_holm < 0.05 else "")
            if cond == "cond1_text":
                delta = "---"
                judge = "---"
            else:
                d = float(r["delta_r_vs_cond1"])
                delta = f"{d:+.3f}"
                judge = L["sup"] if d >= THRESHOLD else L["eq"]
            rows.append(
                f"\\quad {trait} & {r['r_obs']:.3f} & {r['p_value']:.4f} & "
                f"{p_str} & {delta} & {judge} \\\\"
            )
        if cond != "cond3_random":
            rows.append("\\midrule")

    latex = (
        "\\begingroup\n\\small\n"
        "\\begin{tabular}{lrrrrl}\n"
        "\\toprule\n"
        f"{L['dim']} & $r_{{\\mathrm{{obs}}}}$ & $p$ & "
        f"$p_{{\\mathrm{{Holm}}}}$ & {L['delta']} & {L['verdict']} \\\\\n"
        "\\midrule\n"
        + "\n".join(rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\endgroup\n"
    )
    suffix = "" if lang == "ja" else "_en"
    out_path = out_dir / f"tab_baseline_conditions{suffix}.tex"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(latex, encoding="utf-8")
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tsv",
        default="artifacts/analysis/results/baseline_validation/"
                "baseline_conditions_modelB.tsv",
    )
    ap.add_argument("--out_dir", default="reports/paper_figs_v2")
    args = ap.parse_args()

    df = pd.read_csv(args.tsv, sep="\t")
    out_dir = Path(args.out_dir)
    for lang in ("ja", "en"):
        build(df, out_dir, lang)

    # Print a summary for use when writing the prose
    print("\n=== summary ===")
    for cond in ["cond1_text", "cond2_summary", "cond3_random"]:
        sub = df[df["condition"] == cond]
        n_sig = int((sub["p_holm"] < 0.05).sum())
        print(f"{cond}: r={sub['r_obs'].min():.3f}..{sub['r_obs'].max():.3f}, "
              f"Holm-significant {n_sig}/5")
        for _, r in sub.iterrows():
            print(f"    {r['trait']}: r={r['r_obs']:+.3f} p={r['p_value']:.4f} "
                  f"pHolm={r['p_holm']:.4f} dr={r['delta_r_vs_cond1']:+.3f}")


if __name__ == "__main__":
    main()
