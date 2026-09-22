#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Appendix table: the a priori power of the design (Japanese and English twins).

Why this exists
---------------
The Statistical Guidelines for Behavior Research Methods ask for a priori power at
several effect sizes rather than post hoc power computed from an observed effect.

The single source of the numbers is the JSON written by
``scripts/analysis/power_analysis_design.py``:
``artifacts/analysis/results/power_analysis_design_modelB.json``. This script
recomputes nothing; it formats that JSON as a LaTeX table.

    python scripts/analysis/power_analysis_design.py --include_confounds --n_sim 400
    python scripts/paper_figs/gen_tab_power_design.py

Outputs
-------
reports/paper_figs_v2/tab_power_design.tex     (Japanese manuscript)
reports/paper_figs_v2/tab_power_design_en.tex  (English manuscript)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

LABELS = {
    "ja": {
        "rho": r"母集団重相関 $\rho$",
        "median": r"観測される$r_{\text{fold}}$の中央値",
        "p05": r"検出力（$\alpha = 0.05$）",
        "p01": r"検出力（$\alpha = 0.01$）",
    },
    # The English headings are deliberately short. BRM requires 1 inch margins, so
    # the English manuscript's \textwidth is 453pt -- narrower than the Japanese
    # one. A heading such as "Population multiple correlation" overruns it
    # (measured: 16.2pt of Overfull \hbox). The terms are explained in the caption
    # instead.
    "en": {
        "rho": r"$\rho$",
        "median": r"Median $r_{\text{fold}}$",
        "p05": r"Power ($\alpha = .05$)",
        "p01": r"Power ($\alpha = .01$)",
    },
}


def build_latex(payload: dict, lang: str) -> str:
    L = LABELS[lang]
    lv05 = payload["levels"]["0.05"]
    lv01 = payload["levels"]["0.01"]

    rho_grid = lv05["rho_grid"]
    if lv01["rho_grid"] != rho_grid:
        raise SystemExit("the two significance levels use different rho grids; check the JSON")

    # rho runs down the rows, so the column count stays fixed
    rows = []
    for i, rho in enumerate(rho_grid):
        rows.append(
            f"{rho:.2f} & {lv05['median_r_obs'][i]:.3f} & "
            f"{lv05['power'][i] * 100:.1f}\\% & {lv01['power'][i] * 100:.1f}\\% \\\\"
        )

    return (
        "\\begin{tabular}{rrrr}\n"
        "\\toprule\n"
        f"{L['rho']} & {L['median']} & {L['p05']} & {L['p01']} \\\\\n"
        "\\midrule\n"
        + "\n".join(rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json",
                    default="artifacts/analysis/results/power_analysis_design_modelB.json")
    ap.add_argument("--out_dir", default="reports/paper_figs_v2")
    args = ap.parse_args()

    payload = json.loads((REPO_ROOT / args.json).read_text(encoding="utf-8"))
    design = payload["design"]
    if design["n_predictors"] != 21:
        raise SystemExit(
            f"the JSON has {design['n_predictors']} predictors. The reported model has 21 "
            "(features + sex + age), so pass a JSON generated with --include_confounds."
        )

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for lang in ("ja", "en"):
        suffix = "" if lang == "ja" else "_en"
        path = out_dir / f"tab_power_design{suffix}.tex"
        path.write_text(build_latex(payload, lang), encoding="utf-8")
        print(f"wrote {path.relative_to(REPO_ROOT)}")

    lv05 = payload["levels"]["0.05"]
    print(
        "\nValues quoted in the manuscript:\n"
        f"  N={design['n_records']} / predictors={design['n_predictors']} / "
        f"speakers={design['n_speakers']}\n"
        f"  null distribution of r: mean={payload['null']['mean']:+.4f}, "
        f"SD={payload['null']['sd']:.4f}\n"
        f"  critical value r_crit (alpha=.05) = {lv05['r_crit']:.4f}\n"
        f"  50% power at rho={lv05['rho_at_power_50']:.3f} / "
        f"80% at rho={lv05['rho_at_power_80']:.3f}\n"
        f"  {design['n_null']} null iterations / {design['n_sim']} per rho / "
        f"seed={design['seed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
