#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Coefficient-level results for all five Big5 dimensions (O/C/E/A/N).

Why this exists
---------------
The coefficient-level tables and figure were originally produced for C alone
(``tab_permutation_coef.tex``, ``tab_bootstrap_variance.tex``,
``fig_bootstrap_variance.png``). Reporting all five dimensions at equal granularity
requires a five-dimension version, which this script generates.

Inputs
------
``artifacts/analysis/results/coef_bootstrap_all5/``
  - ``permutation_coef_{trait}_ensemble.tsv``   (19 features, alpha=100, 5,000 permutations)
  - ``bootstrap_variance_{trait}_ensemble.tsv`` (19 features, alpha=100, 500 bootstrap resamples)
Produced by ``bash scripts/analysis/run_coef_bootstrap_all5.sh``.

Outputs (``reports/paper_figs_v2/``)
------------------------------------
- ``tab_coef_all5.tex``        Main text. Regression coefficients, 19 features x 5 dimensions.
                               ``*`` marks a *concordant feature* (permutation p<0.05 AND a
                               bootstrap 95% CI excluding zero); ``\dagger`` marks a feature
                               satisfying only one of the two.
- ``tab_coef_all5_detail.tex`` Appendix. Full beta / p / 95% CI per dimension (longtable).
- ``fig_coef_all5.png``        Coefficient plot, five stacked panels (beta +/- 95% CI).

An ``_en`` twin of each table is written as well, for the English manuscript.

Usage
-----
    python scripts/paper_figs/gen_coef_all5.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TRAITS = ["O", "C", "E", "A", "N"]

# Same order as the feature definition table (Classical 10 then Novel 9).
# Must match the order in tab_feature_definitions.tex exactly:
# the captions state that the ordering is shared with that table.
FEATURE_ORDER = [
    "PG_speech_ratio", "PG_pause_mean", "PG_pause_p50", "PG_pause_p90",
    "PG_resp_gap_mean", "PG_resp_gap_p50", "PG_resp_gap_p90",
    "FILL_has_any", "FILL_rate_per_100chars",
    "PG_overlap_rate",
    "IX_oirmarker_rate", "IX_oirmarker_after_question_rate",
    "IX_yesno_rate", "IX_yesno_after_question_rate", "IX_lex_overlap_mean",
    "RESP_NE_AIZUCHI_RATE", "RESP_NE_ENTROPY", "RESP_YO_ENTROPY",
    "PG_pause_variability",
]

NOVEL_FEATURES = {
    "IX_oirmarker_rate", "IX_oirmarker_after_question_rate",
    "IX_yesno_rate", "IX_yesno_after_question_rate", "IX_lex_overlap_mean",
    "RESP_NE_AIZUCHI_RATE", "RESP_NE_ENTROPY", "RESP_YO_ENTROPY",
    "PG_pause_variability",
}


def tex_escape_feature(name: str) -> str:
    return name.replace("_", r"\_")


def load_all5(results_dir: Path) -> dict[str, pd.DataFrame]:
    """trait -> DataFrame indexed by feature, with coef/p/bootstrap columns."""
    out: dict[str, pd.DataFrame] = {}
    for trait in TRAITS:
        p_path = results_dir / f"permutation_coef_{trait}_ensemble.tsv"
        b_path = results_dir / f"bootstrap_variance_{trait}_ensemble.tsv"
        for path in (p_path, b_path):
            if not path.exists():
                raise FileNotFoundError(f"not found: {path}")
        perm = pd.read_csv(p_path, sep="\t").set_index("feature")
        boot = pd.read_csv(b_path, sep="\t").set_index("feature")

        missing = [f for f in FEATURE_ORDER if f not in perm.index or f not in boot.index]
        if missing:
            raise KeyError(f"trait={trait}: missing features {missing}")

        df = pd.DataFrame(index=FEATURE_ORDER)
        df["coef_obs"] = perm.loc[FEATURE_ORDER, "coef_obs"].astype(float)
        df["p_value"] = perm.loc[FEATURE_ORDER, "p_value"].astype(float)
        df["perm_sig"] = df["p_value"] < 0.05
        df["coef_mean"] = boot.loc[FEATURE_ORDER, "coef_mean"].astype(float)
        df["coef_sd"] = boot.loc[FEATURE_ORDER, "coef_sd"].astype(float)
        df["ci_lower"] = boot.loc[FEATURE_ORDER, "ci_lower"].astype(float)
        df["ci_upper"] = boot.loc[FEATURE_ORDER, "ci_upper"].astype(float)
        df["boot_sig"] = boot.loc[FEATURE_ORDER, "ci_excludes_zero"].astype(bool)
        df["both"] = df["perm_sig"] & df["boot_sig"]
        df["either"] = df["perm_sig"] | df["boot_sig"]
        out[trait] = df
    return out


def gen_tab_coef_all5(data: dict[str, pd.DataFrame], out_dir: Path,
                      lang: str = "ja") -> None:
    """Main-text table: coefficients for 19 features x 5 dimensions.

    With lang="en" an English-header twin (tab_coef_all5_en.tex) is written, so
    that the English manuscript cites the same numbers. Same convention as
    tab_feature_definitions_en.tex.
    """
    L = {
        "ja": {"feat": "特徴量", "n_conc": "一致特徴量数",
               "n_novel": "\\quad うち新規（Novel）"},
        "en": {"feat": "Feature", "n_conc": "Concordant features",
               "n_novel": "\\quad of which novel"},
    }[lang]
    lines: list[str] = []
    for feat in FEATURE_ORDER:
        cells = []
        for trait in TRAITS:
            row = data[trait].loc[feat]
            s = f"{row['coef_obs']:+.3f}"
            if row["both"]:
                s = f"{s}$^{{*}}$"
            elif row["either"]:
                s = f"{s}$^{{\\dagger}}$"
            cells.append(s)
        lines.append(f"{tex_escape_feature(feat)} & " + " & ".join(cells) + r" \\")

    # Append per-dimension counts of concordant features as summary rows
    n_both = [int(data[t]["both"].sum()) for t in TRAITS]
    n_novel = [
        int((data[t]["both"] & pd.Series({f: f in NOVEL_FEATURES for f in FEATURE_ORDER})).sum())
        for t in TRAITS
    ]

    body = "\n".join(lines)
    latex = (
        "\\begingroup\n"
        "\\footnotesize\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{lrrrrr}\n"
        "\\toprule\n"
        f"{L['feat']} & O & C & E & A & N \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\midrule\n"
        f"{L['n_conc']} & " + " & ".join(str(v) for v in n_both) + " \\\\\n"
        f"{L['n_novel']} & " + " & ".join(str(v) for v in n_novel) + " \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\endgroup\n"
    )
    suffix = "" if lang == "ja" else "_en"
    path = out_dir / f"tab_coef_all5{suffix}.tex"
    path.write_text(latex, encoding="utf-8")
    print(f"wrote {path}")


def gen_tab_coef_all5_detail(data: dict[str, pd.DataFrame], out_dir: Path,
                             lang: str = "ja") -> None:
    """Appendix table: full beta / p / bootstrap mean / SD / 95% CI per dimension."""
    L = {
        "ja": {
            "feat": "特徴量", "p": "$p$値", "ci": "95\\%CI",
            "cont": "（表\\ref{tab:coef_all5_detail}の続き）",
            "cap": ("各性格次元に対する回帰係数の詳細（19特徴量$\\times$5次元）．"
                    "Ridge回帰（$\\alpha = 100$，全データfit），置換検定5,000回，Bootstrap 500回．"
                    "$p$値の$^{*}$は置換検定$p < 0.05$，95\\%CIの$^{*}$はCIがゼロを除外することを示す．"),
        },
        "en": {
            "feat": "Feature", "p": "$p$", "ci": "95\\% CI",
            "cont": "(Table~\\ref{tab:coef_all5_detail}, continued)",
            "cap": ("Detailed regression coefficients for each trait dimension "
                    "(19 features $\\times$ 5 dimensions). Ridge regression "
                    "($\\alpha = 100$, fitted on the full data), 5,000 permutations, "
                    "500 bootstrap resamples. $^{*}$ on the $p$ value marks a permutation "
                    "$p < 0.05$; $^{*}$ on the 95\\% CI marks a CI excluding zero."),
        },
    }[lang]
    blocks: list[str] = []
    for trait in TRAITS:
        df = data[trait]
        blocks.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{trait}}}}} \\\\")
        for feat in FEATURE_ORDER:
            r = df.loc[feat]
            p_str = f"{r['p_value']:.4f}"
            if r["perm_sig"]:
                p_str = f"{p_str}$^{{*}}$"
            ci_str = f"[{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]"
            if r["boot_sig"]:
                ci_str = f"{ci_str}$^{{*}}$"
            blocks.append(
                f"\\quad {tex_escape_feature(feat)} & {r['coef_obs']:+.4f} & {p_str} & "
                f"{r['coef_mean']:+.4f} & {r['coef_sd']:.4f} & {ci_str} \\\\"
            )
        if trait != TRAITS[-1]:
            blocks.append("\\midrule")

    body = "\n".join(blocks)
    header = (f"{L['feat']} & $\\beta_{{\\mathrm{{obs}}}}$ & {L['p']} "
              f"& $\\bar{{\\beta}}$ & SD & {L['ci']} \\\\")
    caption = L["cap"]
    latex = (
        "\\begingroup\n"
        "\\footnotesize\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{longtable}{lrrrrr}\n"
        f"\\caption{{{caption}}}\\label{{tab:coef_all5_detail}}\\\\\n"
        "\\toprule\n"
        f"{header}\n"
        "\\midrule\n"
        "\\endfirsthead\n"
        f"\\multicolumn{{6}}{{l}}{{{L['cont']}}}\\\\\n"
        "\\toprule\n"
        f"{header}\n"
        "\\midrule\n"
        "\\endhead\n"
        "\\bottomrule\n"
        "\\endlastfoot\n"
        f"{body}\n"
        "\\end{longtable}\n"
        "\\endgroup\n"
    )
    suffix = "" if lang == "ja" else "_en"
    path = out_dir / f"tab_coef_all5_detail{suffix}.tex"
    path.write_text(latex, encoding="utf-8")
    print(f"wrote {path}")


def gen_fig_coef_all5(data: dict[str, pd.DataFrame], out_dir: Path) -> None:
    """Coefficient plot for the five dimensions (stacked panels, features on x).

    Layout rationale
    ----------------
    A wide single-row layout with five panels and the features on the y axis was
    tried first, rotated 90 degrees to fit the page. At a text width of roughly
    12.3 cm that scales the figure to about 0.35, which makes the axis labels
    illegible; rotating fixed legibility but left all the text sideways.

    The stacked layout (five rows, features on the x axis) avoids both problems:
    (1) no rotation, so all text is upright;
    (2) the 19 feature labels appear once, on the bottom panel, which buys back
        scale; and
    (3) reading down a single x position compares one feature across dimensions.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    n = len(FEATURE_ORDER)
    # Text width is about 12.3 cm = 4.85 in. Keep the aspect ratio near 0.74 so the
    # figure fits the page height when included at the full text width.
    fig, axes = plt.subplots(len(TRAITS), 1, figsize=(5.6, 7.5), sharex=True)

    lo = min(data[t]["ci_lower"].min() for t in TRAITS)
    hi = max(data[t]["ci_upper"].max() for t in TRAITS)
    pad = 0.12 * (hi - lo)

    for ax, trait in zip(axes, TRAITS):
        df = data[trait]
        ax.axhline(y=0, color="#999999", linewidth=0.9, linestyle="--", zorder=1)
        for i, feat in enumerate(FEATURE_ORDER):
            r = df.loc[feat]
            both = bool(r["both"])
            color = "#2166ac" if both else "#b2182b"
            ax.plot(
                [i, i], [r["ci_lower"], r["ci_upper"]],
                color=color, linewidth=2.0 if both else 1.3,
                alpha=1.0 if both else 0.5, zorder=2,
            )
            ax.scatter(
                i, r["coef_mean"], color=color, s=26 if both else 14,
                edgecolors="white", linewidths=0.4, zorder=3,
            )
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(-0.7, n - 0.3)
        # Keep the dimension label upright rather than rotated, for legibility
        ax.set_ylabel(trait, fontsize=11, fontweight="bold", rotation=0,
                      labelpad=14, va="center")
        ax.tick_params(axis="y", labelsize=6.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xticks(np.arange(n))
    axes[-1].set_xticklabels(FEATURE_ORDER, rotation=90, fontsize=6.5)
    fig.text(0.005, 0.55, "Ridge coefficient (mean $\\pm$ 95% CI)",
             rotation=90, va="center", fontsize=8)

    legend_handles = [
        Line2D([0], [0], color="#2166ac", linewidth=2.0, marker="o",
               markerfacecolor="#2166ac", markersize=5,
               label="permutation $p<0.05$ and bootstrap 95%CI excludes zero"),
        Line2D([0], [0], color="#b2182b", linewidth=1.3, marker="o",
               markerfacecolor="#b2182b", markersize=4, alpha=0.5,
               label="otherwise"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=1, fontsize=6.5,
               frameon=False, bbox_to_anchor=(0.55, -0.005))

    fig.tight_layout(rect=(0.03, 0.035, 1, 1))
    path = out_dir / "fig_coef_all5.png"
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def print_summary(data: dict[str, pd.DataFrame]) -> None:
    print("\n=== Concordant features (permutation p<0.05 and bootstrap CI excluding zero) ===")
    for trait in TRAITS:
        df = data[trait]
        both = df[df["both"]]
        novel = [f for f in both.index if f in NOVEL_FEATURES]
        classical = [f for f in both.index if f not in NOVEL_FEATURES]
        print(f"[{trait}] n={len(both)} (Classical {len(classical)} / Novel {len(novel)})")
        for f in both.index:
            r = both.loc[f]
            tag = "Novel" if f in NOVEL_FEATURES else "Classical"
            print(f"    {f:38s} beta={r['coef_obs']:+.3f} p={r['p_value']:.4f} "
                  f"CI=[{r['ci_lower']:+.3f},{r['ci_upper']:+.3f}] ({tag})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir",
                    default="artifacts/analysis/results/coef_bootstrap_all5_modelb")
    ap.add_argument("--out_dir", default="reports/paper_figs_v2")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_all5(results_dir)
    # The figure needs no language twin: its labels and legend are already English
    for lang in ("ja", "en"):
        gen_tab_coef_all5(data, out_dir, lang=lang)
        gen_tab_coef_all5_detail(data, out_dir, lang=lang)
    gen_fig_coef_all5(data, out_dir)
    print_summary(data)


if __name__ == "__main__":
    main()
