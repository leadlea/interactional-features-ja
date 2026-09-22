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
``artifacts/analysis/results/coef_bootstrap_all5_modelb/``
  - ``permutation_coef_{trait}_ensemble.tsv``   (21 predictors, alpha=100, 5,000 permutations)
  - ``bootstrap_variance_{trait}_ensemble.tsv`` (21 predictors, alpha=100, 500 bootstrap resamples)
Produced by ``bash scripts/analysis/run_coef_bootstrap_all5_modelb.sh``.

Outputs (``reports/paper_figs_v2/``)
------------------------------------
- ``tab_coef_all5.tex``          Main text. Regression coefficients, 19 features x 5 dimensions.
                                 ``*`` marks a *dually supported feature* (permutation p<0.05 AND
                                 a bootstrap 95% CI excluding zero); ``\dagger`` marks a feature
                                 satisfying only one of the two.
- ``tab_coef_all5_detail.tex``   Appendix. Full beta / p / 95% CI per dimension (longtable).
- ``tab_criterion_counts.tex``   Appendix. How many features each decision rule identifies.
- ``fig_coef_all5.png``          Coefficient plot, one row of five panels (beta +/- 95% CI).

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

# Same order as Table 1 (Classical 10 -> Novel 9), fixed across every dimension and
# every figure. It must match the order in tab_feature_definitions.tex exactly,
# because the manuscript's captions state that the two share an ordering.
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

# Descriptive names for the figure's axis labels. The figure uses these rather than
# the implementation identifiers (PG_speech_ratio and so on); the identifiers appear
# only in Table 1 and the appendices. Same content as Table 1's description column.
FEATURE_LABELS_EN = {
    "PG_speech_ratio": "Speech ratio",
    "PG_pause_mean": "Pause length (mean)",
    "PG_pause_p50": "Pause length (median)",
    "PG_pause_p90": "Pause length (p90)",
    "PG_resp_gap_mean": "Response gap (mean)",
    "PG_resp_gap_p50": "Response gap (median)",
    "PG_resp_gap_p90": "Response gap (p90)",
    "FILL_has_any": "Filler-bearing utterance rate",
    "FILL_rate_per_100chars": "Filler rate per 100 characters",
    "PG_overlap_rate": "Overlap rate",
    "IX_oirmarker_rate": "Repair initiation (OIR) rate",
    "IX_oirmarker_after_question_rate": "OIR rate after a question",
    "IX_yesno_rate": "Yes/no response rate",
    "IX_yesno_after_question_rate": "Yes/no rate after a question",
    "IX_lex_overlap_mean": "Lexical overlap",
    "RESP_NE_AIZUCHI_RATE": "Backchannel rate after $\\it{ne}$",
    "RESP_NE_ENTROPY": "Response diversity after $\\it{ne}$",
    "RESP_YO_ENTROPY": "Response diversity after $\\it{yo}$",
    "PG_pause_variability": "Pause length variability",
}

# Palette. Using hue for the sign of a coefficient collides with the reader's prior
# that red means positive and blue negative, so hue carries no meaning here: shade
# and line weight mark whether a feature is dually supported, and the sign is read
# from which side of the zero line the interval falls on. This also survives
# monochrome printing.
COLOR_SUPPORTED = "#1f3864"   # dark navy: supported by both procedures
COLOR_OTHER = "#b3b3b3"       # light grey: everything else


def tex_escape_feature(name: str) -> str:
    return name.replace("_", r"\_")


def load_all5(results_dir: Path) -> dict[str, pd.DataFrame]:
    """trait -> DataFrame(index=feature, cols=coef_obs,p_value,coef_mean,sd,ci_lo,ci_hi,both)."""
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
    # The summary row's label has to match what the manuscript calls these features.
    # The caption speaks of "the number of dually supported features", so the row
    # label says the same thing; "concordant" is not used, because the name does not
    # say what agrees with what.
    L = {
        "ja": {"feat": "特徴量", "n_conc": "両手法支持特徴量数",
               "n_novel": "\\quad うち新規（Novel）"},
        "en": {"feat": "Feature", "n_conc": "Dually supported features",
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

    # Append per-dimension counts of dually supported features as summary rows
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


def gen_tab_criterion_counts(data: dict[str, pd.DataFrame], out_dir: Path,
                             lang: str = "ja") -> None:
    """Appendix table: how many features each decision rule identifies.

    The manuscript requires both procedures to agree. Taking either one alone is a
    looser rule and identifies more features, so this table discloses the count
    under each of the three rules and makes the effect of the rule's stringency
    visible instead of leaving the reader to wonder why a count changed.
    """
    L = {
        "ja": {
            "dim": "次元", "perm": "置換検定のみ", "boot": "Bootstrapのみ",
            "both": "両方（本文の判定）", "total": "計",
        },
        "en": {
            "dim": "Dimension", "perm": "Permutation only", "boot": "Bootstrap only",
            "both": "Both (criterion used)", "total": "Total",
        },
    }[lang]
    rows: list[str] = []
    tot = [0, 0, 0]
    for trait in TRAITS:
        df = data[trait]
        n_p = int(df["perm_sig"].sum())
        n_b = int(df["boot_sig"].sum())
        n_both = int(df["both"].sum())
        tot[0] += n_p
        tot[1] += n_b
        tot[2] += n_both
        rows.append(f"{trait} & {n_p} & {n_b} & {n_both} \\\\")

    latex = (
        "\\begin{tabular}{lrrr}\n"
        "\\toprule\n"
        f"{L['dim']} & {L['perm']} & {L['boot']} & {L['both']} \\\\\n"
        "\\midrule\n"
        + "\n".join(rows) + "\n"
        "\\midrule\n"
        f"{L['total']} & {tot[0]} & {tot[1]} & {tot[2]} \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )
    suffix = "" if lang == "ja" else "_en"
    path = out_dir / f"tab_criterion_counts{suffix}.tex"
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
    r"""Coefficient forest plot: features on the y axis, O/C/E/A/N across five columns.

    Layout rationale
    ----------------
    A stacked layout (five rows, features on the x axis) was tried first. Each panel
    then comes out wide and short, which flattens the confidence intervals and makes
    the 19 features x 5 dimensions hard to compare.

    So the layout is a one-row, five-column forest plot with the features on the y
    axis. Unlike an earlier attempt, it is not rotated 90 degrees with \rotatebox:
    the figure is placed on its own page and set at width=\linewidth, which keeps
    every label upright while still buying back scale.

    Size
    ----
    The English manuscript's text width is 453pt; the other figures are set between
    0.82\linewidth and \linewidth, so this one stays at \linewidth rather than
    spilling into the 1in margins BRM requires. Holding the canvas to 6.8in puts the
    scale factor near 0.83, so an 8.5pt axis label reads at about 7pt on the page.
    The x-axis label is drawn once for the whole figure rather than five times, which
    returns that width to the panels.

    Palette
    -------
    Hue is not used to carry meaning, because red-for-positive collides with the
    reader's prior. Dually supported features are dark navy with a thick line and a
    filled marker; the rest are light grey with a thin line and an open marker. The
    sign is read from which side of the zero line an interval falls on. Line weight
    and marker fill keep the distinction in monochrome print.

    Axis labels
    -----------
    Descriptive names (FEATURE_LABELS_EN), not implementation identifiers. The
    identifiers appear only in Table 1 and the appendices.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    n = len(FEATURE_ORDER)

    fig, axes = plt.subplots(
        1, len(TRAITS), figsize=(6.8, 5.6), sharey=True,
        gridspec_kw={"wspace": 0.16},
    )

    lo = min(data[t]["ci_lower"].min() for t in TRAITS)
    hi = max(data[t]["ci_upper"].max() for t in TRAITS)
    pad = 0.10 * (hi - lo)

    for ax, trait in zip(axes, TRAITS):
        df = data[trait]
        ax.axvline(x=0, color="#8c8c8c", linewidth=0.7, linestyle="--", zorder=1)

        for i, feat in enumerate(FEATURE_ORDER):
            r = df.loc[feat]
            both = bool(r["both"])
            color = COLOR_SUPPORTED if both else COLOR_OTHER
            ax.plot(
                [r["ci_lower"], r["ci_upper"]], [i, i],
                color=color, linewidth=1.9 if both else 0.9,
                solid_capstyle="round", zorder=2,
            )
            ax.scatter(
                r["coef_mean"], i,
                s=15 if both else 8,
                facecolors=color if both else "white",
                edgecolors=color, linewidths=0.7,
                zorder=3,
            )

        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(n - 0.5, -0.5)  # top to bottom in FEATURE_ORDER
        ax.set_title(trait, fontsize=11, fontweight="bold", pad=5)
        ax.set_xticks([-0.05, 0.0, 0.05])
        ax.set_xticklabels(["$-$.05", "0", ".05"], fontsize=6.5)
        ax.tick_params(axis="x", length=2, pad=1.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)

    axes[0].set_yticks(np.arange(n))
    axes[0].set_yticklabels([FEATURE_LABELS_EN[f] for f in FEATURE_ORDER], fontsize=8.5)

    # One x-axis label for the whole figure
    fig.supxlabel("Ridge coefficient (mean $\\pm$ 95% CI)", fontsize=8.5, y=0.055)

    # No Classical / Novel divider is drawn. Only the layout and the palette changed;
    # every other element is kept as it was in the previous version of the figure.

    legend_handles = [
        Line2D([0], [0], color=COLOR_SUPPORTED, linewidth=2.0, marker="o",
               markerfacecolor=COLOR_SUPPORTED, markeredgecolor=COLOR_SUPPORTED,
               markersize=4.5,
               label="supported by both procedures "
                     "(permutation $p<.05$ and bootstrap 95% CI excluding zero)"),
        Line2D([0], [0], color=COLOR_OTHER, linewidth=1.0, marker="o",
               markerfacecolor="white", markeredgecolor=COLOR_OTHER, markersize=3.8,
               label="not supported by both procedures"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=1, fontsize=7,
               frameon=False, bbox_to_anchor=(0.55, -0.035))

    fig.tight_layout(rect=(0, 0.075, 1, 1))
    path = out_dir / "fig_coef_all5.png"
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def print_summary(data: dict[str, pd.DataFrame]) -> None:
    print("\n=== Dually supported (permutation p<0.05 and bootstrap CI excluding zero) ===")
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
        gen_tab_criterion_counts(data, out_dir, lang=lang)
    gen_fig_coef_all5(data, out_dir)
    print_summary(data)


if __name__ == "__main__":
    main()
