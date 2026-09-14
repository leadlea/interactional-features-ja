#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Headline result (permutation test on the virtual Big5) under subject-wise CV.

Why this exists
---------------
The headline result used to be produced by ``scripts/analysis/ensemble_permutation.py``
and ``gen_paper_figs_v2.py::collect_oof_predictions``, both of which use a plain
``KFold(shuffle=True)``. The manuscript describes GroupKFold over ``cejc_person_id``
(a subject-wise split); with 59.2% of records coming from repeated speakers, the plain
KFold leaks speaker-specific patterns. Since the staged ridge analysis was moved to an
appendix, this permutation test is now the only headline result, so the table and the
figure are regenerated here under the subject-wise split.

Inputs
------
- ``artifacts/analysis/results/ensemble_perm_groupkfold/ensemble_summary_groupkfold.tsv``
  from ``scripts/analysis/ensemble_permutation_groupkfold.py``
- ``artifacts/analysis/datasets/cejc_home2_hq1_XY_{trait}only_ensemble.parquet``
- ``artifacts/analysis/cejc_speaker_metadata.tsv`` (supplies the GroupKFold groups)
- ``artifacts/analysis/results/confound_ensemble_all5.tsv``
  from ``confound_analysis_groupkfold.py --traits O,C,E,A,N --ensemble_only``

Outputs (``reports/paper_figs_v2/``)
------------------------------------
- ``tab_ensemble_permutation.tex``  headline table (r_obs, uncorrected p, Holm p)
- ``fig_predicted_vs_observed.png`` observed versus out-of-fold predicted (5 dimensions)
- ``tab_confound_all5.tex``         confound control (sex, age)

The r annotated on the scatter panels is the same fold-averaged Pearson r as the
table -- the statistic the permutation test is built on -- so the figure and the
table can never disagree.

An ``_en`` twin of each table is written as well, for the English manuscript.

Usage
-----
    python scripts/paper_figs/gen_main_result_groupkfold.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

TRAITS = ["O", "C", "E", "A", "N"]
TRAIT_LABELS = {
    "O": "Openness (O)",
    "C": "Conscientiousness (C)",
    "E": "Extraversion (E)",
    "A": "Agreeableness (A)",
    "N": "Neuroticism (N)",
}
ALL_FEATURES = [
    "PG_speech_ratio", "PG_pause_mean", "PG_pause_p50", "PG_pause_p90",
    "PG_resp_gap_mean", "PG_resp_gap_p50", "PG_resp_gap_p90", "PG_overlap_rate",
    "FILL_has_any", "FILL_rate_per_100chars",
    "IX_oirmarker_rate", "IX_oirmarker_after_question_rate",
    "IX_yesno_rate", "IX_yesno_after_question_rate", "IX_lex_overlap_mean",
    "RESP_NE_AIZUCHI_RATE", "RESP_NE_ENTROPY", "RESP_YO_ENTROPY",
    "PG_pause_variability",
]


def load_xy(datasets_dir: Path, metadata_tsv: Path, trait: str,
            include_confounds: bool = False):
    fpath = datasets_dir / f"cejc_home2_hq1_XY_{trait}only_ensemble.parquet"
    df = pd.read_parquet(fpath).replace([np.inf, -np.inf], np.nan)
    meta = pd.read_csv(metadata_tsv, sep="\t")
    meta_cols = ["conversation_id", "speaker_id", "cejc_person_id"]
    if include_confounds:
        meta_cols += ["gender", "age"]
    merged = df.merge(meta[meta_cols], on=["conversation_id", "speaker_id"], how="left")

    cols = list(ALL_FEATURES)
    if include_confounds:
        merged["confound_gender"] = merged["gender"].map({"M": 0, "F": 1}).astype(float)
        merged["confound_age"] = pd.to_numeric(merged["age"], errors="coerce")
        cols += ["confound_gender", "confound_age"]

    y = merged[f"Y_{trait}"].astype(float).to_numpy()
    X = merged[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    groups = merged["cejc_person_id"].to_numpy()
    ok = ~np.isnan(y)
    return X[ok], y[ok], groups[ok]


def oof_predictions_groupkfold(X, y, groups, folds=5, seed=42, alpha=100.0):
    """Out-of-fold predictions under a subject-wise (GroupKFold) split."""
    y_pred = np.full(len(y), np.nan, dtype=float)
    for tr, te in GroupKFold(n_splits=folds).split(X, y, groups):
        imp = SimpleImputer(strategy="median")
        Xtr = imp.fit_transform(X[tr])
        Xte = imp.transform(X[te])
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xte = sc.transform(Xte)
        m = Ridge(alpha=alpha, random_state=seed)
        m.fit(Xtr, y[tr])
        y_pred[te] = m.predict(Xte)
    return y_pred


def gen_tab_ensemble_permutation(summary: pd.DataFrame, out_dir: Path,
                                 lang: str = "ja") -> None:
    """Headline result table; lang="en" also writes an English-header twin."""
    L = {"ja": {"dim": "次元", "verdict": "判定", "ns": "n.s."},
         "en": {"dim": "Dimension", "verdict": "Verdict", "ns": "n.s."}}[lang]

    # RMSE and the pooled out-of-fold metrics are reported alongside the
    # fold-averaged correlation. Ridge shrinks the predictions toward the mean, so a
    # correlation alone is a weak summary; the fold-averaged r in particular ignores
    # per-fold offsets in the predictions (for agreeableness the fold average is
    # 0.517 while the pooled value is 0.366). The test statistic is the fold-averaged
    # r, so that column is kept for consistency with the permutation test, and the
    # pooled r, the coefficient of determination and the RMSE are placed next to it.
    has_pooled = {"r_oof", "R2_oof", "RMSE"}.issubset(summary.columns)

    rows = []
    for trait in TRAITS:
        r = summary.loc[trait]
        sig = "*" if r["p_groupkfold_holm"] < 0.05 else L["ns"]
        if has_pooled:
            rows.append(
                f"{trait} & {r['r_groupkfold']:.3f} & {r['r_oof']:.3f} & "
                f"{r['R2_oof']:.3f} & {r['RMSE']:.3f} & "
                f"{r['p_groupkfold']:.4f} & {r['p_groupkfold_holm']:.4f} & {sig} \\\\"
            )
        else:
            rows.append(
                f"{trait} & {r['r_groupkfold']:.3f} & {r['p_groupkfold']:.4f} & "
                f"{r['p_groupkfold_holm']:.4f} & {sig} \\\\"
            )

    if has_pooled:
        latex = (
            "\\begin{tabular}{lrrrrrrc}\n"
            "\\toprule\n"
            f"{L['dim']} & $r_{{\\mathrm{{fold}}}}$ & $r_{{\\mathrm{{oof}}}}$ & "
            f"$R^{{2}}_{{\\mathrm{{oof}}}}$ & RMSE & $p$ & "
            f"$p_{{\\mathrm{{corrected}}}}$ & {L['verdict']} \\\\\n"
            "\\midrule\n"
            + "\n".join(rows) + "\n"
            "\\bottomrule\n"
            "\\end{tabular}\n"
        )
    else:
        latex = (
            "\\begin{tabular}{lcccc}\n"
            "\\toprule\n"
            f"{L['dim']} & $r_{{\\mathrm{{obs}}}}$ & $p$ & "
            f"$p_{{\\mathrm{{corrected}}}}$ & {L['verdict']} \\\\\n"
            "\\midrule\n"
            + "\n".join(rows) + "\n"
            "\\bottomrule\n"
            "\\end{tabular}\n"
        )
    suffix = "" if lang == "ja" else "_en"
    path = out_dir / f"tab_ensemble_permutation{suffix}.tex"
    path.write_text(latex, encoding="utf-8")
    print(f"wrote {path}")


def gen_tab_confound_all5(confound: pd.DataFrame, out_dir: Path,
                          lang: str = "ja") -> None:
    """Confound-control table; lang="en" also writes an English-header twin."""
    L = {
        "ja": {"dim": "次元", "a": "Model A（特徴量19変数）",
               "b": "Model B（+性別・年齢）"},
        "en": {"dim": "Dimension", "a": "Model A (19 features)",
               "b": "Model B ($+$ sex, age)"},
    }[lang]
    rows = []
    for trait in TRAITS:
        r = confound.loc[trait]
        rows.append(
            f"{trait} & {r['r_features_only']:.3f} & {r['p_features_only']:.4f} & "
            f"{r['r_with_confounds']:.3f} & {r['p_with_confounds']:.4f} & "
            f"{r['delta_r']:+.3f} \\\\"
        )
    latex = (
        "\\begin{tabular}{lrrrrr}\n"
        "\\toprule\n"
        f" & \\multicolumn{{2}}{{c}}{{{L['a']}}} & "
        f"\\multicolumn{{2}}{{c}}{{{L['b']}}} & \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
        f"{L['dim']} & $r$ & $p$ & $r$ & $p$ & $\\Delta r$ \\\\\n"
        "\\midrule\n"
        + "\n".join(rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )
    suffix = "" if lang == "ja" else "_en"
    path = out_dir / f"tab_confound_all5{suffix}.tex"
    path.write_text(latex, encoding="utf-8")
    print(f"wrote {path}")


def gen_fig_predicted_vs_observed(
    summary: pd.DataFrame, datasets_dir: Path, metadata_tsv: Path,
    out_dir: Path, alpha: float = 100.0, cv_folds: int = 5, seed: int = 42,
    include_confounds: bool = False,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, 3, figsize=(6.6, 5.3))
    axes_flat = axes.flatten()

    for idx, trait in enumerate(TRAITS):
        ax = axes_flat[idx]
        X, y, groups = load_xy(datasets_dir, metadata_tsv, trait,
                               include_confounds=include_confounds)
        y_pred = oof_predictions_groupkfold(
            X, y, groups, folds=cv_folds, seed=seed, alpha=alpha
        )
        ok = ~np.isnan(y_pred)
        y_true, y_hat = y[ok], y_pred[ok]

        row = summary.loc[trait]
        r_val = float(row["r_groupkfold"])
        p_val = float(row["p_groupkfold"])
        p_corr = float(row["p_groupkfold_holm"])
        is_sig = p_corr < 0.05
        color = "#2166ac" if is_sig else "#999999"
        line_style = "-" if is_sig else "--"

        ax.scatter(y_true, y_hat, s=30, alpha=0.6, color=color,
                   edgecolors="white", linewidths=0.3, zorder=3)
        z = np.polyfit(y_true, y_hat, 1)
        x_range = np.linspace(y_true.min(), y_true.max(), 100)
        ax.plot(x_range, np.poly1d(z)(x_range), color=color,
                linewidth=1.8, linestyle=line_style, zorder=4)

        all_vals = np.concatenate([y_true, y_hat])
        lo, hi = all_vals.min(), all_vals.max()
        margin = (hi - lo) * 0.05
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                color="#cccccc", linewidth=1.0, linestyle=":", zorder=2)
        ax.set_xlim(lo - margin, hi + margin)
        ax.set_ylim(lo - margin, hi + margin)
        ax.set_aspect("equal", adjustable="box")

        star = " *" if is_sig else ""
        # Significance is shown by an asterisk and by colour, never by bold
        # (table/figure convention: asterisks throughout).
        ax.text(0.04, 0.965, f"r = {r_val:.3f}, p = {p_val:.4f}{star}",
                transform=ax.transAxes, fontsize=7.2, va="top", ha="left",
                color=color,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="#dddddd", alpha=0.85))
        ax.set_xlabel("Observed", fontsize=11)
        ax.set_ylabel("Predicted", fontsize=11)
        ax.set_title(TRAIT_LABELS[trait], fontsize=12, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax_legend = axes_flat[5]
    ax_legend.axis("off")
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2166ac",
               markeredgecolor="white", markersize=8,
               label="Significant (Holm $p<0.05$)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#999999",
               markeredgecolor="white", markersize=8, label="Non-significant"),
        Line2D([0], [0], color="#2166ac", linewidth=1.8, linestyle="-",
               label="Regression (sig.)"),
        Line2D([0], [0], color="#999999", linewidth=1.8, linestyle="--",
               label="Regression (n.s.)"),
        Line2D([0], [0], color="#cccccc", linewidth=1.0, linestyle=":",
               label="y = x (ideal)"),
    ]
    ax_legend.legend(handles=handles, loc="upper center", fontsize=9.5,
                     frameon=True, framealpha=0.9, edgecolor="#dddddd",
                     title="Legend", title_fontsize=10.5)
    ax_legend.text(
        0.5, 0.06,
        f"Ridge $\\alpha$={alpha:g}, {cv_folds}-fold GroupKFold\n"
        "(subject-wise) out-of-fold predictions",
        ha="center", va="center", transform=ax_legend.transAxes,
        fontsize=9, color="#666666",
    )
    fig.suptitle("Predicted vs Observed: Virtual Big5 (Ridge, subject-wise CV)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = out_dir / "fig_predicted_vs_observed.png"
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_tsv",
                    default="artifacts/analysis/results/ensemble_perm_groupkfold_modelB/"
                            "ensemble_summary_modelB.tsv")
    ap.add_argument("--confound_tsv",
                    default="artifacts/analysis/results/confound_ensemble_all5.tsv")
    ap.add_argument("--datasets_dir", default="artifacts/analysis/datasets")
    ap.add_argument("--metadata_tsv",
                    default="artifacts/analysis/cejc_speaker_metadata.tsv")
    ap.add_argument("--out_dir", default="reports/paper_figs_v2")
    ap.add_argument(
        "--include_confounds", action="store_true", default=True,
        help="散布図の予測を21変数（+性別・年齢）で作る。主結果の設計に合わせる既定",
    )
    ap.add_argument(
        "--no_include_confounds", dest="include_confounds", action="store_false",
        help="19変数で作る（旧主結果の再現用）",
    )
    ap.add_argument(
        "--emit_confound_table", action="store_true",
        help="tab_confound_all5 を生成する。2026-09-13 の方針変更で本文§3.6が"
             "なくなったため既定では作らない（交絡統制済みモデルが主結果になったので"
             "Model A と比較する表そのものが不要）",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(args.summary_tsv, sep="\t").set_index("trait")

    for lang in ("ja", "en"):
        gen_tab_ensemble_permutation(summary, out_dir, lang=lang)
        if args.emit_confound_table:
            confound = pd.read_csv(args.confound_tsv, sep="\t").set_index("trait")
            gen_tab_confound_all5(confound, out_dir, lang=lang)
    gen_fig_predicted_vs_observed(
        summary, Path(args.datasets_dir), Path(args.metadata_tsv), out_dir,
        include_confounds=args.include_confounds,
    )


if __name__ == "__main__":
    main()
