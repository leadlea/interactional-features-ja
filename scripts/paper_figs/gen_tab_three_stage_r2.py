#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""3段階Ridge比較表（R²/RMSE版 + 付録r版）の生成.

方針:
  本文表 tab_three_stage.tex を「OOF-pooled R² / RMSE / ΔR² / ΔRMSE」中心へ更新．
  Stage2→3（Novel追加）のΔにはペア検定(paired bootstrap)の有意性マーカーを付す．
  透明性のため，相関 r 版を付録表 tab_three_stage_r.tex に分離して残す．

入力:
  artifacts/analysis/results/three_stage_metrics/three_stage_metrics_{teacher}.tsv
  artifacts/analysis/results/three_stage_metrics/three_stage_paired_test_{teacher}.tsv
出力:
  reports/paper_figs_v2/tab_three_stage.tex      (本文用: R²/RMSE中心)
  reports/paper_figs_v2/tab_three_stage_r.tex    (付録用: r)

Usage:
    python scripts/paper_figs/gen_tab_three_stage_r2.py --teacher ensemble
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TRAIT_ORDER = ["O", "C", "E", "A", "N"]
STAGE_LABELS = {1: "Demographics", 2: "+Classical", 3: "+Novel"}


def _fmt(v, nd=3, signed=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "---"
    if signed:
        return f"{v:+.{nd}f}"
    return f"{v:.{nd}f}"


def gen_main_table(df, paired_p, out_path):
    """R²/RMSE中心の本文用表."""
    lines = []
    lines.append(r"\begin{tabular}{llrcccc}")
    lines.append(r"\toprule")
    lines.append(
        r"Trait & Stage & $n_{feat}$ & $R^2_{oof}$ & RMSE & "
        r"$\Delta R^2$ & $\Delta$RMSE \\"
    )
    lines.append(r"\midrule")

    for ti, trait in enumerate(TRAIT_ORDER):
        sub = df[df["trait"] == trait].sort_values("stage")
        for _, row in sub.iterrows():
            stage = int(row["stage"])
            stage_label = STAGE_LABELS[stage]
            r2 = _fmt(row["R2_oof"], 3)
            rmse = _fmt(row["RMSE"], 3)

            # ΔR² と ΔRMSE
            if stage == 1:
                d_r2 = "---"
                d_rmse = "---"
            else:
                d_r2 = _fmt(row["dR2_from_prev"], 3, signed=True)
                d_rmse = _fmt(row["dRMSE_from_prev"], 3, signed=True)
                # Stage2→3 のみ有意性マーカーを付す（ペア検定）
                if stage == 3:
                    p_b = paired_p.get(trait, np.nan)
                    if not np.isnan(p_b):
                        marker = r"$^{*}$" if p_b < 0.05 else r"$^{\mathrm{ns}}$"
                        d_r2 = d_r2 + marker
                        d_rmse = d_rmse + marker

            trait_cell = trait if stage == 1 else ""
            lines.append(
                f"{trait_cell} & {stage_label} & {int(row['n_features'])} & "
                f"{r2} & {rmse} & {d_r2} & {d_rmse} \\\\"
            )
        if ti < len(TRAIT_ORDER) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")


def gen_appendix_r_table(df, out_path):
    """相関 r 版の付録表（透明性のため）."""
    lines = []
    lines.append(r"\begin{tabular}{llrccc}")
    lines.append(r"\toprule")
    lines.append(
        r"Trait & Stage & $n_{feat}$ & $r_{oof}$ & $r_{foldmean}$ & "
        r"$\Delta r_{oof}$ \\"
    )
    lines.append(r"\midrule")

    for ti, trait in enumerate(TRAIT_ORDER):
        sub = df[df["trait"] == trait].sort_values("stage")
        for _, row in sub.iterrows():
            stage = int(row["stage"])
            stage_label = STAGE_LABELS[stage]
            r_oof = _fmt(row["r_oof"], 3)
            r_fm = _fmt(row["r_foldmean"], 3)
            d_r = "---" if stage == 1 else _fmt(row["dr_oof_from_prev"], 3, signed=True)
            trait_cell = trait if stage == 1 else ""
            lines.append(
                f"{trait_cell} & {stage_label} & {int(row['n_features'])} & "
                f"{r_oof} & {r_fm} & {d_r} \\\\"
            )
        if ti < len(TRAIT_ORDER) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="ensemble")
    ap.add_argument(
        "--metrics_dir",
        default="artifacts/analysis/results/three_stage_metrics",
    )
    ap.add_argument("--out_dir", default="reports/paper_figs_v2")
    args = ap.parse_args()

    metrics_dir = Path(args.metrics_dir)
    df = pd.read_csv(
        metrics_dir / f"three_stage_metrics_{args.teacher}.tsv", sep="\t"
    )
    paired = pd.read_csv(
        metrics_dir / f"three_stage_paired_test_{args.teacher}.tsv", sep="\t"
    )
    paired_p = dict(zip(paired["trait"], paired["p_boot"]))

    out_dir = Path(args.out_dir)
    gen_main_table(df, paired_p, out_dir / "tab_three_stage.tex")
    gen_appendix_r_table(df, out_dir / "tab_three_stage_r.tex")


if __name__ == "__main__":
    main()
