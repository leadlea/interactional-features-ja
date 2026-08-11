#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""3段階Ridge比較図（R²ベース版）の生成.

方針:
  主指標を fold平均Pearson r から OOF-pooled R²（決定係数）へ変更．
  各次元について Stage 1(人口統計) / Stage 2(+Classical) / Stage 3(+Novel) の
  R²_oof を並列棒で表示し，ΔR² を矢印注記する．

誠実性のための表示方針:
  - Stage2→3 の増分にはペア検定(p_boot)の有意性マーカーを付す．
    本データでは増分は全次元で n.s.（Eのみ有意な悪化）であり，
    「Novelが有意に予測改善」とは主張しない．図はそれを正しく示す．
  - 棒の塗りで増分有意性を偽装しない（全Stage同一塗り）．

入力:
  artifacts/analysis/results/three_stage_metrics/three_stage_metrics_{teacher}.tsv
  artifacts/analysis/results/three_stage_metrics/three_stage_paired_test_{teacher}.tsv
出力:
  reports/paper_figs_v2/fig_three_stage_comparison.png   (R²版・本文用)

Usage:
    python scripts/paper_figs/gen_fig_three_stage_r2.py --teacher ensemble
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TRAIT_ORDER = ["O", "C", "E", "A", "N"]
STAGE_COLORS = {1: "#d4e6f1", 2: "#5dade2", 3: "#2166ac"}
STAGE_LABELS = {
    1: "Stage 1: Demographics",
    2: "Stage 2: +Classical",
    3: "Stage 3: +Novel",
}


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

    x = np.arange(len(TRAIT_ORDER))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9, 5.5))

    bar_info: dict[tuple[str, int], tuple[float, float]] = {}

    for si, stage in enumerate((1, 2, 3)):
        positions = x + (si - 1) * width
        color = STAGE_COLORS[stage]
        for i, trait in enumerate(TRAIT_ORDER):
            row = df[(df["trait"] == trait) & (df["stage"] == stage)]
            r2 = float(row["R2_oof"].values[0]) if len(row) else 0.0
            ax.bar(
                positions[i], r2, width,
                color=color, edgecolor=color, linewidth=1.5,
                label=STAGE_LABELS[stage] if i == 0 else "",
            )
            bar_info[(trait, stage)] = (positions[i], r2)
            ax.text(
                positions[i], r2 + 0.004 if r2 >= 0 else r2 - 0.004,
                f"{r2:.3f}",
                ha="center", va="bottom" if r2 >= 0 else "top",
                fontsize=9, fontweight="medium", color="#333333",
            )

    # ΔR² 注記
    for trait in TRAIT_ORDER:
        rows3 = df[(df["trait"] == trait) & (df["stage"] == 3)]
        rows2 = df[(df["trait"] == trait) & (df["stage"] == 2)]
        if not len(rows3) or not len(rows2):
            continue
        dr12 = float(rows2["dR2_from_prev"].values[0])
        dr23 = float(rows3["dR2_from_prev"].values[0])

        # Stage 1→2
        x1, y1 = bar_info[(trait, 1)]
        x2, y2 = bar_info[(trait, 2)]
        ya = max(y1, y2) + 0.022
        ax.annotate("", xy=(x2, ya), xytext=(x1, ya),
                    arrowprops=dict(arrowstyle="->", color="#5dade2",
                                    lw=1.4, connectionstyle="arc3,rad=0.15"))
        ax.text((x1 + x2) / 2, ya + 0.008,
                f"ΔR²={'+' if dr12 >= 0 else ''}{dr12:.3f}",
                ha="center", va="bottom", fontsize=9,
                fontstyle="italic", color="#5dade2")

        # Stage 2→3 （ペア検定の有意性マーカー付き）
        x3, y3 = bar_info[(trait, 3)]
        ya2 = max(y2, y3) + 0.050
        ax.annotate("", xy=(x3, ya2), xytext=(x2, ya2),
                    arrowprops=dict(arrowstyle="->", color="#2166ac",
                                    lw=1.4, connectionstyle="arc3,rad=0.15"))
        p_b = paired_p.get(trait, np.nan)
        sig = "*" if (not np.isnan(p_b) and p_b < 0.05) else " n.s."
        ax.text((x2 + x3) / 2, ya2 + 0.008,
                f"ΔR²={'+' if dr23 >= 0 else ''}{dr23:.3f}{sig}",
                ha="center", va="bottom", fontsize=9,
                fontstyle="italic", color="#2166ac")

    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(TRAIT_ORDER, fontsize=12)
    ax.set_ylabel("Out-of-fold $R^2$", fontsize=12)
    ax.set_title(
        "Three-Stage Ridge Comparison: Demographics → +Classical → +Novel",
        fontsize=14, pad=12,
    )
    r2max = df["R2_oof"].max()
    ax.set_ylim(min(0, df["R2_oof"].min() - 0.03), r2max * 1.6)
    # 左右に余白を確保し，端の次元のΔR²注記が見切れないようにする
    ax.set_xlim(-0.6, len(TRAIT_ORDER) - 1 + 0.6)
    # 凡例は軸の外（下部）に配置し，上部の注記・矢印との重なりを避ける
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.09),
        ncol=3, fontsize=10, frameon=False,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # 注記は図の最下部にまとめて配置（凡例・プロット領域と干渉しない）
    fig.text(
        0.5, 0.005,
        "ΔR² on arrows = incremental change.  "
        "* = paired bootstrap p<0.05 (Stage2→3);  n.s. = not significant.",
        ha="center", va="bottom", fontsize=8.5, color="#666666",
    )

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out_path = Path(args.out_dir) / "fig_three_stage_comparison.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
