#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Big5 全5次元（O/C/E/A/N）版の係数レベル結果の表・図を生成する.

背景
----
2026-09-06 の山下先生・宗田先生との合議で、論文全体を C（誠実性）偏重の書き振りから
Big5 5次元を均等に扱う書き振りへ改める方針が決まった。従来の
``tab_permutation_coef.tex`` / ``tab_bootstrap_variance.tex`` / ``fig_bootstrap_variance.png``
は C 単独版なので、本スクリプトで5次元版を生成する。

入力
----
``artifacts/analysis/results/coef_bootstrap_all5/``
  - ``permutation_coef_{trait}_ensemble.tsv``   （19特徴量・alpha=100・置換5000回）
  - ``bootstrap_variance_{trait}_ensemble.tsv`` （19特徴量・alpha=100・Bootstrap500回）
生成元: ``bash scripts/analysis/_run_coef_bootstrap_all5.sh``

出力（``reports/paper_figs_v2/``）
--------------------------------
- ``tab_coef_all5.tex``          本文用。19特徴量 × 5次元の回帰係数一覧。
                                 ``*``=置換検定 p<0.05 かつ Bootstrap 95%CI がゼロを除外（両手法一致）、
                                 ``\dagger``=いずれか一方のみ。
- ``tab_coef_all5_detail.tex``   付録用。次元ごとに β・p・95%CI を全記載（longtable）。
- ``fig_coef_all5.png``          5パネルのフォレストプロット（β ± 95%CI）。

再現コマンド
------------
    .venv/bin/python scripts/paper_figs/gen_coef_all5.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TRAITS = ["O", "C", "E", "A", "N"]

# 本文表1と同じ並び（Classical 10 → Novel 9）。全次元・全図表でこの順を固定する。
# 表1（tab_feature_definitions.tex）の掲載順と完全に一致させる。
# 本文キャプションで「表1と共通」と述べているため、ここを崩すと不整合になる。
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

# 図の軸ラベルに使う説明的名称。本文・図では実装上の識別子（PG_speech_ratio 等）ではなく
# こちらを使い、識別子は表1と付録にのみ残す（2026-09-21 山下先生ご指摘）。
# 表1の Description 列と同じ内容を英語で表す。
FEATURE_LABELS_EN = {
    "PG_speech_ratio": "Speech ratio",
    "PG_pause_mean": "Pause length (mean)",
    "PG_pause_p50": "Pause length (median)",
    "PG_pause_p90": "Pause length (90th pct.)",
    "PG_resp_gap_mean": "Response gap (mean)",
    "PG_resp_gap_p50": "Response gap (median)",
    "PG_resp_gap_p90": "Response gap (90th pct.)",
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

# 配色。符号（正負）を色相で表すと「赤＝正／青＝負」という先入観と衝突するため、
# 色相ではなく濃淡と線幅で判定の有無を示す（2026-09-21 山下先生ご指摘）。
# 符号はゼロ線の左右どちらに出るかで読む。
COLOR_SUPPORTED = "#1f3864"   # 濃紺: 両手法で支持された特徴量
COLOR_OTHER = "#b3b3b3"        # 淡グレー: それ以外


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

        # 2026-09-30: 統制変数（confound_age / confound_gender）も読む。宗田先生 #36 で
        # 本文の「統制変数の係数は表の脚に別記」という記述に対応する行が表に無いことが
        # 判明したため、同じ判定規則で脚に出せるようにする。
        # 図（フォレストプロット）は FEATURE_ORDER だけを描くので、行の追加は
        # gen_tab_coef_all5 側でのみ使う。存在しない場合は黙って落とす（旧データ互換）。
        ctrl = [c for c in ("confound_age", "confound_gender")
                if c in perm.index and c in boot.index]
        rows_idx = list(FEATURE_ORDER) + ctrl

        df = pd.DataFrame(index=rows_idx)
        df["coef_obs"] = perm.loc[rows_idx, "coef_obs"].astype(float)
        df["p_value"] = perm.loc[rows_idx, "p_value"].astype(float)
        df["perm_sig"] = df["p_value"] < 0.05
        df["coef_mean"] = boot.loc[rows_idx, "coef_mean"].astype(float)
        df["coef_sd"] = boot.loc[rows_idx, "coef_sd"].astype(float)
        df["ci_lower"] = boot.loc[rows_idx, "ci_lower"].astype(float)
        df["ci_upper"] = boot.loc[rows_idx, "ci_upper"].astype(float)
        df["boot_sig"] = boot.loc[rows_idx, "ci_excludes_zero"].astype(bool)
        df["both"] = df["perm_sig"] & df["boot_sig"]
        df["either"] = df["perm_sig"] | df["boot_sig"]
        out[trait] = df
    return out


def gen_tab_coef_all5(data: dict[str, pd.DataFrame], out_dir: Path,
                      lang: str = "ja") -> None:
    """本文用: 19特徴量 × 5次元の回帰係数一覧（両手法一致を * で示す）.

    lang="en" で英語版（tab_coef_all5_en.tex）を出力する。英語版原稿
    （paper1_en_20260922.tex）は同じ数値を英語ヘッダで参照するため、
    tab_feature_definitions_en.tex と同じ二枚看板の運用にする。
    """
    # 集計行の見出しは本文の呼称に合わせる。2026-09-21 の改称（C103: 「一致特徴量」→
    # 「両手法支持特徴量」）が表の行見出しに反映されておらず、本文キャプションが
    # 「両手法支持特徴量の個数」と述べているのに表側は旧称のままだったため合わせる。
    # 英語版も glossary で concordant を禁止語にしているので dually supported にする。
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

    # 次元ごとの一致特徴量数を集計行として付す
    # 集計は19特徴量のみ。統制変数の行を含めると「両手法支持特徴量数」が変わってしまう。
    n_both = [int(data[t].loc[FEATURE_ORDER, "both"].sum()) for t in TRAITS]
    n_novel = [
        int((data[t]["both"] & pd.Series({f: f in NOVEL_FEATURES for f in FEATURE_ORDER})).sum())
        for t in TRAITS
    ]

    # 2026-09-30 の宗田先生のご指摘 #36。本文は「統制変数（性別・年齢）の係数は表の脚に
    # 別記する」と書いていたが、実際には表に入っていなかった。結果節が具体値
    # （年齢 O −0.058 / C +0.057、性別 A +0.045）を引用しているのに表に無いのは不整合なので、
    # 脚に2行足す。判定は特徴量と同じ規則（両手法一致で *、片方のみで †）。
    ctrl_rows: list[str] = []
    ctrl_labels = {"confound_age": "Age (control)", "confound_gender": "Sex (control)"}
    for key, label in ctrl_labels.items():
        cells = []
        for trait in TRAITS:
            df = data[trait]
            if key not in df.index:
                cells.append("---")
                continue
            row = df.loc[key]
            s = f"{row['coef_obs']:+.3f}"
            if row["both"]:
                s = f"{s}$^{{*}}$"
            elif row["either"]:
                s = f"{s}$^{{\\dagger}}$"
            cells.append(s)
        ctrl_rows.append(f"{label} & " + " & ".join(cells) + r" \\")

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
        + "\n".join(ctrl_rows) + "\n"
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
    """付録用: 寄与判定の基準ごとに同定される特徴量数（置換のみ／Bootstrapのみ／両方）.

    本文は「両方を満たす」を判定条件にしている。どちらか一方だけを基準にすると
    同定数がどう変わるかを開示し、判定の厳しさが結果に与える影響を示す
    （2026-09-21 山下先生のご質問「有意な特徴量の数が変わったのはなぜか」への対応）。
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
    """付録用: 次元ごとに β_obs・p値・Bootstrap平均・SD・95%CI を全記載."""
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
    r"""5次元の係数フォレストプロット（特徴量をy軸・O/C/E/A/N を横に5列）.

    レイアウトの経緯
    ----------------
    2026-09-12 に「5行1列・特徴量をx軸」の縦積みへ変更したが、
    1パネルが横長・低背になるため各CIの長さが潰れて見え、
    19特徴量 × 5次元の違いが読み取りにくくなった（2026-09-21 山下先生ご指摘）。

    そこで「1行5列・特徴量をy軸」のフォレストプロットに戻す。
    ただし以前のように \rotatebox で90度回転させるのではなく、
    figure 環境を単独ページ（[p]）に置いて width=\linewidth で組む。
    これにより全ての文字が正立し、かつ縮小率を確保できる。

    サイズ
    ------
    本文幅は 406.87pt = 5.63in。他の図は width=\linewidth 〜 0.82\linewidth で
    組んでいるので、図4も \linewidth に収める（2026-09-21 ご指摘: 図4だけ幅が
    大きく他の図とバランスが悪い）。キャンバス幅を 6.8in に抑えると縮小率が 0.83 になり、
    8.5pt の軸ラベルが紙面上で約 7pt として読める。
    x 軸ラベルはパネルごとに5回並べず図全体で1つにして、横幅をパネルに回す。

    配色
    ----
    符号を色相で表すと「赤＝正／青＝負」という先入観と衝突するため、
    色相を意味に使わない。両手法で支持された特徴量は濃紺＋太線＋塗りマーカー、
    それ以外は淡グレー＋細線＋白抜きマーカーとし、符号はゼロ線の左右で読む。
    モノクロ印刷でも線幅とマーカーの塗りで区別できる。

    軸ラベル
    --------
    実装上の識別子ではなく説明的名称（FEATURE_LABELS_EN）を使う。
    識別子は表1と付録にのみ残す。
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
        ax.set_ylim(n - 0.5, -0.5)  # 上から FEATURE_ORDER の順
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

    # x 軸ラベルは図全体で1つ
    fig.supxlabel("Ridge coefficient (mean $\\pm$ 95% CI)", fontsize=8.5, y=0.055)

    # Classical / Novel の区切りは図には入れない。
    # 2026-09-21 のご指摘は「レイアウトを縦に戻す」「配色を変える」の2点なので、
    # それ以外の要素は前バージョンの図と同じに保つ。

    # 2026-09-30 の宗田先生のご指摘 #54「レジェンドの灰色の箇所は，意味がないため
    # 削除で良いかと思います」。灰色は「両手法に支持されなかった」＝支持された方の
    # 否定にすぎず、凡例に項目を立てる情報量がない。濃紺の項目だけを残し、
    # 残りが灰色であることはキャプションの1節で述べる。
    legend_handles = [
        Line2D([0], [0], color=COLOR_SUPPORTED, linewidth=2.0, marker="o",
               markerfacecolor=COLOR_SUPPORTED, markeredgecolor=COLOR_SUPPORTED,
               markersize=4.5,
               label="supported by both procedures "
                     "(permutation $p<.05$ and bootstrap 95% CI excluding zero)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=1, fontsize=7,
               frameon=False, bbox_to_anchor=(0.55, -0.035))

    fig.tight_layout(rect=(0, 0.075, 1, 1))
    path = out_dir / "fig_coef_all5.png"
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def print_summary(data: dict[str, pd.DataFrame]) -> None:
    print("\n=== 両手法一致（置換 p<0.05 かつ Bootstrap CI がゼロを除外）===")
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
    # 図は軸ラベル・凡例が元から英語なので言語別の二枚看板は不要
    for lang in ("ja", "en"):
        gen_tab_coef_all5(data, out_dir, lang=lang)
        gen_tab_coef_all5_detail(data, out_dir, lang=lang)
        gen_tab_criterion_counts(data, out_dir, lang=lang)
    gen_fig_coef_all5(data, out_dir)
    print_summary(data)


if __name__ == "__main__":
    main()
