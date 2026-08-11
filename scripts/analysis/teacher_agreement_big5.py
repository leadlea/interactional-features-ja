#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TRAITS = ["C","A","E","N","O"]
TEACHERS = [
    ("sonnet4",    "sonnet4"),
    ("qwen3-235b", "qwen3-235b"),
    ("deepseek-v3","deepseek-v3"),
    ("gpt-oss-120b","gpt-oss-120b"),
]

def teacher_parquet(trait: str, teacher: str) -> Path:
    # teacher_merged の parquet を読む（teacher別に付けたディレクトリ）
    return Path(f"artifacts/big5/llm_scores/dataset=cejc_home2_hq1_v1__items={trait}24__teacher={teacher}/teacher_merged/trait_scores_{trait}_merged.parquet")

def load_trait_scores(trait: str) -> pd.DataFrame:
    dfs=[]
    for label, teacher in TEACHERS:
        p = teacher_parquet(trait, teacher)
        if not p.exists():
            raise FileNotFoundError(f"missing: {p}")
        df = pd.read_parquet(p)[["conversation_id","speaker_id","trait_score"]].copy()
        df = df.rename(columns={"trait_score": label})
        df["conversation_id"]=df["conversation_id"].astype(str)
        df["speaker_id"]=df["speaker_id"].astype(str)
        dfs.append(df)

    out = dfs[0]
    for d in dfs[1:]:
        out = out.merge(d, on=["conversation_id","speaker_id"], how="inner")
    return out

def corr_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cols=[t[0] for t in TEACHERS]
    return df[cols].corr(method="pearson")

def heatmap(corr: pd.DataFrame, title: str, out_png: Path):
    fig = plt.figure(figsize=(6.5, 5.8))
    ax = plt.gca()
    im = ax.imshow(corr.values, vmin=-1, vmax=1)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)

    # 数値アノテ
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center", fontsize=9)

    ax.set_title(title, pad=14)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def main():
    out_fig = Path("reports/model_agreement/figs")
    out_fig.mkdir(parents=True, exist_ok=True)
    out_md = Path("reports/model_agreement/model_agreement_big5.md")

    md=[]
    md.append("# Teacher agreement (Big5)\n\n")
    md.append("Same (conversation_id, speaker_id) trait scores across teachers; Pearson correlation.\n\n")
    md.append("Teachers: Sonnet4 / Qwen3-235B / DeepSeek-V3 / GPT-OSS-120B\n\n")

    summary_rows=[]
    for tr in TRAITS:
        df = load_trait_scores(tr)
        corr = corr_matrix(df)

        # 保存
        tsv = Path(f"reports/model_agreement/teacher_corr_{tr}.tsv")
        tsv.parent.mkdir(parents=True, exist_ok=True)
        corr.to_csv(tsv, sep="\t")

        png = out_fig / f"teacher_corr_{tr}.png"
        heatmap(corr, f"Teacher agreement — {tr} (Pearson r)", png)

        # 平均相関（対角除外）
        vals = corr.values
        m = vals[~np.eye(vals.shape[0], dtype=bool)].mean()
        summary_rows.append((tr, float(m), len(df)))

        md.append(f"## {tr}\n\n")
        md.append(f"- N pairs: {len(df)}\n")
        md.append(f"- Mean off-diagonal r: {m:.3f}\n\n")
        md.append(f"![teacher_corr_{tr}](figs/teacher_corr_{tr}.png)\n\n")

    md.append("## Summary\n\n")
    md.append("| Trait | N | mean off-diagonal r |\n")
    md.append("|---:|---:|---:|\n")
    for tr, m, n in summary_rows:
        md.append(f"| {tr} | {n} | {m:.3f} |\n")

    out_md.write_text("".join(md), encoding="utf-8")
    print("OK:", out_md)

if __name__=="__main__":
    main()
