import argparse
import os
import re
import pandas as pd

TRAITS = ["A", "C", "E", "N", "O"]

def display_model_name(model_id: str) -> str:
    s = model_id
    s = s.replace("global.anthropic.claude-sonnet-4-20250514-v1:0", "Claude Sonnet 4")
    s = s.replace("deepseek.v3-v1:0", "DeepSeek V3")
    s = s.replace("openai.gpt-oss-120b-1:0", "GPT-OSS 120B")
    s = s.replace("qwen.qwen3-235b-a22b-2507-v1:0", "Qwen3-235B")
    return s

def sq_alpha(a: float) -> str:
    if a != a:
        return "⬜"
    if a >= 0.70:
        return "🟩"
    if a >= 0.50:
        return "🟨"
    return "🟥"

def sq_score(x: float) -> str:
    # 8.3 の色：<3.0 green / 3.0-3.4 yellow / >3.4 orange（あなたのスクショに合わせ）
    if x != x:
        return "⬜"
    if x < 3.0:
        return "🟩"
    if x > 3.4:
        return "🟧"
    return "🟨"

def read_model_dirs(root: str):
    # llm_avg_manifest_best.csv があればそれを優先
    mpath = os.path.join(root, "llm_avg_manifest_best.csv")
    if os.path.exists(mpath):
        man = pd.read_csv(mpath)
        return man[["model_id", "path"]].to_dict("records")

    # fallback: model=*/trait_scores.parquet
    rows = []
    for d in sorted(os.listdir(root)):
        if not d.startswith("model="):
            continue
        p = os.path.join(root, d, "trait_scores.parquet")
        if os.path.exists(p):
            df = pd.read_parquet(p)
            mid = df["model_id"].iloc[0] if "model_id" in df.columns and len(df) else d
            rows.append({"model_id": mid, "path": p})
    return rows

def md_table(df: pd.DataFrame) -> str:
    # df columns: Model + A..O (already formatted strings)
    cols = ["Model"] + TRAITS
    out = []
    out.append("| " + " | ".join(cols) + " |")
    out.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="artifacts/big5/llm_scores")
    ap.add_argument("--out_md", default="artifacts/big5/llm_scores/report_tables.md")
    args = ap.parse_args()

    root = args.root
    models = read_model_dirs(root)

    # 8.2 Cronbach alpha table
    alpha_rows = []
    for m in models:
        p = m["path"]
        d = os.path.dirname(p)
        ca = os.path.join(d, "cronbach_alpha.csv")
        if not os.path.exists(ca):
            continue
        adf = pd.read_csv(ca)
        a = {t: float("nan") for t in TRAITS}
        for _, r in adf.iterrows():
            t = str(r["trait"])
            if t in a:
                a[t] = float(r["alpha"]) if r["alpha"] == r["alpha"] else float("nan")
        alpha_rows.append({"Model": display_model_name(m["model_id"]), **a})

    alpha_df = pd.DataFrame(alpha_rows)
    for t in TRAITS:
        alpha_df[t] = alpha_df[t].map(lambda v: f"{sq_alpha(v)}{v:.3f}" if v == v else "⬜NA")

    # 8.3 Model-wise mean Big5 (from subject_mean parquet if exists, else pivot from trait_scores)
    mean_rows = []
    for m in models:
        p = m["path"]
        d = os.path.dirname(p)
        smp = os.path.join(d, "trait_scores_subject_mean.parquet")
        if os.path.exists(smp):
            sm = pd.read_parquet(smp)
            vals = {t: float(sm[t].mean()) for t in TRAITS}
        else:
            ts = pd.read_parquet(p)
            wide = (ts.pivot_table(index=["conversation_id","speaker_id"],
                                   columns="trait", values="trait_score", aggfunc="mean")
                      .reset_index())
            for t in TRAITS:
                if t not in wide.columns:
                    wide[t] = float("nan")
            vals = {t: float(wide[t].mean()) for t in TRAITS}
        mean_rows.append({"Model": display_model_name(m["model_id"]), **vals})

    mean_df = pd.DataFrame(mean_rows)
    for t in TRAITS:
        mean_df[t] = mean_df[t].map(lambda v: f"{sq_score(v)}{v:.2f}" if v == v else "⬜NA")

    # LLM Mean row (if exists)
    llm_mean_path = os.path.join(root, "trait_scores_llm_average_strict_allmodels.parquet")
    sd_row = None
    if os.path.exists(llm_mean_path):
        g = pd.read_parquet(llm_mean_path)
        mean_vals = {t: float(g[t].mean()) for t in TRAITS}
        mean_df = pd.concat([mean_df, pd.DataFrame([{"Model": "LLM Mean (all models)", **mean_vals}])], ignore_index=True)
        for t in TRAITS:
            mean_df.loc[mean_df["Model"] == "LLM Mean (all models)", t] = f"{sq_score(mean_vals[t])}{mean_vals[t]:.2f}"

        sd_vals = {t: float(g[f"sd_{t}"].mean()) for t in TRAITS}
        sd_row = sd_vals

    # write
    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)

    md = []
    md.append("## 8.2 Cronbach’s alpha（モデル別 / trait別）")
    md.append("凡例：🟩>=0.70 / 🟨0.50–0.70 / 🟥<0.50")
    md.append("")
    md.append(md_table(alpha_df))
    md.append("")
    md.append("## 8.3 モデル別：平均Big5（50subjectの平均）")
    md.append("凡例：🟩<3.0 / 🟨3.0–3.4 / 🟧>3.4（※色は見やすさのための区分）")
    md.append("")
    md.append(md_table(mean_df))
    md.append("")
    if sd_row is not None:
        md.append("補足：モデル間のばらつき（subjectごとの4モデルSDの平均）")
        md.append("")
        md.append("| Metric | " + " | ".join(TRAITS) + " |")
        md.append("|---|" + "|".join(["---"] * len(TRAITS)) + "|")
        md.append("| mean(sd across models) | " + " | ".join(f"{sd_row[t]:.3f}" for t in TRAITS) + " |")
        md.append("")

    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print("OK:", args.out_md)

if __name__ == "__main__":
    main()
