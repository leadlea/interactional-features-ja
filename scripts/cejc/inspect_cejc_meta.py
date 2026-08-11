import argparse
import re
import pandas as pd

DEFAULT_URL = "https://www2.ninjal.ac.jp/conversation/cejc/convList.html"

POS_KW = ["自宅","雑談","食事","休息","家族","友人","買い物","移動","夕食","朝食","昼食","くつろ","子ども","子供"]
NEG_KW = ["会議","打合せ","授業","レッスン","講義","職場","学校","研究","面接","インタビュー","病院","歯科","発表","プレゼン"]

def _norm_int(x):
    s = re.sub(r"\D", "", str(x))
    return int(s) if s else None

def diary_score_row(r: pd.Series):
    # CEJC: 個人密着法= Wで始まらない / 特定場面法= Wで始まる（サイト説明に準拠）
    cid = str(r.get("会話ID",""))
    method = "specific_scene" if cid.startswith("W") else "personal_follow"

    score = 0
    if method == "personal_follow":
        score += 2

    blob = " ".join([
        str(r.get("会話概要","")),
        str(r.get("形式","")),
        str(r.get("場所","")),
        str(r.get("活動","")),
        str(r.get("話者間の関係性","")),
    ])

    for kw in POS_KW:
        if kw in blob: score += 1
    for kw in NEG_KW:
        if kw in blob: score -= 2

    n = _norm_int(r.get("話者数",""))
    if n is not None:
        if n == 1: score += 4
        elif n <= 3: score += 1
        elif n >= 6: score -= 2

    return score, method, n

def load_convlist(url: str) -> pd.DataFrame:
    # lxmlが入っていればここが通る
    tables = pd.read_html(url, keep_default_na=False)
    dfs = []
    for t in tables:
        if "会話ID" in t.columns and "会話概要" in t.columns:
            dfs.append(t)
    if not dfs:
        raise RuntimeError("convList table not found. Page structure may have changed.")
    conv = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["会話ID"])
    return conv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--convlist_url", default=DEFAULT_URL)
    ap.add_argument("--out_parquet", default="artifacts/tmp_meta/cejc_convlist.parquet")
    ap.add_argument("--out_top_tsv", default="artifacts/tmp_meta/cejc_diary_candidates_top200.tsv")
    ap.add_argument("--topk", type=int, default=200)
    args = ap.parse_args()

    conv = load_convlist(args.convlist_url)

    scored = conv.copy()
    tmp = scored.apply(lambda r: diary_score_row(r), axis=1, result_type="expand")
    scored["diary_score"] = tmp[0].astype(int)
    scored["method"] = tmp[1].astype(str)
    scored["speaker_n"] = tmp[2]

    scored.to_parquet(args.out_parquet, index=False)

    top = scored.sort_values(["diary_score","method"], ascending=[False, True]).head(args.topk)
    top.to_csv(args.out_top_tsv, sep="\t", index=False)

    print("OK wrote:", args.out_parquet, "rows=", len(scored), "cols=", len(scored.columns))
    print("OK wrote:", args.out_top_tsv, "topk=", args.topk)
    show_cols = [c for c in ["会話ID","diary_score","method","話者数","形式","場所","活動","話者間の関係性","会話概要"] if c in top.columns]
    print(top[show_cols].head(30).to_string(index=False))

if __name__ == "__main__":
    main()
