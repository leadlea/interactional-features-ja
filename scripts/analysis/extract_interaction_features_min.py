#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, math, re
from pathlib import Path
import numpy as np
import pandas as pd

def norm_text(s: str) -> str:
    if s is None: return ""
    s=str(s).replace("\u3000"," ").strip()
    return s

_QMARK_RE = re.compile(r"[?？]")
_Q_END_RE = re.compile(r"(か|かな|かね|でしょう|でしょ|だろう|だろ|の)$")
_SFP_NE_RE = re.compile(r"(よね|だよね|ですよね|だよな|ね)$")
_SFP_YO_RE = re.compile(r"(だよ|ですよ|よ)$")

def is_question(text: str) -> bool:
    t=norm_text(text)
    if not t: return False
    if _QMARK_RE.search(t): return True
    tail=re.sub(r"[。．\.！!、,「」\"\'\s]+$","",t)
    return bool(_Q_END_RE.search(tail))

def sfp_group(text: str) -> str:
    t=norm_text(text)
    if not t: return "NONE"
    tail=re.sub(r"[。．\.！!、,「」\"\'\s]+$","",t)
    q=is_question(tail)
    if _SFP_NE_RE.search(tail): return "NE_Q" if q else "NE"
    if _SFP_YO_RE.search(tail): return "YO"
    return "OTHER"

_HEAD_TOKEN_RE = re.compile(r'^[\s、。,.!?！？「」（）()\[\]{}"\'\-]*([A-Za-z0-9]+|[ぁ-んァ-ヶ一-龠]+)')
def first_token(text: str) -> str:
    t=norm_text(text)
    if not t: return ""
    m=_HEAD_TOKEN_RE.match(t)
    return m.group(1) if m else ""

AIZUCHI_PREFIXES=["はい","うん","ええ","そう","そうそう","そっか","なるほど","へえ","了解","わかった","わかりました","OK","オーケー","あー","ああ","うーん"]
YESNO_PREFIXES=["はい","うん","ええ","そう","そうです","もちろん","了解","OK","オーケー","いいえ","いや","ううん","違う","ちがう","だめ","無理","そうじゃない"]
OIR_PREFIXES=["え？","えっ","えっ？","ん？","は？","なに？","何？","もう一回","もう一度","どういうこと","どういう意味","聞こえ","聞き取","わから","分から"]

def starts_with_any(text: str, prefixes):
    t=norm_text(text)
    return any(t.startswith(p) for p in prefixes) if t else False
def is_aizuchi(t:str)->bool: return starts_with_any(t,AIZUCHI_PREFIXES)
def is_yesno(t:str)->bool: return starts_with_any(t,YESNO_PREFIXES)
def is_oir(t:str)->bool: return starts_with_any(t,OIR_PREFIXES)

def char_bigrams(s:str):
    t=re.sub(r"\s+","",norm_text(s))
    return set() if len(t)<2 else {t[i:i+2] for i in range(len(t)-1)}
def jaccard(a:set,b:set)->float:
    if not a and not b: return float("nan")
    if not a or not b: return 0.0
    return len(a & b)/len(a | b) if len(a|b) else 0.0

def entropy_from_tokens(tokens):
    if len(tokens)==0: return float("nan")
    toks=[t if t else "<EMPTY>" for t in tokens]
    s=pd.Series(toks)
    p=s.value_counts(normalize=True)
    return float(-(p*np.log2(p)).sum())

RE_ETO=re.compile(r"(えっと|えと)")
RE_E=re.compile(r"(えー+|えぇ+|え〜+)")
RE_ANO=re.compile(r"(あの)")
def count_fillers(text:str):
    t=norm_text(text)
    eto=len(RE_ETO.findall(t))
    t2=RE_ETO.sub("",t)
    e=len(RE_E.findall(t2))
    ano=len(RE_ANO.findall(t))
    total=eto+e+ano
    return {"eto":eto,"e":e,"ano":ano,"total":total}

def q_stats(x):
    arr=np.array([v for v in x if v is not None and not (isinstance(v,float) and math.isnan(v))],dtype=float)
    if arr.size==0: return (float("nan"),float("nan"),float("nan"))
    return (float(arr.mean()),float(np.quantile(arr,0.5)),float(np.quantile(arr,0.9)))


def extract_features(
    utterances_df: pd.DataFrame,
    target_pairs: set | None,
    gap_tol: float = 0.05,
) -> pd.DataFrame:
    """Extract interaction features from utterances DataFrame.

    Args:
        utterances_df: DataFrame with columns conversation_id, speaker_id, text,
                       start_time, end_time.
        target_pairs: Set of (conversation_id, speaker_id) tuples to process,
                      or None to process all.
        gap_tol: Minimum gap duration (seconds) to count as a pause/gap.

    Returns:
        DataFrame with one row per (conversation_id, speaker_id) pair.
    """
    utt = utterances_df.copy()
    for c in ["conversation_id", "speaker_id"]:
        utt[c] = utt[c].astype(str)
    utt["text"] = utt.get("text", "").fillna("").astype(str)
    if "start_time" in utt.columns and "end_time" in utt.columns:
        utt["start_time"] = utt["start_time"].astype(float)
        utt["end_time"] = utt["end_time"].astype(float)
    else:
        utt["start_time"] = np.nan
        utt["end_time"] = np.nan

    if target_pairs is not None:
        utt = utt[utt["conversation_id"].isin({c for c, _ in target_pairs})].copy()

    utt["sfp_group"] = utt["text"].map(sfp_group)
    utt["is_question"] = utt["text"].map(is_question)

    sort_cols = ["conversation_id", "start_time", "end_time"]
    utt = utt.sort_values(sort_cols, kind="mergesort")

    rows = []
    for conv_id, u in utt.groupby("conversation_id", sort=False):
        u = u.copy()
        total_time = float(u["end_time"].max() - u["start_time"].min()) if not u["start_time"].isna().all() else float("nan")
        prev = u.shift(1)
        mask = (u["speaker_id"] != prev["speaker_id"]) & prev["speaker_id"].notna()
        pairs = pd.DataFrame({
            "conversation_id": conv_id,
            "prev_speaker_id": prev.loc[mask, "speaker_id"].astype(str).values,
            "resp_speaker_id": u.loc[mask, "speaker_id"].astype(str).values,
            "prev_text": prev.loc[mask, "text"].values,
            "resp_text": u.loc[mask, "text"].values,
            "prev_sfp_group": prev.loc[mask, "sfp_group"].values,
            "prev_is_question": prev.loc[mask, "is_question"].values,
            "prev_end": prev.loc[mask, "end_time"].values,
            "resp_start": u.loc[mask, "start_time"].values,
        })
        if len(pairs) == 0:
            continue
        pairs["resp_first_token"] = pairs["resp_text"].map(first_token)
        pairs["resp_is_aizuchi"] = pairs["resp_text"].map(is_aizuchi).astype(int)
        pairs["resp_is_yesno"] = pairs["resp_text"].map(is_yesno).astype(int)
        pairs["resp_is_oir"] = pairs["resp_text"].map(is_oir).astype(int)
        prev_bg = pairs["prev_text"].map(char_bigrams)
        resp_bg = pairs["resp_text"].map(char_bigrams)
        pairs["lex_overlap"] = [jaccard(a, b) for a, b in zip(prev_bg, resp_bg)]
        pairs["topic_drift"] = 1.0 - pairs["lex_overlap"]

        for spk, p in pairs.groupby("resp_speaker_id", sort=False):
            key = (conv_id, spk)
            if target_pairs is not None and key not in target_pairs:
                continue

            n_pairs_total = int(len(p))
            after_ne = p["prev_sfp_group"].isin(["NE", "NE_Q"])
            after_yo = p["prev_sfp_group"].isin(["YO"])
            after_q = p["prev_is_question"].astype(bool)

            n_after_ne = int(after_ne.sum())
            n_after_yo = int(after_yo.sum())
            n_after_q = int(after_q.sum())

            resp_ne_aiz = float(p.loc[after_ne, "resp_is_aizuchi"].mean()) if n_after_ne > 0 else float("nan")
            resp_ne_ent = float(entropy_from_tokens(p.loc[after_ne, "resp_first_token"].tolist())) if n_after_ne > 0 else float("nan")
            resp_yo_ent = float(entropy_from_tokens(p.loc[after_yo, "resp_first_token"].tolist())) if n_after_yo > 0 else float("nan")

            ix_oir = float(p["resp_is_oir"].mean())
            ix_yesno = float(p["resp_is_yesno"].mean())
            ix_lex = float(np.nanmean(p["lex_overlap"].values))
            ix_drift = float(np.nanmean(p["topic_drift"].values))

            ix_oir_q = float(p.loc[after_q, "resp_is_oir"].mean()) if n_after_q > 0 else float("nan")
            ix_yesno_q = float(p.loc[after_q, "resp_is_yesno"].mean()) if n_after_q > 0 else float("nan")

            us = u[u["speaker_id"] == spk].copy()
            if us.empty:
                pg_speech_ratio = float("nan")
                pause_mean = pause_p50 = pause_p90 = float("nan")
                pause_variability = float("nan")
                text_all = ""
            else:
                dur = (us["end_time"] - us["start_time"]).clip(lower=0.0)
                speech_time = float(dur.sum())
                pg_speech_ratio = float(speech_time / total_time) if total_time and not math.isnan(total_time) and total_time > 0 else float("nan")
                us = us.sort_values(["start_time", "end_time"], kind="mergesort")
                pauses = (us["start_time"].values[1:] - us["end_time"].values[:-1]).tolist()
                pauses = [x for x in pauses if x is not None and not math.isnan(x) and x >= gap_tol]
                pause_mean, pause_p50, pause_p90 = q_stats(pauses)
                # PG_pause_variability: CV = std / mean (mean > 0, len >= 2)
                if len(pauses) >= 2 and pause_mean > 0:
                    pause_variability = float(np.std(pauses, ddof=1) / pause_mean)
                else:
                    pause_variability = float("nan")
                text_all = "\n".join(us["text"].astype(str).tolist())

            gaps = (p["resp_start"].values - p["prev_end"].values).astype(float)
            overlap_rate = float((gaps < -gap_tol).mean()) if len(gaps) > 0 else float("nan")
            gap_list = [float(g) for g in gaps if (not math.isnan(g)) and g >= gap_tol]
            gap_mean, gap_p50, gap_p90 = q_stats(gap_list)

            fill = count_fillers(text_all)
            text_len = int(len(re.sub(r"\s+", "", text_all)))
            fill_rate100 = float(fill["total"] / (text_len / 100.0)) if text_len > 0 else float("nan")
            fill_has_any = float(np.mean([1 if count_fillers(t)["total"] > 0 else 0 for t in us["text"].astype(str).tolist()])) if not us.empty else float("nan")

            rows.append({
                "conversation_id": conv_id, "speaker_id": spk,
                "n_pairs_total": n_pairs_total,
                "n_pairs_after_NE": n_after_ne,
                "n_pairs_after_YO": n_after_yo,
                "IX_n_pairs": n_pairs_total,
                "IX_n_pairs_after_question": n_after_q,

                "RESP_NE_AIZUCHI_RATE": resp_ne_aiz,
                "RESP_NE_ENTROPY": resp_ne_ent,
                "RESP_YO_ENTROPY": resp_yo_ent,

                "IX_oirmarker_rate": ix_oir,
                "IX_yesno_rate": ix_yesno,
                "IX_lex_overlap_mean": ix_lex,
                "IX_topic_drift_mean": ix_drift,
                "IX_oirmarker_after_question_rate": ix_oir_q,
                "IX_yesno_after_question_rate": ix_yesno_q,

                "PG_total_time": total_time,
                "PG_speech_ratio": pg_speech_ratio,
                "PG_pause_mean": pause_mean,
                "PG_pause_p50": pause_p50,
                "PG_pause_p90": pause_p90,
                "PG_pause_variability": pause_variability,
                "PG_resp_gap_mean": gap_mean,
                "PG_resp_gap_p50": gap_p50,
                "PG_resp_gap_p90": gap_p90,
                "PG_overlap_rate": overlap_rate,
                "PG_resp_overlap_rate": overlap_rate,

                "FILL_text_len": text_len,
                "FILL_cnt_total": int(fill["total"]),
                "FILL_cnt_eto": int(fill["eto"]),
                "FILL_cnt_e": int(fill["e"]),
                "FILL_cnt_ano": int(fill["ano"]),
                "FILL_rate_per_100chars": fill_rate100,
                "FILL_has_any": fill_has_any,
            })

    out = pd.DataFrame(rows)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--utterances_parquet", required=True)
    ap.add_argument("--target_pairs_parquet", default="")
    ap.add_argument("--gap_tol", type=float, default=0.05)
    ap.add_argument("--out_parquet", required=True)
    args = ap.parse_args()

    utt = pd.read_parquet(args.utterances_parquet)

    target_pairs = None
    if args.target_pairs_parquet:
        tp = pd.read_parquet(args.target_pairs_parquet)[["conversation_id", "speaker_id"]].drop_duplicates()
        tp["conversation_id"] = tp["conversation_id"].astype(str)
        tp["speaker_id"] = tp["speaker_id"].astype(str)
        target_pairs = set(map(tuple, tp.to_records(index=False)))

    out = extract_features(utt, target_pairs, args.gap_tol)
    if out.empty:
        raise SystemExit("No rows produced.")
    Path(args.out_parquet).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out_parquet, index=False)
    print("OK:", args.out_parquet, "rows=", len(out))
    print(out.head(5).to_string(index=False))

if __name__ == "__main__":
    main()
