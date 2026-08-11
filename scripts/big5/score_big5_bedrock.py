import argparse
import json
import os
import time
import re
import atexit

import boto3
from botocore.config import Config
import numpy as np
import pandas as pd

# ===== Paper-aligned (Sample 2 / IPIP-NEO-120): 0..4 =====
SCALE = [
    "Very Inaccurate",
    "Moderately Inaccurate",
    "Neither Accurate Nor Inaccurate",
    "Moderately Accurate",
    "Very Accurate",
]
S2I = {s: i for i, s in enumerate(SCALE)}  # 0..4

PROMPT_TMPL_PAPER_S2_EN = """Your task is to respond to the following IPIP-NEO-120 question based on the participant’s daily diaries of the most significant event that occurred during the day, provided below. Respond as though you are the individual who generated these thoughts, reflecting their personality traits.
Base your answer on inferred personality traits. Think carefully about what the thoughts imply about tendencies and behaviors.
For each question, select the most appropriate option:
– Very Inaccurate: The statement is definitely false or the participant would strongly disagree with it.
– Moderately Inaccurate: The statement is mostly false or the participant would generally disagree with it.
– Neither Accurate Nor Inaccurate: The participant would be neutral on the statement, cannot decide, or find the statement equally true and false.
– Moderately Accurate: The statement is mostly true or the participant would generally agree with it.
– Very Accurate: The statement is definitely true or the participant would strongly agree with it.
IPIP-NEO-120 question to answer:
{statement}
Participant’s daily diaries:
{transcript}
Your response must be exactly one of:
Very Inaccurate
Moderately Inaccurate
Neither Accurate Nor Inaccurate
Moderately Accurate
Very Accurate
Do not include any explanation, punctuation, or additional text. Return only the exact phrase from the list above.
"""

# 日本語ラッパ（任意）：理解を助けつつ、選択肢は英語のまま固定
PROMPT_TMPL_PAPER_S2_JA = """あなたのタスクは、以下に示す参加者の「その日もっとも重要だった出来事」に関する日記（テキスト）を根拠として、IPIP-NEO-120 の質問項目に回答することです。
あなた自身がこの日記を書いた本人になりきって、その人の性格特性が反映されるように答えてください。
推論される性格特性に基づいて判断し、日記内容が示す傾向や行動をよく考えてください。

質問ごとに、最も適切な選択肢を次から1つだけ選んでください（必ず下の英語の文言をそのまま返してください）:

IPIP-NEO-120 question to answer:
{statement}

Participant’s daily diaries:
{transcript}

Your response must be exactly one of:
Very Inaccurate
Moderately Inaccurate
Neither Accurate Nor Inaccurate
Moderately Accurate
Very Accurate

Do not include any explanation, punctuation, or additional text. Return only the exact phrase from the list above.
"""

def normalize_choice(text: str):
    if text is None:
        return None
    t = " ".join(str(text).strip().split())
    if t in S2I:
        return t
    tl = t.lower()
    for s in SCALE:
        if s.lower() in tl:
            return s
    return None

def cronbach_alpha(item_matrix: np.ndarray):
    k = item_matrix.shape[1]
    if k < 2:
        return np.nan
    item_var = item_matrix.var(axis=0, ddof=1)
    total = item_matrix.sum(axis=1)
    total_var = total.var(ddof=1)
    if total_var <= 0:
        return np.nan
    return (k / (k - 1)) * (1 - item_var.sum() / total_var)

def _extract_text_blocks(converse_response: dict) -> str:
    blocks = (
        converse_response.get("output", {})
        .get("message", {})
        .get("content", [])
    )
    texts = []
    for b in blocks:
        if isinstance(b, dict) and "text" in b:
            t = str(b.get("text") or "").strip()
            if t:
                texts.append(t)
    return "\n".join(texts).strip()

def converse(client, model_id, prompt, max_tokens=128, temperature=0.0, top_p=1.0, seed=None):
    cur_max = int(max_tokens)
    last = None
    for _ in range(3):
        kwargs = {}
        if seed is not None and int(seed) != 0:
            kwargs["additionalModelRequestFields"] = {"seed": int(seed)}

        last = client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={
                "maxTokens": int(cur_max),
                "temperature": float(temperature),
                "topP": float(top_p),
            },
            **kwargs,
        )

        out_text = _extract_text_blocks(last)
        if out_text:
            return out_text

        cur_max = min(cur_max * 4, 512)

    try:
        return json.dumps(last.get("output", {}).get("message", {}).get("content", []), ensure_ascii=False)
    except Exception:
        return ""

def safe_model_dir(model_id: str) -> str:
    # path 用に安全化（. や : を潰す）
    return re.sub(r"[^A-Za-z0-9]+", "_", model_id).strip("_")



### file append cache (avoid open() per line)
_APPEND_FPS = {}

def _close_append_fps():
    for _fp in list(_APPEND_FPS.values()):
        try:
            _fp.close()
        except Exception:
            pass

atexit.register(_close_append_fps)

def _append_line(path: str, line: str) -> None:
    """Append one line to a jsonl/text file (UTF-8) with cached file handles."""
    # ensure parent dir exists
    import os
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    fp = _APPEND_FPS.get(path)
    if fp is None:
        fp = open(path, "a", encoding="utf-8")
        _APPEND_FPS[path] = fp
    fp.write(line)
    # flush lightly: jsonl is our checkpoint
    fp.flush()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--monologues_parquet", required=True)
    ap.add_argument("--items_csv", required=True)
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--region", default="ap-northeast-1")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--limit_subjects", type=int, default=0)
    ap.add_argument("--limit_items", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--prompt_lang", choices=["en", "ja"], default="en")
    ap.add_argument("--max_attempts", type=int, default=20)
    ap.add_argument("--on_fail", choices=["nan", "neutral"], default="nan")
    ap.add_argument("--paper_strict", action="store_true",
                  help="Paper-style strict: accept only exact canonical choice strings; invalid -> retry/NA")
    ap.add_argument("--max_retries", type=int, default=5)
    ap.add_argument("--attempts_jsonl", type=str, default=None,
                  help="Write per-attempt logs as JSON Lines (paper-like invalid->reissue evidence). Default: <out_dir>/attempts.jsonl")
    args = ap.parse_args()

    df = pd.read_parquet(args.monologues_parquet)
    items = pd.read_csv(args.items_csv)

    for c in ["conversation_id", "speaker_id", "text"]:
        if c not in df.columns:
            raise SystemExit(f"monologues missing {c}")
    for c in ["item_id", "trait", "reverse", "text"]:
        if c not in items.columns:
            raise SystemExit(f"items missing {c}")

    if args.limit_subjects and args.limit_subjects > 0:
        df = df.head(args.limit_subjects).copy()
    if args.limit_items and args.limit_items > 0:
        items = items.head(args.limit_items).copy()

    tmpl = PROMPT_TMPL_PAPER_S2_EN if args.prompt_lang == "en" else PROMPT_TMPL_PAPER_S2_JA

    out_rows = []
    done_keys = set()
    item_jsonl_path = os.path.join(args.out_dir, 'item_scores.jsonl')
    if os.path.exists(item_jsonl_path):
        try:
            with open(item_jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line=line.strip()
                    if not line: continue
                    r=json.loads(line)
                    out_rows.append(r)
                    # persist one row for resume
                    try:
                        rec = out_rows[-1]
                        _append_line(item_jsonl_path, json.dumps(rec, ensure_ascii=False) + '\n')
                        try:
                            done_keys.add((str(rec.get('conversation_id','')), str(rec.get('speaker_id','')), int(rec.get('item_id',-1))))
                        except Exception:
                            pass
                    except Exception as e:
                        print('[resume] failed to write item_scores.jsonl:', e)
                    try:
                        done_keys.add((str(r.get('conversation_id','')), str(r.get('speaker_id','')), int(r.get('item_id',-1))))
                    except Exception:
                        pass
            if done_keys:
                print('[resume] loaded', len(done_keys), 'done keys from', item_jsonl_path)
        except Exception as e:
            print('[resume] failed to read item_scores.jsonl:', e)
    client = boto3.client("bedrock-runtime", region_name=args.region, config=Config(connect_timeout=10, read_timeout=120, retries={'max_attempts': 8, 'mode': 'adaptive'}))

    for _, row in df.iterrows():
        cid, sid, transcript = row["conversation_id"], row["speaker_id"], row["text"]

        for _, it in items.iterrows():
            # --- paper-like strict retry + attempts logging (suite-wrap-v2) ---
            import time
            attempts_path = args.attempts_jsonl or os.path.join(getattr(args,'out_dir','.'), 'attempts.jsonl')
            max_try = int(getattr(args,'max_retries',5)) if getattr(args,'paper_strict',False) else 1
            for attempt_i in range(1, max_try + 1):
                item_id, trait, rev, stmt = it["item_id"], it["trait"], int(it["reverse"]), it["text"]
                prompt = tmpl.format(transcript=transcript, statement=stmt)
                
                choice = None
                raw = None
                
                for attempt in range(1, args.max_attempts + 1):
                    # resume-skip
                    try:
                        _key = (str(conversation_id), str(speaker_id), int(item_id))
                    except Exception:
                        _key = None
                    if _key is not None and 'done_keys' in locals() and _key in done_keys:
                        continue
                    raw = converse(
                        client,
                        args.model_id,
                        prompt,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        seed=args.seed,
                    )
                    choice = normalize_choice(raw)
                    if choice:
                        break
                
                    # Paper: invalid -> discard & re-issue (we append a short reminder)
                    prompt = prompt + "\nREMINDER: Return only ONE exact phrase from the list.\n"
                
                if not choice:
                    if args.on_fail == "neutral":
                        choice = "Neither Accurate Nor Inaccurate"
                        score = float(S2I[choice])
                    else:
                        score = np.nan
                else:
                    score = float(S2I[choice])
                
                if rev == 1 and score == score:
                    score = 4.0 - score  # 0..4 の反転
                
                out_rows.append(
                    {
                        "conversation_id": cid,
                        "speaker_id": sid,
                        "item_id": item_id,
                        "trait": trait,
                        "reverse": rev,
                        "choice_raw": raw,
                        "choice_norm": choice,
                        "score": score,
                        "model_id": args.model_id,
                    }
                )
                
                if args.sleep and args.sleep > 0:
                    time.sleep(args.sleep)
                
                _last = out_rows[-1] if out_rows else None
                _raw = str((_last or {}).get('choice_raw','')).strip()
                _norm0 = str((_last or {}).get('choice_norm','')).strip() or _raw
                _norm_cf = ' '.join(_norm0.split()).casefold()
                _CANON = {
                    'very inaccurate': 'Very Inaccurate',
                    'moderately inaccurate': 'Moderately Inaccurate',
                    'neither accurate nor inaccurate': 'Neither Accurate nor Inaccurate',
                    'moderately accurate': 'Moderately Accurate',
                    'very accurate': 'Very Accurate',
                }
                _norm = _CANON.get(_norm_cf)
                if (_last is not None) and (_norm is not None):
                    _last['choice_norm'] = _norm  # force paper canonical casing
                _CHOICES_PAPER = set(_CANON.values())
                _valid_exact = (_raw in _CHOICES_PAPER)
                _valid = (_norm is not None) if getattr(args,'paper_strict',False) else True
                try:
                    rec = {'ts': time.time(), 'model_id': str(getattr(args,'model_id','')), 'conversation_id': str(((_last or {}).get('conversation_id')) or locals().get('conversation_id','')), 'speaker_id': str(((_last or {}).get('speaker_id')) or locals().get('speaker_id','')), 'item_id': (int(((_last or {}).get('item_id')) or locals().get('item_id',-1))           if str(((_last or {}).get('item_id')) or locals().get('item_id',-1)).lstrip('-').isdigit() else -1), 'trait': str(((_last or {}).get('trait')) or locals().get('trait','')), 'reverse': (int(((_last or {}).get('reverse')) or locals().get('reverse',0))            if str(((_last or {}).get('reverse')) or locals().get('reverse',0)).lstrip('-').isdigit() else 0), 'attempt_i': int(attempt_i), 'choice_raw': _raw, 'choice_norm': _norm, 'valid': bool(_valid), 'valid_exact': bool(locals().get('_valid_exact', False))}
                    with open(attempts_path,'a',encoding='utf-8') as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                except Exception:
                    pass
                if getattr(args,'paper_strict',False) and (not _valid):
                    if out_rows: out_rows.pop()
                    continue
                break
    out = pd.DataFrame(out_rows)
    os.makedirs(args.out_dir, exist_ok=True)

    out_path = os.path.join(args.out_dir, "item_scores.parquet")
    out.to_parquet(out_path, index=False)

    trait_scores = (
        out.groupby(["conversation_id", "speaker_id", "trait", "model_id"])["score"]
        .mean()
        .reset_index()
        .rename(columns={"score": "trait_score"})
    )
    ts_path = os.path.join(args.out_dir, "trait_scores.parquet")
    trait_scores.to_parquet(ts_path, index=False)

    alpha_rows = []
    for trait in sorted(out["trait"].unique()):
        sub = out[out["trait"] == trait].copy()
        pivot = sub.pivot_table(
            index=["conversation_id", "speaker_id"],
            columns="item_id",
            values="score",
            aggfunc="mean",
        )
        pivot = pivot.dropna(axis=0)
        if len(pivot) >= 3 and pivot.shape[1] >= 2:
            a = cronbach_alpha(pivot.values.astype(float))
        else:
            a = np.nan
        alpha_rows.append(
            {
                "trait": trait,
                "alpha": float(a) if a == a else np.nan,
                "n_subjects": int(len(pivot)),
                "k_items": int(pivot.shape[1]),
            }
        )

    ar = pd.DataFrame(alpha_rows)
    ar_path = os.path.join(args.out_dir, "cronbach_alpha.csv")
    ar.to_csv(ar_path, index=False)

    print("OK:", out_path)
    print("OK:", ts_path)
    print("OK:", ar_path)
    print(ar.to_string(index=False))

if __name__ == "__main__":
    main()
