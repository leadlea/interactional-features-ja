#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import boto3
import pandas as pd

CHOICES_CANONICAL = [
    "Very Inaccurate",
    "Moderately Inaccurate",
    "Neither Accurate nor Inaccurate",
    "Moderately Accurate",
    "Very Accurate",
]
CHOICE2SCORE = {
    "Very Inaccurate": 0,
    "Moderately Inaccurate": 1,
    "Neither Accurate nor Inaccurate": 2,
    "Moderately Accurate": 3,
    "Very Accurate": 4,
}

# Normalize common variants (case, extra spaces, Nor/nor, punctuation)
def normalize_choice(s: str) -> str:
    if s is None:
        return ""
    t = str(s).strip()

    # take first line only (paper wants single line)
    lines = t.splitlines()
    t = lines[0].strip() if lines else ""

    # remove surrounding quotes
    if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
        t = t[1:-1].strip()

    # collapse spaces
    t = " ".join(t.split())

    # normalize case-insensitive matching
    low = t.lower()

    # numeric shortcuts
    if low in {"0","1","2","3","4"}:
        return CHOICES_CANONICAL[int(low)]

    # map frequent text variants
    variants = {
        "very inaccurate": "Very Inaccurate",
        "moderately inaccurate": "Moderately Inaccurate",
        "neither accurate nor inaccurate": "Neither Accurate nor Inaccurate",
        "neither accurate nor inaccurate.": "Neither Accurate nor Inaccurate",
        "neither accurate nor inaccurate,": "Neither Accurate nor Inaccurate",
        "neither accurate nor inaccurate!": "Neither Accurate nor Inaccurate",
        "neither accurate nor inaccurate?": "Neither Accurate nor Inaccurate",
        "neither accurate nor inaccurate;": "Neither Accurate nor Inaccurate",
        "neither accurate nor inaccurate:": "Neither Accurate nor Inaccurate",
        "neither accurate nor inaccurate (neutral)": "Neither Accurate nor Inaccurate",
        "neither accurate nor inaccurate (neither)": "Neither Accurate nor Inaccurate",
        "neither accurate nor inaccurate (neither accurate nor inaccurate)": "Neither Accurate nor Inaccurate",
        "moderately accurate": "Moderately Accurate",
        "very accurate": "Very Accurate",
    }
    # handle "Neither Accurate Nor Inaccurate" (Nor capitalized) etc.
    low = low.replace(" nor ", " nor ").replace(" nor", " nor").replace("nor ", "nor ")
    if low in variants:
        return variants[low]

    # last: try case-insensitive match against canonical
    for c in CHOICES_CANONICAL:
        if low == c.lower():
            return c

    return t  # return original-ish; caller decides validity


@dataclass
class JsonlWriter:
    path: str
    _fp: Optional[object] = None
    n: int = 0

    def open(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._fp = open(self.path, "a", encoding="utf-8")

    def write_obj(self, obj: Dict):
        if self._fp is None:
            self.open()
        self._fp.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self._fp.flush()
        self.n += 1

    def close(self):
        try:
            if self._fp is not None:
                self._fp.close()
        finally:
            self._fp = None


def bedrock_converse(
    client,
    model_id: str,
    system_text: str,
    user_text: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    resp = client.converse(
        modelId=model_id,
        system=[{"text": system_text}],
        messages=[{"role": "user", "content": [{"text": user_text}]}],
        inferenceConfig={
            "maxTokens": int(max_tokens),
            "temperature": float(temperature),
            "topP": float(top_p),
        },
    )
    # Get assistant text
    try:
        parts = resp["output"]["message"]["content"]
        out = "".join([p.get("text","") for p in parts if isinstance(p, dict)])
    except Exception:
        out = str(resp)
    return out.strip()


def build_prompt_paper_strict(monologue_text: str, item_text: str) -> Tuple[str, str]:
    system = (
        "You are the participant described by the narrative. "
        "Answer the survey item as that person. "
        "You must output ONLY one of the allowed choices, exactly."
    )
    choices = "\n".join(CHOICES_CANONICAL)
    user = (
        "Below is a narrative written by a participant (the person you must respond as).\n\n"
        "=== NARRATIVE (participant diary / transcript) ===\n"
        f"{monologue_text}\n"
        "=== END NARRATIVE ===\n\n"
        "Now answer the following personality item as though you are that individual.\n"
        "Output MUST be exactly one of the following choices (no extra characters, no punctuation, no explanation):\n"
        f"{choices}\n\n"
        f"ITEM: {item_text}\n"
        "ANSWER:"
    )
    return system, user


def load_done_keys(item_jsonl_path: str) -> set:
    done = set()
    if not os.path.exists(item_jsonl_path):
        return done
    with open(item_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done.add((r["conversation_id"], r["speaker_id"], int(r["item_id"])))
            except Exception:
                # ignore broken lines
                continue
    return done


def cronbach_alpha(df_items_wide: pd.DataFrame) -> float:
    # df: rows=subjects, cols=items
    k = df_items_wide.shape[1]
    if k < 2:
        return float("nan")
    # variance per item
    item_var = df_items_wide.var(axis=0, ddof=1)
    total_score = df_items_wide.sum(axis=1)
    total_var = total_score.var(ddof=1)
    if total_var == 0 or pd.isna(total_var):
        return float("nan")
    alpha = (k / (k - 1.0)) * (1.0 - (item_var.sum() / total_var))
    return float(alpha)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--monologues_parquet", required=True)
    ap.add_argument("--items_csv", required=True)
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--region", default="ap-northeast-1")
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--limit_subjects", type=int, default=None)
    ap.add_argument("--limit_items", type=int, default=None)

    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)

    ap.add_argument("--paper_strict", action="store_true")
    ap.add_argument("--max_retries", type=int, default=5)
    ap.add_argument("--attempts_jsonl", default=None)

    ap.add_argument("--on_fail", choices=["nan", "neutral"], default="neutral")

    args = ap.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    item_jsonl_path = os.path.join(out_dir, "item_scores.jsonl")
    attempts_jsonl_path = args.attempts_jsonl or os.path.join(out_dir, "attempts.jsonl")

    # Load data
    mono = pd.read_parquet(args.monologues_parquet)
    items = pd.read_csv(args.items_csv)

    if args.limit_subjects is not None:
        mono = mono.head(args.limit_subjects).copy()
    if args.limit_items is not None:
        items = items.head(args.limit_items).copy()

    # Resume
    done_keys = load_done_keys(item_jsonl_path)
    if done_keys:
        print(f"[resume] loaded {len(done_keys)} done keys from {item_jsonl_path}", file=sys.stderr)

    # Writers
    w_item = JsonlWriter(item_jsonl_path)
    w_attempt = JsonlWriter(attempts_jsonl_path)

    client = boto3.client("bedrock-runtime", region_name=args.region)

    # scoring loop
    for _, srow in mono.iterrows():
        conv_id = str(srow["conversation_id"])
        spk_id = str(srow["speaker_id"])
        narrative = str(srow["text"])

        for _, irow in items.iterrows():
            item_id = int(irow["item_id"])
            trait = str(irow["trait"])
            reverse = int(irow["reverse"])
            item_text = str(irow["text"])

            key = (conv_id, spk_id, item_id)
            if key in done_keys:
                continue

            system_text, user_text = build_prompt_paper_strict(narrative, item_text)

            final_choice_raw = ""
            final_choice_norm = ""
            final_valid = False
            final_score = None

            for attempt_i in range(1, int(args.max_retries) + 1):
                raw = bedrock_converse(
                    client=client,
                    model_id=args.model_id,
                    system_text=system_text,
                    user_text=user_text,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                choice_raw = raw.strip()
                choice_norm = normalize_choice(choice_raw)

                valid = choice_norm in CHOICES_CANONICAL

                # attempts log (small)
                w_attempt.write_obj({
                    "conversation_id": conv_id,
                    "speaker_id": spk_id,
                    "item_id": item_id,
                    "attempt_i": attempt_i,
                    "choice_raw": choice_raw,
                    "choice_norm": choice_norm,
                    "valid": bool(valid),
                    "model_id": args.model_id,
                })

                if valid:
                    score = float(CHOICE2SCORE[choice_norm])
                    if reverse == 1:
                        score = float(4 - score)
                    final_choice_raw = choice_raw
                    final_choice_norm = choice_norm
                    final_valid = True
                    final_score = score
                    break

                if args.sleep:
                    time.sleep(args.sleep)

            if not final_valid:
                if args.on_fail == "nan":
                    final_score = float("nan")
                    final_choice_norm = ""
                    final_choice_raw = ""
                else:
                    # neutral
                    neutral = "Neither Accurate nor Inaccurate"
                    score = float(CHOICE2SCORE[neutral])
                    if reverse == 1:
                        score = float(4 - score)
                    final_choice_raw = final_choice_raw or "NEUTRAL_FALLBACK"
                    final_choice_norm = neutral
                    final_score = score

            # item_scores jsonl (STRICTLY SMALL fields only!)
            rec_small = {
                "conversation_id": conv_id,
                "speaker_id": spk_id,
                "item_id": item_id,
                "trait": trait,
                "reverse": reverse,
                "choice_raw": final_choice_raw,
                "choice_norm": final_choice_norm,
                "score": float(final_score) if final_score is not None else float("nan"),
                "model_id": args.model_id,
            }
            w_item.write_obj(rec_small)
            done_keys.add(key)

            if args.sleep:
                time.sleep(args.sleep)

    # close writers
    w_item.close()
    w_attempt.close()

    # materialize parquet/csv
    df = pd.read_json(item_jsonl_path, lines=True)
    df.to_parquet(os.path.join(out_dir, "item_scores.parquet"), index=False)

    # trait means per subject
    trait_scores = (
        df.groupby(["conversation_id","speaker_id","trait"], as_index=False)["score"]
          .mean()
          .rename(columns={"score":"trait_score"})
    )
    trait_scores.to_parquet(os.path.join(out_dir, "trait_scores.parquet"), index=False)

    # cronbach alpha per trait (need >=2 subjects and >=2 items)
    alphas = []
    for tr, g in df.groupby("trait"):
        wide = g.pivot_table(
            index=["conversation_id","speaker_id"],
            columns="item_id",
            values="score",
            aggfunc="mean",
        )
        n_subj = wide.shape[0]
        k_items = wide.shape[1]
        a = cronbach_alpha(wide.dropna(axis=0, how="any")) if n_subj >= 2 else float("nan")
        alphas.append({"trait": tr, "alpha": a, "n_subjects": int(n_subj), "k_items": int(k_items)})
    pd.DataFrame(alphas).to_csv(os.path.join(out_dir, "cronbach_alpha.csv"), index=False)

    print("OK:", os.path.join(out_dir, "item_scores.parquet"))
    print("OK:", os.path.join(out_dir, "trait_scores.parquet"))
    print("OK:", os.path.join(out_dir, "cronbach_alpha.csv"))

if __name__ == "__main__":
    main()
