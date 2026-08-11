#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1 of the pipeline: build the HQ1 analysis sample (target pairs).

Selects (conversation_id, speaker_id) records from the CEJC corpus that satisfy
the "home2 / HQ1" criteria used in the study:

  Sampling frame (home2)
    - conversation held at home  (metadata place field contains "自宅")
    - exactly 2 speakers         (metadata speaker-count field == 2)

  Quality filter (HQ1)
    - n_pairs_total             >= --min_pairs_total            (default 80)
    - FILL_text_len             >= --min_text_len               (default 2000)
    - IX_n_pairs_after_question >= --min_pairs_after_question    (default 10)

`n_pairs_total` counts speaker-change transitions in which the record's speaker
is the responder; `IX_n_pairs_after_question` counts the subset where the
preceding utterance was classified as a question. `FILL_text_len` is the total
character count of the speaker's utterances with spaces removed.

The thresholds are reported in the manuscript (Method 2.1.2). They are exposed
as CLI options so that the sensitivity of the sample definition can be checked.

Usage
-----
    python scripts/cejc/build_target_pairs_hq1.py \
      --convlist_parquet artifacts/tmp_meta/cejc_convlist.parquet \
      --utterances_parquet artifacts/_tmp_utt/cejc_utterances/part-00000.parquet \
      --out_parquet artifacts/analysis/target_pairs/cejc_home2_hq1_pairs.parquet \
      --out_preview_tsv artifacts/analysis/target_pairs/cejc_home2_hq1_pairs_preview.tsv

Notes
-----
The CEJC corpus itself is not redistributed with this repository. See
docs/data-availability.md for how to obtain it and how the derived files are
version-pinned by sha256.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

# --- column-name resolution -------------------------------------------------
# CEJC metadata ships with Japanese column names. Some intermediate exports use
# romanised names instead, so both are accepted.
ID_COL_CANDIDATES = ("会話ID", "conversation_id")
PLACE_COL_CANDIDATES = ("場所", "place")
SPEAKER_N_COL_CANDIDATES = ("話者数", "speaker_n")

# --- question detection -----------------------------------------------------
# A preceding utterance counts as a question if it carries an explicit question
# mark, or if its (punctuation-stripped) tail ends in a Japanese interrogative
# form. Identical to the rule used by extract_interaction_features_min.py.
QMARK_RE = re.compile(r"[?？]")
QEND_RE = re.compile(r"(か|かな|かね|でしょう|でしょ|だろう|だろ|の)$")
TAIL_PUNCT_RE = re.compile(r"[。．\.！!、,「」\"\'\s]+$")


def pick_column(df: pd.DataFrame, candidates: tuple[str, ...], label: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise SystemExit(
        f"[ERROR] no {label} column found. tried={candidates} available={list(df.columns)}"
    )


def to_speaker_count(value) -> int | None:
    """Coerce values such as '2人' / ' 2 ' / '2名' to an int."""
    s = str(value).strip()
    s = re.sub(r"^[^0-9]*", "", s)
    s = re.sub(r"[^0-9].*$", "", s)
    return int(s) if s.isdigit() else None


def is_question(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if QMARK_RE.search(t):
        return True
    return bool(QEND_RE.search(TAIL_PUNCT_RE.sub("", t)))


def char_len_no_space(texts) -> int:
    return int(sum(len(str(t).replace(" ", "").replace("\u3000", "")) for t in texts))


def select_home2_conversations(convlist: pd.DataFrame, n_speakers: int) -> set[str]:
    id_col = pick_column(convlist, ID_COL_CANDIDATES, "conversation-id")
    place_col = pick_column(convlist, PLACE_COL_CANDIDATES, "place")

    meta = convlist.copy()
    meta[id_col] = meta[id_col].astype(str)
    if "speaker_n" not in meta.columns:
        src = pick_column(meta, SPEAKER_N_COL_CANDIDATES, "speaker-count")
        meta["speaker_n"] = meta[src].map(to_speaker_count)

    at_home = meta[place_col].astype(str).str.contains("自宅", na=False)
    sized = meta["speaker_n"] == n_speakers
    return set(meta.loc[at_home & sized, id_col].dropna().unique().tolist())


def build_candidates(utterances: pd.DataFrame, conv_ids: set[str]) -> pd.DataFrame:
    u = utterances.copy()
    u["conversation_id"] = u["conversation_id"].astype(str)
    u["speaker_id"] = u["speaker_id"].astype(str)
    u = u[u["conversation_id"].isin(conv_ids)].copy()
    u["text"] = u["text"].fillna("").astype(str)

    # Total character count per (conversation, speaker).
    g_text = u.groupby(["conversation_id", "speaker_id"], as_index=False).agg(
        FILL_text_len=("text", char_len_no_space)
    )

    # Adjacent pairs: a transition where the speaker differs from the previous
    # utterance within the same conversation. mergesort keeps the order stable.
    u = u.sort_values(
        ["conversation_id", "start_time", "end_time"], kind="mergesort"
    )
    prev = u.groupby("conversation_id").shift(1)
    prev_is_q = prev["text"].fillna("").astype(str).map(is_question)

    mask = (u["speaker_id"] != prev["speaker_id"]) & prev["speaker_id"].notna()
    pairs = u.loc[mask, ["conversation_id", "speaker_id"]].rename(
        columns={"speaker_id": "resp_speaker_id"}
    )
    pairs["prev_is_question"] = prev_is_q.loc[mask].values

    g_pairs = (
        pairs.groupby(["conversation_id", "resp_speaker_id"], as_index=False)
        .agg(
            n_pairs_total=("prev_is_question", "size"),
            IX_n_pairs_after_question=("prev_is_question", "sum"),
        )
        .rename(columns={"resp_speaker_id": "speaker_id"})
    )

    cand = g_text.merge(
        g_pairs, on=["conversation_id", "speaker_id"], how="left"
    ).fillna({"n_pairs_total": 0, "IX_n_pairs_after_question": 0})
    cand["n_pairs_total"] = cand["n_pairs_total"].astype(int)
    cand["IX_n_pairs_after_question"] = cand["IX_n_pairs_after_question"].astype(int)
    return cand


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--convlist_parquet", required=True, help="CEJC conversation metadata")
    ap.add_argument("--utterances_parquet", required=True, help="CEJC utterance table")
    ap.add_argument("--out_parquet", required=True, help="output: HQ1 target pairs")
    ap.add_argument("--out_preview_tsv", default=None, help="output: human-readable preview")
    ap.add_argument("--n_speakers", type=int, default=2, help="sampling frame: speaker count")
    ap.add_argument("--min_pairs_total", type=int, default=80)
    ap.add_argument("--min_text_len", type=int, default=2000)
    ap.add_argument("--min_pairs_after_question", type=int, default=10)
    ap.add_argument("--preview_rows", type=int, default=200)
    args = ap.parse_args()

    convlist = pd.read_parquet(args.convlist_parquet)
    conv_ids = select_home2_conversations(convlist, args.n_speakers)
    print(f"[1/3] sampling frame: {len(conv_ids)} conversations "
          f"(at home, {args.n_speakers} speakers)")

    utterances = pd.read_parquet(args.utterances_parquet)
    cand = build_candidates(utterances, conv_ids)
    print(f"[2/3] candidate records (conversation x speaker): {len(cand)}")

    hq1 = cand[
        (cand["n_pairs_total"] >= args.min_pairs_total)
        & (cand["FILL_text_len"] >= args.min_text_len)
        & (cand["IX_n_pairs_after_question"] >= args.min_pairs_after_question)
    ].copy()
    print(
        f"[3/3] HQ1 records: {len(hq1)} "
        f"(n_pairs_total>={args.min_pairs_total}, "
        f"FILL_text_len>={args.min_text_len}, "
        f"IX_n_pairs_after_question>={args.min_pairs_after_question})"
    )
    if hq1.empty:
        raise SystemExit("[ERROR] HQ1 filter produced 0 records; check inputs and thresholds")

    out_pairs = Path(args.out_parquet)
    out_pairs.parent.mkdir(parents=True, exist_ok=True)
    hq1[["conversation_id", "speaker_id"]].drop_duplicates().to_parquet(
        out_pairs, index=False
    )
    print(f"OK: {out_pairs}")

    if args.out_preview_tsv:
        prev = Path(args.out_preview_tsv)
        prev.parent.mkdir(parents=True, exist_ok=True)
        hq1.sort_values(
            ["n_pairs_total", "FILL_text_len"], ascending=False
        ).head(args.preview_rows).to_csv(prev, sep="\t", index=False)
        print(f"OK: {prev}")


if __name__ == "__main__":
    main()
