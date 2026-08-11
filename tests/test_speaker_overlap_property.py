#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Property-based test for speaker overlap counting correctness.

# Feature: soda-feedback-v1, Property 1: 話者重複集計の正当性

Validates: Requirements 1.1

任意のcejc_person_idを含むメタデータTSVに対して、speaker_overlap_analysis.pyの出力は以下を満たす:
(a) ユニーク話者数 + 重複レコード数の合計が総レコード数と一致する
(b) 重複話者数は2回以上出現するcejc_person_idの数と一致する
(c) 性別内訳の合計が総レコード数と一致する（ユニーク話者ベース: gender_F + gender_M == unique_speakers）
"""
from __future__ import annotations

from collections import Counter

import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from scripts.analysis.speaker_overlap_analysis import compute_overlap_stats

# ── Strategies ───────────────────────────────────────────────────────

# Generate person IDs: pool of 1-20 unique IDs, then sample 2-50 records
# (with replacement to create duplicates)
_person_id_pool = st.lists(
    st.text(
        alphabet=st.sampled_from("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"),
        min_size=3,
        max_size=8,
    ),
    min_size=1,
    max_size=20,
    unique=True,
)

_gender = st.sampled_from(["F", "M"])


@st.composite
def metadata_dataframes(draw: st.DrawFn) -> pd.DataFrame:
    """Generate random metadata DataFrames with cejc_person_id, conversation_id, gender."""
    pool = draw(_person_id_pool)
    n_records = draw(st.integers(min_value=2, max_value=50))

    person_ids = [draw(st.sampled_from(pool)) for _ in range(n_records)]
    conversation_ids = [f"conv_{i:04d}" for i in range(n_records)]
    genders_map: dict[str, str] = {}
    for pid in pool:
        genders_map[pid] = draw(_gender)
    genders = [genders_map[pid] for pid in person_ids]

    return pd.DataFrame(
        {
            "cejc_person_id": person_ids,
            "conversation_id": conversation_ids,
            "gender": genders,
        }
    )


# ── Property test ────────────────────────────────────────────────────


@given(df=metadata_dataframes())
@settings(max_examples=100)
def test_speaker_overlap_counting_correctness(df: pd.DataFrame) -> None:
    """Property 1: 話者重複集計の正当性

    **Validates: Requirements 1.1**
    """
    result = compute_overlap_stats(df)
    summary = result["summary"]
    detail = result["detail"]

    total = summary["total_records"]
    unique = summary["unique_speakers"]
    dup_speakers = summary["duplicate_speakers"]
    dup_records = summary["duplicate_records"]

    # Ground truth from raw data
    counts = Counter(df["cejc_person_id"])
    expected_unique = len(counts)
    expected_dup_speakers = sum(1 for c in counts.values() if c >= 2)
    expected_dup_records = sum(c for c in counts.values() if c >= 2)

    # (a) ユニーク話者数 + 重複レコード数の整合性
    #     unique_speakers + Σ(n_records - 1) for duplicates == total_records
    #     equivalently: unique_speakers + (duplicate_records - duplicate_speakers) == total
    #     because Σ(n_records - 1) = Σ(n_records) - count_of_dup_speakers = dup_records - dup_speakers
    assert total == len(df), (
        f"total_records mismatch: {total} != {len(df)}"
    )
    assert unique == expected_unique, (
        f"unique_speakers mismatch: {unique} != {expected_unique}"
    )
    # Non-duplicate records + duplicate records == total
    non_dup_records = total - dup_records
    assert non_dup_records + dup_records == total, (
        f"Record accounting: non_dup({non_dup_records}) + dup({dup_records}) != total({total})"
    )
    # Also verify: unique_speakers = (unique - dup_speakers) + dup_speakers
    # i.e. speakers appearing once + speakers appearing 2+ == unique
    single_speakers = unique - dup_speakers
    single_records = sum(1 for c in counts.values() if c == 1)
    assert single_speakers == single_records, (
        f"single speakers: {single_speakers} != {single_records}"
    )
    assert single_records + dup_records == total, (
        f"single_records({single_records}) + dup_records({dup_records}) != total({total})"
    )

    # (b) 重複話者数は2回以上出現するcejc_person_idの数と一致する
    assert dup_speakers == expected_dup_speakers, (
        f"duplicate_speakers mismatch: {dup_speakers} != {expected_dup_speakers}"
    )
    assert dup_records == expected_dup_records, (
        f"duplicate_records mismatch: {dup_records} != {expected_dup_records}"
    )

    # Verify detail DataFrame contains exactly the duplicate speakers
    if dup_speakers > 0:
        assert len(detail) == dup_speakers, (
            f"detail rows ({len(detail)}) != dup_speakers ({dup_speakers})"
        )
        for _, row in detail.iterrows():
            pid = row["cejc_person_id"]
            assert counts[pid] >= 2, f"Non-duplicate speaker {pid} in detail"
            assert row["n_records"] == counts[pid], (
                f"n_records mismatch for {pid}: {row['n_records']} != {counts[pid]}"
            )
    else:
        assert detail.empty, "detail should be empty when no duplicates"

    # (c) 性別内訳の合計がユニーク話者数と一致する
    #     gender_F + gender_M == unique_speakers
    gender_f = summary["gender_F"]
    gender_m = summary["gender_M"]
    assert gender_f + gender_m == unique, (
        f"gender breakdown: F({gender_f}) + M({gender_m}) != unique({unique})"
    )
