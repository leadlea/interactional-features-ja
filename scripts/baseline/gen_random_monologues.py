#!/usr/bin/env python3
"""条件3: Random Assigner — テキストシャッフル（derangement）によるmonologues生成.

monologues parquetのtextカラムを話者間でシャッフルし、
自分自身のテキストが割り当てられない完全順列（derangement）を生成する。

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7
"""
from __future__ import annotations

import argparse
import pathlib
import random
from typing import List

import pandas as pd


# ---------------------------------------------------------------------------
# Derangement アルゴリズム
# ---------------------------------------------------------------------------
def _derange(texts: List[str], seed: int) -> List[int]:
    """インデックスベースの derangement（完全順列）を生成する.

    アルゴリズム:
    1. インデックスリストをコピー
    2. Fisher-Yates シャッフル（seed 固定）
    3. 自分自身に割り当てられた要素（固定点）があれば隣接要素とスワップ
    4. 全要素が元の位置と異なることを検証

    Parameters
    ----------
    texts : list[str]
        元のテキストリスト（長さ N >= 2）
    seed : int
        乱数シード

    Returns
    -------
    list[int]
        割当先インデックスのリスト（perm[i] != i が全 i で成立）

    Raises
    ------
    ValueError
        N=1 の場合（derangement 不可能）
    """
    n = len(texts)
    if n < 2:
        raise ValueError(
            f"Derangement requires N >= 2, got N={n}"
        )

    # Step 1-2: Fisher-Yates シャッフル
    perm = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(perm)

    # Step 3: 固定点を隣接スワップで解消
    for i in range(n):
        if perm[i] == i:
            # 隣接要素とスワップ（末尾の場合は前の要素と）
            j = i + 1 if i < n - 1 else i - 1
            perm[i], perm[j] = perm[j], perm[i]

    # Step 4: 検証
    for i in range(n):
        assert perm[i] != i, f"Fixed point at index {i} after derangement"

    return perm


# ---------------------------------------------------------------------------
# 公開関数: shuffle_texts（プロパティテスト用にインポート可能）
# ---------------------------------------------------------------------------
def shuffle_texts(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """monologues DataFrame のテキストを derangement でシャッフルする.

    Parameters
    ----------
    df : pd.DataFrame
        入力 DataFrame。カラム: conversation_id, speaker_id, text
    seed : int
        乱数シード（デフォルト 42）

    Returns
    -------
    pd.DataFrame
        シャッフル後の DataFrame（conversation_id, speaker_id, text）。
        conversation_id と speaker_id は元のまま、text のみ入れ替え。

    Raises
    ------
    ValueError
        N=1 の場合
    """
    texts = df["text"].tolist()
    perm = _derange(texts, seed)

    result = df[["conversation_id", "speaker_id"]].copy()
    result["text"] = [texts[perm[i]] for i in range(len(texts))]
    return result


# ---------------------------------------------------------------------------
# マッピング CSV 生成
# ---------------------------------------------------------------------------
def build_mapping(df: pd.DataFrame, perm: List[int]) -> pd.DataFrame:
    """シャッフルマッピング CSV 用の DataFrame を構築する.

    Parameters
    ----------
    df : pd.DataFrame
        元の monologues DataFrame
    perm : list[int]
        derangement の割当先インデックス

    Returns
    -------
    pd.DataFrame
        conversation_id, speaker_id, original_index, assigned_index,
        assigned_conversation_id, assigned_speaker_id
    """
    records = []
    for i in range(len(df)):
        row = df.iloc[i]
        assigned_row = df.iloc[perm[i]]
        records.append(
            {
                "conversation_id": row["conversation_id"],
                "speaker_id": row["speaker_id"],
                "original_index": i,
                "assigned_index": perm[i],
                "assigned_conversation_id": assigned_row["conversation_id"],
                "assigned_speaker_id": assigned_row["speaker_id"],
            }
        )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="条件3: テキストシャッフル（derangement）による monologues 生成"
    )
    ap.add_argument(
        "--monologues_parquet",
        default="artifacts/cejc/monologues_cejc_home2_hq1_v1.parquet",
        help="入力 monologues parquet",
    )
    ap.add_argument(
        "--out_parquet",
        default="artifacts/baseline/monologues_random.parquet",
        help="出力 parquet",
    )
    ap.add_argument(
        "--mapping_csv",
        default="artifacts/baseline/shuffle_mapping.csv",
        help="シャッフルマッピング CSV 出力先",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="乱数シード",
    )
    args = ap.parse_args()

    # --- 読み込み ---
    mono_df = pd.read_parquet(args.monologues_parquet)

    # --- derangement ---
    texts = mono_df["text"].tolist()
    perm = _derange(texts, args.seed)

    # --- シャッフル後 DataFrame ---
    out_df = mono_df[["conversation_id", "speaker_id"]].copy()
    out_df["text"] = [texts[perm[i]] for i in range(len(texts))]

    # --- マッピング CSV ---
    mapping_df = build_mapping(mono_df, perm)

    # --- 保存 ---
    pathlib.Path(args.out_parquet).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out_parquet, index=False)
    print(f"✓ {len(out_df)} 件を {args.out_parquet} に保存しました")

    pathlib.Path(args.mapping_csv).parent.mkdir(parents=True, exist_ok=True)
    mapping_df.to_csv(args.mapping_csv, index=False)
    print(f"✓ マッピングを {args.mapping_csv} に保存しました")


if __name__ == "__main__":
    main()
