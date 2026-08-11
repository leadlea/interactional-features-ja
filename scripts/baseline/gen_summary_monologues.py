#!/usr/bin/env python3
"""条件2: Summary Generator — 統計情報テキストのみのmonologues生成.

utterances parquetから conversation_id × speaker_id ごとに4統計量を算出し、
テキストテンプレートに変換して monologues 互換 parquet を出力する。

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6
"""
from __future__ import annotations

import argparse
import re
import warnings

import pandas as pd

# ---------------------------------------------------------------------------
# フィラー検出パターン
# ---------------------------------------------------------------------------
FILLER_RE = re.compile(r"えっと|えー|あの")


# ---------------------------------------------------------------------------
# 純粋関数: テキストテンプレート適用
# ---------------------------------------------------------------------------
def format_summary_text(
    n_utt: int,
    mean_chars: float,
    duration_sec: float,
    filler_count: int,
) -> str:
    """統計量からサマリーテキストを生成する（純粋関数）.

    Parameters
    ----------
    n_utt : int
        発話数
    mean_chars : float
        平均発話長（文字数）
    duration_sec : float
        会話全体の長さ（秒）
    filler_count : int
        フィラー使用回数

    Returns
    -------
    str
        統計情報テキスト
    """
    return (
        f"この話者は合計{n_utt}回発話し、"
        f"平均発話長は{mean_chars:.1f}文字、"
        f"会話全体の長さは{duration_sec:.0f}秒、"
        f"フィラーは{filler_count}回使用しました。"
    )


# ---------------------------------------------------------------------------
# 統計量算出
# ---------------------------------------------------------------------------
def compute_summary_stats(utt_df: pd.DataFrame) -> pd.DataFrame:
    """utterances DataFrameから conversation_id × speaker_id ごとの統計量を算出.

    Parameters
    ----------
    utt_df : pd.DataFrame
        utterances parquet の内容。カラム: conversation_id, speaker_id,
        text, start_time, end_time

    Returns
    -------
    pd.DataFrame
        conversation_id, speaker_id, n_utt, mean_chars, duration_sec, filler_count
    """
    df = utt_df.copy()

    # NaN を 0 として扱う (要件 1.5)
    df["text"] = df["text"].fillna("")
    df["start_time"] = df["start_time"].fillna(0)
    df["end_time"] = df["end_time"].fillna(0)

    # 文字数（空白除去後）
    df["char_count"] = df["text"].apply(lambda t: len(re.sub(r"\s", "", t)))

    # フィラー数
    df["filler_count"] = df["text"].apply(
        lambda t: len(FILLER_RE.findall(t))
    )

    # --- conversation_id 単位の duration_sec ---
    conv_duration = df.groupby("conversation_id").agg(
        conv_start=("start_time", "min"),
        conv_end=("end_time", "max"),
    )
    conv_duration["duration_sec"] = conv_duration["conv_end"] - conv_duration["conv_start"]

    # --- conversation_id × speaker_id 単位の集計 ---
    grouped = df.groupby(["conversation_id", "speaker_id"]).agg(
        n_utt=("text", "count"),
        mean_chars=("char_count", "mean"),
        filler_count=("filler_count", "sum"),
    ).reset_index()

    # duration_sec を結合（同一 conversation_id の全話者で共通）
    grouped = grouped.merge(
        conv_duration[["duration_sec"]],
        left_on="conversation_id",
        right_index=True,
        how="left",
    )

    return grouped


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="条件2: 統計情報テキストのみの monologues 生成"
    )
    ap.add_argument(
        "--utterances_parquet",
        default="artifacts/_tmp_utt/cejc_utterances/part-00000.parquet",
        help="入力発話 parquet",
    )
    ap.add_argument(
        "--monologues_parquet",
        default="artifacts/cejc/monologues_cejc_home2_hq1_v1.parquet",
        help="参照用 monologues parquet（出力対象ペアのフィルタに使用）",
    )
    ap.add_argument(
        "--out_parquet",
        default="artifacts/baseline/monologues_summary.parquet",
        help="出力 parquet",
    )
    ap.add_argument(
        "--expected_n",
        type=int,
        default=120,
        help="期待件数",
    )
    args = ap.parse_args()

    # --- 読み込み ---
    utt_df = pd.read_parquet(args.utterances_parquet)
    mono_df = pd.read_parquet(args.monologues_parquet)

    # --- 統計量算出 ---
    stats = compute_summary_stats(utt_df)

    # --- monologues に存在するペアのみ残す ---
    mono_keys = mono_df[["conversation_id", "speaker_id"]].drop_duplicates()
    stats = stats.merge(mono_keys, on=["conversation_id", "speaker_id"], how="inner")

    # --- テキスト生成 ---
    stats["text"] = stats.apply(
        lambda r: format_summary_text(
            int(r["n_utt"]),
            float(r["mean_chars"]),
            float(r["duration_sec"]),
            int(r["filler_count"]),
        ),
        axis=1,
    )

    # --- 出力 (monologues 互換: conversation_id, speaker_id, text) ---
    out_df = stats[["conversation_id", "speaker_id", "text"]].copy()

    # --- 件数チェック ---
    if len(out_df) != args.expected_n:
        warnings.warn(
            f"生成件数 {len(out_df)} が期待値 {args.expected_n} と異なります"
        )

    # --- 保存 ---
    import pathlib
    pathlib.Path(args.out_parquet).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out_parquet, index=False)
    print(f"✓ {len(out_df)} 件を {args.out_parquet} に保存しました")


if __name__ == "__main__":
    main()
