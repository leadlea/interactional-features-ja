#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CEJC話者メタデータビルダー

S3上のCEJCメタ情報（話者.csv + 話者・会話対応表.csv）と
N=120特徴量parquetを紐付けて、gen_paper_figs_v2.py の --metadata_tsv に
渡せる形式のTSVを生成する。

紐付けロジック:
  1. 話者・会話対応表.csv: (会話ID, 話者ID, 話者ラベル)
     話者ラベル "IC01_玲子" → prefix "IC01" を抽出
  2. features parquet: (conversation_id, speaker_id)
     speaker_id = "IC01" 等の会話内ラベル
  3. JOIN: (conversation_id, speaker_id) = (会話ID, prefix) → 話者ID
  4. 話者.csv: 話者ID → 年齢, 性別, 出身地, 居住地, 職業 等

出力TSV カラム:
  conversation_id, speaker_id, cejc_person_id, gender, age_band,
  age_mid, birthplace, residence, occupation, relationship

Usage:
  python scripts/cejc/build_cejc_speaker_meta.py \
    --features_parquet artifacts/analysis/features_min/features_cejc_home2_hq1.parquet \
    --speaker_csv /tmp/cejc_speaker.csv \
    --mapping_csv /tmp/cejc_speaker_conversation.csv \
    --out artifacts/analysis/cejc_speaker_metadata.tsv
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# 年齢バンド → 中央値 変換
# ---------------------------------------------------------------------------
AGE_BAND_PATTERN = re.compile(r"(\d+)-(\d+)")


def age_band_to_midpoint(band: str) -> float | None:
    """'40-44歳' → 42.0"""
    if pd.isna(band):
        return None
    m = AGE_BAND_PATTERN.search(str(band))
    if m:
        return (int(m.group(1)) + int(m.group(2))) / 2.0
    return None


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="CEJC話者メタデータビルダー")
    ap.add_argument(
        "--features_parquet",
        required=True,
        help="N=120 特徴量 parquet (conversation_id, speaker_id)",
    )
    ap.add_argument(
        "--speaker_csv",
        required=True,
        help="CEJC 話者.csv (CP932)",
    )
    ap.add_argument(
        "--mapping_csv",
        required=True,
        help="CEJC 話者・会話対応表.csv (CP932)",
    )
    ap.add_argument(
        "--encoding",
        default="cp932",
        help="CSV encoding (default: cp932)",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="出力TSVパス",
    )
    args = ap.parse_args()

    # --- Load ---
    feat = pd.read_parquet(args.features_parquet)[["conversation_id", "speaker_id"]]
    print(f"features: {len(feat)} rows", file=sys.stderr)

    mapping = pd.read_csv(args.mapping_csv, encoding=args.encoding)
    mapping["speaker_label_prefix"] = mapping["話者ラベル"].str.extract(
        r"^([A-Z]+\d+[A-Z]?)"
    )
    print(f"mapping: {len(mapping)} rows", file=sys.stderr)

    speaker = pd.read_csv(args.speaker_csv, encoding=args.encoding)
    print(f"speaker: {len(speaker)} rows", file=sys.stderr)

    # --- Step 1: conversation_id + speaker_id → 話者ID ---
    merged = feat.merge(
        mapping[["会話ID", "話者ID", "speaker_label_prefix"]],
        left_on=["conversation_id", "speaker_id"],
        right_on=["会話ID", "speaker_label_prefix"],
        how="left",
    )
    n_unmatched_step1 = merged["話者ID"].isna().sum()
    if n_unmatched_step1 > 0:
        print(
            f"WARNING: {n_unmatched_step1} rows unmatched at mapping step",
            file=sys.stderr,
        )

    # --- Step 2: 話者ID → 属性 ---
    merged = merged.merge(speaker, on="話者ID", how="left")
    n_unmatched_step2 = merged["性別"].isna().sum()
    if n_unmatched_step2 > 0:
        print(
            f"WARNING: {n_unmatched_step2} rows unmatched at speaker step",
            file=sys.stderr,
        )

    # --- 整形 ---
    merged["age_mid"] = merged["年齢"].apply(age_band_to_midpoint)

    out = merged[
        [
            "conversation_id",
            "speaker_id",
            "話者ID",
            "性別",
            "年齢",
            "age_mid",
            "出身地",
            "居住地",
            "職業",
            "協力者からみた関係性",
        ]
    ].rename(
        columns={
            "話者ID": "cejc_person_id",
            "性別": "gender",
            "年齢": "age_band",
            "出身地": "birthplace",
            "居住地": "residence",
            "職業": "occupation",
            "協力者からみた関係性": "relationship",
        }
    )

    # gender を英語に変換（gen_paper_figs_v2.py が期待する形式）
    out["gender"] = out["gender"].map({"女性": "F", "男性": "M"})

    # age カラム追加（gen_paper_figs_v2.py が期待する形式）
    out["age"] = out["age_mid"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, sep="\t", index=False, encoding="utf-8")

    print(f"\nwrote: {out_path}", file=sys.stderr)
    print(f"rows: {len(out)}", file=sys.stderr)
    print(f"gender: {out['gender'].value_counts().to_dict()}", file=sys.stderr)
    print(f"age_mid: min={out['age_mid'].min()}, max={out['age_mid'].max()}, "
          f"mean={out['age_mid'].mean():.1f}", file=sys.stderr)
    print(f"birthplace missing: {out['birthplace'].isna().sum()}", file=sys.stderr)


if __name__ == "__main__":
    main()
