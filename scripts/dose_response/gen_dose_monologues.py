#!/usr/bin/env python3
"""Dose Level別monologues_parquet生成スクリプト.

入力monologues_parquet（N=120）の全行に対して、指定Target_Featureの
各Dose Level（×0, ×1, ×3）で操作済みテキストを生成し、
条件別parquetファイルとして出力する。

Requirements: 4.1-4.7, 9.1, 9.2
"""
from __future__ import annotations

import argparse
import pathlib
import re
import warnings

import pandas as pd

# manipulate_text.py は同一パッケージ内
from scripts.dose_response.manipulate_text import (
    manipulate_fillers,
    manipulate_oir,
    manipulate_yesno,
)

# ── Target Feature → manipulator 関数のマッピング ──
MANIPULATOR_MAP = {
    "FILL": manipulate_fillers,
    "YESNO": manipulate_yesno,
    "OIR": manipulate_oir,
}

VALID_FEATURES = list(MANIPULATOR_MAP.keys())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI引数をパースする."""
    ap = argparse.ArgumentParser(
        description="Dose Level別monologues_parquet生成",
    )
    ap.add_argument(
        "--monologues_parquet",
        default="artifacts/cejc/monologues_cejc_home2_hq1_v1.parquet",
        help="入力monologues parquet (default: %(default)s)",
    )
    ap.add_argument(
        "--target_feature",
        required=True,
        choices=VALID_FEATURES,
        help="操作対象特徴量 (FILL / YESNO / OIR)",
    )
    ap.add_argument(
        "--dose_levels",
        default="0,1,3",
        help="Dose Level一覧 (カンマ区切り, default: %(default)s)",
    )
    ap.add_argument(
        "--out_dir",
        default="artifacts/dose_response",
        help="出力ディレクトリ (default: %(default)s)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="乱数シード (default: %(default)s)",
    )
    ap.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="サンプルペア出力数 (default: %(default)s)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="dry-runモード（テキスト操作のみ実行、LLM呼び出しなし）",
    )
    return ap.parse_args(argv)


def generate_dose_monologues(
    df: pd.DataFrame,
    target_feature: str,
    dose_level: int,
    seed: int,
) -> tuple[pd.DataFrame, list[dict]]:
    """1つのDose Levelに対して全行を操作し、操作済みDataFrameとログを返す.

    Args:
        df: 入力monologues DataFrame
        target_feature: 操作対象特徴量名 (FILL / YESNO / OIR)
        dose_level: 操作倍率 (0, 1, 3)
        seed: 乱数シード

    Returns:
        (操作済みDataFrame, 操作ログリスト)
    """
    manipulate_fn = MANIPULATOR_MAP[target_feature]
    result_rows = []
    log_entries: list[dict] = []

    for idx, row in df.iterrows():
        text = row["text"]
        # 各話者に対して seed + row index でシードを分散
        # （全話者同一シードだと同一パターンになるため）
        speaker_seed = seed + idx  # type: ignore[operator]

        manipulated_text, op_log = manipulate_fn(text, dose_level, seed=speaker_seed)

        # 操作済み行を構築
        # n_chars は空白除去後の文字数（元データと同一の計算方法）
        new_n_chars = len(re.sub(r"\s", "", manipulated_text))
        new_row = {
            "conversation_id": row["conversation_id"],
            "speaker_id": row["speaker_id"],
            "n_utt": row["n_utt"],  # 元の値を保持
            "n_chars": new_n_chars,  # 操作後テキスト文字数で更新
            "text": manipulated_text,
        }
        result_rows.append(new_row)

        # 操作ログ
        char_change_pct = (
            abs(op_log["new_chars"] - op_log["original_chars"])
            / op_log["original_chars"]
            * 100
            if op_log["original_chars"] > 0
            else 0.0
        )
        warning_msg = ""
        if char_change_pct > 5.0:
            warning_msg = (
                f"char_change {char_change_pct:.1f}% exceeds ±5% limit"
            )

        log_entry = {
            "conversation_id": row["conversation_id"],
            "speaker_id": row["speaker_id"],
            "dose_level": dose_level,
            "original_count": op_log["original_count"],
            "new_count": op_log["new_count"],
            "inserted": op_log["inserted"],
            "deleted": op_log["deleted"],
            "original_chars": op_log["original_chars"],
            "new_chars": op_log["new_chars"],
            "char_change_pct": round(char_change_pct, 2),
            "warning": warning_msg,
        }
        log_entries.append(log_entry)

    result_df = pd.DataFrame(result_rows)
    # dtypeを入力と揃える
    result_df["n_utt"] = result_df["n_utt"].astype("int64")
    result_df["n_chars"] = result_df["n_chars"].astype("int64")

    return result_df, log_entries


def write_sample_pairs(
    original_df: pd.DataFrame,
    manipulated_df: pd.DataFrame,
    target_feature: str,
    dose_level: int,
    n_samples: int,
    out_dir: pathlib.Path,
) -> None:
    """操作前後のテキストペアをテキストファイルとして出力する（目視確認用）.

    Args:
        original_df: 元のmonologues DataFrame
        manipulated_df: 操作済みDataFrame
        target_feature: 操作対象特徴量名
        dose_level: 操作倍率
        n_samples: 出力するサンプル数
        out_dir: 出力ディレクトリ
    """
    # ×1 はコピーなのでサンプルペア不要
    if dose_level == 1:
        return

    sample_dir = out_dir / "sample_pairs"
    sample_dir.mkdir(parents=True, exist_ok=True)

    n = min(n_samples, len(original_df))
    out_path = sample_dir / f"{target_feature}_sample_x{dose_level}.txt"

    lines: list[str] = []
    lines.append(f"# Sample Pairs: {target_feature} ×{dose_level}")
    lines.append(f"# {n} samples")
    lines.append("")

    for i in range(n):
        orig_row = original_df.iloc[i]
        manip_row = manipulated_df.iloc[i]

        lines.append(f"{'='*60}")
        lines.append(
            f"Speaker: {orig_row['conversation_id']} / {orig_row['speaker_id']}"
        )
        lines.append(f"{'='*60}")
        lines.append("")

        # テキストの先頭500文字を表示（長すぎる場合は省略）
        orig_text = orig_row["text"]
        manip_text = manip_row["text"]
        max_preview = 500

        lines.append("--- ORIGINAL ---")
        if len(orig_text) > max_preview:
            lines.append(orig_text[:max_preview])
            lines.append(f"... (truncated, total {len(orig_text)} chars)")
        else:
            lines.append(orig_text)
        lines.append("")

        lines.append(f"--- MANIPULATED (×{dose_level}) ---")
        if len(manip_text) > max_preview:
            lines.append(manip_text[:max_preview])
            lines.append(f"... (truncated, total {len(manip_text)} chars)")
        else:
            lines.append(manip_text)
        lines.append("")

        lines.append(
            f"Chars: {len(orig_text)} → {len(manip_text)} "
            f"(Δ{len(manip_text) - len(orig_text):+d})"
        )
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  ✓ サンプルペア: {out_path}")


def main(argv: list[str] | None = None) -> None:
    """メインエントリポイント."""
    args = parse_args(argv)

    # Dose Level のパース
    dose_levels = [int(x.strip()) for x in args.dose_levels.split(",")]

    # 入力parquet読み込み
    input_path = pathlib.Path(args.monologues_parquet)
    if not input_path.exists():
        raise FileNotFoundError(f"入力parquetが見つかりません: {input_path}")

    df = pd.read_parquet(input_path)
    n_rows = len(df)
    print(f"入力: {input_path} ({n_rows} 行)")

    if n_rows == 0:
        warnings.warn("入力parquetが空です")

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 全Dose Levelの操作ログを集約
    all_log_entries: list[dict] = []

    for level in dose_levels:
        print(f"\n--- {args.target_feature} ×{level} ---")

        # 操作実行
        result_df, log_entries = generate_dose_monologues(
            df,
            target_feature=args.target_feature,
            dose_level=level,
            seed=args.seed,
        )

        all_log_entries.extend(log_entries)

        # parquet出力
        out_parquet = (
            out_dir / f"monologues_dose_{args.target_feature}_x{level}.parquet"
        )
        result_df.to_parquet(out_parquet, index=False)
        print(f"  ✓ {out_parquet} ({len(result_df)} 行)")

        # サンプルペア出力
        write_sample_pairs(
            original_df=df,
            manipulated_df=result_df,
            target_feature=args.target_feature,
            dose_level=level,
            n_samples=args.n_samples,
            out_dir=out_dir,
        )

    # 操作ログCSV出力
    log_df = pd.DataFrame(all_log_entries)
    log_csv = out_dir / f"manipulation_log_{args.target_feature}.csv"
    log_df.to_csv(log_csv, index=False)
    print(f"\n✓ 操作ログ: {log_csv} ({len(log_df)} 行)")

    # サマリー表示
    print(f"\n--- サマリー ---")
    for level in dose_levels:
        level_log = log_df[log_df["dose_level"] == level]
        n_warnings = (level_log["warning"] != "").sum()
        mean_change = level_log["char_change_pct"].mean()
        print(
            f"  ×{level}: "
            f"mean_char_change={mean_change:.2f}%, "
            f"warnings={n_warnings}/{len(level_log)}"
        )

    if args.dry_run:
        print("\n[dry-run] テキスト操作のみ実行しました（LLM呼び出しなし）")

    print("\n✓ 完了")


if __name__ == "__main__":
    main()
