#!/usr/bin/env python3
"""特徴量検証スクリプト.

操作済みmonologuesに対して、対象特徴量が意図通りに変化しているかを
直接テキストカウントで検証する。

extract_interaction_features_min.py の extract_features() は
utterances（個別発話 + start_time/end_time）を必要とするが、
monologues は連結テキストのため、タイミング情報がない。
そのため、テキストベースで直接カウント可能な特徴量
（フィラー数、YES/NO数、OIR数）を検証対象とする。

PG（タイミング）特徴量はテキスト操作では変化しないため検証不要。

Requirements: 5.1-5.5
"""
from __future__ import annotations

import argparse
import pathlib
import re

import pandas as pd

from scripts.dose_response.manipulate_text import (
    YESNO_PREFIXES,
    OIR_PREFIXES,
    count_fillers_in_text,
)


# ── Target Feature → カウント関数のマッピング ──

def count_yesno_in_text(text: str) -> int:
    """テキスト中のYES/NO応答行数をカウントする.

    各行（発話）の先頭がYESNO_PREFIXESのいずれかで始まるかを判定。
    extract_interaction_features_min.py の is_yesno() と同一ロジック。

    Args:
        text: 会話テキスト（複数行、1行=1発話）

    Returns:
        YES/NO応答行数
    """
    if not text:
        return 0
    count = 0
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped and any(stripped.startswith(p) for p in YESNO_PREFIXES):
            count += 1
    return count


def count_oir_in_text(text: str) -> int:
    """テキスト中のOIRマーカー行数をカウントする.

    各行（発話）の先頭がOIR_PREFIXESのいずれかで始まるかを判定。
    extract_interaction_features_min.py の is_oir() と同一ロジック。

    Args:
        text: 会話テキスト（複数行、1行=1発話）

    Returns:
        OIRマーカー行数
    """
    if not text:
        return 0
    count = 0
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped and any(stripped.startswith(p) for p in OIR_PREFIXES):
            count += 1
    return count


def count_lines(text: str) -> int:
    """テキストの行数（発話数）をカウントする."""
    if not text:
        return 0
    return len(text.split("\n"))


def count_chars_no_ws(text: str) -> int:
    """空白除去後の文字数をカウントする."""
    return len(re.sub(r"\s", "", text))


# Target Feature名 → カウント関数
COUNTER_MAP: dict[str, callable] = {
    "FILL": count_fillers_in_text,
    "YESNO": count_yesno_in_text,
    "OIR": count_oir_in_text,
}

# 全検証対象の特徴量名リスト（target + non-target）
ALL_FEATURE_NAMES = ["FILL_count", "YESNO_count", "OIR_count", "n_lines", "n_chars"]


def compute_features_for_df(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrameの各行に対して全特徴量をカウントする.

    Args:
        df: monologues DataFrame（text カラム必須）

    Returns:
        特徴量カラムを追加したDataFrame
    """
    texts = df["text"].fillna("").astype(str)
    result = df[["conversation_id", "speaker_id"]].copy()
    result["FILL_count"] = texts.map(count_fillers_in_text)
    result["YESNO_count"] = texts.map(count_yesno_in_text)
    result["OIR_count"] = texts.map(count_oir_in_text)
    result["n_lines"] = texts.map(count_lines)
    result["n_chars"] = texts.map(count_chars_no_ws)
    return result


def _target_feature_to_count_col(target_feature: str) -> str:
    """Target Feature名をカウントカラム名に変換する."""
    return f"{target_feature}_count"


def parse_dose_level_from_filename(filename: str) -> int | None:
    """ファイル名からDose Levelを抽出する.

    例: monologues_dose_FILL_x3.parquet → 3

    Args:
        filename: ファイル名

    Returns:
        Dose Level (int) or None
    """
    m = re.search(r"_x(\d+)\.parquet$", filename)
    if m:
        return int(m.group(1))
    return None


def find_dose_parquets(
    dose_dir: pathlib.Path,
    target_feature: str,
) -> dict[int, pathlib.Path]:
    """dose_dir内のDose Level別parquetファイルを検索する.

    Args:
        dose_dir: Dose Level別parquet格納ディレクトリ
        target_feature: 操作対象特徴量名 (FILL / YESNO / OIR)

    Returns:
        {dose_level: parquet_path} のdict
    """
    pattern = f"monologues_dose_{target_feature}_x*.parquet"
    files = sorted(dose_dir.glob(pattern))
    result: dict[int, pathlib.Path] = {}
    for f in files:
        level = parse_dose_level_from_filename(f.name)
        if level is not None:
            result[level] = f
    return result


def verify_features(
    original_df: pd.DataFrame,
    dose_dfs: dict[int, pd.DataFrame],
    target_feature: str,
) -> pd.DataFrame:
    """操作前後の特徴量を比較し、検証レポートを生成する.

    Args:
        original_df: 元のmonologues DataFrame
        dose_dfs: {dose_level: 操作済みDataFrame} のdict
        target_feature: 操作対象特徴量名 (FILL / YESNO / OIR)

    Returns:
        検証レポートDataFrame
        カラム: dose_level, feature_name, mean_original, mean_manipulated,
                delta, delta_pct, is_target
    """
    # 元データの特徴量を計算
    original_features = compute_features_for_df(original_df)
    target_col = _target_feature_to_count_col(target_feature)

    rows: list[dict] = []

    for dose_level in sorted(dose_dfs.keys()):
        dose_df = dose_dfs[dose_level]
        dose_features = compute_features_for_df(dose_df)

        for feat_name in ALL_FEATURE_NAMES:
            mean_orig = float(original_features[feat_name].mean())
            mean_manip = float(dose_features[feat_name].mean())
            delta = mean_manip - mean_orig
            delta_pct = (delta / mean_orig * 100) if mean_orig != 0 else 0.0
            is_target = feat_name == target_col

            rows.append({
                "dose_level": dose_level,
                "feature_name": feat_name,
                "mean_original": round(mean_orig, 4),
                "mean_manipulated": round(mean_manip, 4),
                "delta": round(delta, 4),
                "delta_pct": round(delta_pct, 2),
                "is_target": is_target,
            })

    return pd.DataFrame(rows)


def check_monotonicity(
    report_df: pd.DataFrame,
    target_feature: str,
) -> list[str]:
    """対象特徴量がDose Level間で単調増加しているかを検証する.

    Args:
        report_df: verify_features() の出力DataFrame
        target_feature: 操作対象特徴量名

    Returns:
        検証メッセージのリスト
    """
    target_col = _target_feature_to_count_col(target_feature)
    target_rows = report_df[report_df["feature_name"] == target_col].sort_values(
        "dose_level"
    )

    messages: list[str] = []

    if len(target_rows) < 2:
        messages.append(f"⚠ {target_col}: Dose Levelが2つ未満のため単調性検証不可")
        return messages

    values = target_rows.set_index("dose_level")["mean_manipulated"]
    dose_levels = sorted(values.index)

    # 単調非減少チェック: ×0 ≤ ×1 ≤ ×3
    is_monotonic = True
    for i in range(len(dose_levels) - 1):
        d_low = dose_levels[i]
        d_high = dose_levels[i + 1]
        if values[d_low] > values[d_high]:
            is_monotonic = False
            messages.append(
                f"✗ {target_col}: ×{d_low} ({values[d_low]:.4f}) > "
                f"×{d_high} ({values[d_high]:.4f}) — 単調性違反"
            )

    if is_monotonic:
        vals_str = ", ".join(
            f"×{d}={values[d]:.4f}" for d in dose_levels
        )
        messages.append(f"✓ {target_col}: 単調非減少 ({vals_str})")

    return messages


def check_side_effects(
    report_df: pd.DataFrame,
    target_feature: str,
    threshold_pct: float = 10.0,
) -> list[str]:
    """非対象特徴量の変化が閾値以内かを検証する.

    Args:
        report_df: verify_features() の出力DataFrame
        target_feature: 操作対象特徴量名
        threshold_pct: 副作用警告閾値（%）

    Returns:
        検証メッセージのリスト
    """
    target_col = _target_feature_to_count_col(target_feature)
    non_target = report_df[report_df["feature_name"] != target_col]

    messages: list[str] = []

    for _, row in non_target.iterrows():
        abs_delta_pct = abs(row["delta_pct"])
        if abs_delta_pct > threshold_pct:
            messages.append(
                f"⚠ 副作用: {row['feature_name']} ×{row['dose_level']} "
                f"delta={row['delta_pct']:+.2f}% (閾値 ±{threshold_pct}%)"
            )

    if not messages:
        messages.append("✓ 非対象特徴量の変化は全て閾値以内")

    return messages


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI引数をパースする."""
    ap = argparse.ArgumentParser(
        description="操作済みmonologuesの特徴量検証",
    )
    ap.add_argument(
        "--original_parquet",
        default="artifacts/cejc/monologues_cejc_home2_hq1_v1.parquet",
        help="元のmonologues parquet (default: %(default)s)",
    )
    ap.add_argument(
        "--dose_dir",
        default="artifacts/dose_response",
        help="Dose Level別monologues格納ディレクトリ (default: %(default)s)",
    )
    ap.add_argument(
        "--target_feature",
        required=True,
        choices=list(COUNTER_MAP.keys()),
        help="操作対象特徴量 (FILL / YESNO / OIR)",
    )
    ap.add_argument(
        "--out_tsv",
        default=None,
        help="検証レポート出力先TSV (default: {dose_dir}/feature_verification_{target}.tsv)",
    )
    ap.add_argument(
        "--side_effect_threshold",
        type=float,
        default=10.0,
        help="副作用警告閾値 (%%) (default: %(default)s)",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """メインエントリポイント."""
    args = parse_args(argv)

    # 入力parquet読み込み
    original_path = pathlib.Path(args.original_parquet)
    if not original_path.exists():
        raise FileNotFoundError(f"元parquetが見つかりません: {original_path}")

    original_df = pd.read_parquet(original_path)
    print(f"元データ: {original_path} ({len(original_df)} 行)")

    # Dose Level別parquetを検索
    dose_dir = pathlib.Path(args.dose_dir)
    dose_parquets = find_dose_parquets(dose_dir, args.target_feature)

    if not dose_parquets:
        raise FileNotFoundError(
            f"Dose Level別parquetが見つかりません: "
            f"{dose_dir}/monologues_dose_{args.target_feature}_x*.parquet"
        )

    print(f"検出されたDose Level: {sorted(dose_parquets.keys())}")

    # 各Dose Levelのparquetを読み込み
    dose_dfs: dict[int, pd.DataFrame] = {}
    for level, path in sorted(dose_parquets.items()):
        df = pd.read_parquet(path)
        dose_dfs[level] = df
        print(f"  ×{level}: {path} ({len(df)} 行)")

    # 特徴量検証
    print(f"\n--- 特徴量検証: {args.target_feature} ---")
    report_df = verify_features(original_df, dose_dfs, args.target_feature)

    # 単調性チェック
    print("\n[単調性チェック]")
    mono_msgs = check_monotonicity(report_df, args.target_feature)
    for msg in mono_msgs:
        print(f"  {msg}")

    # 副作用チェック
    print("\n[副作用チェック]")
    side_msgs = check_side_effects(
        report_df, args.target_feature, args.side_effect_threshold
    )
    for msg in side_msgs:
        print(f"  {msg}")

    # TSV出力
    if args.out_tsv is None:
        out_tsv = dose_dir / f"feature_verification_{args.target_feature}.tsv"
    else:
        out_tsv = pathlib.Path(args.out_tsv)

    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(out_tsv, sep="\t", index=False)
    print(f"\n✓ 検証レポート: {out_tsv} ({len(report_df)} 行)")

    # サマリーテーブル表示
    print("\n--- 検証サマリー ---")
    target_col = _target_feature_to_count_col(args.target_feature)
    summary = report_df[report_df["feature_name"] == target_col][
        ["dose_level", "mean_original", "mean_manipulated", "delta", "delta_pct"]
    ]
    print(summary.to_string(index=False))

    print("\n✓ 完了")


if __name__ == "__main__":
    main()
