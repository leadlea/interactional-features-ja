#!/usr/bin/env python3
"""Comparison Reporter — 3条件の ensemble_summary.tsv を比較するレポート生成.

3条件（テキストあり / 要約のみ / ランダムテキスト）の ensemble_summary.tsv を
読み込み、trait 単位で結合して比較テーブルを生成する。

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6
"""
from __future__ import annotations

import argparse
import math
import pathlib

import pandas as pd


# ---------------------------------------------------------------------------
# 純粋関数: ラベル判定（プロパティテスト用にインポート可能）
# ---------------------------------------------------------------------------
def assign_label(
    r1: float,
    r2: float,
    r3: float,
    threshold: float,
) -> str:
    """3条件の r_obs に基づき判定ラベルを付与する（純粋関数）.

    Parameters
    ----------
    r1 : float
        条件1（テキストあり）の r_obs
    r2 : float
        条件2（要約のみ）の r_obs
    r3 : float
        条件3（ランダムテキスト）の r_obs
    threshold : float
        ≈ vs >> 判定閾値（デフォルト 0.1）

    Returns
    -------
    str
        判定ラベル:
        - "テキスト内容依存": |r1 - r2| >= threshold AND |r1 - r3| >= threshold
        - "表層統計量依存の可能性": |r1 - r2| < threshold
        - "テキスト内容が寄与": |r2 - r3| < threshold
        - "N/A": いずれかの r_obs が NaN
    """
    # NaN チェック
    if math.isnan(r1) or math.isnan(r2) or math.isnan(r3):
        return "N/A"

    diff_12 = abs(r1 - r2)
    diff_13 = abs(r1 - r3)
    diff_23 = abs(r2 - r3)

    if diff_12 >= threshold and diff_13 >= threshold:
        return "テキスト内容依存"
    if diff_12 < threshold:
        return "表層統計量依存の可能性"
    if diff_23 < threshold:
        return "テキスト内容が寄与"

    # フォールバック（上記条件で網羅されるが安全のため）
    return "N/A"


# ---------------------------------------------------------------------------
# TSV 読み込み
# ---------------------------------------------------------------------------
def _read_condition_tsv(path: str, condition_name: str) -> pd.DataFrame:
    """ensemble_summary.tsv を読み込み condition カラムを付与する.

    Parameters
    ----------
    path : str
        TSV ファイルパス
    condition_name : str
        条件名（text / summary / random）

    Returns
    -------
    pd.DataFrame
        trait, r_obs, p_value, p_corrected, condition
    """
    df = pd.read_csv(path, sep="\t")
    df["condition"] = condition_name
    return df


# ---------------------------------------------------------------------------
# 比較テーブル生成
# ---------------------------------------------------------------------------
def build_comparison(
    cond1_df: pd.DataFrame,
    cond2_df: pd.DataFrame,
    cond3_df: pd.DataFrame,
    threshold: float = 0.1,
) -> pd.DataFrame:
    """3条件の DataFrame を結合し比較テーブルを生成する.

    Parameters
    ----------
    cond1_df, cond2_df, cond3_df : pd.DataFrame
        各条件の ensemble_summary データ（trait, r_obs, p_value, p_corrected, condition）
    threshold : float
        ラベル判定閾値

    Returns
    -------
    pd.DataFrame
        condition, trait, r_obs, p_value, p_corrected, delta_r_vs_text, label
    """
    # 条件1 の r_obs を trait → r_obs の辞書に
    r1_map = dict(zip(cond1_df["trait"], cond1_df["r_obs"]))
    # 条件2 の r_obs を trait → r_obs の辞書に
    r2_map = dict(zip(cond2_df["trait"], cond2_df["r_obs"]))
    # 条件3 の r_obs を trait → r_obs の辞書に
    r3_map = dict(zip(cond3_df["trait"], cond3_df["r_obs"]))

    # 全 trait の和集合
    all_traits = sorted(
        set(r1_map.keys()) | set(r2_map.keys()) | set(r3_map.keys())
    )

    # trait ごとにラベルを算出
    label_map: dict[str, str] = {}
    for trait in all_traits:
        r1 = r1_map.get(trait, float("nan"))
        r2 = r2_map.get(trait, float("nan"))
        r3 = r3_map.get(trait, float("nan"))
        label_map[trait] = assign_label(r1, r2, r3, threshold)

    # 3条件を結合
    combined = pd.concat([cond1_df, cond2_df, cond3_df], ignore_index=True)

    # delta_r_vs_text: 条件1 の r_obs との差分
    def _calc_delta(row: pd.Series) -> float:
        r1 = r1_map.get(row["trait"], float("nan"))
        if row["condition"] == "text":
            return 0.0
        if math.isnan(r1) or (isinstance(row["r_obs"], float) and math.isnan(row["r_obs"])):
            return float("nan")
        return r1 - row["r_obs"]

    combined["delta_r_vs_text"] = combined.apply(_calc_delta, axis=1)

    # label: trait ごとの共通ラベル
    combined["label"] = combined["trait"].map(label_map)

    # カラム順序を整理
    combined = combined[
        ["condition", "trait", "r_obs", "p_value", "p_corrected", "delta_r_vs_text", "label"]
    ]

    # ソート: condition → trait
    condition_order = {"text": 0, "summary": 1, "random": 2}
    combined["_cond_sort"] = combined["condition"].map(condition_order).fillna(3)
    combined = combined.sort_values(["_cond_sort", "trait"]).drop(columns=["_cond_sort"])
    combined = combined.reset_index(drop=True)

    return combined


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="3条件の ensemble_summary.tsv を比較するレポート生成"
    )
    ap.add_argument(
        "--cond1_tsv",
        default="artifacts/analysis/results/ensemble_perm_v4/ensemble_summary.tsv",
        help="条件1（テキストあり）の ensemble_summary.tsv",
    )
    ap.add_argument(
        "--cond2_tsv",
        default="artifacts/analysis/results/baseline_validation/condition_summary/ensemble_summary.tsv",
        help="条件2（要約のみ）の ensemble_summary.tsv",
    )
    ap.add_argument(
        "--cond3_tsv",
        default="artifacts/analysis/results/baseline_validation/condition_random/ensemble_summary.tsv",
        help="条件3（ランダムテキスト）の ensemble_summary.tsv",
    )
    ap.add_argument(
        "--out_tsv",
        default="artifacts/analysis/results/baseline_validation/comparison_summary.tsv",
        help="比較テーブル出力先",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="≈ vs >> 判定閾値",
    )
    args = ap.parse_args()

    # --- 読み込み ---
    cond1 = _read_condition_tsv(args.cond1_tsv, "text")
    cond2 = _read_condition_tsv(args.cond2_tsv, "summary")
    cond3 = _read_condition_tsv(args.cond3_tsv, "random")

    # --- 比較テーブル生成 ---
    comparison = build_comparison(cond1, cond2, cond3, args.threshold)

    # --- 保存 ---
    pathlib.Path(args.out_tsv).parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.out_tsv, sep="\t", index=False)
    print(f"✓ 比較テーブルを {args.out_tsv} に保存しました（{len(comparison)} 行）")


if __name__ == "__main__":
    main()
