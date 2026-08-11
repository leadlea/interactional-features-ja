#!/usr/bin/env bash
# Feature Dose-Response 実験: ensemble_permutation.py 実行コマンド
# 3特徴量 × 2条件（×0, ×3）= 6回の ensemble_permutation.py 実行
# ×1 は既存結果（ensemble_perm_v4）を再利用するため実行不要
#
# 注意: features_parquet は ORIGINAL（操作前）の特徴量を使用する。
# Dose-Response実験では LLM スコアの変化を見るため、説明変数は固定。
#
# 手動ターミナル実行用（Permutation test 5000回は計算時間がかかる）
#
# Usage:
#   # 全条件実行:
#   bash scripts/dose_response/run_ensemble_dose.sh
#
#   # 特定の特徴量のみ:
#   bash scripts/dose_response/run_ensemble_dose.sh FILL
#   bash scripts/dose_response/run_ensemble_dose.sh YESNO
#   bash scripts/dose_response/run_ensemble_dose.sh OIR

set -euo pipefail

# ── 設定 ──
SCORES_BASE="artifacts/big5"
FEATURES_PQ="artifacts/analysis/features_min/features_cejc_home2_hq1.parquet"
RESULTS_BASE="artifacts/analysis/results/dose_response"
ENSEMBLE_SCRIPT="scripts/analysis/ensemble_permutation.py"

N_PERM=5000
SEED=42
ALPHA=100.0
CV_FOLDS=5

# 対象特徴量（引数で絞り込み可能）
TARGET_FEATURE="${1:-ALL}"

run_ensemble() {
  local feature="$1"
  local dose="$2"

  local items_dir="${SCORES_BASE}/llm_scores_dose_${feature}_x${dose}"
  local out_dir="${RESULTS_BASE}/dose_${feature}_x${dose}"

  if [ ! -d "$items_dir" ]; then
    echo "SKIP: $items_dir not found (run prepare_ensemble_dirs.py first)"
    return
  fi

  echo ""
  echo "============================================================"
  echo "  ${feature} ×${dose}"
  echo "  items_dir:      ${items_dir}"
  echo "  features_pq:    ${FEATURES_PQ}"
  echo "  out_dir:        ${out_dir}"
  echo "============================================================"

  python "${ENSEMBLE_SCRIPT}" \
    --items_dir "${items_dir}" \
    --features_parquet "${FEATURES_PQ}" \
    --alpha "${ALPHA}" \
    --cv_folds "${CV_FOLDS}" \
    --n_perm "${N_PERM}" \
    --seed "${SEED}" \
    --out_dir "${out_dir}"

  echo ""
  echo "  ✓ ${feature} ×${dose} 完了 → ${out_dir}/ensemble_summary.tsv"
}

run_feature() {
  local feature="$1"
  # ×0 と ×3 のみ（×1 は既存結果 ensemble_perm_v4 を再利用）
  for dose in 0 3; do
    run_ensemble "$feature" "$dose"
  done
}

echo "Feature Dose-Response: ensemble_permutation.py 実行開始"
echo "  Features PQ: ${FEATURES_PQ} (ORIGINAL, 操作前)"
echo "  Results:     ${RESULTS_BASE}"
echo "  n_perm:      ${N_PERM}"
echo ""

if [ "$TARGET_FEATURE" = "ALL" ]; then
  run_feature "FILL"
  run_feature "YESNO"
  run_feature "OIR"
else
  run_feature "$TARGET_FEATURE"
fi

echo ""
echo "✓ Feature Dose-Response ensemble 実行完了"
echo "  ×1 ベースライン: artifacts/analysis/results/ensemble_perm_v4/ensemble_summary.tsv"
echo "  次のステップ: dose_response_report.py"
