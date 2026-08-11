#!/usr/bin/env bash
# Feature Dose-Response 実験: LLM Big5採点コマンド
# 3特徴量 × 2条件（×0, ×3）× 4教師 × 5trait = 120回の score_big5_bedrock.py 実行
# ×1 は既存結果を再利用するため採点不要
#
# 手動ターミナル実行用（AWS Bedrock API経由）
#
# Usage:
#   # 全条件実行:
#   bash scripts/dose_response/run_scoring_dose.sh
#
#   # 特定の特徴量のみ:
#   bash scripts/dose_response/run_scoring_dose.sh FILL
#   bash scripts/dose_response/run_scoring_dose.sh YESNO
#   bash scripts/dose_response/run_scoring_dose.sh OIR

set -euo pipefail

DOSE_DIR="artifacts/dose_response"
ITEMS_DIR="artifacts/big5"
OUT_BASE="artifacts/big5/llm_scores"
TRAITS="O C E A N"

# 対象特徴量（引数で絞り込み可能）
TARGET_FEATURE="${1:-ALL}"

run_teacher() {
  local teacher="$1"
  local model_id="$2"
  local feature="$3"
  local dose="$4"

  local mono="${DOSE_DIR}/monologues_dose_${feature}_x${dose}.parquet"

  if [ ! -f "$mono" ]; then
    echo "SKIP: $mono not found"
    return
  fi

  for trait in $TRAITS; do
    local out_dir="${OUT_BASE}/dataset=cejc_home2_hq1_v1__items=${trait}24__teacher=${teacher}__dose=${feature}_x${dose}"
    local items_csv="${ITEMS_DIR}/items_ipipneo120_ja_${trait}24.csv"

    echo "=== ${feature} ×${dose} / ${teacher} / ${trait} ==="
    python scripts/big5/score_big5_bedrock_v2.py \
      --monologues_parquet "${mono}" \
      --items_csv "${items_csv}" \
      --model_id "${model_id}" \
      --out_dir "${out_dir}" \
      --temperature 0.0 \
      --paper_strict \
      --max_retries 5
    echo ""
  done
}

run_feature() {
  local feature="$1"
  # ×0 と ×3 のみ（×1 は既存結果を再利用）
  for dose in 0 3; do
    echo ""
    echo "============================================================"
    echo "  ${feature} ×${dose}"
    echo "============================================================"
    run_teacher "sonnet4"      "global.anthropic.claude-sonnet-4-20250514-v1:0" "$feature" "$dose"
    run_teacher "qwen3-235b"   "qwen.qwen3-235b-a22b-2507-v1:0"               "$feature" "$dose"
    run_teacher "deepseek-v3"  "deepseek.v3-v1:0"                              "$feature" "$dose"
    run_teacher "gpt-oss-120b" "openai.gpt-oss-120b-1:0"                       "$feature" "$dose"
  done
}

echo "Feature Dose-Response: LLM Big5 採点開始"
echo "  Dose Dir: ${DOSE_DIR}"
echo "  Output:   ${OUT_BASE}"
echo ""

if [ "$TARGET_FEATURE" = "ALL" ]; then
  run_feature "FILL"
  run_feature "YESNO"
  run_feature "OIR"
else
  run_feature "$TARGET_FEATURE"
fi

echo ""
echo "✓ Feature Dose-Response 採点完了"
echo "  次のステップ: prepare_ensemble_dirs.py → ensemble_permutation.py → dose_response_report.py"
