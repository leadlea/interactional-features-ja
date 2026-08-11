#!/usr/bin/env bash
# 条件3（random）のLLM Big5採点コマンド一覧
# 4教師 × 5trait = 20回の score_big5_bedrock.py 実行
# 手動ターミナル実行用（AWS Bedrock API経由）
#
# Usage: bash scripts/baseline/run_scoring_random.sh

set -euo pipefail

MONO="artifacts/baseline/monologues_random.parquet"
ITEMS_DIR="artifacts/big5"
OUT_BASE="artifacts/big5/llm_scores"

TRAITS="O C E A N"

run_teacher() {
  local teacher="$1"
  local model_id="$2"
  for trait in $TRAITS; do
    out_dir="${OUT_BASE}/dataset=cejc_home2_hq1_v1__items=${trait}24__teacher=${teacher}__condition=random"
    items_csv="${ITEMS_DIR}/items_ipipneo120_ja_${trait}24.csv"
    echo "=== ${teacher} / ${trait} (condition=random) ==="
    python scripts/big5/score_big5_bedrock.py \
      --monologues_parquet "${MONO}" \
      --items_csv "${items_csv}" \
      --model_id "${model_id}" \
      --out_dir "${out_dir}" \
      --seed 0 \
      --temperature 0.0 \
      --prompt_lang en \
      --paper_strict \
      --max_retries 5
    echo ""
  done
}

run_teacher "sonnet4"      "global.anthropic.claude-sonnet-4-20250514-v1:0"
run_teacher "qwen3-235b"   "qwen.qwen3-235b-a22b-2507-v1:0"
run_teacher "deepseek-v3"  "deepseek.v3-v1:0"
run_teacher "gpt-oss-120b" "openai.gpt-oss-120b-1:0"

echo "✓ 条件3（random）の全20回の採点が完了しました"
