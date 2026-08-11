#!/usr/bin/env bash
# ===========================================================================
#  Score IPIP-NEO-120 with Amazon Bedrock: trait x model x shard.
#
#  BILLABLE. Every invocation sends transcript text to Bedrock model endpoints.
#  A full pass is (traits) x (models) x (shards) runs; with the reported
#  configuration that is 5 x 4 x 12 = 240 script invocations.
#
#  Output layout (this is what the downstream analysis globs for -- do not
#  change it without updating ensemble_permutation.py and
#  teacher_agreement_big5.py):
#
#    artifacts/big5/llm_scores/
#      dataset=<DATASET>__items=<TRAIT>24__teacher=<LABEL>/
#        shard=<SID>/
#          model=<sanitised model id>/
#            trait_scores.parquet
#            attempts.jsonl
#
#  Resumable: a (trait, model, shard) whose trait_scores.parquet already exists
#  is skipped, so an interrupted run is resumed by re-invoking this script. That
#  also means a stale output file is silently kept -- delete the directory rather
#  than editing it if you need a genuine re-run. Afterwards run
#  merge_teacher_scores.py with --expected_records to confirm that nothing was
#  skipped or double-counted.
#
#  Models file (default artifacts/big5/models.txt): one model per line as
#
#      <label><whitespace><bedrock model id>
#
#  where <label> is the short name used in the results and figures.
#  '#' starts a comment. Example:
#
#      sonnet4       global.anthropic.claude-sonnet-4-20250514-v1:0
#      qwen3-235b    qwen.qwen3-235b-a22b-2507-v1:0
#      deepseek-v3   deepseek.v3-v1:0
#      gpt-oss-120b  openai.gpt-oss-120b-1:0
#
#  Usage
#    bash scripts/big5/run_scoring_sharded.sh
#    TRAITS="C" bash scripts/big5/run_scoring_sharded.sh     # one trait
#    DRY_RUN=1 bash scripts/big5/run_scoring_sharded.sh      # plan only
#
#  Environment overrides
#    TRAITS      traits to score                  (default: "O C E A N")
#    SHARD_DIR   directory of monologue shards
#    ITEMS_DIR   directory holding <ITEMS_STEM>_<TRAIT>NN.csv
#    ITEMS_STEM  stem of the per-trait item files (default: items_ipipneo120_ja)
#    MODELS_FILE label + model id per line
#    OUT_ROOT    root of the score tree
#    DATASET     dataset label embedded in the output path
#    CONDITION   appends __condition=<value> (used by the baseline experiment)
#    REGION      AWS region                       (default: ap-northeast-1)
#    MAX_TOKENS / SLEEP / SEED / PROMPT_LANG      passed to the scorer
#    DRY_RUN=1   print what would run, call nothing
# ===========================================================================
set -euo pipefail

PYTHON="${PYTHON:-python}"
TRAITS="${TRAITS:-O C E A N}"
SHARD_DIR="${SHARD_DIR:-artifacts/cejc/shards_home2_hq1}"
ITEMS_DIR="${ITEMS_DIR:-artifacts/big5}"
ITEMS_STEM="${ITEMS_STEM:-items_ipipneo120_ja}"
MODELS_FILE="${MODELS_FILE:-artifacts/big5/models.txt}"
OUT_ROOT="${OUT_ROOT:-artifacts/big5/llm_scores}"
DATASET="${DATASET:-cejc_home2_hq1_v1}"
CONDITION="${CONDITION:-}"
REGION="${REGION:-${AWS_REGION:-ap-northeast-1}}"
MAX_TOKENS="${MAX_TOKENS:-64}"   # 64 keeps GPT-OSS from padding the reply
SLEEP="${SLEEP:-0.0}"
SEED="${SEED:-0}"
PROMPT_LANG="${PROMPT_LANG:-en}"
DRY_RUN="${DRY_RUN:-}"

# Bedrock throttles under sustained load; back off adaptively rather than fail.
export AWS_RETRY_MODE="${AWS_RETRY_MODE:-adaptive}"
export AWS_MAX_ATTEMPTS="${AWS_MAX_ATTEMPTS:-10}"

# --- preflight -------------------------------------------------------------
[[ -d "$SHARD_DIR" ]]   || { echo "[NG] shard dir not found: $SHARD_DIR" >&2; exit 1; }
[[ -f "$MODELS_FILE" ]] || { echo "[NG] models file not found: $MODELS_FILE" >&2; exit 1; }

shopt -s nullglob
SHARDS=("$SHARD_DIR"/*shard*.parquet)
shopt -u nullglob
[[ ${#SHARDS[@]} -gt 0 ]] || { echo "[NG] no *shard*.parquet in $SHARD_DIR" >&2; exit 1; }

LABELS=(); MODEL_IDS=()
while IFS= read -r line || [[ -n "$line" ]]; do
  line="${line%%#*}"
  line="$(printf '%s' "$line" | awk '{$1=$1; print}')"
  [[ -z "$line" ]] && continue
  label="$(printf '%s' "$line" | awk '{print $1}')"
  model="$(printf '%s' "$line" | awk '{print $2}')"
  [[ -n "$model" ]] || {
    echo "[NG] $MODELS_FILE: expected '<label> <model id>', got: $line" >&2
    exit 1
  }
  LABELS+=("$label"); MODEL_IDS+=("$model")
done < "$MODELS_FILE"
[[ ${#MODEL_IDS[@]} -gt 0 ]] || { echo "[NG] no models listed in $MODELS_FILE" >&2; exit 1; }

# Resolve the per-trait item file. Kept as a function rather than an
# associative array so the script runs on bash 3.2 (the macOS default).
items_file_for_trait() {
  local trait="$1" found
  shopt -s nullglob
  found=("$ITEMS_DIR/${ITEMS_STEM}_${trait}"*.csv)
  shopt -u nullglob
  if [[ ${#found[@]} -eq 0 ]]; then
    echo "[NG] no item file for trait $trait: $ITEMS_DIR/${ITEMS_STEM}_${trait}*.csv" >&2
    echo "     run scripts/big5/subset_items_by_trait.py first" >&2
    return 1
  fi
  printf '%s' "${found[0]}"
}

for T in $TRAITS; do
  items_file_for_trait "$T" >/dev/null || exit 1
done

n_traits=$(printf '%s\n' $TRAITS | wc -l | tr -d ' ')
echo "traits=${n_traits}  models=${#MODEL_IDS[@]}  shards=${#SHARDS[@]}"
echo "planned runs: $(( n_traits * ${#MODEL_IDS[@]} * ${#SHARDS[@]} ))"
[[ -n "$CONDITION" ]] && echo "condition: ${CONDITION}"
[[ -n "$DRY_RUN" ]] && echo "DRY_RUN: nothing will be sent to Bedrock"
echo

# --- run -------------------------------------------------------------------
n_run=0; n_skip=0; n_fail=0
cond_suffix=""
[[ -n "$CONDITION" ]] && cond_suffix="__condition=${CONDITION}"

for T in $TRAITS; do
  ITEMS="$(items_file_for_trait "$T")"
  n_items="$($PYTHON -c "import pandas as pd,sys; print(len(pd.read_csv(sys.argv[1])))" "$ITEMS")"

  for i in "${!MODEL_IDS[@]}"; do
    LABEL="${LABELS[$i]}"
    MODEL="${MODEL_IDS[$i]}"
    SAFE="$($PYTHON -c "import re,sys; print(re.sub(r'[^A-Za-z0-9._-]+','_',sys.argv[1]))" "$MODEL")"
    TRAIT_ROOT="${OUT_ROOT}/dataset=${DATASET}__items=${T}${n_items}__teacher=${LABEL}${cond_suffix}"

    for SHARD in "${SHARDS[@]}"; do
      SID="$(basename "$SHARD" .parquet | sed 's/.*shard//')"
      OUTDIR="${TRAIT_ROOT}/shard=${SID}/model=${SAFE}"

      if [[ -f "$OUTDIR/trait_scores.parquet" ]]; then
        n_skip=$((n_skip + 1))
        continue
      fi

      echo "== ${T} / ${LABEL} / shard=${SID}"
      if [[ -n "$DRY_RUN" ]]; then
        echo "   -> ${OUTDIR}"
        n_run=$((n_run + 1))
        continue
      fi

      mkdir -p "$OUTDIR"
      if $PYTHON scripts/big5/score_big5_bedrock.py \
            --monologues_parquet "$SHARD" \
            --items_csv "$ITEMS" \
            --model_id "$MODEL" \
            --region "$REGION" \
            --out_dir "$OUTDIR" \
            --max_tokens "$MAX_TOKENS" \
            --sleep "$SLEEP" \
            --seed "$SEED" \
            --prompt_lang "$PROMPT_LANG" \
            --paper_strict \
            --on_fail nan \
            --attempts_jsonl "$OUTDIR/attempts.jsonl"; then
        n_run=$((n_run + 1))
      else
        echo "[WARN] failed: ${T} / ${LABEL} / shard=${SID}" >&2
        n_fail=$((n_fail + 1))
      fi
    done
  done
done

echo
echo "ran=${n_run}  skipped=${n_skip}  failed=${n_fail}"
if [[ $n_fail -gt 0 ]]; then
  echo "re-invoke this script to retry only the failures" >&2
  exit 1
fi
exit 0
