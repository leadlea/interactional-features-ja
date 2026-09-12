#!/bin/bash
# Coefficient-level permutation test and bootstrap stability for all five traits.
#
# Why this exists
# ---------------
# The coefficient-level results were originally reported for C only. Reporting all
# five dimensions at equal granularity requires rerunning both procedures for
# O, C, E, A and N.
#
# The earlier five-dimension result files could not be reused: they were computed
# with the superseded 18-feature specification (which included IX_topic_drift_mean
# and lacked PG_overlap_rate and PG_pause_variability). Both scripts below already
# implement the final 19-feature set, so this script simply runs them uniformly.
#
# Outputs
# -------
#   artifacts/analysis/results/coef_bootstrap_all5/
#     permutation_coef_{trait}_ensemble.tsv
#     bootstrap_variance_{trait}_ensemble.tsv
#
# Usage
# -----
#   bash scripts/analysis/run_coef_bootstrap_all5.sh
set -euo pipefail

cd "$(dirname "$0")/../.."

PY=${PY:-python}
DATASETS=artifacts/analysis/datasets
OUT=artifacts/analysis/results/coef_bootstrap_all5
ALPHA=${ALPHA:-100}
N_PERM=${N_PERM:-5000}
N_BOOT=${N_BOOT:-500}
SEED=${SEED:-42}

mkdir -p "$OUT"

for TRAIT in O C E A N; do
  PARQUET="$DATASETS/cejc_home2_hq1_XY_${TRAIT}only_ensemble.parquet"

  echo "=== ${TRAIT}: permutation test on the coefficients (n_perm=${N_PERM}) ==="
  $PY -u scripts/analysis/permutation_coef_test.py \
    --xy_parquet "$PARQUET" \
    --y_col "Y_${TRAIT}" \
    --alpha "$ALPHA" \
    --n_perm "$N_PERM" \
    --seed "$SEED" \
    --out_dir "$OUT"

  echo "=== ${TRAIT}: bootstrap coefficient stability (n_boot=${N_BOOT}) ==="
  $PY -u scripts/analysis/bootstrap_variance.py \
    --xy_parquet "$PARQUET" \
    --y_col "Y_${TRAIT}" \
    --alpha "$ALPHA" \
    --n_boot "$N_BOOT" \
    --seed "$SEED" \
    --out_dir "$OUT"
done

echo "DONE -> $OUT"
