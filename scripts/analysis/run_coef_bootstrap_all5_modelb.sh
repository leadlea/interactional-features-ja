#!/bin/bash
# Coefficient-level analysis for the confound-controlled design (19 features + sex + age
# = 21 predictors), run over all five traits.
#
# Why this exists
# ---------------
# The headline result is reported from an "interactional features + sex + age" model.
# "Which interactional features contribute to each trait dimension" has to come from the
# same design, so the coefficient permutation test and the bootstrap are recomputed with
# 21 predictors.
#
# The 19-predictor script (run_coef_bootstrap_all5.sh) is kept: the appendix uses it for
# the cross-validation design comparison, and it keeps the 19-predictor results exactly
# reproducible.
#
# Prerequisite
# ------------
#   python scripts/analysis/build_modelb_datasets.py
# creates the XYB_* datasets this script reads.
#
# Outputs
# -------
#   artifacts/analysis/results/coef_bootstrap_all5_modelb/
#     permutation_coef_{trait}_ensemble.tsv
#     bootstrap_variance_{trait}_ensemble.tsv
#
# Usage
# -----
#   bash scripts/analysis/run_coef_bootstrap_all5_modelb.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

PY=${PY:-python}
DATASETS=artifacts/analysis/datasets
OUT=artifacts/analysis/results/coef_bootstrap_all5_modelb
ALPHA=${ALPHA:-100}
N_PERM=${N_PERM:-5000}
N_BOOT=${N_BOOT:-500}
SEED=${SEED:-42}

mkdir -p "$OUT"

for TRAIT in O C E A N; do
  PARQUET="$DATASETS/cejc_home2_hq1_XYB_${TRAIT}only_ensemble.parquet"
  if [[ ! -f "$PARQUET" ]]; then
    echo "[ERROR] $PARQUET is missing; run build_modelb_datasets.py first" >&2
    exit 1
  fi

  echo "=== ${TRAIT}: permutation test on the coefficients (21 predictors, n_perm=${N_PERM}) ==="
  $PY -u scripts/analysis/permutation_coef_test.py \
    --xy_parquet "$PARQUET" \
    --y_col "Y_${TRAIT}" \
    --alpha "$ALPHA" \
    --n_perm "$N_PERM" \
    --seed "$SEED" \
    --include_confounds \
    --out_dir "$OUT"

  echo "=== ${TRAIT}: bootstrap coefficient stability (21 predictors, n_boot=${N_BOOT}) ==="
  $PY -u scripts/analysis/bootstrap_variance.py \
    --xy_parquet "$PARQUET" \
    --y_col "Y_${TRAIT}" \
    --alpha "$ALPHA" \
    --n_boot "$N_BOOT" \
    --seed "$SEED" \
    --include_confounds \
    --out_dir "$OUT"
done

echo "DONE -> $OUT"
