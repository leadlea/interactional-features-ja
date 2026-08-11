#!/bin/bash
# Run baseline_vs_extended.py for all trait × teacher combinations
# 5 traits × 4 teachers = 20 runs

PYTHON="${PYTHON:-python}"
SCRIPT="scripts/analysis/baseline_vs_extended.py"
OUT_DIR="artifacts/analysis/results"
N_PERM=5000
SEED=42

TRAITS=("C" "A" "E" "N" "O")
TEACHERS=("sonnet" "qwen3-235b" "gpt-oss-120b" "deepseek-v3")
Y_COLS=("Y_C" "Y_A" "Y_E" "Y_N" "Y_O")

for i in "${!TRAITS[@]}"; do
    trait="${TRAITS[$i]}"
    y_col="${Y_COLS[$i]}"
    for teacher in "${TEACHERS[@]}"; do
        parquet="artifacts/analysis/datasets/cejc_home2_hq1_XY_${trait}only_${teacher}.parquet"
        if [ ! -f "$parquet" ]; then
            echo "SKIP: $parquet not found"
            continue
        fi
        echo "=== ${trait} / ${teacher} ==="
        $PYTHON $SCRIPT \
            --xy_parquet "$parquet" \
            --y_col "$y_col" \
            --out_dir "$OUT_DIR" \
            --n_perm $N_PERM \
            --seed $SEED
    done
done

echo "All done."
