#!/usr/bin/env bash
# 条件2・条件3の ensemble_permutation.py 実行コマンド
# 手動ターミナル実行用（Permutation test 5000回は計算時間がかかる）
#
# 前提: 
#   1. run_scoring_summary.sh / run_scoring_random.sh が完了済み
#   2. prepare_ensemble_dirs.py が実行済み
#
# Usage: bash scripts/baseline/run_ensemble_conditions.sh
# または各コマンドを個別にコピー＆ペーストして実行

set -euo pipefail

echo "=== Step 1: prepare_ensemble_dirs.py ==="
python scripts/baseline/prepare_ensemble_dirs.py
echo ""

echo "=== Step 2: 条件2（summary）の ensemble_permutation.py ==="
python scripts/analysis/ensemble_permutation.py \
  --items_dir artifacts/big5/llm_scores_summary \
  --out_dir artifacts/analysis/results/baseline_validation/condition_summary \
  --n_perm 5000 \
  --seed 42 \
  --alpha 100.0 \
  --cv_folds 5
echo ""

echo "=== Step 3: 条件3（random）の ensemble_permutation.py ==="
python scripts/analysis/ensemble_permutation.py \
  --items_dir artifacts/big5/llm_scores_random \
  --out_dir artifacts/analysis/results/baseline_validation/condition_random \
  --n_perm 5000 \
  --seed 42 \
  --alpha 100.0 \
  --cv_folds 5
echo ""

echo "✓ 条件2・条件3の ensemble_permutation.py が完了しました"
echo ""
echo "条件1の結果は既存を再利用:"
echo "  artifacts/analysis/results/ensemble_perm_v4/ensemble_summary.tsv"
echo ""
echo "次のステップ: compare_conditions.py を実行してください"
