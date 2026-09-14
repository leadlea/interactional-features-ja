# ===========================================================================
#  Reproduction driver
#
#  The pipeline has three kinds of step:
#    (a) deterministic local computation  -> runnable with `make`
#    (b) AWS Bedrock model scoring        -> costs money, run explicitly
#    (c) figure / table generation        -> runnable with `make`
#
#  Steps of kind (b) are NOT wired into aggregate targets, so no `make` target
#  can start a paid scoring run by accident. See docs/reproduction.md.
#
#  Corpus files are not distributed with this repository; see
#  docs/data-availability.md.
# ===========================================================================

PYTHON ?= python

# --- inputs (local, not tracked) -------------------------------------------
CONVLIST_PQ := artifacts/tmp_meta/cejc_convlist.parquet
UTT_PQ      := artifacts/_tmp_utt/cejc_utterances/part-00000.parquet
ITEMS_CSV   := artifacts/big5/items_ipipneo120_ja.csv

# --- derived files ---------------------------------------------------------
PAIRS_PQ    := artifacts/analysis/target_pairs/cejc_home2_hq1_pairs.parquet
MONO_PQ0    := artifacts/cejc/monologues_cejc_home2_hq1.parquet
MONO_PQ     := artifacts/cejc/monologues_cejc_home2_hq1_v1.parquet
MONO_SHA    := artifacts/cejc/monologues_cejc_home2_hq1_v1.sha256.txt
SHARD_DIR   := artifacts/cejc/shards_home2_hq1
FEATURES_PQ := artifacts/analysis/features_min/features_cejc_home2_hq1.parquet
META_TSV    := artifacts/analysis/cejc_speaker_metadata.tsv
RESULTS_DIR := artifacts/analysis/results
FIG_DIR     := reports/paper_figs_v2

# permutation / bootstrap iteration counts as reported in the manuscript
N_PERM  := 5000
N_BOOT  := 500
SEED    := 42

# Features excluded from the design matrix: raw counts, exposure measures, and
# IX_topic_drift_mean (perfectly collinear with IX_lex_overlap_mean). They stay
# in the feature table for provenance. Keep this on ONE line -- make joins
# backslash continuations with a space, which would corrupt the comma list.
EXCLUDE_COLS := n_pairs_total,n_pairs_after_NE,n_pairs_after_YO,IX_n_pairs,IX_n_pairs_after_question,PG_total_time,PG_resp_overlap_rate,FILL_text_len,FILL_cnt_total,FILL_cnt_eto,FILL_cnt_e,FILL_cnt_ano,IX_topic_drift_mean

.PHONY: help setup test \
        pairs monologues shards items features metadata \
        analysis modelb-datasets permutation permutation-groupkfold three-stage \
        coefficients coefficients-all5 sensitivity sensitivity-alpha power \
        confound confound-all5 \
        groupkfold-compare baseline-vs-extended speaker-overlap teacher-agreement \
        figures slides \
        verify verify-consistency \
        baseline-prep baseline-dirs baseline-compare baseline-conditions \
        dose-manipulate dose-verify dose-dirs dose-report \
        clean-figures

help:
	@echo "Setup"
	@echo "  make setup                install pinned dependencies"
	@echo "  make test                 run the unit / property tests"
	@echo ""
	@echo "Pipeline (local, deterministic)"
	@echo "  make pairs                step 1  HQ1 sample selection"
	@echo "  make monologues           step 2  pseudo-monologues (+ sha256 pin)"
	@echo "  make shards               step 3  shard for resumable scoring"
	@echo "  make features             step 5  19 interaction features"
	@echo "  make metadata             step 6  speaker metadata table"
	@echo "  make analysis             step 7  all statistical analyses"
	@echo "  make figures              step 8  manuscript figures and tables"
	@echo "  make verify               step 9  reproducibility checks"
	@echo ""
	@echo "Model scoring (AWS Bedrock, billable) -- see docs/reproduction.md"
	@echo "  DRY_RUN=1 bash scripts/big5/run_scoring_sharded.sh   (plan only)"
	@echo "  bash scripts/big5/run_scoring_sharded.sh"
	@echo ""
	@echo ""
	@echo "Individual analysis steps (all included in 'make analysis')"
	@echo "  make permutation-groupkfold  main result, 21 predictors, subject-wise"
	@echo "  make coefficients-all5       coefficient tests, five dimensions"
	@echo "  make sensitivity-alpha       alpha grid x five dimensions"
	@echo "  make power                   what this design can detect"
	@echo "  make analysis-unreported     analyses kept for review responses"
	@echo "  make groupkfold-compare      KFold vs GroupKFold (longest step)"
	@echo ""
	@echo "Validation experiments"
	@echo "  make baseline-prep        3-condition baseline: build inputs"
	@echo "  make baseline-compare     3-condition baseline: comparison table"
	@echo "  make baseline-conditions  3-condition baseline: subject-wise split"
	@echo "  make dose-manipulate      dose-response: manipulate text"
	@echo "  make dose-verify          dose-response: check manipulation worked"

setup:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest tests -q

# ---------------------------------------------------------------------------
# Step 1-3: corpus -> scoring inputs
# ---------------------------------------------------------------------------
pairs:
	$(PYTHON) scripts/cejc/build_target_pairs_hq1.py \
	  --convlist_parquet $(CONVLIST_PQ) \
	  --utterances_parquet $(UTT_PQ) \
	  --out_parquet $(PAIRS_PQ) \
	  --out_preview_tsv artifacts/analysis/target_pairs/cejc_home2_hq1_pairs_preview.tsv

monologues:
	$(PYTHON) scripts/cejc/build_monologues_hq1.py \
	  --utterances_parquet $(UTT_PQ) \
	  --target_pairs_parquet $(PAIRS_PQ) \
	  --out_parquet $(MONO_PQ0) \
	  --out_pinned_parquet $(MONO_PQ) \
	  --out_sha256 $(MONO_SHA) \
	  --out_preview_tsv artifacts/cejc/monologues_cejc_home2_hq1_preview.tsv

shards:
	$(PYTHON) scripts/cejc/shard_monologues.py \
	  --monologues_parquet $(MONO_PQ) \
	  --out_dir $(SHARD_DIR) \
	  --shard_size 10

items:
	$(PYTHON) scripts/big5/subset_items_by_trait.py \
	  --items_csv $(ITEMS_CSV) \
	  --out_dir artifacts/big5 \
	  --traits O C E A N

# ---------------------------------------------------------------------------
# Step 5-6: features and speaker metadata
# ---------------------------------------------------------------------------
features:
	$(PYTHON) scripts/analysis/extract_interaction_features_min.py \
	  --utterances_parquet $(UTT_PQ) \
	  --target_pairs_parquet $(PAIRS_PQ) \
	  --out_parquet $(FEATURES_PQ)

metadata:
	$(PYTHON) scripts/cejc/build_cejc_speaker_meta.py \
	  --features_parquet $(FEATURES_PQ) \
	  --speaker_csv artifacts/tmp_meta/cejc_speaker.csv \
	  --mapping_csv artifacts/tmp_meta/cejc_speaker_conversation.csv \
	  --out $(META_TSV)

# ---------------------------------------------------------------------------
# Step 7: statistical analyses
# ---------------------------------------------------------------------------
DATASET_DIR := artifacts/analysis/datasets
TRAITS      := O C E A N
TEACHER     := ensemble
XY          = $(DATASET_DIR)/cejc_home2_hq1_XY_$(1)only_$(TEACHER).parquet

# The reported model has 21 predictors: the 19 interactional features plus speaker
# sex and age. Every analysis in the main-result chain takes --include_confounds and
# writes to a *_modelB path, so the 19-predictor outputs stay intact for the appendix
# comparisons. Forgetting the flag silently produces the 19-predictor numbers, which
# is why the paths differ rather than being overwritten in place.
PERM_GK_DIR       := $(RESULTS_DIR)/ensemble_perm_groupkfold
PERM_MB_DIR       := $(RESULTS_DIR)/ensemble_perm_groupkfold_modelB
COEF_ALL5_DIR     := $(RESULTS_DIR)/coef_bootstrap_all5
COEF_ALL5_MB_DIR  := $(RESULTS_DIR)/coef_bootstrap_all5_modelb
CONFOUND_ALL5_TSV := $(RESULTS_DIR)/confound_ensemble_all5.tsv
GK_CMP_TSV        := $(RESULTS_DIR)/groupkfold_vs_kfold_all_nperm$(N_PERM).tsv
BL_COND_GK_TSV    := $(RESULTS_DIR)/baseline_validation/baseline_conditions_groupkfold.tsv
BL_COND_MB_TSV    := $(RESULTS_DIR)/baseline_validation/baseline_conditions_modelB.tsv
POWER_MB_JSON     := $(RESULTS_DIR)/power_analysis_design_modelB.json
ALPHA_GRID        := 10,50,100,200,500

# Datasets carrying the sex and age columns, read by the coefficient-level scripts.
# Everything downstream of coefficients-all5 depends on this target.
modelb-datasets:
	$(PYTHON) scripts/analysis/build_modelb_datasets.py \
	  --datasets_dir $(DATASET_DIR) --metadata_tsv $(META_TSV) --teacher $(TEACHER)

# superseded main result: permutation test under a plain KFold. Kept because the
# alpha sensitivity table and the appendix report the KFold-vs-GroupKFold gap, so
# the KFold numbers have to remain reproducible. Not the reported main result.
permutation:
	$(PYTHON) scripts/analysis/ensemble_permutation.py \
	  --items_dir artifacts/big5/llm_scores \
	  --features_parquet $(FEATURES_PQ) \
	  --exclude_cols "$(EXCLUDE_COLS)" \
	  --out_dir $(RESULTS_DIR)

# main result: permutation test on the model-ensemble scores, all five traits,
# 21 predictors (features + sex + age), subject-wise GroupKFold over cejc_person_id,
# Holm-corrected across traits. Also writes the pooled out-of-fold r, R2, RMSE and
# SD(y) that the headline table reports next to the fold-averaged r.
permutation-groupkfold:
	$(PYTHON) scripts/analysis/ensemble_permutation_groupkfold.py \
	  --datasets_dir $(DATASET_DIR) \
	  --metadata_tsv $(META_TSV) \
	  --include_confounds \
	  --out_tsv $(PERM_MB_DIR)/ensemble_summary_modelB.tsv \
	  --alpha 100 --cv_folds 5 --n_perm $(N_PERM) --seed $(SEED)

# the same test on the 19 features only. Not reported; kept so the effect of adding
# the covariates stays reproducible if a reviewer asks.
permutation-groupkfold-featuresonly:
	$(PYTHON) scripts/analysis/ensemble_permutation_groupkfold.py \
	  --datasets_dir $(DATASET_DIR) \
	  --metadata_tsv $(META_TSV) \
	  --out_tsv $(PERM_GK_DIR)/ensemble_summary_groupkfold.tsv \
	  --alpha 100 --cv_folds 5 --n_perm $(N_PERM) --seed $(SEED)

# three-stage model comparison: demographics -> +classical -> +novel.
# Not reported: stage 3 is the same model as the main result, and at N=120 the design
# cannot judge the stage 2 -> 3 increment. Kept runnable for review responses, and
# because comparing its stage 3 against the main result is a useful cross-check
# (they agree to four decimal places).
THREE_STAGE_DIR := $(RESULTS_DIR)/three_stage_metrics
three-stage:
	@for t in $(TRAITS); do \
	  $(PYTHON) scripts/analysis/three_stage_ridge.py \
	    --xy_parquet $(DATASET_DIR)/cejc_home2_hq1_XY_$${t}only_$(TEACHER).parquet \
	    --y_col Y_$${t} \
	    --metadata_tsv $(META_TSV) \
	    --out_dir $(RESULTS_DIR)/three_stage || exit 1; \
	done
	$(PYTHON) scripts/analysis/three_stage_metrics_diag.py \
	  --teacher $(TEACHER) --metadata_tsv $(META_TSV) \
	  --seed $(SEED) --out_dir $(THREE_STAGE_DIR)
	$(PYTHON) scripts/analysis/three_stage_paired_test.py \
	  --teacher $(TEACHER) --metadata_tsv $(META_TSV) \
	  --n_boot $(N_PERM) --out_dir $(THREE_STAGE_DIR)

# which individual features carry the association, C only (superseded scope;
# retained because the appendix keeps the single-trait tables)
COEF_DIR := $(RESULTS_DIR)/cejc_home2_hq1_Conly_$(TEACHER)_controls_excluded
coefficients:
	$(PYTHON) scripts/analysis/permutation_coef_test.py \
	  --xy_parquet $(call XY,C) --y_col Y_C \
	  --n_perm $(N_PERM) --seed $(SEED) --out_dir $(COEF_DIR)
	$(PYTHON) scripts/analysis/bootstrap_variance.py \
	  --xy_parquet $(call XY,C) --y_col Y_C \
	  --n_boot $(N_BOOT) --seed $(SEED) --out_dir $(RESULTS_DIR)

# reported scope: coefficient permutation test + bootstrap stability for all five
# dimensions, over the 21 predictors of the reported model. Requires modelb-datasets.
coefficients-all5: modelb-datasets
	PY=$(PYTHON) ALPHA=100 N_PERM=$(N_PERM) N_BOOT=$(N_BOOT) SEED=$(SEED) \
	  bash scripts/analysis/run_coef_bootstrap_all5_modelb.sh

# the same procedures on the 19 features only. Not reported; kept for comparison.
coefficients-all5-featuresonly:
	PY=$(PYTHON) ALPHA=100 N_PERM=$(N_PERM) N_BOOT=$(N_BOOT) SEED=$(SEED) \
	  bash scripts/analysis/run_coef_bootstrap_all5.sh

# robustness of C to the arbitrary choices in the feature definitions
sensitivity:
	$(PYTHON) scripts/analysis/sensitivity_analysis.py \
	  --trait C --n_perm $(N_PERM) --seed $(SEED) \
	  --out_dir $(RESULTS_DIR)/sensitivity

# how much the regularisation strength matters, five dimensions x alpha grid,
# under the same subject-wise split as the main result
sensitivity-alpha:
	$(PYTHON) scripts/analysis/ensemble_permutation_groupkfold.py \
	  --datasets_dir $(DATASET_DIR) \
	  --metadata_tsv $(META_TSV) \
	  --include_confounds \
	  --alpha_sweep $(ALPHA_GRID) \
	  --alpha_sweep_tsv $(PERM_MB_DIR)/sensitivity_alpha_modelB.tsv \
	  --cv_folds 5 --n_perm $(N_PERM) --seed $(SEED)

# what magnitude of association this design can detect, for the 21-predictor model.
# Not post hoc power: the outcome is generated synthetically, so the estimate does
# not depend on the observed effect sizes.
power:
	$(PYTHON) scripts/analysis/power_analysis_design.py \
	  --datasets_dir $(DATASET_DIR) --metadata_tsv $(META_TSV) \
	  --include_confounds --seed $(SEED)

# The reported model already controls for sex and age, so neither target below is
# reported. They compare a features-only model against a features-plus-demographics
# model, which the reported design makes redundant. Kept runnable for review responses.
#
# all trait x model combinations
confound:
	$(PYTHON) scripts/analysis/confound_analysis_groupkfold.py \
	  --datasets_dir $(DATASET_DIR) \
	  --metadata_tsv $(META_TSV) \
	  --out_tsv $(RESULTS_DIR)/confound_groupkfold_all.tsv \
	  --n_perm 1000

# five dimensions, ensemble scores only
confound-all5:
	$(PYTHON) scripts/analysis/confound_analysis_groupkfold.py \
	  --datasets_dir $(DATASET_DIR) \
	  --metadata_tsv $(META_TSV) \
	  --traits O,C,E,A,N --ensemble_only \
	  --out_tsv $(CONFOUND_ALL5_TSV) \
	  --n_perm 1000

# how much does the subject-wise split change the estimates (KFold vs GroupKFold).
# The appendix reports this at the manuscript iteration count, so the output file
# name carries N_PERM: 5 traits x 4 models x 2 designs, the longest step here.
groupkfold-compare:
	$(PYTHON) scripts/analysis/groupkfold_all.py \
	  --datasets_dir $(DATASET_DIR) \
	  --metadata_tsv $(META_TSV) \
	  --out_tsv $(GK_CMP_TSV) \
	  --n_perm $(N_PERM)

# appendix: classical-only vs classical+novel feature sets
baseline-vs-extended:
	PYTHON=$(PYTHON) bash scripts/analysis/run_baseline_vs_extended_all.sh

# how many speakers appear in more than one conversation (motivates GroupKFold)
speaker-overlap:
	$(PYTHON) scripts/analysis/speaker_overlap_analysis.py \
	  --metadata_tsv $(META_TSV) \
	  --out $(RESULTS_DIR)

# agreement between the four models, per trait
teacher-agreement:
	$(PYTHON) scripts/analysis/teacher_agreement_big5.py

analysis: modelb-datasets permutation permutation-groupkfold \
          coefficients-all5 \
          sensitivity sensitivity-alpha power \
          groupkfold-compare speaker-overlap teacher-agreement
	@echo "analysis: done -> $(RESULTS_DIR)"

# analyses that are no longer reported but stay reproducible for review responses
analysis-unreported: three-stage coefficients confound confound-all5 \
                     permutation-groupkfold-featuresonly \
                     coefficients-all5-featuresonly baseline-vs-extended
	@echo "analysis-unreported: done -> $(RESULTS_DIR)"

# ---------------------------------------------------------------------------
# Step 8: figures and tables
# ---------------------------------------------------------------------------
figures:
	$(PYTHON) scripts/paper_figs/gen_paper_figs_v2.py \
	  --metadata_tsv $(META_TSV) \
	  --out_dir $(FIG_DIR)
	@# Main-result table and scatter, from the 21-predictor chain.
	@# Every file has one writer, so these can run in any order relative to the
	@# batch generator above. The incremental-validity table and the separate
	@# confound table are no longer in the manuscript and are not produced here.
	$(PYTHON) scripts/paper_figs/gen_main_result_groupkfold.py \
	  --summary_tsv $(PERM_MB_DIR)/ensemble_summary_modelB.tsv \
	  --datasets_dir $(DATASET_DIR) --metadata_tsv $(META_TSV) \
	  --out_dir $(FIG_DIR)
	$(PYTHON) scripts/paper_figs/gen_coef_all5.py \
	  --results_dir $(COEF_ALL5_MB_DIR) --out_dir $(FIG_DIR)
	$(PYTHON) scripts/paper_figs/gen_sensitivity_tables.py \
	  --sweep_tsv $(PERM_MB_DIR)/sensitivity_alpha_modelB.tsv \
	  --summary_tsv $(PERM_MB_DIR)/ensemble_summary_modelB.tsv \
	  --per_model_tsv $(GK_CMP_TSV) \
	  --out_dir $(FIG_DIR)
	$(PYTHON) scripts/paper_figs/gen_tab_baseline_conditions.py \
	  --tsv $(BL_COND_MB_TSV) --out_dir $(FIG_DIR)

slides:
	$(PYTHON) scripts/paper_figs/gen_kamishibai_slides.py

clean-figures:
	rm -f $(FIG_DIR)/fig_*.png $(FIG_DIR)/tab_*.tex

# ---------------------------------------------------------------------------
# Step 9: reproducibility checks
# ---------------------------------------------------------------------------
verify:
	$(PYTHON) scripts/analysis/verify_reproducibility.py

verify-consistency:
	$(PYTHON) scripts/analysis/verify_numerical_consistency.py
	$(PYTHON) scripts/analysis/verify_predicted_vs_three_stage.py

# ---------------------------------------------------------------------------
# Validation experiment A: 3-condition baseline
#   condition 1 = real transcript   (main analysis)
#   condition 2 = summary statistics only, no transcript
#   condition 3 = transcript of a different speaker (derangement)
# Scoring for conditions 2 and 3 is billable and run explicitly.
# ---------------------------------------------------------------------------
BL_DIR    := artifacts/baseline
COND1_TSV := $(RESULTS_DIR)/ensemble_perm_v4/ensemble_summary.tsv
COND2_TSV := $(RESULTS_DIR)/baseline_validation/condition_summary/ensemble_summary.tsv
COND3_TSV := $(RESULTS_DIR)/baseline_validation/condition_random/ensemble_summary.tsv
COMP_TSV  := $(RESULTS_DIR)/baseline_validation/comparison_summary.tsv

baseline-prep:
	$(PYTHON) scripts/baseline/gen_summary_monologues.py \
	  --utterances_parquet $(UTT_PQ) \
	  --monologues_parquet $(MONO_PQ) \
	  --out_parquet $(BL_DIR)/monologues_summary.parquet
	$(PYTHON) scripts/baseline/gen_random_monologues.py \
	  --monologues_parquet $(MONO_PQ) \
	  --out_parquet $(BL_DIR)/monologues_random.parquet \
	  --mapping_csv $(BL_DIR)/shuffle_mapping.csv \
	  --seed $(SEED)

baseline-dirs:
	$(PYTHON) scripts/baseline/prepare_ensemble_dirs.py \
	  --scores_dir artifacts/big5/llm_scores \
	  --out_summary_dir artifacts/big5/llm_scores_summary \
	  --out_random_dir artifacts/big5/llm_scores_random

baseline-compare:
	$(PYTHON) scripts/baseline/compare_conditions.py \
	  --cond1_tsv $(COND1_TSV) \
	  --cond2_tsv $(COND2_TSV) \
	  --cond3_tsv $(COND3_TSV) \
	  --out_tsv $(COMP_TSV) \
	  --threshold 0.1

# reported version: all three conditions recomputed under the same subject-wise
# split as the main result, so that the delta r across conditions is comparable
baseline-conditions:
	PYTHONPATH=scripts/analysis $(PYTHON) \
	  scripts/analysis/baseline_conditions_groupkfold.py \
	  --items_dir artifacts/big5/llm_scores \
	  --features_parquet $(FEATURES_PQ) \
	  --metadata_tsv $(META_TSV) \
	  --include_confounds \
	  --out_tsv $(BL_COND_MB_TSV) \
	  --alpha 100 --cv_folds 5 --n_perm $(N_PERM) --seed $(SEED)

# ---------------------------------------------------------------------------
# Validation experiment B: feature dose-response
#   manipulate one feature at x0 / x1 / x3 and re-score, to test whether the
#   model's trait estimate tracks the manipulated feature.
#   DRYRUN=1 make dose-manipulate  reports the planned edits without writing.
# ---------------------------------------------------------------------------
DOSE_OUT      := artifacts/dose_response
DOSE_SCORES   := artifacts/big5/llm_scores
DOSE_RESULTS  := $(RESULTS_DIR)/dose_response
DOSE_BASELINE := $(COND1_TSV)
DOSE_FEATURES := FILL YESNO OIR

ifdef DRYRUN
  DOSE_DRYRUN := --dry-run
else
  DOSE_DRYRUN :=
endif

dose-manipulate:
	@for f in $(DOSE_FEATURES); do \
	  PYTHONPATH=. $(PYTHON) scripts/dose_response/gen_dose_monologues.py \
	    --monologues_parquet $(MONO_PQ) \
	    --target_feature $$f \
	    --dose_levels 0,1,3 \
	    --out_dir $(DOSE_OUT) \
	    --seed $(SEED) $(DOSE_DRYRUN) || exit 1; \
	done

dose-verify:
	@for f in $(DOSE_FEATURES); do \
	  PYTHONPATH=. $(PYTHON) scripts/dose_response/verify_features.py \
	    --original_parquet $(MONO_PQ) \
	    --dose_dir $(DOSE_OUT) \
	    --target_feature $$f || exit 1; \
	done

dose-dirs:
	@for f in $(DOSE_FEATURES); do \
	  PYTHONPATH=. $(PYTHON) scripts/dose_response/prepare_ensemble_dirs.py \
	    --scores_dir $(DOSE_SCORES) \
	    --dose_levels 0,3 \
	    --target_feature $$f \
	    --out_dir artifacts/big5 || exit 1; \
	done

dose-report:
	@for f in FILL YESNO; do \
	  PYTHONPATH=. $(PYTHON) scripts/dose_response/dose_response_report.py \
	    --results_dir $(DOSE_RESULTS) \
	    --baseline_tsv $(DOSE_BASELINE) \
	    --target_feature $$f \
	    --control_feature OIR \
	    --verification_tsv $(DOSE_OUT)/feature_verification_$$f.tsv \
	    --out_dir $(DOSE_OUT) || exit 1; \
	done
