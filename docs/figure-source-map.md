# Figure and table provenance

Every figure and table in the manuscript, with the script that emitted it and the
result file it read. Use this to trace a number in the paper back to the code
that produced it.

Terms:

- **generator** — the code that writes the `.png` / `.tex` under `reports/paper_figs_v2/`
- **source data** — the file the generator reads; the primary source for the values
- **upstream** — the analysis script that computed the source data

Paths in the *source data* column are relative to `artifacts/analysis/results/`
unless stated otherwise. Everything lands in `reports/paper_figs_v2/`.

The manuscript uses 13 generated tables and 8 generated figures. Three further
sets of values are typed into the text rather than generated; they are listed
under [hand-entered values](#hand-entered-values).

## Main text

| Label | Kind | Content | Generator | Source data | Upstream |
|---|---|---|---|---|---|
| `tab:feature_def` | table | 19 feature definitions | `gen_paper_figs_v2.py::gen_tab_feature_definitions` | `scripts/paper_figs/feature_definitions.py` (static) | — |
| `tab:desc_stats_full` | table | descriptive statistics | `gen_paper_figs_v2.py::gen_descriptive_stats_full_table` | `features_min/features_cejc_home2_hq1.parquet` | `extract_interaction_features_min.py` |
| `fig:feature_dist` | figure | feature distributions by category | `gen_paper_figs_v2.py::gen_feature_distribution` | same features parquet | same |
| `fig:corr_heatmap` | figure | correlation heatmap, block structure | `gen_paper_figs_v2.py::gen_corr_heatmap_block` | same features parquet | same |
| `tab:metadata_tests` | table | sex (Mann-Whitney U) and age (Spearman rho) | `gen_paper_figs_v2.py::gen_tab_metadata_tests` | features parquet + `cejc_speaker_metadata.tsv` | tests run inside the generator, pairwise deletion |
| `tab:ensemble_perm` | table | **main result**: permutation test, five traits, subject-wise split | `gen_main_result_groupkfold.py` | `ensemble_perm_groupkfold/ensemble_summary_groupkfold.tsv` | `ensemble_permutation_groupkfold.py` |
| `fig:predicted_vs_observed` | figure | observed vs out-of-fold predicted, five panels | `gen_main_result_groupkfold.py` | same summary TSV + `datasets/cejc_home2_hq1_XY_{trait}only_ensemble.parquet` + `cejc_speaker_metadata.tsv` | same |
| `tab:coef_all5` | table | coefficients, 19 features x 5 dimensions, concordance marks | `gen_coef_all5.py` | `coef_bootstrap_all5/permutation_coef_{trait}_ensemble.tsv` + `bootstrap_variance_{trait}_ensemble.tsv` | `run_coef_bootstrap_all5.sh` (`permutation_coef_test.py`, `bootstrap_variance.py`) |
| `fig:coef_all5` | figure | coefficient plot, five stacked panels, beta ± 95% CI | `gen_coef_all5.py` | same | same |
| `tab:confound_all5` | table | sex and age added as predictors, five dimensions | `gen_main_result_groupkfold.py` | `confound_ensemble_all5.tsv` | `confound_analysis_groupkfold.py --traits O,C,E,A,N --ensemble_only` |

A feature is marked `*` in `tab:coef_all5` only when the permutation test and the
bootstrap interval agree (a *concordant feature*); `\dagger` marks a feature that
satisfies one of the two. The rule is stated in
[evaluation-design.md](evaluation-design.md#which-features-carry-the-association)
and implemented once, in `gen_coef_all5.py`.

## Appendix

| Label | Kind | Content | Generator | Source data | Upstream |
|---|---|---|---|---|---|
| `tab:corr_matrix` | table | 19x19 correlation matrix | `gen_paper_figs_v2.py::_gen_tab_corr_matrix` | features parquet | `extract_interaction_features_min.py` |
| `fig:metadata_gender` | figure | features by sex | `gen_paper_figs_v2.py::gen_metadata_gender` | features parquet + `cejc_speaker_metadata.tsv` | tests inside the generator |
| `fig:metadata_age` | figure | features by age | `gen_paper_figs_v2.py::gen_metadata_age` | same | same |
| `tab:coef_all5_detail` | table | full beta / p / 95% CI per dimension (longtable) | `gen_coef_all5.py` | `coef_bootstrap_all5/` | `run_coef_bootstrap_all5.sh` |
| `tab:sensitivity_alpha` | table | alpha grid x five dimensions, Holm-corrected | `gen_sensitivity_tables.py` | `ensemble_perm_groupkfold/sensitivity_alpha_groupkfold.tsv` | `ensemble_permutation_groupkfold.py --alpha_sweep` |
| `tab:cv_sensitivity` | table | KFold vs GroupKFold | `gen_sensitivity_tables.py` | `groupkfold_vs_kfold_all_nperm5000.tsv` + the GroupKFold summary TSV | `groupkfold_all.py --n_perm 5000` |
| `tab:three_stage` | table | incremental validity, R² / RMSE, exploratory | `gen_tab_three_stage_r2.py --teacher ensemble` | `three_stage_metrics/three_stage_metrics_ensemble.tsv` + `three_stage_paired_test_ensemble.tsv` | `three_stage_metrics_diag.py`, `three_stage_paired_test.py` |
| `fig:teacher_corr_matrix_appendix` | figure | 4x4 model correlation matrix | `gen_paper_figs_v2.py::gen_fig_teacher_corr_matrix` | `reports/model_agreement/teacher_corr_{trait}.tsv` | `teacher_agreement_big5.py` |
| `fig:teacher_heatmap` | figure | mean between-model agreement | `gen_paper_figs_v2.py::gen_fig_teacher_heatmap` | same | same |
| `tab:perm_all` | table | permutation test per model, ordered O,C,E,A,N | `gen_sensitivity_tables.py` | `groupkfold_vs_kfold_all_nperm5000.tsv` | `groupkfold_all.py --n_perm 5000` |
| `tab:baseline_conditions` | table | three-condition baseline, subject-wise split | `gen_tab_baseline_conditions.py` | `baseline_validation/baseline_conditions_groupkfold.tsv` | `baseline_conditions_groupkfold.py` |

## Japanese and English twins

Both manuscripts read the same generated files. Where a table has caption or
header text, the generator writes two files in one run — `tab_x.tex` with
Japanese labels and `tab_x_en.tex` with English ones — from a single computation,
so the two versions cannot drift numerically:

| Generator | `_en` twins |
|---|---|
| `gen_paper_figs_v2.py::gen_tab_feature_definitions` | `tab_feature_definitions_en.tex` |
| `gen_main_result_groupkfold.py` | `tab_ensemble_permutation_en.tex`, `tab_confound_all5_en.tex` |
| `gen_coef_all5.py` | `tab_coef_all5_en.tex`, `tab_coef_all5_detail_en.tex` |
| `gen_sensitivity_tables.py` | `tab_sensitivity_alpha_en.tex`, `tab_cv_sensitivity_en.tex`, `tab_permutation_all_en.tex` |
| `gen_tab_baseline_conditions.py` | `tab_baseline_conditions_en.tex` |

Four generated tables (`tab_descriptive_stats_full`, `tab_metadata_tests`,
`tab_corr_matrix`, `tab_three_stage`) and all eight figures need no twin: their
labels, axis titles and legends are already English, and the only data-derived
labels are gender `F`/`M` and numeric age.

## Hand-entered values

Three sets of values are typed into the manuscript rather than generated. When
any of them change, the files below are authoritative and the text has to be
updated by hand. These three are **not** covered by `make verify`, which checks
the main-result values, the concordant feature sets, the KFold comparison and the
between-model agreement; extending it to these is an open item.

| Where | Content | Source data | Upstream |
|---|---|---|---|
| `tab:score_stats` | trait score descriptives, Cronbach alpha | `artifacts/big5/llm_scores/` | `score_big5_bedrock.py`, `teacher_agreement_big5.py` |
| appendix, per-model confound breakdown | sex/age control per model | `confound_groupkfold_all.tsv` | `confound_analysis_groupkfold.py` |
| appendix, dose-response | feature manipulation experiment | `artifacts/dose_response/` + scores | `scripts/dose_response/`, `score_big5_bedrock.py` |

## Regenerating

```bash
make figures
```

Every file in `reports/paper_figs_v2/` has exactly one writer, so the generators
can be run in any order and none overwrites another's output.

`gen_tab_three_stage_r2.py` sits outside the batch on purpose: it produces the
R²/RMSE version of the incremental-validity table, and keeping it separate means
rerunning the batch cannot revert the table to the correlation-based version. The
old functions inside `gen_paper_figs_v2.py` (`gen_fig_three_stage_comparison`,
`gen_tab_three_stage`) are never called.

`gen_paper_figs_v2.py` accepts `--bootstrap_dir` but no longer reads it, because
the only generator that used it is no longer in the batch. The option is kept so
existing invocations do not break.

Two scripts are not part of `make figures`:

```bash
make slides                                             # the explainer deck
python scripts/paper_figs/gen_fig_three_stage_r2.py \
  --teacher ensemble                                    # incremental-validity figure
```

The incremental-validity figure is not in the manuscript — the appendix carries
only the table — so `make figures` does not produce it.

## The explainer deck

`make slides` writes `reports/paper_figs_v2/kamishibai_slides.html`, a
self-contained nine-slide walkthrough in Japanese: two method slides, five slides
covering the main-text results, and two appendix slides. It embeds only figures
the manuscript uses and its numbers are transcribed from the generated tables, so
it has the same failure mode as the hand-entered manuscript values below: when a
result changes, `gen_kamishibai_slides.py` has to be updated by hand. The
docstring at the top of that script records which table each slide's numbers come
from.

## Not cited by the manuscript

One generated file remains without a manuscript reference:
`tab_three_stage_r.tex`, the correlation version of the incremental-validity
table, which `gen_tab_three_stage_r2.py` writes alongside the cited
`tab_three_stage.tex`.

Outputs that nothing writes and nothing consumes were removed when the analysis
moved to five dimensions: the C-only coefficient tables and figures
(`tab_permutation_coef.tex`, `tab_bootstrap_variance.tex`,
`fig_bootstrap_variance.png`, `fig_permutation_C_bar.png`,
`fig_bootstrap_C_radar.png`), the classical-vs-extended appendix pair
(`tab_baseline_vs_extended.tex`, `fig_baseline_vs_extended.png`), the short
descriptive table superseded by the `_full` version
(`tab_descriptive_stats.tex`), the sample-selection flowchart
(`fig_consort_flowchart.png`), the bar version of the permutation result
(`fig_ensemble_permutation.png`), the incremental-validity figure
(`fig_three_stage_comparison.png`), and two `_legacy` three-stage files. Their
generator functions are still in `gen_paper_figs_v2.py`, listed in the comment at
the top of its batch list, but are not called.
