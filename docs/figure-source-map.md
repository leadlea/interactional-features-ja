# Figure and table provenance

Every figure and table in the manuscript, with the script that emitted it and the
result file it read. Use this to trace a number in the paper back to the code
that produced it.

Terms:

- **generator** — the code that writes the `.png` / `.tex` under `reports/paper_figs_v2/`
- **source data** — the file the generator reads; the primary source for the values
- **upstream** — the analysis script that computed the source data

## Main text

| Label | Kind | Content | Generator | Source data | Upstream |
|---|---|---|---|---|---|
| `tab:feature_def` | table | 19 feature definitions | `gen_paper_figs_v2.py::gen_tab_feature_definitions` | `scripts/paper_figs/feature_definitions.py` (static) | — |

`tab_feature_definitions.tex` is the only generated table whose caption is in
Japanese. `gen_tab_feature_definitions(out_dir, lang="en")` writes
`tab_feature_definitions_en.tex`, which the English manuscript uses; the table
body is byte-identical and only the caption and the longtable continuation
markers differ. Every other generated table, and every figure, contains no
Japanese text: axis labels, legends and table bodies are English, and the
data-derived labels are gender `F`/`M` and numeric age.
| `fig:feature_dist` | figure | feature distributions by category | `gen_paper_figs_v2.py::gen_feature_distribution` | `features_min/features_cejc_home2_hq1.parquet` | `extract_interaction_features_min.py` |
| `tab:desc_stats_full` | table | descriptive statistics | `gen_paper_figs_v2.py::gen_descriptive_stats_full_table` | same features parquet | same |
| `fig:corr_heatmap` | figure | correlation heatmap, block structure | `gen_paper_figs_v2.py::gen_corr_heatmap_block` | same features parquet | same |
| `tab:corr_matrix` | table | 19x19 correlation matrix | `gen_paper_figs_v2.py::_gen_tab_corr_matrix` | same features parquet | same |
| `tab:metadata_tests` | table | sex (Mann-Whitney U) and age (Spearman rho) | `gen_paper_figs_v2.py::gen_tab_metadata_tests` | features parquet + `cejc_speaker_metadata.tsv` | tests run inside the generator, pairwise deletion |
| `tab:three_stage` | table | three-stage Ridge, R² / RMSE | `gen_tab_three_stage_r2.py --teacher ensemble` | `three_stage_metrics/three_stage_metrics_ensemble.tsv` + `three_stage_paired_test_ensemble.tsv` | `three_stage_metrics_diag.py`, `three_stage_paired_test.py` |
| `fig:three_stage` | figure | three-stage Ridge, R² bars | `gen_fig_three_stage_r2.py --teacher ensemble` | same | same |
| `tab:ensemble_perm` | table | permutation test, all five traits | `gen_paper_figs_v2.py::gen_tab_ensemble_permutation` | `ensemble_perm_v4/ensemble_summary.tsv` | `ensemble_permutation.py` |
| `fig:predicted_vs_observed` | figure | predicted vs observed, out-of-fold | `gen_paper_figs_v2.py::gen_fig_predicted_vs_observed` | `ensemble_perm_v4/ensemble_summary.tsv` + `datasets/cejc_home2_hq1_XY_{trait}only_ensemble.parquet` | `ensemble_permutation.py` |
| `tab:permutation_coef` | table | per-coefficient permutation test (C) | `gen_paper_figs_v2.py::gen_tab_permutation_coef` | `cejc_home2_hq1_Conly_*_controls_excluded/` | `permutation_coef_test.py` |
| `tab:bootstrap_variance` | table | bootstrap coefficient stability (C) | `gen_paper_figs_v2.py::gen_tab_bootstrap_variance` | `bootstrap_variance_{trait}_{model}.tsv` | `bootstrap_variance.py` |
| `fig:bootstrap_variance` | figure | bootstrap CI forest plot | `gen_paper_figs_v2.py::gen_fig_bootstrap_variance` | same | same |

## Appendix

| Label | Kind | Content | Generator | Source data | Upstream |
|---|---|---|---|---|---|
| `fig:metadata_gender` | figure | features by sex | `gen_paper_figs_v2.py::gen_metadata_gender` | features parquet + `cejc_speaker_metadata.tsv` | tests inside the generator |
| `fig:metadata_age` | figure | features by age | `gen_paper_figs_v2.py::gen_metadata_age` | same | same |
| `tab:score_stats` | inline table | trait score descriptives | **hand-entered in the manuscript** | `artifacts/big5/llm_scores/` + Cronbach alpha | `score_big5_bedrock.py`, `teacher_agreement_big5.py` |
| `tab:three_stage_r` | table | three-stage Ridge, correlation version | `gen_tab_three_stage_r2.py` | `three_stage_{trait}_{model}.tsv` (r column) | `three_stage_ridge.py` |
| `tab:sensitivity_alpha` | table | alpha sensitivity | `gen_paper_figs_v2.py::gen_tab_sensitivity_alpha` | `sensitivity/sensitivity_results.tsv` (`analysis_type=alpha`) | `sensitivity_analysis.py` |
| (confound values) | inline | sex/age controlled models | **hand-entered in the manuscript** | `confound_groupkfold_all.tsv` | `confound_analysis_groupkfold.py` |
| `fig:teacher_corr_matrix_appendix` | figure | 4x4 model correlation matrix | `gen_paper_figs_v2.py::gen_fig_teacher_corr_matrix` | `reports/model_agreement/teacher_corr_{trait}.tsv` | `teacher_agreement_big5.py` |
| `fig:teacher_heatmap` | figure | mean between-model agreement | `gen_paper_figs_v2.py::gen_fig_teacher_heatmap` | same | same |
| `tab:perm_all` | table | permutation test per model | `gen_paper_figs_v2.py::gen_tab_permutation_all` | `cejc_home2_hq1_*only_*_controls_excluded/` | `ensemble_permutation.py` (per model) |
| `tab:baseline_comparison` | inline table | three-condition baseline | **hand-entered in the manuscript** | `baseline_validation/comparison_summary.tsv`; condition 1 = `ensemble_perm_v4/ensemble_summary.tsv` | `scripts/baseline/compare_conditions.py` |
| (dose-response values) | inline | feature manipulation experiment | **hand-entered in the manuscript** | `artifacts/dose_response/` + scores | `scripts/dose_response/`, `score_big5_bedrock.py` |

Paths above are relative to `artifacts/analysis/results/` unless stated
otherwise. Figures and tables land in `reports/paper_figs_v2/`.

## Regenerating

```bash
make figures        # everything except the three-stage R² figure and table
                    # (those have their own generators and are also invoked here)
```

The three-stage R² figure and table are produced by dedicated scripts rather than
by the batch generator. Older functions inside `gen_paper_figs_v2.py`
(`gen_fig_three_stage_comparison`, `gen_tab_three_stage`) produce the
correlation-based version and are excluded from the batch, so that rerunning the
batch cannot revert the manuscript to the superseded metric.

## The four hand-entered tables

Four sets of values are typed into the manuscript rather than generated: trait
score descriptives, the confound-controlled models, the three-condition
baseline, and the dose-response results. When any of those change, the source
files listed above are authoritative and the manuscript has to be updated by
hand. `make verify` exists partly to catch the case where that update was
missed.
