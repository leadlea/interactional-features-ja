# Reproduction

Nine steps. Steps 1-3 and 5-9 are deterministic and free; step 4 calls Amazon
Bedrock and is billable, which is why no `make` target chains into it.

Before starting, obtain the corpus and lay out the input files as described in
[`../artifacts/README.md`](../artifacts/README.md), then:

```bash
python -m venv .venv && source .venv/bin/activate
make setup          # pinned dependencies (Python 3.12)
make test           # unit and property tests, no corpus needed
```

`make help` lists every target. All commands are run from the repository root.

---

## Step 1 — select the analysis sample

```bash
make pairs
```

Applies the sampling frame (conversations at home with exactly two speakers) and
the HQ1 quality filter to produce the target-pair table. Expected output for the
reported run: **120 records** from **66 conversations**.

The three HQ1 thresholds (`>=80` adjacent pairs, `>=2000` characters, `>=10`
post-question pairs) are CLI options, so the sample definition itself can be
varied:

```bash
python scripts/cejc/build_target_pairs_hq1.py --help
```

## Step 2 — build the pseudo-monologues

```bash
make monologues
```

Concatenates each selected speaker's utterances in chronological order. Writes a
version-pinned copy plus its sha256; compare the digest against the one recorded
in [data-availability.md](data-availability.md) before going further.

## Step 3 — prepare the scoring inputs

```bash
make shards     # split into 10-record shards so scoring is resumable
make items      # split IPIP-NEO-120 into per-trait subsets of 24 items
```

## Step 4 — score the traits (Bedrock, billable)

Not wired into a `make` target. Run it deliberately:

First list the models, as `<label> <bedrock model id>` in
`artifacts/big5/models.txt`. The label is the short name that appears in the
results:

```
sonnet4       global.anthropic.claude-sonnet-4-20250514-v1:0
qwen3-235b    qwen.qwen3-235b-a22b-2507-v1:0
deepseek-v3   deepseek.v3-v1:0
gpt-oss-120b  openai.gpt-oss-120b-1:0
```

Check the plan before spending anything, then run it:

```bash
DRY_RUN=1 bash scripts/big5/run_scoring_sharded.sh   # 240 runs, sends nothing
bash scripts/big5/run_scoring_sharded.sh
```

The runner skips a (trait, model, shard) whose output already exists, so an
interrupted run is resumed by re-invoking it, and a single trait can be redone
with `TRAITS="C" bash scripts/big5/run_scoring_sharded.sh`. Protocol details,
prompt structure, and the baseline conditions are in
[llm-scoring-protocol.md](llm-scoring-protocol.md).

Then merge the shards back and check nothing was scored twice or skipped:

```bash
python scripts/big5/merge_teacher_scores.py \
  --scores_root "artifacts/big5/llm_scores/dataset=cejc_home2_hq1_v1__items=C24__teacher=sonnet4" \
  --out_parquet "artifacts/big5/llm_scores/dataset=cejc_home2_hq1_v1__items=C24__teacher=sonnet4/teacher_merged/trait_scores_C_merged.parquet" \
  --expected_records 120
```

## Step 5 — extract the interaction features

```bash
make features
```

Computes the 19 explanatory features and the control variables from the
utterance table. This reads the utterances directly, not the monologues, so the
feature side and the model side of the analysis share no preprocessing.
Definitions: [feature-dictionary.md](feature-dictionary.md).

## Step 6 — assemble the analysis tables

```bash
make metadata     # speaker sex, age, and cejc_person_id per record
```

Then join features with each trait score:

```bash
for T in O C E A N; do
  python scripts/analysis/build_xy_dataset.py \
    --features_parquet artifacts/analysis/features_min/features_cejc_home2_hq1.parquet \
    --scores_parquet "artifacts/big5/llm_scores/dataset=cejc_home2_hq1_v1__items=${T}24__teacher=ensemble/teacher_merged/trait_scores_${T}_merged.parquet" \
    --trait "$T" \
    --out_parquet "artifacts/analysis/datasets/cejc_home2_hq1_XY_${T}only_ensemble.parquet"
done
```

The script prints how many records were lost on each side of the join. The
reported analyses use all 120, so any loss here needs explaining before moving
on.

## Step 7 — run the analyses

```bash
make analysis
```

which is the following, in order:

| Target | What it produces | Reported as |
|---|---|---|
| `make permutation` | permutation test on the ensemble scores, five traits, Holm-corrected | main result |
| `make three-stage` | demographics -> +classical -> +novel, R² / RMSE, paired bootstrap | incremental validity |
| `make coefficients` | per-feature permutation test and bootstrap CIs for C | which features carry it |
| `make sensitivity` | `gap_tol`, yes/no lexicon, ne/yo matching variants | robustness |
| `make confound` | sex and age as additional predictors, GroupKFold | confound control |
| `make speaker-overlap` | unique speakers and repeat appearances | justifies GroupKFold |
| `make teacher-agreement` | between-model correlation per trait | how stable the outcome is |

Additional analyses not in the aggregate target:

```bash
make groupkfold-compare      # how much the subject-wise split changes estimates
make baseline-vs-extended    # classical-only vs classical+novel (appendix)
```

Iteration counts (`N_PERM=5000`, `N_BOOT=500`) and `SEED=42` are Makefile
variables and match the manuscript. Lower them for a quick smoke test:

```bash
make coefficients N_PERM=50 N_BOOT=20
```

## Step 8 — regenerate the figures and tables

```bash
make figures
```

Overwrites `reports/paper_figs_v2/` with the 14 figures and 10 LaTeX tables used
in the manuscript. Which figure comes from which script and which result file is
tabulated in [figure-source-map.md](figure-source-map.md).

## Step 9 — check the reported numbers

```bash
make verify              # manuscript values vs result files
make verify-consistency  # cross-checks between analyses that should agree
```

`verify` writes `artifacts/analysis/results/reproducibility_check.tsv` with one
row per checked value and a match flag. `verify-consistency` covers two places
where the same quantity is computed differently — the three-stage Stage 3
correlation versus the predicted-vs-observed scatter — and reports which of the
two axes (feature set, or fold-averaged versus pooled out-of-fold aggregation)
accounts for the difference.

---

## Validation experiments

Both test whether the model scores actually depend on the transcript, rather than
on surface quantities that correlate with the features.

**Three-condition baseline.** Condition 1 is the real transcript, condition 2
replaces it with summary statistics only, condition 3 gives each record another
speaker's transcript (a derangement, so no record keeps its own).

```bash
make baseline-prep       # build the condition 2 and 3 inputs
# score conditions 2 and 3 (billable):
bash scripts/baseline/run_scoring_summary.sh
bash scripts/baseline/run_scoring_random.sh
bash scripts/baseline/run_ensemble_conditions.sh
make baseline-dirs baseline-compare
```

**Feature dose-response.** Manipulate one feature at x0 / x1 / x3 in the text
and re-score, to see whether the trait estimate moves with it.

```bash
DRYRUN=1 make dose-manipulate    # report planned edits without writing
make dose-manipulate dose-verify
# score the manipulated texts (billable):
bash scripts/dose_response/run_scoring_dose.sh
bash scripts/dose_response/run_ensemble_dose.sh
make dose-dirs dose-report
```

`dose-verify` confirms the manipulation changed the intended feature and left the
control feature alone. Run it before spending money on scoring.

---

## What will not reproduce exactly

- **Model scores.** Bedrock endpoints are not version-frozen; the same prompt can
  return different scores later. The scored outputs are the record of what the
  models returned at the time.
- **Everything downstream of fixed inputs.** Given the same feature table and the
  same scores, the permutation tests, bootstrap intervals, and figures are
  seeded and reproduce exactly. Library versions matter here, which is why
  `requirements.txt` is pinned rather than ranged.
