# interactional-features-ja

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22888170.svg)](https://doi.org/10.5281/zenodo.22888170)

Reproducible measurement of **19 interactional features** from Japanese everyday
conversation, and the analyses that validate them.

The features describe *how* people talk to each other rather than what they say:
pause and gap timing, filler use, how a response relates to what preceded it, and
what follows the sentence-final particles ne / yo. All of them are computed from
transcripts with timestamps, with no manual coding step.

**Two sets of category abbreviations are in use, on purpose.** The manuscript labels
the four categories PG / FU / SO / RT, while the column names in this repository keep
the implementation's original prefixes. The prefixes are deliberately *not* renamed,
so that a value reported in the manuscript can still be traced to the released code
and to the SHA-256-pinned artefacts.

| Category | Label in the manuscript | Column prefix here | Count |
|---|---|---|---:|
| pause / gap | `PG` | `PG_` | 9 |
| filler use | `FU` | `FILL_` | 2 |
| sequence organisation | `SO` | `IX_` | 5 |
| response typing | `RT` | `RESP_` | 3 |

The mapping is defined in one place, `CATEGORY_LABELS` in
[`scripts/paper_figs/feature_definitions.py`](scripts/paper_figs/feature_definitions.py),
and is what the generated tables and figures render.

This repository is the code and provenance record for a manuscript being prepared
for *Behavior Research Methods*. It contains the full pipeline, the tests, the
generated figures and tables, and the documentation needed to check any reported
number against the code that produced it. It does not contain the corpus or the
manuscript.

- Pipeline and commands: [docs/reproduction.md](docs/reproduction.md)
- Feature definitions: [docs/feature-dictionary.md](docs/feature-dictionary.md)
- Statistical design: [docs/evaluation-design.md](docs/evaluation-design.md)
- Outcome variable: [docs/llm-scoring-protocol.md](docs/llm-scoring-protocol.md)
- Figure-to-code map: [docs/figure-source-map.md](docs/figure-source-map.md)
- Data access and licensing: [docs/data-availability.md](docs/data-availability.md)

## What the study does

1. Select a homogeneous sample from the Corpus of Everyday Japanese Conversation
   (CEJC): conversations at home between two speakers, passing a transcript
   quality filter. **120 records** (66 conversations x speaker).
2. Compute the 19 features from the utterance table.
3. Separately, have four language models complete IPIP-NEO-120 on behalf of each
   speaker, reading only that speaker's concatenated utterances. The averaged
   item-level responses are the external criterion.
4. Test whether the features predict that criterion, under a subject-wise
   split, with permutation tests, bootstrap coefficient stability, sensitivity
   analyses, and baseline conditions that check whether the models are using the
   transcript at all. All five trait dimensions are analysed and reported at the
   same granularity, down to the coefficient level.

Speaker sex and age are entered as predictors alongside the 19 features, so the
reported model has **21 predictors** and every association is estimated with those
attributes held constant. There is no separate confound-control analysis: checking
for confounding only requires entering the confounders simultaneously. The analysis
scripts take `--include_confounds` for this design and write to `*_modelB` paths.

The contribution is the measurement instrument and its validation procedure, not
a personality prediction model. The trait scores are a criterion for validating
the features, not a measurement of the speakers' personalities, and nothing here
is diagnostic.

## Repository layout

```
scripts/
  cejc/            sample selection, monologue construction, sharding, speaker metadata
  big5/            IPIP-NEO-120 scoring via Amazon Bedrock, item subsetting, score merging
  analysis/        features, Ridge + permutation + bootstrap, sensitivity, power, verification
  baseline/        three-condition baseline validation
  dose_response/   feature manipulation experiment
  paper_figs/      figure and table generation
  synthetic/       structural stand-in for the inputs, from published statistics only
tests/             unit and property-based tests (no corpus needed)
docs/              methods documentation
reports/
  paper_figs_v2/   the manuscript's 8 figures and 13 LaTeX tables; tables with
                   Japanese labels also have an `_en` twin written in the same run,
                   for the English version. Also holds kamishibai_slides.html, a
                   nine-slide walkthrough of the study written in Japanese
  model_agreement/ between-model correlation tables
artifacts/         local working directory, not tracked (see artifacts/README.md)
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
make setup
make test        # runs without corpus data
make help        # every pipeline target
```

Reproducing the analyses requires the CEJC corpus. Reproducing the model scoring
additionally requires AWS credentials and incurs Bedrock charges; no `make`
target starts a scoring run, so that step is always explicit.

Without the corpus you can still run the whole pipeline:

```bash
make synthetic        # a dataset of the same shape, from published statistics only
make synthetic-check  # the headline analysis on it
```

The licence does not permit redistributing the feature matrix or the trait scores, so
`artifacts/synthetic/` stands in for them. It does not reproduce the reported results and
is not meant to — by default its outcome is independent of the features, so every test
comes out non-significant, which is how you confirm the pipeline runs and the null case
behaves. See [docs/data-availability.md](docs/data-availability.md).

## Requirements

Python 3.12. Dependencies are pinned exactly in `requirements.txt`, because the
reported p-values come from seeded permutation and bootstrap procedures and
depend on the estimator implementation.

## Reproducibility notes

- The text shown to the models is pinned by sha256; the digest for the reported
  run is in [docs/data-availability.md](docs/data-availability.md).
- Seeds are passed explicitly (`SEED=42`); iteration counts (`N_PERM=5000`,
  `N_BOOT=500`) are Makefile variables and match the manuscript.
- `make verify` compares the values reported in the manuscript against the result
  files and writes a per-value match report (59 checks): every value the headline
  table prints for all five dimensions (fold-averaged r, Holm-corrected p, pooled
  out-of-fold r, R² and RMSE), the dually supported feature set per dimension, the
  count under each decision rule, the plain-KFold values the appendix compares
  against, the a priori power values, and the between-model agreement. It reads the
  21-predictor result paths, matching the reported model.
- `make power` reports what magnitude of association this design can detect: under
  the null the fold-averaged r has SD = 0.128, giving a two-sided 5% critical value
  of r = 0.250, and power reaches 50% at rho = 0.349 and 80% at rho = 0.438. This is
  the sensitivity of the design, not post hoc power from an observed effect size.
  `make figures` turns the same JSON into the appendix table.
- Model endpoints are not version-frozen by the provider, so re-scoring may not
  return identical values. Everything downstream of fixed inputs does.

## Citation

See [CITATION.cff](CITATION.cff). Please cite the manuscript once published; this
repository can be cited alongside it for the implementation.

Each release is archived on Zenodo, which mints two DOIs. They are not
interchangeable:

| DOI | Resolves to | Use for |
|---|---|---|
| [10.5281/zenodo.22888170](https://doi.org/10.5281/zenodo.22888170) | whichever release is latest | the badge above, citing the project in general |
| [10.5281/zenodo.22888171](https://doi.org/10.5281/zenodo.22888171) | v1.3.0 specifically | reproducing the reported numbers |

The manuscript cites the version DOI, because a reader has to be able to reach the
exact code that produced the values in the article.

## License

Code (`scripts/`, `tests/`): MIT — see [LICENSE](LICENSE).
Documentation and generated figures (`docs/`, `reports/`): CC BY 4.0.
The CEJC corpus is covered by neither and is not redistributed here.
