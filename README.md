# interactional-features-ja

Reproducible measurement of **19 interactional features** from Japanese everyday
conversation, and the analyses that validate them.

The features describe *how* people talk to each other rather than what they say:
speech timing and silences (`PG`), fillers (`FILL`), how a response relates to
what preceded it (`IX`), and what follows the sentence-final particles ne / yo
(`RESP`). All of them are computed from transcripts with timestamps, with no
manual coding step.

This repository is the code and provenance record for a manuscript submitted to
*Behavior Research Methods*. It contains the full pipeline, the tests, the
generated figures and tables, and the documentation needed to check any reported
number against the code that produced it. It does not contain the corpus.

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
   analyses, confound control, and baseline conditions that check whether the
   models are using the transcript at all.

The contribution is the measurement instrument and its validation procedure, not
a personality prediction model. The trait scores are a criterion for validating
the features, not a measurement of the speakers' personalities, and nothing here
is diagnostic.

## Repository layout

```
scripts/
  cejc/            sample selection, monologue construction, sharding, speaker metadata
  big5/            IPIP-NEO-120 scoring via Amazon Bedrock, item subsetting, score merging
  analysis/        features, Ridge + permutation + bootstrap, sensitivity, confounds, verification
  baseline/        three-condition baseline validation
  dose_response/   feature manipulation experiment
  paper_figs/      figure and table generation
tests/             unit and property-based tests (no corpus needed)
docs/              methods documentation
reports/
  paper_figs_v2/   generated figures and LaTeX tables as they appear in the manuscript
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
  files and writes a per-value match report.
- Model endpoints are not version-frozen by the provider, so re-scoring may not
  return identical values. Everything downstream of fixed inputs does.

## Citation

See [CITATION.cff](CITATION.cff). Please cite the manuscript once published; this
repository can be cited alongside it for the implementation.

## License

Code (`scripts/`, `tests/`): MIT — see [LICENSE](LICENSE).
Documentation and generated figures (`docs/`, `reports/`): CC BY 4.0.
The CEJC corpus is covered by neither and is not redistributed here.
