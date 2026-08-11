# Data availability

## What is and is not in this repository

| | Included | Why |
|---|---|---|
| Analysis and figure code | yes | `scripts/`, `tests/` |
| Generated figures and LaTeX tables | yes | `reports/` — lets a reader check the outputs without corpus access |
| Model-agreement correlation tables | yes | `reports/model_agreement/` — aggregate statistics only |
| CEJC transcripts | **no** | distributed under a data use agreement; redistribution is not permitted |
| Utterance tables, monologues, feature matrices | **no** | derived from the transcripts, so the same restriction applies |
| Language-model item responses and trait scores | **no** | contain no transcript text, but are keyed to corpus records; withheld with the rest of `artifacts/` |

## Obtaining the corpus

The Corpus of Everyday Japanese Conversation (CEJC) is published by the National
Institute for Japanese Language and Linguistics (NINJAL) and is available to
researchers under NINJAL's terms of use:
<https://www2.ninjal.ac.jp/conversation/cejc/>

Once you have the corpus, `artifacts/README.md` describes the file layout the
pipeline expects, and [reproduction.md](reproduction.md) gives the commands.

## Identity of the derived inputs

The analysis sample is a fixed set of 120 records (66 conversations x speaker;
several conversations contribute both speakers). The text that the language
models were shown is pinned by digest:

```
file    artifacts/cejc/monologues_cejc_home2_hq1_v1.parquet
rows    120
sha256  01b86e7d52bf0e1e76e7cc6c28fca7e210ce56decb6b64db4340eb6f00ae9f72
```

Check a rebuild against it with:

```bash
shasum -a 256 artifacts/cejc/monologues_cejc_home2_hq1_v1.parquet
```

A mismatch means the corpus export, the sample filter, or the concatenation
order differs from the reported run. In that case the numbers will not reproduce
exactly, and the difference should be resolved before comparing results.

Note that `speaker_id` in the corpus is a within-conversation label (`IC01`,
`IC02`, ...), so it is not a person identifier. Person identity comes from the
speaker metadata table (`cejc_person_id`), which is what the subject-wise
GroupKFold split groups on. 74 unique speakers produce the 120 records; 25
speakers appear in more than one conversation.

## Language-model access

Trait scoring runs against Amazon Bedrock in `ap-northeast-1`. Transcript text
is sent to Bedrock model endpoints and to no other service. Reproducing the
scoring step requires your own AWS credentials and incurs charges; the
deterministic parts of the pipeline do not.

Models used, with the identifiers passed to Bedrock:

| Label in the results | Bedrock model id |
|---|---|
| Sonnet 4 | `global.anthropic.claude-sonnet-4-20250514-v1:0` |
| Qwen3-235B | `qwen.qwen3-235b-a22b-2507-v1:0` |
| DeepSeek-V3 | `deepseek.v3-v1:0` |
| GPT-OSS-120B | `openai.gpt-oss-120b-1:0` |

Model endpoints change over time and are not version-frozen by the provider, so
a rerun may not return identical scores even with identical inputs. This is a
limitation of the design, not something the repository can pin. The scored
outputs are therefore treated as the fixed record of what the models returned.

## Ethics

The corpus is publicly released for secondary research use under NINJAL's terms;
no new data were collected from participants for this study. The analysis
estimates conversational and personality-descriptive measures, not clinical
status, and the outputs are not diagnostic.
