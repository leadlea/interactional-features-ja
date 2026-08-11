# Language-model scoring protocol

The outcome variable is not a measurement. It is what four language models
answered when asked to fill in a personality inventory on behalf of a speaker,
having read only that speaker's transcript. This document records the protocol
precisely enough to judge and repeat it.

Implementation: [`scripts/big5/score_big5_bedrock.py`](../scripts/big5/score_big5_bedrock.py)

## Instrument

IPIP-NEO-120: five traits x 24 items, each answered on a five-point scale from
"Very Inaccurate" to "Very Accurate". 55 items are reverse-keyed and inverted as
`4 - score` before aggregation.

Items are supplied as a CSV (`artifacts/big5/items_ipipneo120_ja.csv`) and split
into per-trait files of 24 items by
[`scripts/big5/subset_items_by_trait.py`](../scripts/big5/subset_items_by_trait.py).
Each request covers one item for one speaker, so a full pass is
120 items x 120 speakers x 4 models.

Traits are scored one at a time rather than all 120 items in one request. That
keeps each response inside the output token budget and makes a failed trait cheap
to redo, at the cost of the model not seeing the other traits' items in the same
context. The runner is
[`scripts/big5/run_scoring_sharded.sh`](../scripts/big5/run_scoring_sharded.sh),
which iterates trait x model x shard and skips work that already has output.

## Models

Four models from different families, on Amazon Bedrock in `ap-northeast-1`:

| Label | Bedrock model id |
|---|---|
| Sonnet 4 | `global.anthropic.claude-sonnet-4-20250514-v1:0` |
| Qwen3-235B | `qwen.qwen3-235b-a22b-2507-v1:0` |
| DeepSeek-V3 | `deepseek.v3-v1:0` (671B total parameters) |
| GPT-OSS-120B | `openai.gpt-oss-120b-1:0` |

Four models rather than one, because a single model's estimate has no way to
distinguish "this trait is recoverable from the transcript" from "this model
happens to answer this way". Agreement between independent models is the check.

Inference parameters, identical across models:

```
temperature  0.0
top_p        1.0
max_tokens   128
seed         0        (where the model accepts one)
```

`temperature = 0.0` is for determinism, not quality: the same input should give
the same answer within a single model version.

## Prompt

The model is asked to answer as the person who produced the text. English is
used for the response options even when the instruction text is Japanese
(`--prompt_lang ja`), because free-form Japanese answers vary in orthography and
are unreliable to map onto five fixed categories.

```
Your task is to respond to the following IPIP-NEO-120 question based on the
participant's daily diaries of the most significant event that occurred during
the day, provided below. Respond as though you are the individual who generated
these thoughts, reflecting their personality traits.

Base your answer on inferred personality traits. Think carefully about what the
thoughts imply about tendencies and behaviors.

For each question, select the most appropriate option:
- Very Inaccurate: ...
- Moderately Inaccurate: ...
- Neither Accurate Nor Inaccurate: ...
- Moderately Accurate: ...
- Very Accurate: ...

IPIP-NEO-120 question to answer:
{statement}

Participant's daily diaries:
{transcript}

Your response must be exactly one of:
Very Inaccurate
Moderately Inaccurate
Neither Accurate Nor Inaccurate
Moderately Accurate
Very Accurate

Do not include any explanation, punctuation, or additional text.
Return only the exact phrase from the list above.
```

`{statement}` is one inventory item; `{transcript}` is the speaker's
pseudo-monologue. The full text of both prompt variants is in the scoring script
and reproduced in the manuscript's supplementary material.

## Handling invalid responses

A response that does not normalise to one of the five options is not silently
coerced:

- exact match first, then case-insensitive substring match
- if neither applies, the request is retried, up to 5 attempts under
  `--paper_strict`, with `REMINDER: Return only ONE exact phrase from the list.`
  appended
- if every attempt fails, the item is recorded as missing (`--on_fail nan`,
  default) or as the scale midpoint (`--on_fail neutral`)
- every attempt, including the failures, is appended to `attempts.jsonl`

The default is missing rather than midpoint because filling in a neutral value
would pull the trait score toward the centre and quietly reduce variance.

## Aggregation

Item-level scores are averaged across the four models to form the ensemble
outcome, which is what the main analysis uses. Averaging at the item level rather
than at the trait level keeps the internal-consistency calculation meaningful:
Cronbach's alpha is computed on the ensemble item responses and written to
`cronbach_alpha.csv`.

## Between-model agreement

Mean off-diagonal Pearson r across the four models, N = 120, computed from
`reports/model_agreement/teacher_corr_{trait}.tsv`:

| Trait | mean r |
|---|---|
| C | 0.699 |
| E | 0.640 |
| N | 0.603 |
| O | 0.559 |
| A | 0.435 |

Read this as a ceiling on how much of the outcome is a property of the
transcript rather than of the model. C is where the four models most agree, which
is why it is the trait the manuscript reports in the main text. A is where they
agree least, so results for A are treated as exploratory regardless of their
p-value.

High agreement is not accuracy. Four large language models can share training
data and inductive biases, so a shared error would look like agreement. That is a
stated limitation, and the two validation experiments below are the partial
answer to it.

## Validation experiments

**Three conditions.** The same scoring protocol is run on three inputs:

| Condition | Input | What it distinguishes |
|---|---|---|
| 1 | the speaker's real transcript | the reported analysis |
| 2 | summary statistics only, no transcript text | whether surface quantities (length, filler counts) suffice |
| 3 | another speaker's transcript, assigned by derangement | whether any transcript produces the same result |

Condition 3 is the important one: if the association survived when transcripts
are shuffled between speakers, it would not be about the speakers at all. The
derangement guarantees no record keeps its own text.
Scripts: [`scripts/baseline/`](../scripts/baseline/).

**Feature dose-response.** One feature is manipulated in the text at x0 / x1 / x3
and the text is re-scored, with a second feature left untouched as a control.
This asks whether the trait estimate tracks the feature causally, or whether the
two merely covary in natural speech.
Scripts: [`scripts/dose_response/`](../scripts/dose_response/).
`make dose-verify` confirms the manipulation moved the intended feature and left
the control alone; run it before scoring.

## What this outcome is and is not

The trait scores are used as an external criterion for validating the interaction
features, not as measurements of personality. No self-report or observer-rated
personality data exist for these speakers, so the models' estimates cannot be
checked against a ground truth here. Comparison with human ratings is the
obvious next step and has not been done.
