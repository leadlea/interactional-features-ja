# Feature dictionary — 19 interaction features

The definitive definitions live in code, not in this file:

- computation: [`scripts/analysis/extract_interaction_features_min.py`](../scripts/analysis/extract_interaction_features_min.py)
- metadata (category, classical/novel, prose description):
  [`scripts/paper_figs/feature_definitions.py`](../scripts/paper_figs/feature_definitions.py)

The manuscript's feature table (`tab_feature_definitions.tex`) is generated from
the same module, so the paper, this document, and the code cannot drift apart.

## Categories

| Prefix | Category | Count | What the category captures |
|---|---|---|---|
| `PG` | speech timing | 9 | who holds the floor, and the silences around turn transitions |
| `FILL` | fillers | 2 | hesitation markers (etto / ee / ano) |
| `IX` | sequence organisation | 5 | how a response relates to what preceded it |
| `RESP` | response types | 3 | what follows the sentence-final particles ne / yo |

## Classical vs Novel

`Classical` (10) are measures with established precedent in the conversation and
speech-timing literature. `Novel` (9) are the measures this study proposes.

The split is by variable, not by category: `PG` contains both classical
variables and one novel one (`PG_pause_variability`), `FILL` is entirely
classical, and `IX` and `RESP` are entirely novel. Category and novelty are not
in one-to-one correspondence.

## The 19 explanatory variables

| Feature | Category | Class | What it measures | How it is computed |
|---|---|---|---|---|
| `PG_speech_ratio` | PG | Classical | Speech ratio | Speaker's total speech time / total conversation time. missing if total_time is 0 or missing. |
| `PG_pause_mean` | PG | Classical | Mean pause duration | Mean of intra-speaker consecutive utterance gaps (>=gap_tol sec). missing if no qualifying gaps. |
| `PG_pause_p50` | PG | Classical | Median pause | 50th percentile of intra-speaker gaps. missing if no qualifying gaps. |
| `PG_pause_p90` | PG | Classical | 90th percentile pause | 90th percentile of intra-speaker gaps. missing if no qualifying gaps. |
| `PG_resp_gap_mean` | PG | Classical | Mean response gap | Mean of turn-taking gaps (prev_end -> resp_start, >=gap_tol sec). missing if no qualifying gaps. |
| `PG_resp_gap_p50` | PG | Classical | Median response gap | 50th percentile of turn-taking gaps. missing if no qualifying gaps. |
| `PG_resp_gap_p90` | PG | Classical | 90th percentile response gap | 90th percentile of turn-taking gaps. missing if no qualifying gaps. |
| `FILL_has_any` | FILL | Classical | Filler utterance rate | Proportion of speaker's utterances containing >=1 filler (etto/ee/ano). missing if speaker has no utterances. |
| `FILL_rate_per_100chars` | FILL | Classical | Filler rate per 100 chars | Total filler count / (text character count / 100). missing if text_len is 0. |
| `PG_overlap_rate` | PG | Classical | Overlap rate | Proportion of turn-taking gaps < -gap_tol (overlaps). |
| `IX_oirmarker_rate` | IX | Novel | OIR marker rate | Proportion of responses starting with OIR markers (e?/eQ/nani? etc). Computed over all adjacent pairs where speaker is responder. |
| `IX_oirmarker_after_question_rate` | IX | Novel | Post-question OIR rate | OIR marker rate when previous utterance is a question. missing if no question-preceded pairs. |
| `IX_yesno_rate` | IX | Novel | Yes/No response rate | Proportion of responses starting with yes/no prefixes (hai/un/ee/iie etc). |
| `IX_yesno_after_question_rate` | IX | Novel | Post-question Yes/No rate | Yes/No rate when previous utterance is a question. missing if no question-preceded pairs. |
| `IX_lex_overlap_mean` | IX | Novel | Lexical overlap | Mean character-bigram Jaccard coefficient between previous utterance and response. |
| `RESP_NE_AIZUCHI_RATE` | RESP | Novel | Post-NE aizuchi rate | Proportion of responses that start with aizuchi prefixes when previous utterance ends with NE particle. missing if n_pairs_after_NE is 0. |
| `RESP_NE_ENTROPY` | RESP | Novel | Post-NE response entropy | Shannon entropy of response-initial tokens after NE sentence-final particle. missing if n_pairs_after_NE is 0. |
| `RESP_YO_ENTROPY` | RESP | Novel | Post-YO response entropy | Shannon entropy of response-initial tokens after YO sentence-final particle. missing if n_pairs_after_YO is 0. |
| `PG_pause_variability` | PG | Novel | Pause duration CV | Coefficient of variation (std / mean) of intra-speaker pause durations. missing if fewer than 2 pauses or mean is 0. |

Notes on reading the table:

- "missing" means the variable is left as missing rather than filled with 0.
  A speaker with no qualifying pauses has an undefined mean pause, which is not
  the same as a pause of length zero. Missing values are imputed with the median
  inside the cross-validation pipeline, so imputation is fitted on training folds
  only.
- `gap_tol` is the tolerance for treating a measured gap as a real gap rather
  than annotation noise. Its default and the sensitivity of the results to it
  are covered in [evaluation-design.md](evaluation-design.md).
- A "question" is an utterance carrying an explicit question mark, or one whose
  punctuation-stripped tail ends in an interrogative form
  (か / かな / かね / でしょう / だろう / の). The same rule is used when selecting
  the analysis sample, so the two never disagree.

## Control variables (computed, not used as predictors)

These are written into the feature table for provenance and for the sample
filter, but excluded from the design matrix. They are raw counts and exposure
measures: including them would let the model predict the outcome from how much
the speaker talked rather than from how they talked.

| Variable | What it holds | Why it is excluded from the design matrix |
|---|---|---|
| `IX_topic_drift_mean` | Topic drift | 1 - IX_lex_overlap_mean. Collinear with IX_lex_overlap_mean by construction. |
| `n_pairs_total` | Total adjacent pairs | Count of all adjacent speaker-switch pairs. |
| `n_pairs_after_NE` | Pairs after NE particle | Count of pairs where previous utterance ends with NE. |
| `n_pairs_after_YO` | Pairs after YO particle | Count of pairs where previous utterance ends with YO. |
| `IX_n_pairs` | IX pair count | Total adjacent pairs (same as n_pairs_total). |
| `IX_n_pairs_after_question` | Pairs after question | Count of pairs where previous utterance is a question. |
| `PG_total_time` | Total conversation time | end_time.max() - start_time.min() for the conversation. |
| `PG_resp_overlap_rate` | Response overlap rate | Same as PG_overlap_rate in current implementation. |
| `FILL_text_len` | Text character count | Total whitespace-stripped character count of speaker's utterances. |
| `FILL_cnt_total` | Total filler count | Sum of etto + ee + ano filler occurrences. |
| `FILL_cnt_eto` | Etto filler count | Count of etto/eto fillers in speaker's text. |
| `FILL_cnt_e` | Ee filler count | Count of ee/e~ fillers in speaker's text. |
| `FILL_cnt_ano` | Ano filler count | Count of ano fillers in speaker's text. |

`IX_topic_drift_mean` is a special case. It is defined as
`1 - IX_lex_overlap_mean`, so it is perfectly collinear with a variable that is
already in the design matrix (r = -1.00). It is retained as a control for
transparency and excluded from the predictors.

The exact exclusion list used in the reported analyses is the `EXCLUDE_COLS`
variable in the [Makefile](../Makefile), which is passed to the analysis scripts
verbatim.
