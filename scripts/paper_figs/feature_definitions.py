"""19 interaction feature definitions + EXCL3 control variables.

Derived from ``scripts/analysis/extract_interaction_features_min.py``.
Each entry documents name, category, summary, algorithm, and whether
the variable is a control (EXCL3) or an explanatory variable.

Usage::

    from scripts.paper_figs.feature_definitions import (
        FEATURE_DEFINITIONS,
        get_explanatory_features,
        get_classical_features,
        get_novel_features,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class FeatureDefinition:
    """Single feature definition entry."""

    name: str
    category: str  # PG / FILL / IX / RESP / CTRL
    summary: str
    algorithm: str
    is_control: bool
    classification: str  # "Classical" / "Novel" / "Control"


# ------------------------------------------------------------------
# 19 explanatory features (is_control=False)
# Classical: 10 (PG 8 + FILL 2), Novel: 9 (IX 5 + RESP 3 + PG_pause_variability)
# ------------------------------------------------------------------
_EXPLANATORY: List[FeatureDefinition] = [
    FeatureDefinition(
        name="PG_speech_ratio",
        category="PG",
        summary="Speech ratio",
        algorithm=(
            "Speaker's total speech time / total conversation time. "
            "missing if total_time is 0 or missing."
        ),
        is_control=False,
        classification="Classical",
    ),
    FeatureDefinition(
        name="PG_pause_mean",
        category="PG",
        summary="Mean pause duration",
        algorithm=(
            "Mean of intra-speaker consecutive utterance gaps "
            "(>=gap_tol sec). missing if no qualifying gaps."
        ),
        is_control=False,
        classification="Classical",
    ),
    FeatureDefinition(
        name="PG_pause_p50",
        category="PG",
        summary="Median pause",
        algorithm=(
            "50th percentile of intra-speaker gaps. "
            "missing if no qualifying gaps."
        ),
        is_control=False,
        classification="Classical",
    ),
    FeatureDefinition(
        name="PG_pause_p90",
        category="PG",
        summary="90th percentile pause",
        algorithm=(
            "90th percentile of intra-speaker gaps. "
            "missing if no qualifying gaps."
        ),
        is_control=False,
        classification="Classical",
    ),
    FeatureDefinition(
        name="PG_resp_gap_mean",
        category="PG",
        summary="Mean response gap",
        algorithm=(
            "Mean of turn-taking gaps (prev_end -> resp_start, "
            ">=gap_tol sec). missing if no qualifying gaps."
        ),
        is_control=False,
        classification="Classical",
    ),
    FeatureDefinition(
        name="PG_resp_gap_p50",
        category="PG",
        summary="Median response gap",
        algorithm=(
            "50th percentile of turn-taking gaps. "
            "missing if no qualifying gaps."
        ),
        is_control=False,
        classification="Classical",
    ),
    FeatureDefinition(
        name="PG_resp_gap_p90",
        category="PG",
        summary="90th percentile response gap",
        algorithm=(
            "90th percentile of turn-taking gaps. "
            "missing if no qualifying gaps."
        ),
        is_control=False,
        classification="Classical",
    ),
    FeatureDefinition(
        name="FILL_has_any",
        category="FILL",
        summary="Filler utterance rate",
        algorithm=(
            "Proportion of speaker's utterances containing >=1 filler "
            "(etto/ee/ano). missing if speaker has no utterances."
        ),
        is_control=False,
        classification="Classical",
    ),
    FeatureDefinition(
        name="FILL_rate_per_100chars",
        category="FILL",
        summary="Filler rate per 100 chars",
        algorithm=(
            "Total filler count / (text character count / 100). "
            "missing if text_len is 0."
        ),
        is_control=False,
        classification="Classical",
    ),
    FeatureDefinition(
        name="PG_overlap_rate",
        category="PG",
        summary="Overlap rate",
        algorithm="Proportion of turn-taking gaps < -gap_tol (overlaps).",
        is_control=False,
        classification="Classical",
    ),
    FeatureDefinition(
        name="IX_oirmarker_rate",
        category="IX",
        summary="OIR marker rate",
        algorithm=(
            "Proportion of responses starting with OIR markers "
            "(e?/eQ/nani? etc). Computed over all adjacent pairs "
            "where speaker is responder."
        ),
        is_control=False,
        classification="Novel",
    ),
    FeatureDefinition(
        name="IX_oirmarker_after_question_rate",
        category="IX",
        summary="Post-question OIR rate",
        algorithm=(
            "OIR marker rate when previous utterance is a question. "
            "missing if no question-preceded pairs."
        ),
        is_control=False,
        classification="Novel",
    ),
    FeatureDefinition(
        name="IX_yesno_rate",
        category="IX",
        summary="Yes/No response rate",
        algorithm=(
            "Proportion of responses starting with yes/no prefixes "
            "(hai/un/ee/iie etc)."
        ),
        is_control=False,
        classification="Novel",
    ),
    FeatureDefinition(
        name="IX_yesno_after_question_rate",
        category="IX",
        summary="Post-question Yes/No rate",
        algorithm=(
            "Yes/No rate when previous utterance is a question. "
            "missing if no question-preceded pairs."
        ),
        is_control=False,
        classification="Novel",
    ),
    FeatureDefinition(
        name="IX_lex_overlap_mean",
        category="IX",
        summary="Lexical overlap",
        algorithm=(
            "Mean character-bigram Jaccard coefficient between "
            "previous utterance and response."
        ),
        is_control=False,
        classification="Novel",
    ),
    FeatureDefinition(
        name="RESP_NE_AIZUCHI_RATE",
        category="RESP",
        summary="Post-NE aizuchi rate",
        algorithm=(
            "Proportion of responses that start with aizuchi prefixes "
            "when previous utterance ends with NE particle. "
            "missing if n_pairs_after_NE is 0."
        ),
        is_control=False,
        classification="Novel",
    ),
    FeatureDefinition(
        name="RESP_NE_ENTROPY",
        category="RESP",
        summary="Post-NE response entropy",
        algorithm=(
            "Shannon entropy of response-initial tokens after NE "
            "sentence-final particle. missing if n_pairs_after_NE is 0."
        ),
        is_control=False,
        classification="Novel",
    ),
    FeatureDefinition(
        name="RESP_YO_ENTROPY",
        category="RESP",
        summary="Post-YO response entropy",
        algorithm=(
            "Shannon entropy of response-initial tokens after YO "
            "sentence-final particle. missing if n_pairs_after_YO is 0."
        ),
        is_control=False,
        classification="Novel",
    ),
    FeatureDefinition(
        name="PG_pause_variability",
        category="PG",
        summary="Pause duration CV",
        algorithm=(
            "Coefficient of variation (std / mean) of intra-speaker "
            "pause durations. missing if fewer than 2 pauses or mean is 0."
        ),
        is_control=False,
        classification="Novel",
    ),
]

# ------------------------------------------------------------------
# EXCL3 control variables (is_control=True)
# 13 controls: IX_topic_drift_mean (collinear) + 12 original controls
# (PG_overlap_rate moved to explanatory)
# ------------------------------------------------------------------
_CONTROLS: List[FeatureDefinition] = [
    FeatureDefinition(
        name="IX_topic_drift_mean",
        category="IX",
        summary="Topic drift",
        algorithm=(
            "1 - IX_lex_overlap_mean. Collinear with "
            "IX_lex_overlap_mean by construction."
        ),
        is_control=True,
        classification="Control",
    ),
    FeatureDefinition(
        name="n_pairs_total",
        category="CTRL",
        summary="Total adjacent pairs",
        algorithm="Count of all adjacent speaker-switch pairs.",
        is_control=True,
        classification="Control",
    ),
    FeatureDefinition(
        name="n_pairs_after_NE",
        category="CTRL",
        summary="Pairs after NE particle",
        algorithm="Count of pairs where previous utterance ends with NE.",
        is_control=True,
        classification="Control",
    ),
    FeatureDefinition(
        name="n_pairs_after_YO",
        category="CTRL",
        summary="Pairs after YO particle",
        algorithm="Count of pairs where previous utterance ends with YO.",
        is_control=True,
        classification="Control",
    ),
    FeatureDefinition(
        name="IX_n_pairs",
        category="CTRL",
        summary="IX pair count",
        algorithm="Total adjacent pairs (same as n_pairs_total).",
        is_control=True,
        classification="Control",
    ),
    FeatureDefinition(
        name="IX_n_pairs_after_question",
        category="CTRL",
        summary="Pairs after question",
        algorithm="Count of pairs where previous utterance is a question.",
        is_control=True,
        classification="Control",
    ),
    FeatureDefinition(
        name="PG_total_time",
        category="CTRL",
        summary="Total conversation time",
        algorithm="end_time.max() - start_time.min() for the conversation.",
        is_control=True,
        classification="Control",
    ),
    FeatureDefinition(
        name="PG_resp_overlap_rate",
        category="CTRL",
        summary="Response overlap rate",
        algorithm="Same as PG_overlap_rate in current implementation.",
        is_control=True,
        classification="Control",
    ),
    FeatureDefinition(
        name="FILL_text_len",
        category="CTRL",
        summary="Text character count",
        algorithm="Total whitespace-stripped character count of speaker's utterances.",
        is_control=True,
        classification="Control",
    ),
    FeatureDefinition(
        name="FILL_cnt_total",
        category="CTRL",
        summary="Total filler count",
        algorithm="Sum of etto + ee + ano filler occurrences.",
        is_control=True,
        classification="Control",
    ),
    FeatureDefinition(
        name="FILL_cnt_eto",
        category="CTRL",
        summary="Etto filler count",
        algorithm="Count of etto/eto fillers in speaker's text.",
        is_control=True,
        classification="Control",
    ),
    FeatureDefinition(
        name="FILL_cnt_e",
        category="CTRL",
        summary="Ee filler count",
        algorithm="Count of ee/e~ fillers in speaker's text.",
        is_control=True,
        classification="Control",
    ),
    FeatureDefinition(
        name="FILL_cnt_ano",
        category="CTRL",
        summary="Ano filler count",
        algorithm="Count of ano fillers in speaker's text.",
        is_control=True,
        classification="Control",
    ),
]

# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

FEATURE_DEFINITIONS: List[FeatureDefinition] = _EXPLANATORY + _CONTROLS
"""All feature definitions (19 explanatory + 13 controls)."""


def get_explanatory_features() -> List[FeatureDefinition]:
    """Return only the 19 explanatory features (is_control=False)."""
    return [f for f in FEATURE_DEFINITIONS if not f.is_control]


def get_classical_features() -> List[FeatureDefinition]:
    """Return Classical features (PG + FILL, 10 features)."""
    return [f for f in FEATURE_DEFINITIONS if f.classification == "Classical"]


def get_novel_features() -> List[FeatureDefinition]:
    """Return Novel features (IX + RESP + PG_pause_variability, 9 features)."""
    return [f for f in FEATURE_DEFINITIONS if f.classification == "Novel"]
