#!/usr/bin/env python3
"""Unit tests for extract_interaction_features_min.extract_features() and PG_pause_variability."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "analysis"))
from extract_interaction_features_min import extract_features


def _make_utterances(rows):
    """Helper: build utterances DataFrame from list of (conv_id, spk, start, end, text)."""
    return pd.DataFrame(rows, columns=["conversation_id", "speaker_id", "start_time", "end_time", "text"])


class TestExtractFeaturesFunction:
    """Test that extract_features() is callable and returns expected structure."""

    def test_returns_dataframe(self):
        utt = _make_utterances([
            ("c1", "A", 0.0, 1.0, "こんにちは"),
            ("c1", "B", 1.5, 2.5, "はい"),
        ])
        result = extract_features(utt, None, gap_tol=0.05)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_has_pause_variability_column(self):
        utt = _make_utterances([
            ("c1", "A", 0.0, 1.0, "こんにちは"),
            ("c1", "B", 1.5, 2.5, "はい"),
            ("c1", "A", 3.0, 4.0, "元気ですか"),
            ("c1", "B", 4.5, 5.5, "うん"),
        ])
        result = extract_features(utt, None, gap_tol=0.05)
        assert "PG_pause_variability" in result.columns

    def test_has_topic_drift_column(self):
        """IX_topic_drift_mean must still be present in output."""
        utt = _make_utterances([
            ("c1", "A", 0.0, 1.0, "こんにちは"),
            ("c1", "B", 1.5, 2.5, "はい"),
        ])
        result = extract_features(utt, None, gap_tol=0.05)
        assert "IX_topic_drift_mean" in result.columns

    def test_target_pairs_filtering(self):
        utt = _make_utterances([
            ("c1", "A", 0.0, 1.0, "こんにちは"),
            ("c1", "B", 1.5, 2.5, "はい"),
            ("c2", "C", 0.0, 1.0, "おはよう"),
            ("c2", "D", 1.5, 2.5, "うん"),
        ])
        target = {("c1", "B")}
        result = extract_features(utt, target, gap_tol=0.05)
        assert len(result) == 1
        assert result.iloc[0]["conversation_id"] == "c1"
        assert result.iloc[0]["speaker_id"] == "B"

    def test_empty_input(self):
        utt = _make_utterances([])
        result = extract_features(utt, None, gap_tol=0.05)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


class TestPGPauseVariability:
    """Test PG_pause_variability (CV = std/mean) calculation."""

    def test_constant_pauses_cv_zero(self):
        """If all pauses are identical, CV should be 0."""
        # Speaker A has 3 utterances with equal gaps (1.0s each)
        utt = _make_utterances([
            ("c1", "A", 0.0, 1.0, "あ"),
            ("c1", "B", 1.0, 1.5, "うん"),
            ("c1", "A", 2.0, 3.0, "い"),
            ("c1", "B", 3.0, 3.5, "うん"),
            ("c1", "A", 4.0, 5.0, "う"),
            ("c1", "B", 5.0, 5.5, "うん"),
        ])
        result = extract_features(utt, None, gap_tol=0.05)
        row_a = result[result["speaker_id"] == "A"].iloc[0]
        # A's intra-speaker pauses: start[1]-end[0]=2.0-1.0=1.0, start[2]-end[1]=4.0-3.0=1.0
        # Both are 1.0, so CV = 0
        assert row_a["PG_pause_variability"] == pytest.approx(0.0, abs=1e-9)

    def test_varied_pauses_cv_positive(self):
        """If pauses vary, CV should be positive."""
        utt = _make_utterances([
            ("c1", "A", 0.0, 1.0, "あ"),
            ("c1", "B", 1.0, 1.5, "うん"),
            ("c1", "A", 2.0, 3.0, "い"),  # pause from A: 2.0-1.0=1.0
            ("c1", "B", 3.0, 3.5, "うん"),
            ("c1", "A", 6.0, 7.0, "う"),  # pause from A: 6.0-3.0=3.0
            ("c1", "B", 7.0, 7.5, "うん"),
        ])
        result = extract_features(utt, None, gap_tol=0.05)
        row_a = result[result["speaker_id"] == "A"].iloc[0]
        # pauses = [1.0, 3.0], mean=2.0, std(ddof=1)=sqrt(2)≈1.414
        # CV = 1.414/2.0 ≈ 0.707
        expected_cv = float(np.std([1.0, 3.0], ddof=1) / np.mean([1.0, 3.0]))
        assert row_a["PG_pause_variability"] == pytest.approx(expected_cv, rel=1e-6)

    def test_single_pause_nan(self):
        """With fewer than 2 pauses, CV should be NaN."""
        utt = _make_utterances([
            ("c1", "A", 0.0, 1.0, "あ"),
            ("c1", "B", 1.5, 2.5, "うん"),
            ("c1", "A", 3.0, 4.0, "い"),  # only 1 intra-speaker pause for A
            ("c1", "B", 4.5, 5.5, "うん"),
        ])
        result = extract_features(utt, None, gap_tol=0.05)
        row_a = result[result["speaker_id"] == "A"].iloc[0]
        assert math.isnan(row_a["PG_pause_variability"])

    def test_no_utterances_nan(self):
        """Speaker with no solo utterances should get NaN."""
        utt = _make_utterances([
            ("c1", "A", 0.0, 1.0, "あ"),
            ("c1", "B", 1.5, 2.5, "うん"),
        ])
        result = extract_features(utt, None, gap_tol=0.05)
        # B has only 1 utterance, so no intra-speaker pauses
        row_b = result[result["speaker_id"] == "B"].iloc[0]
        assert math.isnan(row_b["PG_pause_variability"])
