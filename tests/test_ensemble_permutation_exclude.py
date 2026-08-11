"""Unit tests for ensemble_permutation.py DEFAULT_EXCLUDE logic.

Validates Requirements 1.3 and 5.3:
- IX_topic_drift_mean is excluded from features (Req 1.3)
- PG_overlap_rate is NOT excluded — used as explanatory variable (Req 5.3)
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure scripts/analysis is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "analysis"))

from ensemble_permutation import DEFAULT_EXCLUDE


def test_ix_topic_drift_mean_in_default_exclude():
    """IX_topic_drift_mean must be excluded (collinear with IX_lex_overlap_mean)."""
    assert "IX_topic_drift_mean" in DEFAULT_EXCLUDE


def test_pg_overlap_rate_not_in_default_exclude():
    """PG_overlap_rate must NOT be excluded — it is a Classical explanatory variable."""
    assert "PG_overlap_rate" not in DEFAULT_EXCLUDE


def test_id_columns_in_default_exclude():
    """conversation_id and speaker_id must always be excluded."""
    assert "conversation_id" in DEFAULT_EXCLUDE
    assert "speaker_id" in DEFAULT_EXCLUDE


def test_control_columns_in_default_exclude():
    """All 12 original control columns must be in DEFAULT_EXCLUDE."""
    expected_controls = {
        "n_pairs_total",
        "n_pairs_after_NE",
        "n_pairs_after_YO",
        "IX_n_pairs",
        "IX_n_pairs_after_question",
        "PG_total_time",
        "PG_resp_overlap_rate",
        "FILL_text_len",
        "FILL_cnt_total",
        "FILL_cnt_eto",
        "FILL_cnt_e",
        "FILL_cnt_ano",
    }
    assert expected_controls.issubset(DEFAULT_EXCLUDE)


def test_cli_exclude_cols_merge():
    """CLI --exclude_cols should be additive to DEFAULT_EXCLUDE."""
    # Simulate the merge logic from main()
    cli_input = "custom_col_1, custom_col_2"
    excl = set(DEFAULT_EXCLUDE)
    excl |= set(c.strip() for c in cli_input.split(",") if c.strip())

    # Original defaults still present
    assert "IX_topic_drift_mean" in excl
    assert "conversation_id" in excl
    # CLI extras added
    assert "custom_col_1" in excl
    assert "custom_col_2" in excl
    # PG_overlap_rate still not excluded
    assert "PG_overlap_rate" not in excl
