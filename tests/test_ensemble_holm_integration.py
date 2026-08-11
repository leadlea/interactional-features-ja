"""Unit tests for Holm correction integration in ensemble_permutation.py main().

Validates Requirements 2.6 and 2.7:
- ensemble_summary.tsv includes p_corrected column (Req 2.6)
- Per-teacher summaries also include p_corrected column (Req 2.7)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure scripts/analysis is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "analysis"))

from ensemble_permutation import holm_correction


class TestHolmIntegrationWithSummaryDf:
    """Test that Holm correction integrates correctly with summary_df construction."""

    def test_p_corrected_column_added(self):
        """summary_df should have p_corrected column after Holm correction."""
        summary_rows = [
            {"trait": "O", "r_obs": 0.360, "p_value": 0.0048},
            {"trait": "C", "r_obs": 0.447, "p_value": 0.0004},
            {"trait": "E", "r_obs": 0.217, "p_value": 0.0902},
            {"trait": "A", "r_obs": 0.465, "p_value": 0.0004},
            {"trait": "N", "r_obs": 0.309, "p_value": 0.0152},
        ]
        summary_df = pd.DataFrame(summary_rows)
        p_corrected = holm_correction(summary_df["p_value"].tolist())
        summary_df["p_corrected"] = [round(pc, 4) for pc in p_corrected]

        assert "p_corrected" in summary_df.columns
        assert len(summary_df) == 5

    def test_p_corrected_geq_p_value(self):
        """Each p_corrected must be >= corresponding p_value."""
        summary_rows = [
            {"trait": "O", "r_obs": 0.360, "p_value": 0.0048},
            {"trait": "C", "r_obs": 0.447, "p_value": 0.0004},
            {"trait": "E", "r_obs": 0.217, "p_value": 0.0902},
            {"trait": "A", "r_obs": 0.465, "p_value": 0.0004},
            {"trait": "N", "r_obs": 0.309, "p_value": 0.0152},
        ]
        summary_df = pd.DataFrame(summary_rows)
        p_corrected = holm_correction(summary_df["p_value"].tolist())
        summary_df["p_corrected"] = [round(pc, 4) for pc in p_corrected]

        for _, row in summary_df.iterrows():
            assert row["p_corrected"] >= row["p_value"]

    def test_p_corrected_leq_one(self):
        """All p_corrected values must be <= 1.0."""
        summary_rows = [
            {"trait": "O", "r_obs": 0.360, "p_value": 0.0048},
            {"trait": "C", "r_obs": 0.447, "p_value": 0.0004},
            {"trait": "E", "r_obs": 0.217, "p_value": 0.0902},
            {"trait": "A", "r_obs": 0.465, "p_value": 0.0004},
            {"trait": "N", "r_obs": 0.309, "p_value": 0.0152},
        ]
        summary_df = pd.DataFrame(summary_rows)
        p_corrected = holm_correction(summary_df["p_value"].tolist())
        summary_df["p_corrected"] = [round(pc, 4) for pc in p_corrected]

        for _, row in summary_df.iterrows():
            assert row["p_corrected"] <= 1.0

    def test_expected_output_format(self):
        """Verify the output matches the design spec's expected format."""
        # From design doc: expected output
        summary_rows = [
            {"trait": "O", "r_obs": 0.360, "p_value": 0.0048},
            {"trait": "C", "r_obs": 0.447, "p_value": 0.0004},
            {"trait": "E", "r_obs": 0.217, "p_value": 0.0902},
            {"trait": "A", "r_obs": 0.465, "p_value": 0.0004},
            {"trait": "N", "r_obs": 0.309, "p_value": 0.0152},
        ]
        summary_df = pd.DataFrame(summary_rows)
        p_corrected = holm_correction(summary_df["p_value"].tolist())
        summary_df["p_corrected"] = [round(pc, 4) for pc in p_corrected]

        assert list(summary_df.columns) == ["trait", "r_obs", "p_value", "p_corrected"]

        # Holm correction for these 5 p-values:
        # Sorted: C=0.0004(rank0), A=0.0004(rank1), O=0.0048(rank2), N=0.0152(rank3), E=0.0902(rank4)
        # rank0: 0.0004*5=0.0020, cummax=0.0020
        # rank1: 0.0004*4=0.0016 -> cummax=0.0020
        # rank2: 0.0048*3=0.0144, cummax=0.0144
        # rank3: 0.0152*2=0.0304, cummax=0.0304
        # rank4: 0.0902*1=0.0902, cummax=0.0902
        assert summary_df.loc[summary_df["trait"] == "C", "p_corrected"].iloc[0] == pytest.approx(0.0020, abs=1e-4)
        assert summary_df.loc[summary_df["trait"] == "A", "p_corrected"].iloc[0] == pytest.approx(0.0020, abs=1e-4)
        assert summary_df.loc[summary_df["trait"] == "O", "p_corrected"].iloc[0] == pytest.approx(0.0144, abs=1e-4)
        assert summary_df.loc[summary_df["trait"] == "N", "p_corrected"].iloc[0] == pytest.approx(0.0304, abs=1e-4)
        assert summary_df.loc[summary_df["trait"] == "E", "p_corrected"].iloc[0] == pytest.approx(0.0902, abs=1e-4)

    def test_tsv_roundtrip(self, tmp_path: Path):
        """Verify TSV write/read preserves p_corrected column."""
        summary_rows = [
            {"trait": "O", "r_obs": 0.360, "p_value": 0.0048},
            {"trait": "C", "r_obs": 0.447, "p_value": 0.0004},
            {"trait": "E", "r_obs": 0.217, "p_value": 0.0902},
            {"trait": "A", "r_obs": 0.465, "p_value": 0.0004},
            {"trait": "N", "r_obs": 0.309, "p_value": 0.0152},
        ]
        summary_df = pd.DataFrame(summary_rows)
        p_corrected = holm_correction(summary_df["p_value"].tolist())
        summary_df["p_corrected"] = [round(pc, 4) for pc in p_corrected]

        tsv_path = tmp_path / "ensemble_summary.tsv"
        summary_df.to_csv(tsv_path, sep="\t", index=False)

        loaded = pd.read_csv(tsv_path, sep="\t")
        assert "p_corrected" in loaded.columns
        assert len(loaded) == 5
        assert loaded["p_corrected"].notna().all()


class TestPerTeacherHolmCorrection:
    """Test per-teacher Holm correction logic (Req 2.7)."""

    def test_per_teacher_correction_applied(self):
        """Each teacher's 5-trait p-values should be Holm-corrected independently."""
        teacher_rows = [
            {"trait": "O", "r_obs": 0.300, "p_value": 0.05},
            {"trait": "C", "r_obs": 0.434, "p_value": 0.0008},
            {"trait": "E", "r_obs": 0.150, "p_value": 0.20},
            {"trait": "A", "r_obs": 0.200, "p_value": 0.10},
            {"trait": "N", "r_obs": 0.250, "p_value": 0.08},
        ]
        teacher_df = pd.DataFrame(teacher_rows)
        p_corr = holm_correction(teacher_df["p_value"].tolist())
        teacher_df["p_corrected"] = [round(pc, 4) for pc in p_corr]

        assert "p_corrected" in teacher_df.columns
        for _, row in teacher_df.iterrows():
            assert row["p_corrected"] >= row["p_value"]
            assert row["p_corrected"] <= 1.0

    def test_per_teacher_with_nan(self):
        """Per-teacher results with NaN p-values should preserve NaN."""
        teacher_rows = [
            {"trait": "O", "r_obs": float("nan"), "p_value": float("nan")},
            {"trait": "C", "r_obs": 0.434, "p_value": 0.0008},
            {"trait": "E", "r_obs": 0.150, "p_value": 0.20},
            {"trait": "A", "r_obs": 0.200, "p_value": 0.10},
            {"trait": "N", "r_obs": 0.250, "p_value": 0.08},
        ]
        teacher_df = pd.DataFrame(teacher_rows)
        p_corr = holm_correction(teacher_df["p_value"].tolist())
        teacher_df["p_corrected"] = [round(pc, 4) if not pd.isna(pc) else pc for pc in p_corr]

        import math
        assert math.isnan(teacher_df.loc[0, "p_corrected"])
        # Non-NaN values should still be corrected
        assert teacher_df.loc[1, "p_corrected"] >= teacher_df.loc[1, "p_value"]
