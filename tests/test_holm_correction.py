"""Unit tests for holm_correction() in ensemble_permutation.py."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

# Ensure scripts/analysis is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "analysis"))

from ensemble_permutation import holm_correction


class TestHolmCorrectionBasic:
    """Basic correctness tests."""

    def test_empty_list(self):
        assert holm_correction([]) == []

    def test_single_value(self):
        # Single p-value: corrected = p * 1 = p (no adjustment needed)
        assert holm_correction([0.05]) == [0.05]

    def test_single_value_one(self):
        assert holm_correction([1.0]) == [1.0]

    def test_single_value_zero(self):
        assert holm_correction([0.0]) == [0.0]

    def test_known_example(self):
        """Known Holm-Bonferroni example with 5 p-values."""
        p_values = [0.01, 0.04, 0.03, 0.5, 0.8]
        result = holm_correction(p_values)

        # Sorted: (0, 0.01), (2, 0.03), (1, 0.04), (3, 0.5), (4, 0.8)
        # rank=0: 0.01 * 5 = 0.05, cummax=0.05
        # rank=1: 0.03 * 4 = 0.12, cummax=0.12
        # rank=2: 0.04 * 3 = 0.12, cummax=0.12
        # rank=3: 0.5  * 2 = 1.0,  cummax=1.0
        # rank=4: 0.8  * 1 = 0.8,  cummax=1.0, clip=1.0
        assert result[0] == pytest.approx(0.05)
        assert result[2] == pytest.approx(0.12)
        assert result[1] == pytest.approx(0.12)
        assert result[3] == pytest.approx(1.0)
        assert result[4] == pytest.approx(1.0)

    def test_all_same_p(self):
        """All identical p-values."""
        result = holm_correction([0.05, 0.05, 0.05])
        # Sorted all same: rank0: 0.05*3=0.15, rank1: 0.05*2=0.10 -> cummax=0.15, rank2: 0.05*1=0.05 -> cummax=0.15
        for v in result:
            assert v == pytest.approx(0.15)

    def test_preserves_order(self):
        """Output order matches input order."""
        p_values = [0.5, 0.01, 0.1]
        result = holm_correction(p_values)
        assert len(result) == 3
        # Smallest p (0.01 at index 1) gets largest multiplier
        assert result[1] <= result[0]
        assert result[1] <= result[2]


class TestHolmCorrectionEdgeCases:
    """Edge case tests."""

    def test_value_error_negative(self):
        with pytest.raises(ValueError, match="outside \\[0, 1\\]"):
            holm_correction([-0.1, 0.5])

    def test_value_error_above_one(self):
        with pytest.raises(ValueError, match="outside \\[0, 1\\]"):
            holm_correction([0.5, 1.1])

    def test_nan_preserved(self):
        """NaN values are preserved in output."""
        result = holm_correction([0.01, float("nan"), 0.05])
        assert result[0] == pytest.approx(0.02)  # 0.01 * 2 (only 2 non-NaN)
        assert math.isnan(result[1])
        assert result[2] == pytest.approx(0.05)  # 0.05 * 1 = 0.05, cummax=max(0.02,0.05)=0.05

    def test_all_nan(self):
        """All NaN input returns all NaN."""
        result = holm_correction([float("nan"), float("nan")])
        assert all(math.isnan(v) for v in result)

    def test_clipped_at_one(self):
        """Corrected values are clipped at 1.0."""
        result = holm_correction([0.9, 0.8])
        # rank0: 0.8*2=1.6 -> clip=1.0, rank1: 0.9*1=0.9 -> cummax=1.0
        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(1.0)

    def test_monotonicity(self):
        """Corrected p-values >= original p-values."""
        p_values = [0.001, 0.01, 0.05, 0.1, 0.5]
        result = holm_correction(p_values)
        for orig, corr in zip(p_values, result):
            assert corr >= orig

    def test_bounded_zero_to_one(self):
        """All corrected values in [0, 1]."""
        p_values = [0.0, 0.001, 0.5, 1.0]
        result = holm_correction(p_values)
        for v in result:
            assert 0.0 <= v <= 1.0
