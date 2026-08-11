#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for verify_numerical_consistency.py."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from scripts.analysis.verify_numerical_consistency import (
    ALL_INTERACTION_FEATURES,
    verify_consistency,
    write_report,
)


# ── Helpers ──────────────────────────────────────────────────────────
def _write_three_stage_tsv(path: Path, trait: str = "C", teacher: str = "sonnet"):
    """Write a minimal three_stage result TSV."""
    lines = [
        "trait\tteacher\tstage\tn_features\tfeature_set\tr_obs\tp_value\tdelta_r_from_prev",
        f"{trait}\t{teacher}\t1\t2\tdemographics\t0.392\t0.0008\t",
        f"{trait}\t{teacher}\t2\t12\tdemographics+classical\t0.3914\t0.0016\t-0.0006",
        f"{trait}\t{teacher}\t3\t21\tdemographics+classical+novel\t0.4067\t0.0016\t0.0153",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_perm_coef_tsv(
    path: Path,
    features: list[str] | None = None,
):
    """Write a minimal permutation_coef result TSV."""
    if features is None:
        features = ALL_INTERACTION_FEATURES
    lines = ["feature\tcoef_obs\tp_value\tsignificant"]
    for feat in features:
        lines.append(f"{feat}\t0.03\t0.05\tFalse")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── Tests ────────────────────────────────────────────────────────────
class TestVerifyConsistency:
    """Tests for verify_consistency function."""

    def test_consistent_feature_sets(self, tmp_path: Path):
        """When both analyses use the same 19 interaction features, report consistent."""
        ts_dir = tmp_path / "three_stage"
        pc_dir = tmp_path / "permutation_coef"
        ts_dir.mkdir()
        pc_dir.mkdir()

        _write_three_stage_tsv(ts_dir / "three_stage_C_sonnet.tsv")
        _write_perm_coef_tsv(pc_dir / "permutation_coef_C_sonnet.tsv")

        result = verify_consistency(str(ts_dir), str(pc_dir))

        assert result["consistent"] is True
        assert len(result["trait_results"]) == 1
        tr = result["trait_results"][0]
        assert tr["feature_set_match"] is True
        assert tr["three_stage_r_obs"] == pytest.approx(0.4067, abs=1e-4)

    def test_inconsistent_feature_sets(self, tmp_path: Path):
        """When permutation_coef has old features, report inconsistent."""
        ts_dir = tmp_path / "three_stage"
        pc_dir = tmp_path / "permutation_coef"
        ts_dir.mkdir()
        pc_dir.mkdir()

        _write_three_stage_tsv(ts_dir / "three_stage_C_sonnet.tsv")

        # Old feature set: has IX_topic_drift_mean, missing PG_pause_variability
        old_features = [f for f in ALL_INTERACTION_FEATURES if f != "PG_pause_variability"]
        old_features.append("IX_topic_drift_mean")
        _write_perm_coef_tsv(
            pc_dir / "permutation_coef_C_sonnet.tsv",
            features=old_features,
        )

        result = verify_consistency(str(ts_dir), str(pc_dir))

        assert result["consistent"] is False
        tr = result["trait_results"][0]
        assert tr["feature_set_match"] is False
        assert "IX_topic_drift_mean" in tr["feature_set_diff"]
        assert "PG_pause_variability" in tr["feature_set_diff"]
        assert "Re-run" in tr["recommendations"][0]

    def test_missing_three_stage_dir(self, tmp_path: Path):
        """When three_stage dir has no files, report error."""
        ts_dir = tmp_path / "three_stage"
        pc_dir = tmp_path / "permutation_coef"
        ts_dir.mkdir()
        pc_dir.mkdir()

        _write_perm_coef_tsv(pc_dir / "permutation_coef_C_sonnet.tsv")

        result = verify_consistency(str(ts_dir), str(pc_dir))

        assert result["consistent"] is False
        assert "No three_stage result files" in result["summary"]

    def test_missing_permutation_coef_dir(self, tmp_path: Path):
        """When permutation_coef dir has no files, report error."""
        ts_dir = tmp_path / "three_stage"
        pc_dir = tmp_path / "permutation_coef"
        ts_dir.mkdir()
        pc_dir.mkdir()

        _write_three_stage_tsv(ts_dir / "three_stage_C_sonnet.tsv")

        result = verify_consistency(str(ts_dir), str(pc_dir))

        assert result["consistent"] is False
        assert "No permutation_coef result files" in result["summary"]

    def test_multiple_traits(self, tmp_path: Path):
        """Verify all 5 traits are checked when present."""
        ts_dir = tmp_path / "three_stage"
        pc_dir = tmp_path / "permutation_coef"
        ts_dir.mkdir()
        pc_dir.mkdir()

        for trait in ["O", "C", "E", "A", "N"]:
            _write_three_stage_tsv(
                ts_dir / f"three_stage_{trait}_sonnet.tsv", trait=trait
            )
            _write_perm_coef_tsv(
                pc_dir / f"permutation_coef_{trait}_sonnet.tsv"
            )

        result = verify_consistency(str(ts_dir), str(pc_dir))

        assert result["consistent"] is True
        assert len(result["trait_results"]) == 5
        traits_checked = {r["trait"] for r in result["trait_results"]}
        assert traits_checked == {"O", "C", "E", "A", "N"}

    def test_no_common_traits(self, tmp_path: Path):
        """When trait-teacher pairs don't overlap, report error."""
        ts_dir = tmp_path / "three_stage"
        pc_dir = tmp_path / "permutation_coef"
        ts_dir.mkdir()
        pc_dir.mkdir()

        _write_three_stage_tsv(
            ts_dir / "three_stage_C_sonnet.tsv", trait="C"
        )
        _write_perm_coef_tsv(
            pc_dir / "permutation_coef_O_qwen.tsv"
        )

        result = verify_consistency(str(ts_dir), str(pc_dir))

        assert result["consistent"] is False
        assert "No common trait-teacher pairs" in result["summary"]


class TestWriteReport:
    """Tests for write_report function."""

    def test_generates_both_files(self, tmp_path: Path):
        """write_report should create both .json and .txt files."""
        result = {
            "consistent": True,
            "timestamp": "2026-04-15T00:00:00+00:00",
            "three_stage_dir": "/tmp/ts",
            "permutation_coef_dir": "/tmp/pc",
            "summary": "All good.",
            "trait_results": [],
        }

        txt_path = write_report(result, tmp_path)

        assert txt_path.exists()
        assert (tmp_path / "consistency_report.json").exists()
        assert "All good." in txt_path.read_text(encoding="utf-8")
