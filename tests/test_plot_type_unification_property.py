"""Property test: plot type unification between feature distribution and gender comparison.

# Feature: soda-feedback-v1, Property 6: 図表プロットタイプの統一

**Validates: Requirements 11.3**

Static analysis test — reads the source code of gen_paper_figs_v2.py and
verifies that gen_feature_distribution() and gen_metadata_gender() use the
same primary plot type (both violinplot or both boxplot).
"""
from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path

import pytest

# Path to the source file under test
_SRC_PATH = Path(__file__).resolve().parent.parent / "scripts" / "paper_figs" / "gen_paper_figs_v2.py"


def _read_source() -> str:
    """Read the full source of gen_paper_figs_v2.py."""
    assert _SRC_PATH.exists(), f"Source file not found: {_SRC_PATH}"
    return _SRC_PATH.read_text(encoding="utf-8")


def _extract_function_body(source: str, func_name: str) -> str:
    """Extract the source text of a top-level function by name using AST."""
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            start = node.lineno - 1  # 0-indexed
            end = node.end_lineno  # end_lineno is 1-indexed inclusive
            return "".join(lines[start:end])
    raise ValueError(f"Function '{func_name}' not found in source")


def _detect_plot_types(func_body: str) -> set[str]:
    """Detect plot type calls (violinplot / boxplot) in a function body.

    Returns a set of detected primary plot types, e.g. {"violinplot"}.
    """
    types: set[str] = set()
    # Match calls like ax.violinplot(...) or ax.boxplot(...)
    if re.search(r"\bviolinplot\s*\(", func_body):
        types.add("violinplot")
    if re.search(r"\bboxplot\s*\(", func_body):
        types.add("boxplot")
    return types


class TestPlotTypeUnification:
    """Verify that feature distribution and gender comparison use the same plot type."""

    def test_both_functions_exist(self) -> None:
        """Both gen_feature_distribution and gen_metadata_gender must exist."""
        source = _read_source()
        _extract_function_body(source, "gen_feature_distribution")
        _extract_function_body(source, "gen_metadata_gender")

    def test_primary_plot_type_matches(self) -> None:
        """The primary plot type used in both functions must be the same.

        Both functions should use violinplot as their primary (default) plot
        type.  gen_metadata_gender may have a boxplot fallback for very small
        samples, but its primary path must still be violinplot.
        """
        source = _read_source()

        dist_body = _extract_function_body(source, "gen_feature_distribution")
        gender_body = _extract_function_body(source, "gen_metadata_gender")

        dist_types = _detect_plot_types(dist_body)
        gender_types = _detect_plot_types(gender_body)

        # Both must include the same primary plot type
        assert len(dist_types) > 0, (
            "gen_feature_distribution does not call any known plot type "
            "(violinplot or boxplot)"
        )
        assert len(gender_types) > 0, (
            "gen_metadata_gender does not call any known plot type "
            "(violinplot or boxplot)"
        )

        # The intersection must be non-empty — they share at least one plot type
        shared = dist_types & gender_types
        assert len(shared) > 0, (
            f"Plot type mismatch: gen_feature_distribution uses {dist_types}, "
            f"gen_metadata_gender uses {gender_types}. "
            f"They must share the same primary plot type."
        )

    def test_feature_distribution_uses_violinplot(self) -> None:
        """gen_feature_distribution should use violinplot."""
        source = _read_source()
        body = _extract_function_body(source, "gen_feature_distribution")
        types = _detect_plot_types(body)
        assert "violinplot" in types, (
            f"gen_feature_distribution uses {types} but expected violinplot"
        )

    def test_metadata_gender_uses_violinplot(self) -> None:
        """gen_metadata_gender should use violinplot as its primary plot type."""
        source = _read_source()
        body = _extract_function_body(source, "gen_metadata_gender")
        types = _detect_plot_types(body)
        assert "violinplot" in types, (
            f"gen_metadata_gender uses {types} but expected violinplot"
        )
