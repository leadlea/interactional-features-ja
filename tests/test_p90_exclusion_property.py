#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Property-based test for p90 exclusion in descriptive stats table.

# Feature: soda-feedback-v1, Property 5: 記述統計テーブルのp90除外

Validates: Requirements 10.1, 10.2

ランダム生成した有効な特徴量DataFrameに対して、gen_descriptive_stats_full_table()の
出力LaTeXにp90列が含まれないことを検証する。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from scripts.paper_figs.gen_paper_figs_v2 import (
    FEATURE_COLUMNS,
    gen_descriptive_stats_full_table,
)

# ── Strategy: use a single seed to generate the entire DataFrame at once ──


@st.composite
def feature_dataframes(draw: st.DrawFn) -> pd.DataFrame:
    """Generate random DataFrames with all 18 FEATURE_COLUMNS via numpy RNG seed."""
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    n = draw(st.integers(min_value=10, max_value=50))
    rng = np.random.default_rng(seed)

    data: dict[str, np.ndarray] = {}
    for col in FEATURE_COLUMNS:
        data[col] = rng.standard_normal(n)

    return pd.DataFrame(data)


@given(df=feature_dataframes())
@settings(max_examples=10, deadline=None)
def test_p90_excluded_from_descriptive_stats_table(
    df: pd.DataFrame, tmp_path_factory: object
) -> None:
    """Property 5: 記述統計テーブルのp90除外

    **Validates: Requirements 10.1, 10.2**
    """
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = Path(tmp_dir)
        gen_descriptive_stats_full_table(df, out_dir)

        tex_path = out_dir / "tab_descriptive_stats_full.tex"
        assert tex_path.exists(), "tab_descriptive_stats_full.tex was not created"

        content = tex_path.read_text(encoding="utf-8")

        # Extract the header line (between \toprule and \midrule)
        lines = content.splitlines()
        header_lines = []
        in_header = False
        for line in lines:
            if r"\toprule" in line:
                in_header = True
                continue
            if r"\midrule" in line:
                in_header = False
                continue
            if in_header:
                header_lines.append(line)

        header_text = " ".join(header_lines)

        # p90 must NOT appear as a column header
        assert "p90" not in header_text, (
            f"p90 found in table header: {header_text}"
        )

        # Also verify the table has exactly 8 data columns (N, Mean, SD, Min, p25, p50, p75, Max)
        # by checking the column spec: lrrrrrrrr = 1 text + 8 numeric = 9 total
        assert r"\begin{tabular}{lrrrrrrrr}" in content, (
            "Expected 9-column tabular (l + 8r), got different column spec"
        )
