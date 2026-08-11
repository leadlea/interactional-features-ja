#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Property-based test for bootstrap variance CI correctness.

# Feature: soda-feedback-v1, Property 4: Bootstrap分散分析のCI正当性

Validates: Requirements 4.2, 4.3, 4.4

任意の有効なXYデータに対して、bootstrap_variance.pyの出力は以下を満たす:
(a) 19特徴量すべてについてcoef_mean, coef_sd, ci_lower, ci_upperが含まれる
(b) ci_lower <= coef_mean <= ci_upperが成り立つ
(c) ci_excludes_zeroがTrueであるのは、ci_lower > 0 または ci_upper < 0 の場合に限る
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from scripts.analysis.bootstrap_variance import (
    ALL_FEATURES,
    CLASSICAL_FEATURES,
    NOVEL_FEATURES,
    run_bootstrap_variance,
)

# ── Strategy: use a single seed to generate the entire DataFrame at once ──


@st.composite
def xy_dataframes(draw: st.DrawFn) -> pd.DataFrame:
    """Generate random DataFrames via a single numpy RNG seed (fast)."""
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    n = draw(st.integers(min_value=30, max_value=50))
    rng = np.random.default_rng(seed)

    data: dict[str, np.ndarray] = {}

    # All 18 features: standard normal
    for feat in CLASSICAL_FEATURES + NOVEL_FEATURES:
        data[feat] = rng.standard_normal(n)

    # Target variable y: standard normal
    data["y"] = rng.standard_normal(n)

    return pd.DataFrame(data)


@given(df=xy_dataframes())
@settings(max_examples=5, deadline=None)
def test_bootstrap_variance_ci_correctness(df: pd.DataFrame) -> None:
    """Property 4: Bootstrap分散分析のCI正当性

    **Validates: Requirements 4.2, 4.3, 4.4**
    """
    result = run_bootstrap_variance(
        df=df,
        y_col="y",
        feature_cols=ALL_FEATURES,
        alpha=100.0,
        n_boot=20,
        seed=42,
    )

    # (a) 19特徴量すべてについてcoef_mean, coef_sd, ci_lower, ci_upperが含まれる
    assert len(result) == 19, f"Expected 19 rows, got {len(result)}"
    assert set(result["feature"].tolist()) == set(ALL_FEATURES)

    for col in ("coef_mean", "coef_sd", "ci_lower", "ci_upper", "ci_excludes_zero"):
        assert col in result.columns, f"Missing column: {col}"

    # coef_mean, coef_sd, ci_lower, ci_upper must be finite
    for col in ("coef_mean", "coef_sd", "ci_lower", "ci_upper"):
        assert result[col].notna().all(), f"{col} contains NaN"
        assert np.isfinite(result[col].to_numpy()).all(), f"{col} contains inf"

    # (b) ci_lower <= coef_mean <= ci_upper for all features
    for _, row in result.iterrows():
        feat = row["feature"]
        assert row["ci_lower"] <= row["coef_mean"], (
            f"{feat}: ci_lower ({row['ci_lower']}) > coef_mean ({row['coef_mean']})"
        )
        assert row["coef_mean"] <= row["ci_upper"], (
            f"{feat}: coef_mean ({row['coef_mean']}) > ci_upper ({row['ci_upper']})"
        )

    # (c) ci_excludes_zero iff (ci_lower > 0 or ci_upper < 0)
    for _, row in result.iterrows():
        feat = row["feature"]
        expected_excludes = (row["ci_lower"] > 0) or (row["ci_upper"] < 0)
        assert row["ci_excludes_zero"] == expected_excludes, (
            f"{feat}: ci_excludes_zero={row['ci_excludes_zero']} but "
            f"ci_lower={row['ci_lower']}, ci_upper={row['ci_upper']} "
            f"(expected {expected_excludes})"
        )
