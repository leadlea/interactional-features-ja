#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Property-based test for permutation coefficient test output completeness.

# Feature: soda-feedback-v1, Property 3: Permutation回帰係数検定の出力完全性

Validates: Requirements 4.1

任意の有効なXYデータに対して、permutation_coef_test.pyの出力TSVは19特徴量すべてについて
coef_obsとp_valueを含み、各p_valueは0以上1以下の範囲にある。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from scripts.analysis.permutation_coef_test import (
    ALL_FEATURES,
    CLASSICAL_FEATURES,
    NOVEL_FEATURES,
    run_permutation_coef_test,
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
def test_permutation_coef_output_completeness(df: pd.DataFrame) -> None:
    """Property 3: Permutation回帰係数検定の出力完全性

    **Validates: Requirements 4.1**
    """
    result = run_permutation_coef_test(
        df=df,
        y_col="y",
        feature_cols=ALL_FEATURES,
        alpha=100.0,
        n_perm=0,
        seed=42,
    )

    # Output should contain exactly 19 rows (one per feature)
    assert len(result) == 19, f"Expected 19 rows, got {len(result)}"

    # All 19 features must be present
    assert set(result["feature"].tolist()) == set(ALL_FEATURES)

    # Required columns must exist
    assert "coef_obs" in result.columns
    assert "p_value" in result.columns

    # coef_obs must be finite numbers (not NaN or inf)
    assert result["coef_obs"].notna().all(), "coef_obs contains NaN"
    assert np.isfinite(result["coef_obs"].to_numpy()).all(), "coef_obs contains inf"

    # p_value must be in [0, 1] for all features
    p_values = result["p_value"].to_numpy()
    assert (p_values >= 0.0).all(), f"p_value < 0 found: {p_values[p_values < 0]}"
    assert (p_values <= 1.0).all(), f"p_value > 1 found: {p_values[p_values > 1]}"
