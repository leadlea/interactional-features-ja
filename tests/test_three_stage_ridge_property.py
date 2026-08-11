#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Property-based test for three-stage Ridge regression feature set correctness and delta_r arithmetic.

# Feature: soda-feedback-v1, Property 2: 3段階Ridge回帰の特徴量セット正当性とΔr算術

Validates: Requirements 3.1, 3.2

任意の有効なXYデータとメタデータの組み合わせに対して、three_stage_ridge.pyの出力は以下を満たす:
(a) Stage 1は正確に2個の人口統計変数のみを使用する
(b) Stage 2は正確に12個（人口統計2 + Classical 10）の変数を使用する
(c) Stage 3は正確に21個（人口統計2 + Classical 10 + Novel 9）の変数を使用する
(d) delta_r_from_prev（Stage 2）== r_obs(Stage 2) - r_obs(Stage 1)が浮動小数点精度の範囲内で成り立つ
(e) delta_r_from_prev（Stage 3）== r_obs(Stage 3) - r_obs(Stage 2)が同様に成り立つ
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from scripts.analysis.three_stage_ridge import (
    CLASSICAL_FEATURES,
    NOVEL_FEATURES,
    run_three_stage_analysis,
)

# ── Strategy: use a single seed to generate the entire DataFrame at once ──

@st.composite
def xy_dataframes(draw: st.DrawFn) -> pd.DataFrame:
    """Generate random DataFrames via a single numpy RNG seed (fast)."""
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    n = draw(st.integers(min_value=30, max_value=50))
    rng = np.random.default_rng(seed)

    data: dict[str, np.ndarray] = {}

    # Gender: binary 0/1, ensure both values present
    g = rng.choice([0.0, 1.0], size=n)
    if np.all(g == g[0]):
        g[0] = 1.0 - g[0]
    data["confound_gender"] = g

    # Age: uniform 20-80
    data["confound_age"] = rng.uniform(20, 80, size=n)

    # Classical + Novel features + y: standard normal
    for feat in CLASSICAL_FEATURES + NOVEL_FEATURES:
        data[feat] = rng.standard_normal(n)
    data["y"] = rng.standard_normal(n)

    return pd.DataFrame(data)


@given(df=xy_dataframes())
@settings(max_examples=5, deadline=None)
def test_three_stage_feature_set_correctness_and_delta_r_arithmetic(
    df: pd.DataFrame,
) -> None:
    """Property 2: 3段階Ridge回帰の特徴量セット正当性とΔr算術"""
    result = run_three_stage_analysis(
        df=df, y_col="y", alpha=100.0, cv_folds=3, n_perm=0, seed=42,
        trait="test", teacher="test",
    )

    assert len(result) == 3, f"Expected 3 stages, got {len(result)}"

    s1 = result[result["stage"] == 1].iloc[0]
    s2 = result[result["stage"] == 2].iloc[0]
    s3 = result[result["stage"] == 3].iloc[0]

    # (a-c) Feature set sizes
    assert s1["n_features"] == 2
    assert s2["n_features"] == 12
    assert s3["n_features"] == 21

    # (d-e) Δr arithmetic
    assert np.isclose(s2["delta_r_from_prev"], s2["r_obs"] - s1["r_obs"], atol=1e-3)
    assert np.isclose(s3["delta_r_from_prev"], s3["r_obs"] - s2["r_obs"], atol=1e-3)

    # Stage 1 delta should be NaN
    assert np.isnan(s1["delta_r_from_prev"])
