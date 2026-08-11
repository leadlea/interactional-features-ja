#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sensitivity Analysis for interaction feature parameters.

Run Ridge + Permutation test under varying parameter conditions
to assess robustness of main results (trait C).

Supported analysis types:
  - gap_tol:     gap_tol ∈ {0.05, 0.1, 0.2, 0.3, 0.5, 1.0}
  - yesno_list:  narrow (5) vs broad (17)  [Task 6.2]
  - ne_yo_match: regex vs 1-char vs 2-char [Task 6.3]
  - all:         run all of the above

Usage:
    python scripts/analysis/sensitivity_analysis.py \
        --utterances_parquet artifacts/cejc/utterances_home2_hq1.parquet \
        --features_parquet artifacts/analysis/features_min/features_cejc_home2_hq1.parquet \
        --items_dir artifacts/big5/llm_scores \
        --analysis_type gap_tol \
        --out_dir artifacts/analysis/results/sensitivity
"""
from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ── Sibling imports ──────────────────────────────────────────────────
# Scripts in this directory are not a package; add parent to sys.path.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import extract_interaction_features_min as eim  # noqa: E402
from extract_interaction_features_min import extract_features  # noqa: E402
from ensemble_permutation import (  # noqa: E402
    DEFAULT_EXCLUDE,
    TEACHERS,
    cv_ridge_r,
    load_trait_scores,
)

# ── Constants ────────────────────────────────────────────────────────
GAP_TOL_CONDITIONS: list[float] = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
DEFAULT_TRAIT = "C"
DEFAULT_ALPHA = 100.0
DEFAULT_N_PERM = 5000
DEFAULT_SEED = 42
DEFAULT_CV_FOLDS = 5


# ── Core analysis runner ─────────────────────────────────────────────

def _run_permutation(
    X: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float = DEFAULT_ALPHA,
    n_perm: int = DEFAULT_N_PERM,
    seed: int = DEFAULT_SEED,
    cv_folds: int = DEFAULT_CV_FOLDS,
) -> tuple[float, float]:
    """Run Ridge CV + permutation test, return (r_obs, p_value)."""
    r_obs = cv_ridge_r(X, y, cv_folds, seed, alpha)

    rng = np.random.default_rng(seed)
    r_perm = np.empty(n_perm, float)
    for i in range(n_perm):
        yp = rng.permutation(y)
        r_perm[i] = cv_ridge_r(X, yp, cv_folds, seed, alpha)

    p_val = float((np.sum(np.abs(r_perm) >= abs(r_obs)) + 1.0) / (n_perm + 1.0))
    return r_obs, p_val


def _prepare_Xy(
    features_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    exclude: set[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Merge features with trait scores and return (X, y, feat_cols)."""
    merged = scores_df.merge(
        features_df, on=["conversation_id", "speaker_id"], how="inner",
    )
    y_col = "trait_score"
    y = merged[y_col].astype(float).to_numpy()

    feat_cols = [
        c for c in merged.columns if c not in exclude and c != y_col
    ]
    X = merged[feat_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    ok = ~np.isnan(y)
    return X[ok], y[ok], feat_cols


# ── gap_tol sensitivity ─────────────────────────────────────────────
def run_gap_tol_sensitivity(
    utterances_df: pd.DataFrame,
    base_features_df: pd.DataFrame,
    items_dir: str,
    *,
    trait: str = DEFAULT_TRAIT,
    alpha: float = DEFAULT_ALPHA,
    n_perm: int = DEFAULT_N_PERM,
    seed: int = DEFAULT_SEED,
    cv_folds: int = DEFAULT_CV_FOLDS,
) -> list[dict]:
    """Run gap_tol sensitivity analysis.

    For each gap_tol value, re-extract features and run Ridge + Permutation
    test against ensemble Big5 trait scores.

    Args:
        utterances_df: Raw utterances DataFrame.
        base_features_df: Base features parquet (used to derive target_pairs).
        items_dir: Root directory for LLM trait score parquets.
        trait: Big5 trait to test (default "C").
        alpha: Ridge alpha.
        n_perm: Number of permutation rounds.
        seed: Random seed.
        cv_folds: Number of CV folds.

    Returns:
        List of dicts with keys: analysis_type, condition, trait, r_obs, p_value.
    """
    # Derive target_pairs from base features
    target_pairs: set[tuple[str, str]] = set()
    for _, row in base_features_df[["conversation_id", "speaker_id"]].iterrows():
        target_pairs.add((str(row["conversation_id"]), str(row["speaker_id"])))

    # Load ensemble trait scores once
    scores_df = load_trait_scores(items_dir, trait, TEACHERS)

    exclude = set(DEFAULT_EXCLUDE) | {"trait_score"}

    results: list[dict] = []

    for gap_tol in GAP_TOL_CONDITIONS:
        print(f"\n--- gap_tol = {gap_tol} ---")

        # Re-extract features with this gap_tol
        feat_df = extract_features(utterances_df, target_pairs, gap_tol=gap_tol)
        feat_df = feat_df.replace([np.inf, -np.inf], np.nan)

        if feat_df.empty:
            warnings.warn(f"gap_tol={gap_tol}: no features extracted")
            results.append({
                "analysis_type": "gap_tol",
                "condition": str(gap_tol),
                "trait": trait,
                "r_obs": float("nan"),
                "p_value": float("nan"),
            })
            continue

        X, y, feat_cols = _prepare_Xy(feat_df, scores_df, exclude)
        print(f"  Features: {len(feat_cols)} cols, {X.shape[0]} samples")

        r_obs, p_val = _run_permutation(
            X, y, alpha=alpha, n_perm=n_perm, seed=seed, cv_folds=cv_folds,
        )
        print(f"  r_obs = {r_obs:.3f}, p = {p_val:.4f}")

        results.append({
            "analysis_type": "gap_tol",
            "condition": str(gap_tol),
            "trait": trait,
            "r_obs": round(r_obs, 4),
            "p_value": round(p_val, 4),
        })

    return results


# ── YES/NO list sensitivity ──────────────────────────────────────────
YESNO_NARROW: list[str] = ["はい", "うん", "ええ", "いいえ", "いや"]
YESNO_BROAD: list[str] = list(eim.YESNO_PREFIXES)  # current 17 entries


def run_yesno_list_sensitivity(
    utterances_df: pd.DataFrame,
    base_features_df: pd.DataFrame,
    items_dir: str,
    *,
    trait: str = DEFAULT_TRAIT,
    alpha: float = DEFAULT_ALPHA,
    n_perm: int = DEFAULT_N_PERM,
    seed: int = DEFAULT_SEED,
    cv_folds: int = DEFAULT_CV_FOLDS,
) -> list[dict]:
    """Run YES/NO prefix list sensitivity analysis.

    Compare narrow list (5 items) vs broad list (current 17 items) by
    monkey-patching YESNO_PREFIXES in extract_interaction_features_min,
    re-extracting features, and running Ridge + Permutation test.

    Args:
        utterances_df: Raw utterances DataFrame.
        base_features_df: Base features parquet (used to derive target_pairs).
        items_dir: Root directory for LLM trait score parquets.
        trait: Big5 trait to test (default "C").
        alpha: Ridge alpha.
        n_perm: Number of permutation rounds.
        seed: Random seed.
        cv_folds: Number of CV folds.

    Returns:
        List of dicts with keys: analysis_type, condition, trait, r_obs, p_value.
    """
    # Derive target_pairs from base features
    target_pairs: set[tuple[str, str]] = set()
    for _, row in base_features_df[["conversation_id", "speaker_id"]].iterrows():
        target_pairs.add((str(row["conversation_id"]), str(row["speaker_id"])))

    # Load ensemble trait scores once
    scores_df = load_trait_scores(items_dir, trait, TEACHERS)

    exclude = set(DEFAULT_EXCLUDE) | {"trait_score"}

    conditions: list[tuple[str, list[str]]] = [
        ("narrow_5", YESNO_NARROW),
        ("broad_17", YESNO_BROAD),
    ]

    results: list[dict] = []
    original_prefixes = eim.YESNO_PREFIXES

    for cond_name, prefix_list in conditions:
        print(f"\n--- yesno_list = {cond_name} ({len(prefix_list)} items) ---")

        # Monkey-patch YESNO_PREFIXES for feature extraction
        eim.YESNO_PREFIXES = prefix_list
        try:
            feat_df = extract_features(utterances_df, target_pairs, gap_tol=0.05)
        finally:
            eim.YESNO_PREFIXES = original_prefixes  # always restore

        feat_df = feat_df.replace([np.inf, -np.inf], np.nan)

        if feat_df.empty:
            warnings.warn(f"yesno_list={cond_name}: no features extracted")
            results.append({
                "analysis_type": "yesno_list",
                "condition": cond_name,
                "trait": trait,
                "r_obs": float("nan"),
                "p_value": float("nan"),
            })
            continue

        X, y, feat_cols = _prepare_Xy(feat_df, scores_df, exclude)
        print(f"  Features: {len(feat_cols)} cols, {X.shape[0]} samples")

        r_obs, p_val = _run_permutation(
            X, y, alpha=alpha, n_perm=n_perm, seed=seed, cv_folds=cv_folds,
        )
        print(f"  r_obs = {r_obs:.3f}, p = {p_val:.4f}")

        results.append({
            "analysis_type": "yesno_list",
            "condition": cond_name,
            "trait": trait,
            "r_obs": round(r_obs, 4),
            "p_value": round(p_val, 4),
        })

    return results


# ── NE/YO matching sensitivity ────────────────────────────────────────
NE_YO_CONDITIONS: list[tuple[str, re.Pattern, re.Pattern]] = [
    (
        "regex",
        re.compile(r"(よね|だよね|ですよね|だよな|ね)$"),
        re.compile(r"(だよ|ですよ|よ)$"),
    ),
    (
        "1char",
        re.compile(r"ね$"),
        re.compile(r"よ$"),
    ),
    (
        "2char",
        re.compile(r"(よね|だね|ですね|ね)$"),
        re.compile(r"(だよ|ですよ|よ)$"),
    ),
]


def run_ne_yo_match_sensitivity(
    utterances_df: pd.DataFrame,
    base_features_df: pd.DataFrame,
    items_dir: str,
    *,
    trait: str = DEFAULT_TRAIT,
    alpha: float = DEFAULT_ALPHA,
    n_perm: int = DEFAULT_N_PERM,
    seed: int = DEFAULT_SEED,
    cv_folds: int = DEFAULT_CV_FOLDS,
) -> list[dict]:
    """Run NE/YO matching method sensitivity analysis.

    Compare three matching conditions for sentence-final particles:
      - "regex"  (current): multi-pattern regex
      - "1char":  single-char match (末尾「ね」/「よ」のみ)
      - "2char":  two-char-aware patterns

    Monkey-patches eim._SFP_NE_RE / eim._SFP_YO_RE, re-extracts features,
    and runs Ridge + Permutation test for each condition.

    Args:
        utterances_df: Raw utterances DataFrame.
        base_features_df: Base features parquet (used to derive target_pairs).
        items_dir: Root directory for LLM trait score parquets.
        trait: Big5 trait to test (default "C").
        alpha: Ridge alpha.
        n_perm: Number of permutation rounds.
        seed: Random seed.
        cv_folds: Number of CV folds.

    Returns:
        List of dicts with keys: analysis_type, condition, trait, r_obs, p_value.
    """
    # Derive target_pairs from base features
    target_pairs: set[tuple[str, str]] = set()
    for _, row in base_features_df[["conversation_id", "speaker_id"]].iterrows():
        target_pairs.add((str(row["conversation_id"]), str(row["speaker_id"])))

    # Load ensemble trait scores once
    scores_df = load_trait_scores(items_dir, trait, TEACHERS)

    exclude = set(DEFAULT_EXCLUDE) | {"trait_score"}

    results: list[dict] = []
    original_ne_re = eim._SFP_NE_RE
    original_yo_re = eim._SFP_YO_RE

    for cond_name, ne_re, yo_re in NE_YO_CONDITIONS:
        print(f"\n--- ne_yo_match = {cond_name} ---")
        print(f"  NE pattern: {ne_re.pattern}")
        print(f"  YO pattern: {yo_re.pattern}")

        # Monkey-patch _SFP_NE_RE / _SFP_YO_RE for feature extraction
        eim._SFP_NE_RE = ne_re
        eim._SFP_YO_RE = yo_re
        try:
            feat_df = extract_features(utterances_df, target_pairs, gap_tol=0.05)
        finally:
            eim._SFP_NE_RE = original_ne_re  # always restore
            eim._SFP_YO_RE = original_yo_re

        feat_df = feat_df.replace([np.inf, -np.inf], np.nan)

        if feat_df.empty:
            warnings.warn(f"ne_yo_match={cond_name}: no features extracted")
            results.append({
                "analysis_type": "ne_yo_match",
                "condition": cond_name,
                "trait": trait,
                "r_obs": float("nan"),
                "p_value": float("nan"),
            })
            continue

        X, y, feat_cols = _prepare_Xy(feat_df, scores_df, exclude)
        print(f"  Features: {len(feat_cols)} cols, {X.shape[0]} samples")

        r_obs, p_val = _run_permutation(
            X, y, alpha=alpha, n_perm=n_perm, seed=seed, cv_folds=cv_folds,
        )
        print(f"  r_obs = {r_obs:.3f}, p = {p_val:.4f}")

        results.append({
            "analysis_type": "ne_yo_match",
            "condition": cond_name,
            "trait": trait,
            "r_obs": round(r_obs, 4),
            "p_value": round(p_val, 4),
        })

    return results


# ── Regularization alpha sensitivity ─────────────────────────────────
ALPHA_CONDITIONS: list[float] = [10.0, 50.0, 100.0, 200.0, 500.0]


def run_alpha_sensitivity(
    utterances_df: pd.DataFrame,
    base_features_df: pd.DataFrame,
    items_dir: str,
    *,
    trait: str = DEFAULT_TRAIT,
    alpha: float = DEFAULT_ALPHA,
    n_perm: int = DEFAULT_N_PERM,
    seed: int = DEFAULT_SEED,
    cv_folds: int = DEFAULT_CV_FOLDS,
) -> list[dict]:
    """Run Ridge regularization strength (alpha) sensitivity analysis.

    Unlike gap_tol / yesno_list / ne_yo_match, the feature set is fixed to the
    canonical final features parquet (no re-extraction); only the Ridge alpha
    is varied over ALPHA_CONDITIONS. At alpha=100 this reproduces the main
    model exactly, so the sweep quantifies how r_obs / p_value shift with
    regularization strength.

    Note: the ``alpha`` keyword argument is ignored here (the sweep defines its
    own alpha grid); it is accepted only for a uniform runner signature.
    """
    # Load ensemble trait scores once
    scores_df = load_trait_scores(items_dir, trait, TEACHERS)
    exclude = set(DEFAULT_EXCLUDE) | {"trait_score"}

    # Feature set is fixed to the canonical parquet (no re-extraction).
    feat_df = base_features_df.replace([np.inf, -np.inf], np.nan)
    X, y, feat_cols = _prepare_Xy(feat_df, scores_df, exclude)
    print(f"alpha sweep: {len(feat_cols)} features, {X.shape[0]} samples")

    results: list[dict] = []
    for a in ALPHA_CONDITIONS:
        print(f"\n--- alpha = {a} ---")
        r_obs, p_val = _run_permutation(
            X, y, alpha=a, n_perm=n_perm, seed=seed, cv_folds=cv_folds,
        )
        print(f"  r_obs = {r_obs:.3f}, p = {p_val:.4f}")
        results.append({
            "analysis_type": "alpha",
            "condition": str(a),
            "trait": trait,
            "r_obs": round(r_obs, 4),
            "p_value": round(p_val, 4),
        })
    return results


# ── Dispatcher ───────────────────────────────────────────────────────
ANALYSIS_RUNNERS = {
    "gap_tol": "run_gap_tol_sensitivity",
    "yesno_list": "run_yesno_list_sensitivity",
    "ne_yo_match": "run_ne_yo_match_sensitivity",
    "alpha": "run_alpha_sensitivity",
}


def run_analysis(
    analysis_type: str,
    utterances_df: pd.DataFrame,
    base_features_df: pd.DataFrame,
    items_dir: str,
    **kwargs,
) -> list[dict]:
    """Dispatch to the appropriate sensitivity analysis runner."""
    if analysis_type == "all":
        all_results: list[dict] = []
        for atype in ANALYSIS_RUNNERS:
            all_results.extend(
                run_analysis(atype, utterances_df, base_features_df, items_dir, **kwargs)
            )
        return all_results

    if analysis_type not in ANALYSIS_RUNNERS:
        raise ValueError(
            f"Unknown analysis_type: {analysis_type!r}. "
            f"Available: {list(ANALYSIS_RUNNERS.keys()) + ['all']}"
        )

    runner_name = ANALYSIS_RUNNERS[analysis_type]
    runner_fn = globals()[runner_name]
    return runner_fn(utterances_df, base_features_df, items_dir, **kwargs)


# ── CLI ──────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Sensitivity analysis for interaction feature parameters",
    )
    ap.add_argument(
        "--utterances_parquet", default=None,
        help="Path to utterances parquet (required for gap_tol/yesno_list/"
             "ne_yo_match/all; not needed for the 'alpha' sweep, which reuses "
             "the fixed feature set)",
    )
    ap.add_argument(
        "--features_parquet", required=True,
        help="Path to base features parquet (for target_pairs derivation)",
    )
    ap.add_argument(
        "--items_dir", default="artifacts/big5/llm_scores",
        help="Root dir containing LLM trait score parquets",
    )
    ap.add_argument(
        "--analysis_type", required=True,
        choices=list(ANALYSIS_RUNNERS.keys()) + ["all"],
        help="Which sensitivity analysis to run",
    )
    ap.add_argument("--trait", default=DEFAULT_TRAIT, help="Big5 trait (default: C)")
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="Ridge alpha")
    ap.add_argument("--n_perm", type=int, default=DEFAULT_N_PERM, help="Permutation rounds")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    ap.add_argument("--cv_folds", type=int, default=DEFAULT_CV_FOLDS, help="CV folds")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    args = ap.parse_args()

    # Load data. The 'alpha' sweep reuses the fixed feature set and does not
    # re-extract features, so utterances are not required for it.
    needs_utterances = args.analysis_type != "alpha"
    if needs_utterances and not args.utterances_parquet:
        ap.error(
            f"--utterances_parquet is required for analysis_type="
            f"{args.analysis_type!r} (only 'alpha' can omit it)"
        )
    if args.utterances_parquet:
        print(f"Loading utterances: {args.utterances_parquet}")
        utterances_df = pd.read_parquet(args.utterances_parquet)
        print(f"  {len(utterances_df)} rows")
    else:
        utterances_df = None

    print(f"Loading base features: {args.features_parquet}")
    base_features_df = pd.read_parquet(args.features_parquet)
    print(f"  {len(base_features_df)} rows")

    # Run analysis
    results = run_analysis(
        analysis_type=args.analysis_type,
        utterances_df=utterances_df,
        base_features_df=base_features_df,
        items_dir=args.items_dir,
        trait=args.trait,
        alpha=args.alpha,
        n_perm=args.n_perm,
        seed=args.seed,
        cv_folds=args.cv_folds,
    )

    # Write output
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)
    out_path = out_dir / "sensitivity_results.tsv"

    # Append if file exists (for incremental runs of different analysis_types)
    if out_path.exists() and args.analysis_type != "all":
        existing = pd.read_csv(out_path, sep="\t")
        # Remove old rows for this analysis_type to avoid duplicates
        existing = existing[existing["analysis_type"] != args.analysis_type]
        results_df = pd.concat([existing, results_df], ignore_index=True)

    results_df.to_csv(out_path, sep="\t", index=False)
    print(f"\nWrote {out_path}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
