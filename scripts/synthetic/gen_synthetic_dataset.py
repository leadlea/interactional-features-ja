#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Generate a synthetic dataset with the same structure as the analysis inputs.

Why this exists
---------------
The CEJC licence agreement prohibits providing derived data to third parties, and its
definition of derived data covers the feature matrix and the language-model trait scores
as well as the transcripts. So none of the real analysis inputs can be archived here (see
docs/data-availability.md). That leaves a reader unable to run the pipeline at all, even
just to confirm their environment is set up and the code executes.

This script closes that gap. It builds a dataset of matching shape and schema from
**published summary statistics only**, so nothing derived from the corpus is
redistributed. Two published tables are the entire input:

- ``reports/paper_figs_v2/tab_descriptive_stats_full.tex`` — N, mean, SD and the
  five-number summary for each of the 19 features
- ``reports/paper_figs_v2/tab_corr_matrix.tex`` — the 19x19 Pearson correlation matrix,
  printed to two decimals

Both are already in this repository as aggregate statistics. The speaker structure comes
from counts stated in the manuscript: 120 records from 66 conversations and 74 speakers,
of whom 25 appear in two or more conversations (71 records, 59.2%) and 49 appear once;
66 records are from women and 54 from men.

What this is not
----------------
**It does not reproduce the reported results, and it is not meant to.** By default the
outcome is drawn independently of the features, so every analysis run on this dataset
should come out non-significant. That is the useful behaviour: it tells you the pipeline
executes end to end and that the null case behaves as it should. Passing ``--rho`` injects
an association of known strength, which is how you check that the pipeline can detect one.

Neither mode says anything about the real data. A synthetic dataset built from marginals
and a correlation matrix cannot carry the structure the findings rest on.

Method
------
1. Parse the two published tables.
2. Project the correlation matrix onto the nearest positive semi-definite matrix. This is
   necessary because printing to two decimals can push the matrix slightly indefinite.
3. Draw from a Gaussian copula: sample from a multivariate normal with that correlation,
   convert to uniforms, then map each margin through a piecewise-linear quantile function
   built from the published min / p25 / p50 / p75 / max. This reproduces the published
   marginal shape approximately and the rank correlations approximately; it does not
   reproduce the joint distribution, which no summary statistic determines.
4. Rescale each column so its mean and SD match the published values.
5. Build the speaker grouping to the published counts, assign sex to the published record
   split, and draw ages from a configurable distribution. **The age distribution is not
   taken from the corpus** — the manuscript does not publish age descriptives — so its
   default is a generic adult range and is documented as arbitrary.
6. Write the feature parquet, the speaker metadata table, and the XY / XYB datasets for
   all five traits, into a separate root so they cannot be confused with real inputs.

Usage
-----
    python scripts/synthetic/gen_synthetic_dataset.py
    python scripts/synthetic/gen_synthetic_dataset.py --rho 0.45   # inject a known signal

Then point the analyses at the synthetic root:

    make analysis DATASET_DIR=artifacts/synthetic/datasets \
                  META_TSV=artifacts/synthetic/synthetic_speaker_metadata.tsv
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

TRAITS = ["O", "C", "E", "A", "N"]

# The 19 predictors, in the order the manuscript's tables use.
ALL_FEATURES = [
    "PG_speech_ratio", "PG_pause_mean", "PG_pause_p50", "PG_pause_p90",
    "PG_resp_gap_mean", "PG_resp_gap_p50", "PG_resp_gap_p90", "PG_overlap_rate",
    "FILL_has_any", "FILL_rate_per_100chars",
    "IX_oirmarker_rate", "IX_oirmarker_after_question_rate",
    "IX_yesno_rate", "IX_yesno_after_question_rate", "IX_lex_overlap_mean",
    "RESP_NE_AIZUCHI_RATE", "RESP_NE_ENTROPY", "RESP_YO_ENTROPY",
    "PG_pause_variability",
]

# Speaker structure, from the manuscript (section 2.1).
N_RECORDS = 120
N_CONVERSATIONS = 66
N_SPEAKERS = 74
# 25 speakers contribute 71 records; the remaining 49 contribute one each.
N_REPEAT_SPEAKERS = 25
N_REPEAT_RECORDS = 71
N_FEMALE_RECORDS = 66

# Features bounded to [0, 1] by construction; the sampler clips them so the synthetic
# values stay in a range the real measure could produce.
UNIT_INTERVAL_FEATURES = {
    "PG_speech_ratio", "PG_overlap_rate", "FILL_has_any",
    "IX_oirmarker_rate", "IX_oirmarker_after_question_rate",
    "IX_yesno_rate", "IX_yesno_after_question_rate", "IX_lex_overlap_mean",
    "RESP_NE_AIZUCHI_RATE",
}

# Columns the analysis scripts expect to exist but exclude from the design matrix. They
# are filled with NaN: nothing downstream reads them, and a plausible-looking fake count
# would be more misleading than an obvious absence.
PASSTHROUGH_NAN_COLS = [
    "n_pairs_total", "n_pairs_after_NE", "n_pairs_after_YO",
    "IX_n_pairs", "IX_n_pairs_after_question",
    "PG_total_time", "PG_resp_overlap_rate",
    "FILL_text_len", "FILL_cnt_total", "FILL_cnt_eto", "FILL_cnt_e", "FILL_cnt_ano",
    "IX_topic_drift_mean",
]


def _unescape(name: str) -> str:
    return name.replace("\\_", "_").strip()


def parse_descriptives(path: Path) -> pd.DataFrame:
    """Read the published descriptive-statistics table."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if "&" not in line or line.lstrip().startswith("%"):
            continue
        cells = [c.strip() for c in line.rstrip("\\ ").split("&")]
        name = _unescape(cells[0])
        if name not in ALL_FEATURES or len(cells) < 9:
            continue
        try:
            nums = [float(c.rstrip("\\ ")) for c in cells[1:9]]
        except ValueError:
            continue
        rows.append({
            "feature": name, "n": int(nums[0]), "mean": nums[1], "sd": nums[2],
            "min": nums[3], "p25": nums[4], "p50": nums[5], "p75": nums[6],
            "max": nums[7],
        })
    df = pd.DataFrame(rows).set_index("feature")
    missing = [f for f in ALL_FEATURES if f not in df.index]
    if missing:
        raise SystemExit(f"features missing from {path.name}: {missing}")
    return df.loc[ALL_FEATURES]


def parse_correlations(path: Path) -> np.ndarray:
    """Read the published 19x19 correlation matrix.

    The diagonal is printed as ``---``; off-diagonal cells are two-decimal numbers.
    """
    k = len(ALL_FEATURES)
    rows: list[list[float]] = []
    row_re = re.compile(r"^\((\d+)\)$")
    for line in path.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.rstrip("\\ ").split("&")]
        if not cells or not row_re.match(cells[0]):
            continue
        vals = []
        for c in cells[1:]:
            c = c.rstrip("\\ ").strip()
            vals.append(1.0 if c.startswith("-" * 3) else float(c))
        if len(vals) != k:
            raise SystemExit(f"row {cells[0]} in {path.name} has {len(vals)} cells, want {k}")
        rows.append(vals)
    if len(rows) != k:
        raise SystemExit(f"{path.name} has {len(rows)} rows, want {k}")
    R = np.asarray(rows, float)
    return (R + R.T) / 2.0


def nearest_psd_correlation(R: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, float]:
    """Project onto the nearest PSD correlation matrix; return it and the adjustment.

    Two-decimal rounding can leave the published matrix slightly indefinite, which makes
    it unusable as a covariance. Clipping the eigenvalues at ``eps`` and renormalising the
    diagonal is the standard fix. The returned float is the largest absolute change to any
    entry, so the caller can report how much the input had to be moved.
    """
    w, V = np.linalg.eigh(R)
    if w.min() >= eps:
        return R, 0.0
    A = V @ np.diag(np.clip(w, eps, None)) @ V.T
    d = np.sqrt(np.diag(A))
    A = A / np.outer(d, d)
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 1.0)
    return A, float(np.max(np.abs(A - R)))


def quantile_map(u: np.ndarray, stats: pd.Series) -> np.ndarray:
    """Map uniforms through a piecewise-linear quantile function from the five-number summary."""
    probs = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
    knots = np.array([stats["min"], stats["p25"], stats["p50"], stats["p75"], stats["max"]],
                     float)
    # np.interp needs the knots non-decreasing; a degenerate summary would otherwise
    # produce a non-monotone map.
    knots = np.maximum.accumulate(knots)
    return np.interp(u, probs, knots)


def build_features(desc: pd.DataFrame, R: np.ndarray, n: int,
                   rng: np.random.Generator) -> pd.DataFrame:
    """Gaussian copula draw, then match the published mean and SD per column."""
    k = len(ALL_FEATURES)
    z = rng.multivariate_normal(np.zeros(k), R, size=n, method="eigh")
    u = norm.cdf(z)

    out = {}
    for j, feat in enumerate(ALL_FEATURES):
        stats = desc.loc[feat]
        x = quantile_map(u[:, j], stats)
        sd = x.std(ddof=1)
        if sd > 0:
            x = (x - x.mean()) / sd * stats["sd"] + stats["mean"]
        else:
            x = np.full(n, stats["mean"], float)
        lo, hi = stats["min"], stats["max"]
        if feat in UNIT_INTERVAL_FEATURES:
            lo, hi = max(lo, 0.0), min(hi, 1.0)
        out[feat] = np.clip(x, lo, hi)
    return pd.DataFrame(out)


def build_speaker_frame(rng: np.random.Generator, age_mean: float,
                        age_sd: float, age_min: int, age_max: int) -> pd.DataFrame:
    """Conversation, speaker and person ids plus sex and age, at the published counts."""
    # 25 speakers share 71 records; distribute them as evenly as the totals allow.
    base, extra = divmod(N_REPEAT_RECORDS, N_REPEAT_SPEAKERS)
    counts = [base + (1 if i < extra else 0) for i in range(N_REPEAT_SPEAKERS)]
    counts += [1] * (N_SPEAKERS - N_REPEAT_SPEAKERS)
    if sum(counts) != N_RECORDS:
        raise SystemExit(f"speaker counts sum to {sum(counts)}, want {N_RECORDS}")

    person_ids = []
    for i, c in enumerate(counts):
        person_ids.extend([f"SYN_P{i + 1:03d}"] * c)
    person_ids = np.array(person_ids)

    # 66 conversations yield 120 records, so not every conversation contributes both
    # speakers: 54 contribute two and 12 contribute one.
    n_two = N_RECORDS - N_CONVERSATIONS
    n_one = N_CONVERSATIONS - n_two
    conv_ids = []
    for c in range(n_two):
        conv_ids.extend([f"SYN_C{c + 1:03d}"] * 2)
    for c in range(n_two, n_two + n_one):
        conv_ids.append(f"SYN_C{c + 1:03d}")
    conv_ids = np.array(conv_ids)
    if len(conv_ids) != N_RECORDS:
        raise SystemExit(f"built {len(conv_ids)} records, want {N_RECORDS}")

    # Assign people to slots so that nobody appears twice in the same conversation, which
    # the real data cannot contain either.
    for _ in range(1000):
        person_ids = person_ids[rng.permutation(N_RECORDS)]
        pairs = pd.DataFrame({"conv": conv_ids, "person": person_ids})
        if not pairs.duplicated(["conv", "person"]).any():
            break
    else:
        raise SystemExit("could not place speakers without a within-conversation "
                         "duplicate; try another --seed")

    # speaker_id is a within-conversation label in CEJC, so it repeats across conversations.
    speaker_ids = []
    seen: dict[str, int] = {}
    for cid in conv_ids:
        seen[cid] = seen.get(cid, 0) + 1
        speaker_ids.append(f"IC{seen[cid]:02d}")

    # Sex is fixed per person, so assign it at the person level and let the record split
    # land as close to the published 66 / 54 as the grouping allows.
    persons = pd.unique(person_ids)
    sizes = pd.Series(person_ids).value_counts()
    gender_by_person: dict[str, str] = {}
    female_records = 0
    for pid in rng.permutation(persons):
        if female_records + sizes[pid] <= N_FEMALE_RECORDS:
            gender_by_person[pid] = "F"
            female_records += sizes[pid]
        else:
            gender_by_person[pid] = "M"
    age_by_person = {
        pid: int(np.clip(round(rng.normal(age_mean, age_sd)), age_min, age_max))
        for pid in persons
    }

    return pd.DataFrame({
        "conversation_id": conv_ids,
        "speaker_id": speaker_ids,
        "cejc_person_id": person_ids,
        "gender": [gender_by_person[p] for p in person_ids],
        "age": [age_by_person[p] for p in person_ids],
    })


def build_outcome(X: np.ndarray, rho: float, rng: np.random.Generator,
                  y_mean: float, y_sd: float) -> np.ndarray:
    """Draw the outcome. rho = 0 makes it independent of X.

    With rho > 0 the outcome is a random linear combination of the standardised features
    plus noise, scaled so that Var(signal) / Var(y) = rho^2. That gives a dataset where
    the pipeline should detect an association of roughly known strength.
    """
    n = X.shape[0]
    if rho <= 0:
        y = rng.standard_normal(n)
    else:
        Z = X.copy()
        for j in range(Z.shape[1]):
            col = Z[:, j]
            sd = col.std()
            Z[:, j] = (col - col.mean()) / sd if sd > 0 else 0.0
        b = rng.standard_normal(Z.shape[1])
        signal = Z @ b
        sd_sig = signal.std()
        signal = signal / sd_sig if sd_sig > 0 else signal
        noise_sd = np.sqrt((1.0 - rho ** 2) / rho ** 2)
        y = signal + rng.standard_normal(n) * noise_sd
    sd = y.std(ddof=1)
    y = (y - y.mean()) / sd if sd > 0 else y
    return y * y_sd + y_mean


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fig_dir", default="reports/paper_figs_v2",
                    help="where the published tables are")
    ap.add_argument("--out_dir", default="artifacts/synthetic",
                    help="output root; kept separate from the real artifacts tree")
    ap.add_argument("--n", type=int, default=N_RECORDS)
    ap.add_argument("--rho", type=float, default=0.0,
                    help="population multiple correlation to inject; 0 (the default) "
                         "makes the outcome independent of the features")
    ap.add_argument("--y_mean", type=float, default=3.0,
                    help="mean of the synthetic trait score (IPIP-NEO-120 is on a 1-5 scale)")
    ap.add_argument("--y_sd", type=float, default=0.4)
    ap.add_argument("--age_mean", type=float, default=45.0,
                    help="NOT taken from the corpus; the manuscript does not publish age "
                         "descriptives, so this default is arbitrary")
    ap.add_argument("--age_sd", type=float, default=15.0)
    ap.add_argument("--age_min", type=int, default=18)
    ap.add_argument("--age_max", type=int, default=85)
    ap.add_argument("--teacher", default="ensemble")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    fig_dir = REPO_ROOT / args.fig_dir
    out_dir = REPO_ROOT / args.out_dir
    datasets_dir = out_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    desc = parse_descriptives(fig_dir / "tab_descriptive_stats_full.tex")
    R_raw = parse_correlations(fig_dir / "tab_corr_matrix.tex")
    R, adjustment = nearest_psd_correlation(R_raw)
    if adjustment > 0:
        print(f"correlation matrix projected onto the nearest PSD matrix "
              f"(largest entry change {adjustment:.4f}); the published matrix is printed "
              f"to two decimals, which can leave it slightly indefinite")

    rng = np.random.default_rng(args.seed)

    feats = build_features(desc, R, args.n, rng)
    speakers = build_speaker_frame(rng, args.age_mean, args.age_sd,
                                   args.age_min, args.age_max)
    if args.n != N_RECORDS:
        # The speaker structure is fixed to the published counts, so a different n needs
        # the frame resampled rather than reused.
        speakers = speakers.sample(args.n, replace=args.n > N_RECORDS,
                                   random_state=args.seed).reset_index(drop=True)

    X = feats[ALL_FEATURES].to_numpy(float)

    meta = speakers.copy()
    meta["age_band"] = pd.NA
    meta["age_mid"] = meta["age"]
    for col in ["birthplace", "residence", "occupation", "relationship"]:
        meta[col] = pd.NA
    meta_path = out_dir / "synthetic_speaker_metadata.tsv"
    meta.to_csv(meta_path, sep="\t", index=False)

    features_path = out_dir / "synthetic_features.parquet"
    features_out = pd.concat([speakers[["conversation_id", "speaker_id"]], feats], axis=1)
    for col in PASSTHROUGH_NAN_COLS:
        features_out[col] = np.nan
    features_out.to_parquet(features_path, index=False)

    written = []
    for trait in TRAITS:
        y = build_outcome(X, args.rho, rng, args.y_mean, args.y_sd)
        base = pd.concat([speakers[["conversation_id", "speaker_id"]], feats], axis=1)
        base.insert(2, f"Y_{trait}", y)
        for col in PASSTHROUGH_NAN_COLS:
            base[col] = np.nan

        xy = datasets_dir / f"cejc_home2_hq1_XY_{trait}only_{args.teacher}.parquet"
        base.to_parquet(xy, index=False)
        written.append(xy)

        xyb = base.copy()
        xyb["confound_gender"] = speakers["gender"].map({"M": 0, "F": 1}).astype(float)
        xyb["confound_age"] = speakers["age"].astype(float)
        xybp = datasets_dir / f"cejc_home2_hq1_XYB_{trait}only_{args.teacher}.parquet"
        xyb.to_parquet(xybp, index=False)
        written.append(xybp)

    provenance = {
        "purpose": "structural stand-in for the real analysis inputs, which the CEJC "
                   "licence does not permit redistributing",
        "inputs": [
            f"{args.fig_dir}/tab_descriptive_stats_full.tex",
            f"{args.fig_dir}/tab_corr_matrix.tex",
            "record and speaker counts stated in the manuscript (section 2.1)",
        ],
        "not_derived_from_corpus": [
            "the age distribution (see --age_mean / --age_sd; the manuscript does not "
            "publish age descriptives)",
            "the trait score location and scale (see --y_mean / --y_sd)",
        ],
        "reproduces_reported_results": False,
        "outcome": ("independent of the features" if args.rho <= 0
                    else f"linear in the features at rho={args.rho}"),
        "n_records": int(args.n),
        "n_predictors_XY": len(ALL_FEATURES),
        "n_predictors_XYB": len(ALL_FEATURES) + 2,
        "correlation_psd_adjustment": adjustment,
        "seed": args.seed,
    }
    (out_dir / "PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\nwrote {len(written)} dataset files to {datasets_dir.relative_to(REPO_ROOT)}")
    print(f"  {features_path.relative_to(REPO_ROOT)}")
    print(f"  {meta_path.relative_to(REPO_ROOT)}")
    print(f"  {(out_dir / 'PROVENANCE.json').relative_to(REPO_ROOT)}")
    print("\nrun the analyses against it with:")
    print(f"  make analysis DATASET_DIR={args.out_dir}/datasets \\")
    print(f"                META_TSV={args.out_dir}/synthetic_speaker_metadata.tsv")
    if args.rho <= 0:
        print("\nthe outcome is independent of the features, so every test should come "
              "out non-significant. That is the expected result, not a failure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
