#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--xy_parquet", required=True)
    ap.add_argument("--y_col", default="Y_C")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_boot", type=int, default=500)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude_cols", default="", help="comma-separated columns to exclude from X")
    args=ap.parse_args()

    df=pd.read_parquet(args.xy_parquet).reset_index(drop=True)

    excl=set([c.strip() for c in args.exclude_cols.split(",") if c.strip()])
    drop=set(["conversation_id","speaker_id",args.y_col])
    Xcols=[c for c in df.columns if (c not in drop) and (c not in excl)]
    X=df[Xcols].apply(pd.to_numeric, errors="coerce")
    y=df[args.y_col].astype(float).values
    n=len(df)

    out_dir=Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    rng=np.random.default_rng(args.seed)

    # model: ridge with internal CV for alpha
    alphas=np.logspace(-3, 3, 25)
    model=Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("ridge", RidgeCV(alphas=alphas, cv=5)),
    ])

    coef_mat=np.zeros((args.n_boot, X.shape[1]), dtype=float)
    alpha_vec=np.zeros(args.n_boot, dtype=float)
    topk_counts=np.zeros(X.shape[1], dtype=int)

    for b in range(args.n_boot):
        idx=rng.integers(0, n, size=n)  # bootstrap sample
        Xb=X.iloc[idx]
        yb=y[idx]
        model.fit(Xb, yb)
        ridge=model.named_steps["ridge"]
        coef=ridge.coef_.astype(float)
        coef_mat[b,:]=coef
        alpha_vec[b]=float(ridge.alpha_)

        # top-k by abs coef
        top_idx=np.argsort(np.abs(coef))[::-1][:args.topk]
        topk_counts[top_idx]+=1

    feats=np.array(Xcols)
    # stability metrics
    sign_mean=np.mean(np.sign(coef_mat), axis=0)          # close to 1/-1 => sign stable
    sign_agree=np.mean(np.sign(coef_mat)==np.sign(np.mean(coef_mat,axis=0)), axis=0)  # agree with average sign
    coef_mean=np.mean(coef_mat, axis=0)
    coef_std=np.std(coef_mat, axis=0)
    coef_q05=np.quantile(coef_mat, 0.05, axis=0)
    coef_q95=np.quantile(coef_mat, 0.95, axis=0)

    res=pd.DataFrame({
        "feature": feats,
        "coef_mean": coef_mean,
        "coef_std": coef_std,
        "coef_q05": coef_q05,
        "coef_q95": coef_q95,
        "sign_mean": sign_mean,
        "sign_agree_rate": sign_agree,
        "topk_rate": topk_counts / args.n_boot,
    }).sort_values(["topk_rate","sign_agree_rate","coef_mean"], ascending=[False,False,False])

    res.to_csv(out_dir/"bootstrap_summary.tsv", sep="\t", index=False)
    pd.Series(alpha_vec).describe().to_csv(out_dir/"alpha_describe.tsv", sep="\t")

    print("OK:", (out_dir/"bootstrap_summary.tsv").as_posix())
    print(res.head(20).to_string(index=False))
    print("\nOK:", (out_dir/"alpha_describe.tsv").as_posix())
    print(pd.Series(alpha_vec).describe().to_string())

if __name__=="__main__":
    main()
