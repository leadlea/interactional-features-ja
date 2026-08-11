#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

def pearsonr(a, b) -> float:
    a=np.asarray(a,float); b=np.asarray(b,float)
    a=a-a.mean(); b=b-b.mean()
    den=np.sqrt((a*a).sum())*np.sqrt((b*b).sum())
    return float((a*b).sum()/den) if den!=0 else float("nan")

def cv_ridge_r(X, y, folds, seed, alpha):
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    rs=[]
    for tr, te in kf.split(X):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]

        imp = SimpleImputer(strategy="median")
        Xtr = imp.fit_transform(Xtr)
        Xte = imp.transform(Xte)

        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xte = sc.transform(Xte)

        m = Ridge(alpha=alpha, random_state=seed)
        m.fit(Xtr, ytr)
        yh = m.predict(Xte)
        rs.append(pearsonr(yte, yh))
    return float(np.mean(rs))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--xy_parquet", required=True)
    ap.add_argument("--y_col", required=True)
    ap.add_argument("--exclude_cols", default="")
    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--n_perm", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, required=True)
    args=ap.parse_args()

    df=pd.read_parquet(args.xy_parquet)

    # inf→NaN にしておく（rate系の事故対策）
    df = df.replace([np.inf, -np.inf], np.nan)

    excl=set([c.strip() for c in args.exclude_cols.split(",") if c.strip()])
    excl |= {"conversation_id","speaker_id"}

    if args.y_col not in df.columns:
        raise SystemExit(f"y_col not found: {args.y_col}")

    y=df[args.y_col].astype(float).to_numpy()

    feat_cols=[c for c in df.columns if c not in excl and c!=args.y_col]
    X=df[feat_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    # 目的変数NaN行があれば落とす（基本ないはずだが保険）
    ok = ~np.isnan(y)
    X = X[ok]
    y = y[ok]

    r_obs=cv_ridge_r(X,y,args.cv_folds,args.seed,args.alpha)

    rng=np.random.default_rng(args.seed)
    r_perm=np.empty(args.n_perm,float)
    for i in range(args.n_perm):
        yp=rng.permutation(y)
        r_perm[i]=cv_ridge_r(X,yp,args.cv_folds,args.seed,args.alpha)

    p=(np.sum(np.abs(r_perm)>=abs(r_obs))+1.0)/(args.n_perm+1.0)
    print(f"alpha={args.alpha}")
    print(f"r_obs={r_obs:.3f}")
    print(f"p(|r|)={p:.4f}  (n_perm={args.n_perm})")

if __name__=="__main__":
    main()
