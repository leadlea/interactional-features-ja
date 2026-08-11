#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

def pearsonr(a,b):
    a=np.asarray(a); b=np.asarray(b)
    a=a-np.nanmean(a); b=b-np.nanmean(b)
    denom=np.sqrt(np.nansum(a*a))*np.sqrt(np.nansum(b*b))
    return float(np.nansum(a*b)/denom) if denom>0 else float("nan")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--xy_parquet", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--min_pairs_total", type=int, default=20)
    ap.add_argument("--min_text_len", type=int, default=800)
    ap.add_argument("--cv_folds", type=int, default=5)
    args=ap.parse_args()

    df=pd.read_parquet(args.xy_parquet).copy()

    keep = (df["n_pairs_total"]>=args.min_pairs_total) & (df["FILL_text_len"]>=args.min_text_len)
    d=df[keep].reset_index(drop=True)

    if len(d) < 5:
        raise SystemExit(f"Too few rows after filter: n={len(d)}. Lower thresholds (min_pairs_total/min_text_len).")

    y_cols=[c for c in d.columns if c.startswith("Y_")]
    id_cols=["conversation_id","speaker_id"]
    drop=set(id_cols + y_cols)

    X=d[[c for c in d.columns if c not in drop]].apply(pd.to_numeric, errors="coerce")

    out_dir=Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    folds=min(args.cv_folds, len(d))
    if folds < 2:
        raise SystemExit(f"cv_folds too large for n={len(d)} (folds={folds}).")

    alphas=np.logspace(-3, 3, 25)
    model=Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("ridge", RidgeCV(alphas=alphas, cv=folds)),
    ])
    cv = KFold(n_splits=folds, shuffle=True, random_state=42)

    rows=[]
    top_tables=[]
    for yc in y_cols:
        y=d[yc].astype(float).values

        yhat=cross_val_predict(model, X, y, cv=cv)
        r2=float(r2_score(y, yhat))
        r=float(pearsonr(y, yhat))

        model.fit(X,y)
        ridge=model.named_steps["ridge"]
        alpha=float(ridge.alpha_)
        coef=ridge.coef_.astype(float)

        trait=yc.replace("Y_","")
        coef_df=pd.DataFrame({"feature":X.columns, "coef":coef, "abs":np.abs(coef)}).sort_values("abs", ascending=False)
        coef_df.insert(0,"trait",trait)
        coef_df["alpha"]=alpha
        coef_df.to_csv(out_dir/f"coef_{trait}.tsv", sep="\t", index=False)

        pred=pd.DataFrame({"conversation_id":d["conversation_id"],"speaker_id":d["speaker_id"],"y":y,"yhat_cv":yhat})
        pred.to_csv(out_dir/f"pred_{trait}.tsv", sep="\t", index=False)

        rows.append({"trait":trait, "n":len(d), "r2_cv":r2, "pearson_cv":r, "alpha":alpha})
        top_tables.append(coef_df.head(15)[["trait","feature","coef","alpha"]])

    summary=pd.DataFrame(rows).sort_values("trait")
    summary.to_csv(out_dir/"summary.tsv", sep="\t", index=False)

    top=pd.concat(top_tables, ignore_index=True)
    top.to_csv(out_dir/"top15_features_all_traits.tsv", sep="\t", index=False)

    print("OK:", (out_dir/"summary.tsv").as_posix())
    print(summary.to_string(index=False))
    print("\nOK:", (out_dir/"top15_features_all_traits.tsv").as_posix())
    print(top.head(25).to_string(index=False))

if __name__=="__main__":
    main()
