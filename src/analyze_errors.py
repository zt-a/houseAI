#!/usr/bin/env python3
import json

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold

from config import DATA_PROCESSED, MODELS, TARGET


def target_encode_kfold(df, col, target, n_splits=5, seed=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    encoded = pd.Series([np.nan] * len(df), index=df.index)
    for train_idx, val_idx in kf.split(df):
        means = df.iloc[train_idx].groupby(col)[target].mean()
        global_mean = df.iloc[train_idx][target].mean()
        encoded.iloc[val_idx] = df.iloc[val_idx][col].map(means).fillna(global_mean)
    return encoded


def main():
    df = pd.read_parquet(DATA_PROCESSED / "apartments_bishkek_v3.parquet")
    print(f"Data: {len(df)} rows")

    model_cb = CatBoostRegressor()
    model_cb.load_model(str(MODELS / "catboost_apartments_bishkek_v3.cbm"))
    model_lgb = __import__('joblib').load(MODELS / "lgbm_apartments_bishkek_v3.pkl")

    with open(MODELS / "ensemble_apartments_bishkek_v3_meta.json") as f:
        meta = json.load(f)

    feature_cols = meta["feature_cols"]
    cat_features = meta["cat_features"]
    num_features = meta["num_features"]
    w_cb = meta["blend_w_cb"]

    df["price_log"] = np.log1p(df[TARGET])
    df["age"] = 2026 - pd.to_numeric(df["year_built_num"], errors="coerce")
    df["age"] = df["age"].clip(lower=0, upper=80)
    df["age2"] = df["age"] ** 2
    df["area2"] = df["total_area"] ** 2
    df["floor_ratio"] = df["floor_num"] / df["floors_total_num"]
    df["floor_ratio"] = df["floor_ratio"].clip(lower=0, upper=1)
    df["floor_ratio2"] = df["floor_ratio"] ** 2
    df["area_per_room"] = df["total_area"] / df["rooms_num"].replace(0, np.nan)
    df["area_per_room"] = df["area_per_room"].fillna(df["total_area"])
    df["living_ratio"] = df["living_area_num"] / df["total_area"]
    df["kitchen_ratio"] = df["kitchen_area_num"] / df["total_area"]
    df["is_first_floor"] = (df["floor_num"] == 1).astype(int)
    df["is_last_floor"] = (df["floor_num"] == df["floors_total_num"]).astype(int)
    df["is_studio"] = (df["rooms_num"] == 0).astype(int)
    df["area_x_rooms"] = df["total_area"] * df["rooms_num"].fillna(1)
    df["area_x_dist"] = df["total_area"] / (df["dist_bishkek_center_km"] + 1)

    df["district_te"] = target_encode_kfold(df, "district_clean", "price_log", n_splits=5)

    X = df[feature_cols].copy()
    for c in num_features:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")
            med = X[c].median()
            X[c] = X[c].fillna(med)
    if "district_te" in X.columns:
        X["district_te"] = pd.to_numeric(X["district_te"], errors="coerce").fillna(X["district_te"].median())
    for c in cat_features:
        if c in X.columns:
            X[c] = X[c].fillna("unknown").astype(str)

    X_lgb = X.copy()
    for c in cat_features:
        if c in X_lgb.columns:
            X_lgb[c] = X_lgb[c].astype('category')

    p_cb = np.expm1(model_cb.predict(X))
    p_lgb = np.expm1(model_lgb.predict(X_lgb))
    pred = w_cb * p_cb + (1 - w_cb) * p_lgb

    df["pred"] = pred
    df["err_abs"] = abs(df["pred"] - df[TARGET])
    df["ape"] = df["err_abs"] / df[TARGET] * 100

    print("\n=== По районам (топ по ошибке) ===")
    g = df.groupby("district_clean").agg(
        cnt=("pred", "count"),
        mae=("err_abs", "mean"),
        mape=("ape", "mean"),
        median_price=("price_som", "median"),
    ).sort_values("mae", ascending=False)
    print(g.head(15).to_string())

    print("\n=== По площади ===")
    df["area_bin"] = pd.cut(df["total_area"], bins=[0,40,60,80,100,150,250])
    g2 = df.groupby("area_bin").agg(cnt=("pred","count"), mae=("err_abs","mean"), mape=("ape","mean"))
    print(g2.to_string())

    print("\n=== Топ 10 самых больших ошибок ===")
    top = df.nlargest(10, "err_abs")[["url_slug", "district_clean", "total_area", "rooms", "price_som", "pred", "err_abs", "ape"]]
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()
