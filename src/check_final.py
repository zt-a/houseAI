#!/usr/bin/env python3
import json

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from config import DATA_PROCESSED, MODELS, TARGET


def main():
    df = pd.read_parquet(DATA_PROCESSED / "apartments_bishkek_v3.parquet")
    drop_slugs = ["62777276a2163a2a358b0-54473163", "356786369cf6c933bda54-32226356", "36098406a97b43a989c25-36318215"]
    df = df[~df["url_slug"].isin(drop_slugs)].copy()

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

    global_mean_log = df["price_log"].mean()
    district_means = df.groupby("district_clean")["price_log"].mean()
    df["district_te"] = df["district_clean"].map(district_means).fillna(global_mean_log)

    model_cb = CatBoostRegressor()
    model_cb.load_model(str(MODELS / "catboost_apartments_bishkek_final.cbm"))
    model_lgb = __import__('joblib').load(MODELS / "lgbm_apartments_bishkek_final.pkl")

    with open(MODELS / "ensemble_apartments_bishkek_final_meta.json") as f:
        meta = json.load(f)

    feature_cols = meta["feature_cols"]
    cat_features = meta["cat_features"]
    num_features = meta["num_features"]
    w_cb = meta["blend_w_cb"]

    X = df[feature_cols].copy()
    for c in num_features:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(X[c].median())
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

    print(f"MAE (на чистых данных): {df['err_abs'].mean():,.0f} сом")
    print(f"MAPE: {df['ape'].mean():.2f}%")
    print(f"Median APE: {df['ape'].median():.2f}%")


if __name__ == "__main__":
    main()
