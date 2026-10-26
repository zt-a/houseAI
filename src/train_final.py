#!/usr/bin/env python3
import json

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from config import DATA_PROCESSED, MODELS, TARGET


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA_PROCESSED / "apartments_bishkek_v3.parquet")
    print(f"Data: {len(df)} rows")

    # Удалим топ выбросы по ошибке из прошлого анализа + экстремальные большие квартиры с аномальной ценой
    # Критерий - очень большие ошибки
    df = df.copy()
    df["price_m2"] = df[TARGET] / df["total_area"]
    
    # Жёстче чистим
    # Удалим объекты где price_m2 очень высокий при относительно обычной площади или наоборот
    # Проще - удалим 2-3 самых явных выбросов из топа
    drop_slugs = ["62777276a2163a2a358b0-54473163", "356786369cf6c933bda54-32226356", "36098406a97b43a989c25-36318215"]
    df = df[~df["url_slug"].isin(drop_slugs)].copy()
    print(f"After removing big outliers: {len(df)} rows")

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

    # Target encoding на всех данных
    global_mean_log = df["price_log"].mean()
    district_means = df.groupby("district_clean")["price_log"].mean()
    df["district_te"] = df["district_clean"].map(district_means).fillna(global_mean_log)

    num_features = [
        "total_area", "rooms_num", "floor_num", "floors_total_num",
        "year_built_num", "ceiling_height_num", "living_area_num", "kitchen_area_num",
        "dist_bishkek_center_km", "age", "age2", "area2", "floor_ratio", "floor_ratio2",
        "area_per_room", "living_ratio", "kitchen_ratio",
        "is_first_floor", "is_last_floor", "is_studio",
        "area_x_rooms", "area_x_dist", "district_te",
    ]
    cat_features = [
        "district_clean", "geo_cluster",
        "condition_cat", "seller_type_cat", "documents_cat",
        "heating_cat", "gas_cat", "water_cat", "electricity_cat", "canalization_cat",
        "furniture_cat", "toilet_cat", "security_cat",
    ]
    feat_features = [c for c in df.columns if c.startswith("feat_")]
    feature_cols = num_features + cat_features + feat_features
    feature_cols = [c for c in feature_cols if c in df.columns]
    print(f"Features: {len(feature_cols)}")

    X = df[feature_cols].copy()
    for c in num_features:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(X[c].median())
    for c in cat_features:
        if c in X.columns:
            X[c] = X[c].fillna("unknown").astype(str)

    y = df["price_log"].copy()

    X_lgb = X.copy()
    for c in cat_features:
        if c in X_lgb.columns:
            X_lgb[c] = X_lgb[c].astype('category')

    print("\n=== Training CatBoost on full data ===")
    cat_indices = [i for i, c in enumerate(feature_cols) if c in cat_features]
    model_cb = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=4,
        loss_function="RMSE",
        random_seed=9999,
        verbose=300,
    )
    model_cb.fit(X, y, cat_features=cat_indices)

    print("\n=== Training LightGBM on full data ===")
    model_lgb = LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.02,
        max_depth=9,
        num_leaves=80,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.2,
        reg_lambda=1.2,
        random_state=9999,
        n_jobs=-1,
        verbose=-1,
    )
    model_lgb.fit(X_lgb, y)

    model_cb.save_model(str(MODELS / "catboost_apartments_bishkek_final.cbm"))
    import joblib
    joblib.dump(model_lgb, MODELS / "lgbm_apartments_bishkek_final.pkl")

    meta = {
        "feature_cols": feature_cols,
        "cat_features": cat_features,
        "num_features": num_features,
        "feat_features": feat_features,
        "cat_indices": cat_indices,
        "blend_w_cb": 0.707,
        "trained_on": "full_cleaned_v3_no_outliers",
        "rows": len(df),
    }
    with open(MODELS / "ensemble_apartments_bishkek_final_meta.json", "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\nSaved final models")


if __name__ == "__main__":
    main()
