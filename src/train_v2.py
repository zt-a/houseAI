#!/usr/bin/env python3
import json

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from config import DATA_PROCESSED, MODELS, TARGET


def main():
    df = pd.read_parquet(DATA_PROCESSED / "apartments_bishkek_raw.parquet")
    print(f"Data: {len(df)} rows")

    # Новые фичи
    df["price_log"] = np.log1p(df[TARGET])
    df["age"] = 2026 - pd.to_numeric(df["year_built_num"], errors="coerce")
    df["age"] = df["age"].clip(lower=0, upper=80)
    df["floor_ratio"] = df["floor_num"] / df["floors_total_num"]
    df["floor_ratio"] = df["floor_ratio"].clip(lower=0, upper=1)
    df["area_per_room"] = df["total_area"] / df["rooms_num"].replace(0, np.nan)
    df["area_per_room"] = df["area_per_room"].fillna(df["total_area"])
    df["living_ratio"] = df["living_area_num"] / df["total_area"]
    df["kitchen_ratio"] = df["kitchen_area_num"] / df["total_area"]
    df["is_first_floor"] = (df["floor_num"] == 1).astype(int)
    df["is_last_floor"] = (df["floor_num"] == df["floors_total_num"]).astype(int)
    df["is_studio"] = (df["rooms_num"] == 0).astype(int)

    num_features = [
        "total_area",
        "rooms_num",
        "floor_num",
        "floors_total_num",
        "year_built_num",
        "ceiling_height_num",
        "living_area_num",
        "kitchen_area_num",
        "dist_bishkek_center_km",
        "age",
        "floor_ratio",
        "area_per_room",
        "living_ratio",
        "kitchen_ratio",
        "is_first_floor",
        "is_last_floor",
        "is_studio",
    ]

    cat_features = [
        "district_cat",
        "condition_cat",
        "seller_type_cat",
        "documents_cat",
        "heating_cat",
        "gas_cat",
        "water_cat",
        "electricity_cat",
        "canalization_cat",
        "furniture_cat",
        "toilet_cat",
        "security_cat",
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=123)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1765, random_state=123)

    cat_indices = [i for i, c in enumerate(feature_cols) if c in cat_features]

    model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.03,
        depth=7,
        l2_leaf_reg=5,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=123,
        verbose=300,
        early_stopping_rounds=150,
        bagging_temperature=1,
        random_strength=1,
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_indices,
        eval_set=(X_val, y_val),
        use_best_model=True,
    )

    def pred_price(X):
        return np.expm1(model.predict(X))

    y_pred_test = pred_price(X_test)
    y_test_real = np.expm1(y_test)

    mae = mean_absolute_error(y_test_real, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_test))
    r2 = r2_score(y_test_real, y_pred_test)
    mape = np.mean(np.abs((y_test_real - y_pred_test) / y_test_real)) * 100

    print("\n=== Test metrics (improved) ===")
    print(f"MAE: {mae:,.0f} сом")
    print(f"RMSE: {rmse:,.0f} сом")
    print(f"MAPE: {mape:.2f}%")
    print(f"R2: {r2:.4f}")

    model_path = MODELS / "catboost_apartments_bishkek_v2.cbm"
    model.save_model(str(model_path))

    meta = {
        "feature_cols": feature_cols,
        "cat_features": cat_features,
        "num_features": num_features,
        "feat_features": feat_features,
        "cat_indices": cat_indices,
        "metrics": {"mae": float(mae), "rmse": float(rmse), "mape": float(mape), "r2": float(r2)},
        "version": "v2",
    }
    with open(MODELS / "catboost_apartments_bishkek_v2_meta.json", "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {model_path}")


if __name__ == "__main__":
    main()
