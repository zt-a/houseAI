#!/usr/bin/env python3
import json

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

from config import DATA_PROCESSED, MODELS, TARGET


def target_encode_kfold(df, col, target, n_splits=5, seed=42):
    """Target encoding с CV чтобы избежать утечки"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    encoded = pd.Series([np.nan] * len(df), index=df.index)
    for train_idx, val_idx in kf.split(df):
        means = df.iloc[train_idx].groupby(col)[target].mean()
        global_mean = df.iloc[train_idx][target].mean()
        encoded.iloc[val_idx] = df.iloc[val_idx][col].map(means).fillna(global_mean)
    return encoded


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA_PROCESSED / "apartments_bishkek_v3.parquet")
    print(f"Data: {len(df)} rows")

    df["price_log"] = np.log1p(df[TARGET])

    # Новые фичи
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

    # Target encoding для district_clean
    print("Target encoding...")
    df["district_te"] = target_encode_kfold(df, "district_clean", "price_log", n_splits=5)

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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=777)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1765, random_state=777)

    cat_indices = [i for i, c in enumerate(feature_cols) if c in cat_features]

    print("\n=== Training CatBoost ===")
    model_cb = CatBoostRegressor(
        iterations=4000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=4,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=777,
        verbose=400,
        early_stopping_rounds=200,
        bagging_temperature=0.8,
        random_strength=0.8,
    )
    model_cb.fit(X_train, y_train, cat_features=cat_indices, eval_set=(X_val, y_val), use_best_model=True)

    print("\n=== Training LightGBM ===")
    # Для LGBM категории как category
    X_train_lgb = X_train.copy()
    X_val_lgb = X_val.copy()
    X_test_lgb = X_test.copy()
    for c in cat_features:
        if c in X_train_lgb.columns:
            X_train_lgb[c] = X_train_lgb[c].astype('category')
            X_val_lgb[c] = X_val_lgb[c].astype('category')
            X_test_lgb[c] = X_test_lgb[c].astype('category')

    model_lgb = LGBMRegressor(
        n_estimators=4000,
        learning_rate=0.02,
        max_depth=8,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1,
        reg_lambda=1,
        random_state=777,
        n_jobs=-1,
        verbose=-1,
    )
    model_lgb.fit(X_train_lgb, y_train, eval_set=[(X_val_lgb, y_val)], callbacks=[])

    def pred_price_blend(X_cb, X_lgb, w_cb=0.7):
        p_cb = np.expm1(model_cb.predict(X_cb))
        p_lgb = np.expm1(model_lgb.predict(X_lgb))
        return w_cb * p_cb + (1 - w_cb) * p_lgb

    y_test_real = np.expm1(y_test)
    y_pred = pred_price_blend(X_test, X_test_lgb, w_cb=0.7)

    mae = mean_absolute_error(y_test_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred))
    r2 = r2_score(y_test_real, y_pred)
    mape = np.mean(np.abs((y_test_real - y_pred) / y_test_real)) * 100

    print("\n=== Test metrics (v3 blend) ===")
    print(f"MAE: {mae:,.0f} сом")
    print(f"RMSE: {rmse:,.0f} сом")
    print(f"MAPE: {mape:.2f}%")
    print(f"R2: {r2:.4f}")

    model_cb.save_model(str(MODELS / "catboost_apartments_bishkek_v3.cbm"))
    import joblib
    joblib.dump(model_lgb, MODELS / "lgbm_apartments_bishkek_v3.pkl")

    meta = {
        "feature_cols": feature_cols,
        "cat_features": cat_features,
        "num_features": num_features,
        "feat_features": feat_features,
        "cat_indices": cat_indices,
        "blend_w_cb": 0.7,
        "metrics": {"mae": float(mae), "rmse": float(rmse), "mape": float(mape), "r2": float(r2)},
        "version": "v3_blend",
    }
    with open(MODELS / "ensemble_apartments_bishkek_v3_meta.json", "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\nSaved v3 models")


if __name__ == "__main__":
    main()
