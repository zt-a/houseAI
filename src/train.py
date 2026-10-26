#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib

from config import DATA_PROCESSED, MODELS, TARGET


def main():
    MODELS.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA_PROCESSED / "apartments_bishkek_raw.parquet")
    print(f"Data: {len(df)} rows")

    # Целевая - логарифм
    df["price_log"] = np.log1p(df[TARGET])

    # Фичи
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
    # оставим только те что есть
    feature_cols = [c for c in feature_cols if c in df.columns]
    print(f"Features: {len(feature_cols)} (num={len([c for c in feature_cols if c in num_features])}, cat={len([c for c in feature_cols if c in cat_features])}, feat={len([c for c in feature_cols if c in feat_features])})")

    # Заполним пропуски
    X = df[feature_cols].copy()
    for c in num_features:
        if c in X.columns:
            X[c] = X[c].fillna(X[c].median())
    for c in cat_features:
        if c in X.columns:
            X[c] = X[c].fillna("unknown")

    y = df["price_log"].copy()

    # Сплит
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1765, random_state=42)  # ~0.15

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # CatBoost - определим cat индексы
    cat_indices = [i for i, c in enumerate(feature_cols) if c in cat_features]

    model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42,
        verbose=200,
        early_stopping_rounds=100,
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_indices,
        eval_set=(X_val, y_val),
        use_best_model=True,
    )

    # Предсказания в лог шкале -> обратно
    def pred_price(X):
        p_log = model.predict(X)
        return np.expm1(p_log)

    y_pred_test = pred_price(X_test)
    y_test_real = np.expm1(y_test)

    mae = mean_absolute_error(y_test_real, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_test))
    r2 = r2_score(y_test_real, y_pred_test)
    mape = np.mean(np.abs((y_test_real - y_pred_test) / y_test_real)) * 100

    print("\n=== Test metrics ===")
    print(f"MAE: {mae:,.0f} сом")
    print(f"RMSE: {rmse:,.0f} сом")
    print(f"MAPE: {mape:.2f}%")
    print(f"R2: {r2:.4f}")

    # Сохраняем
    model_path = MODELS / "catboost_apartments_bishkek.cbm"
    model.save_model(str(model_path))
    
    meta = {
        "feature_cols": feature_cols,
        "cat_features": cat_features,
        "num_features": num_features,
        "feat_features": feat_features,
        "cat_indices": cat_indices,
        "metrics": {"mae": float(mae), "rmse": float(rmse), "mape": float(mape), "r2": float(r2)},
    }
    with open(MODELS / "catboost_apartments_bishkek_meta.json", "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nSaved model: {model_path}")


if __name__ == "__main__":
    main()
