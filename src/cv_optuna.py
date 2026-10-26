#!/usr/bin/env python3
import json

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

import optuna
from config import DATA_PROCESSED, MODELS, TARGET


def target_encode_fold(train_df, val_df, col, target):
    means = train_df.groupby(col)[target].mean()
    global_mean = train_df[target].mean()
    return val_df[col].map(means).fillna(global_mean)


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    df = pd.read_parquet(DATA_PROCESSED / "apartments_bishkek_v3.parquet")
    print(f"Data: {len(df)} rows")

    df["price_log"] = np.log1p(df[TARGET])

    # Фичи
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

    num_features = [
        "total_area", "rooms_num", "floor_num", "floors_total_num",
        "year_built_num", "ceiling_height_num", "living_area_num", "kitchen_area_num",
        "dist_bishkek_center_km", "age", "age2", "area2", "floor_ratio", "floor_ratio2",
        "area_per_room", "living_ratio", "kitchen_ratio",
        "is_first_floor", "is_last_floor", "is_studio",
        "area_x_rooms", "area_x_dist",
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

    df_base = df.copy()

    kf = KFold(n_splits=10, shuffle=True, random_state=1234)
    oof_cb = np.zeros(len(df))
    oof_lgb = np.zeros(len(df))

    print("10-fold CV...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(df_base)):
        print(f"Fold {fold+1}/10")
        train_df = df_base.iloc[train_idx].copy()
        val_df = df_base.iloc[val_idx].copy()

        # Target encoding per fold
        train_df["district_te"] = train_df.groupby("district_clean")["price_log"].transform("mean")
        global_mean = train_df["price_log"].mean()
        val_df["district_te"] = val_df["district_clean"].map(train_df.groupby("district_clean")["price_log"].mean()).fillna(global_mean)

        X_train = train_df[feature_cols + ["district_te"]].copy()
        X_val = val_df[feature_cols + ["district_te"]].copy()
        y_train = train_df["price_log"].copy()
        y_val = val_df["price_log"].copy()

        for c in num_features + ["district_te"]:
            if c in X_train.columns:
                med = X_train[c].median()
                X_train[c] = pd.to_numeric(X_train[c], errors="coerce").fillna(med)
                X_val[c] = pd.to_numeric(X_val[c], errors="coerce").fillna(med)
        for c in cat_features:
            if c in X_train.columns:
                X_train[c] = X_train[c].fillna("unknown").astype(str)
                X_val[c] = X_val[c].fillna("unknown").astype(str)

        cat_indices = [i for i, c in enumerate(X_train.columns) if c in cat_features]

        model_cb = CatBoostRegressor(
            iterations=5000,
            learning_rate=0.015,
            depth=8,
            l2_leaf_reg=4,
            loss_function="RMSE",
            random_seed=1234 + fold,
            verbose=False,
            early_stopping_rounds=250,
        )
        model_cb.fit(X_train, y_train, cat_features=cat_indices, eval_set=(X_val, y_val), use_best_model=True)
        oof_cb[val_idx] = model_cb.predict(X_val)

        X_train_lgb = X_train.copy()
        X_val_lgb = X_val.copy()
        for c in cat_features:
            if c in X_train_lgb.columns:
                X_train_lgb[c] = X_train_lgb[c].astype('category')
                X_val_lgb[c] = X_val_lgb[c].astype('category')

        model_lgb = LGBMRegressor(
            n_estimators=5000,
            learning_rate=0.015,
            max_depth=9,
            num_leaves=80,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.2,
            reg_lambda=1.2,
            random_state=1234 + fold,
            n_jobs=-1,
            verbose=-1,
        )
        model_lgb.fit(X_train_lgb, y_train)
        oof_lgb[val_idx] = model_lgb.predict(X_val_lgb)

    # Найдём оптимальные веса
    y_real_log = df_base["price_log"].values
    y_real = np.expm1(y_real_log)

    def objective(trial):
        w = trial.suggest_float("w_cb", 0.3, 0.9)
        pred_log = w * oof_cb + (1 - w) * oof_lgb
        pred = np.expm1(pred_log)
        return mean_absolute_error(y_real, pred)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)
    best_w = study.best_params["w_cb"]
    print(f"\nBest weight CatBoost: {best_w:.3f}")

    pred_log_best = best_w * oof_cb + (1 - best_w) * oof_lgb
    pred_best = np.expm1(pred_log_best)

    mae = mean_absolute_error(y_real, pred_best)
    rmse = np.sqrt(mean_squared_error(y_real, pred_best))
    r2 = r2_score(y_real, pred_best)
    mape = np.mean(np.abs((y_real - pred_best) / y_real)) * 100

    print("\n=== 10-Fold CV OOF ===")
    print(f"MAE: {mae:,.0f} сом")
    print(f"RMSE: {rmse:,.0f} сом")
    print(f"MAPE: {mape:.2f}%")
    print(f"R2: {r2:.4f}")

    meta_cv = {
        "cv": "10-fold",
        "best_w_cb": best_w,
        "oof_mae": float(mae),
        "oof_rmse": float(rmse),
        "oof_mape": float(mape),
        "oof_r2": float(r2),
    }
    with open(MODELS / "cv_results.json", "w") as f:
        json.dump(meta_cv, f, ensure_ascii=False, indent=2)

    print("Saved cv_results.json")


if __name__ == "__main__":
    main()
