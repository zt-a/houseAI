#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from config import DATA_PROCESSED, MODELS


class PricePredictor:
    def __init__(self):
        self.model = CatBoostRegressor()
        self.model.load_model(str(MODELS / "catboost_apartments_bishkek.cbm"))
        with open(MODELS / "catboost_apartments_bishkek_meta.json") as f:
            self.meta = json.load(f)
        self.feature_cols = self.meta["feature_cols"]
        self.cat_features = self.meta["cat_features"]
        self.num_features = self.meta["num_features"]

    def predict(self, X):
        X = X.copy()
        # Заполним
        for c in self.num_features:
            if c in X.columns:
                X[c] = pd.to_numeric(X[c], errors="coerce")
                X[c] = X[c].fillna(X[c].median() if c in X else 0)
        for c in self.cat_features:
            if c in X.columns:
                X[c] = X[c].fillna("unknown").astype(str)
        # Добавим недостающие фичи
        for c in self.feature_cols:
            if c not in X.columns:
                if c.startswith("feat_"):
                    X[c] = 0
                elif c in self.num_features:
                    X[c] = 0
                elif c in self.cat_features:
                    X[c] = "unknown"
                else:
                    X[c] = 0
        X = X[self.feature_cols]
        p_log = self.model.predict(X)
        return np.expm1(p_log)


def main():
    pred = PricePredictor()
    df = pd.read_parquet(DATA_PROCESSED / "apartments_bishkek_raw.parquet")

    # Предскажем на всех и сравним с реальной ценой
    price_pred = pred.predict(df)
    df["price_pred"] = price_pred
    df["error"] = df["price_pred"] - df["price_som"]
    df["abs_error"] = abs(df["error"])
    df["ape"] = df["abs_error"] / df["price_som"] * 100

    print("Пример предсказаний:")
    sample = df[["url_slug", "district", "total_area", "rooms", "floor", "price_som", "price_pred", "abs_error", "ape"]].head(10)
    print(sample.to_string(index=False))
    print(f"\nСредняя APE: {df['ape'].mean():.2f}%")
    print(f"Медианная APE: {df['ape'].median():.2f}%")


if __name__ == "__main__":
    main()
