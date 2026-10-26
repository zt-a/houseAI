#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib

# Добавим src в путь
sys.path.insert(0, '/mnt/hdd/Projects/tools/houseAI/src')
from config import DATA_PROCESSED, MODELS, BISHKEK_CENTER_LAT, BISHKEK_CENTER_LNG

app = FastAPI(title="HouseKG Price Predictor")

templates = Jinja2Templates(directory="/mnt/hdd/Projects/tools/houseAI/web/templates")
app.mount("/static", StaticFiles(directory="/mnt/hdd/Projects/tools/houseAI/web/static"), name="static")

# Загружаем модель
model_cb = CatBoostRegressor()
model_cb.load_model(str(MODELS / "catboost_apartments_bishkek_final.cbm"))
model_lgb = joblib.load(MODELS / "lgbm_apartments_bishkek_final.pkl")

with open(MODELS / "ensemble_apartments_bishkek_final_meta.json") as f:
    meta = json.load(f)

feature_cols = meta["feature_cols"]
cat_features = meta["cat_features"]
num_features = meta["num_features"]
w_cb = meta["blend_w_cb"]


def haversine(lat1, lon1, lat2, lon2):
    import math
    if pd.isna(lat1) or pd.isna(lon1):
        return 5.0  # дефолт
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {"result": None})


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    total_area: float = Form(...),
    rooms_num: float = Form(...),
    floor_num: float = Form(...),
    floors_total_num: float = Form(...),
    year_built_num: float = Form(...),
    lat: float = Form(42.8746),
    lng: float = Form(74.5698),
    district_clean: str = Form("unknown"),
    condition_cat: str = Form("unknown"),
    geo_cluster: str = Form("0"),
    heating_cat: str = Form("unknown"),
    has_balcony: int = Form(0),
    has_elevator: int = Form(0),
):
    try:
        df = pd.DataFrame([{
            'total_area': total_area,
            'rooms_num': rooms_num,
            'floor_num': floor_num,
            'floors_total_num': floors_total_num,
            'year_built_num': year_built_num,
            'ceiling_height_num': 2.7,
            'living_area_num': total_area * 0.65,
            'kitchen_area_num': total_area * 0.15,
            'lat_num': lat,
            'lng_num': lng,
            'dist_bishkek_center_km': haversine(lat, lng, BISHKEK_CENTER_LAT, BISHKEK_CENTER_LNG),
            'district_clean': district_clean,
            'condition_cat': condition_cat,
            'geo_cluster': str(geo_cluster),
            'heating_cat': heating_cat,
            'gas_cat': 'unknown',
            'water_cat': 'unknown',
            'electricity_cat': 'unknown',
            'canalization_cat': 'unknown',
            'furniture_cat': 'unknown',
            'toilet_cat': 'unknown',
            'security_cat': 'unknown',
            'seller_type_cat': 'unknown',
            'documents_cat': 'unknown',
            'feat_балкон': has_balcony,
            'feat_лоджия': has_balcony,
            'feat_лифт': has_elevator,
        }])

        df["age"] = 2026 - df["year_built_num"]
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
        df["is_first_floor"] = int(df["floor_num"].iloc[0] == 1)
        df["is_last_floor"] = int(df["floor_num"].iloc[0] == df["floors_total_num"].iloc[0])
        df["is_studio"] = int(df["rooms_num"].iloc[0] == 0)
        df["area_x_rooms"] = df["total_area"] * df["rooms_num"]
        df["area_x_dist"] = df["total_area"] / (df["dist_bishkek_center_km"] + 1)
        df["district_te"] = 0

        # Добавим все недостающие фичи из модели
        for c in feature_cols:
            if c not in df.columns:
                if c.startswith("feat_"):
                    df[c] = 0
                elif c in cat_features:
                    df[c] = "unknown"
                elif c in num_features:
                    df[c] = 0
                else:
                    df[c] = 0

        X = df[feature_cols].copy()
        for c in num_features:
            if c in X.columns:
                X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
        if "district_te" in X.columns:
            X["district_te"] = pd.to_numeric(X["district_te"], errors="coerce").fillna(0)
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

        pred_som = int(pred[0])
        pred_usd = int(pred_som / 87)

        return templates.TemplateResponse(request, "index.html", {
            "result": {
                "price_som": f"{pred_som:,}",
                "price_usd": f"{pred_usd:,}",
                "price_m2": f"{int(pred_som/total_area):,}",
                "confidence": "Хорошая"
            }
        })
    except Exception as e:
        return templates.TemplateResponse(request, "index.html", {
            "result": {"error": str(e)}
        })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
