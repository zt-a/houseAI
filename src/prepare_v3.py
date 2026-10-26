#!/usr/bin/env python3
import json
import math
import re
import sqlite3

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm

from config import (
    APT_MAX_AREA,
    APT_MAX_PRICE,
    APT_MAX_PRICE_M2,
    APT_MIN_AREA,
    APT_MIN_PRICE,
    APT_MIN_PRICE_M2,
    BISHKEK_CENTER_LAT,
    BISHKEK_CENTER_LNG,
    DATA_PROCESSED,
    DB_PATH,
    TARGET,
)


def haversine(lat1, lon1, lat2, lon2):
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return np.nan
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def parse_rooms(x):
    if pd.isna(x) or x == "":
        return np.nan
    s = str(x).strip().lower()
    if s in ["студия", "студийная", "studio"]:
        return 0.0
    m = re.findall(r"[-+]?\d*\.?\d+", s)
    if m:
        try:
            return float(m[0])
        except Exception:
            pass
    return np.nan


def parse_num(x):
    if pd.isna(x) or x == "":
        return np.nan
    m = re.findall(r"[-+]?\d*\.?\d+", str(x))
    if m:
        try:
            return float(m[0])
        except Exception:
            pass
    return np.nan


def extract_district_from_address(addr):
    if pd.isna(addr) or not str(addr).strip():
        return "unknown"
    s = str(addr).lower().replace("ё", "е")

    # микрорайоны
    m = re.findall(r"(?:мкр\.?|микрорайон)\s*([^\s,/;()\-]+(?:\-[^\s,/;()\-]+)?)", s)
    if m:
        cand = m[0].strip().replace(".", "")
        if cand and len(cand) > 1:
            return f"mkr_{cand}"

    # известные районы/названия
    patterns = [
        "кок-жар", "кок жар", "арча-бешик", "арча бешик", "ак ордо", "ак-ордо",
        "джал", "ж/м", "восток", "запад", "север", "юг", "калыка акиева",
        "алма-Ата", "алтын ордо", "бишкек-парк", "горка", "тунгуч", "учкун",
        "аламедин", "орто-сай", "чон арык"
    ]
    for p in patterns:
        if p in s:
            return f"area_{p.replace(' ', '_').replace('-', '_')}"

    # первые слова могут быть улицей, но иногда попадаются названия
    # попробуем взять что-то осмысленное из начала
    parts = re.split(r"[,;/]", s)
    first = parts[0].strip()
    first = re.sub(r"ул\.?|улица|проспект|пр\.|пер\.?", "", first).strip()
    # если короткое и не номер
    if first and not re.match(r"^\d", first) and len(first) >= 3 and len(first) <= 20:
        # пропустим совсем общие
        if first not in ["бишкек", "кыргызстан"]:
            return f"addr_{first.replace(' ', '_')}"

    return "unknown"


def load_from_db():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM properties", conn)
    conn.close()
    return df


def main():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    print(f"Loading from {DB_PATH}")
    df = load_from_db()
    print(f"Loaded: {len(df)} rows")

    # Квартиры Бишкека
    df = df[(df["property_type"] == "Apartment") & (df["city"] == "Бишкек")].copy()
    print(f"After filter: {len(df)} rows")

    # Фильтры
    df = df[df[TARGET].notna() & (df[TARGET] > 0)].copy()
    df = df[df["total_area"].notna() & (df["total_area"] > 0)].copy()
    df = df[(df["total_area"] >= APT_MIN_AREA) & (df["total_area"] <= APT_MAX_AREA)].copy()
    df = df[(df[TARGET] >= APT_MIN_PRICE) & (df[TARGET] <= APT_MAX_PRICE)].copy()

    df["price_m2_som_calc"] = df[TARGET] / df["total_area"]
    df = df[(df["price_m2_som_calc"] >= APT_MIN_PRICE_M2) & (df["price_m2_som_calc"] <= APT_MAX_PRICE_M2)].copy()
    print(f"After cleaning: {len(df)} rows")

    # Базовые числовые
    df["rooms_num"] = df["rooms"].apply(parse_rooms)
    df["floor_num"] = df["floor"].apply(parse_num)
    df["floors_total_num"] = df["floors_total"].apply(parse_num)
    df["year_built_num"] = pd.to_numeric(df["year_built"], errors="coerce")
    df["ceiling_height_num"] = pd.to_numeric(df["ceiling_height"], errors="coerce")
    df["living_area_num"] = pd.to_numeric(df["living_area"], errors="coerce")
    df["kitchen_area_num"] = pd.to_numeric(df["kitchen_area"], errors="coerce")

    # Гео
    df["dist_bishkek_center_km"] = df.apply(lambda r: haversine(r["lat"], r["lng"], BISHKEK_CENTER_LAT, BISHKEK_CENTER_LNG), axis=1)
    df["lat_num"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lng_num"] = pd.to_numeric(df["lng"], errors="coerce")

    # Кластеры по координатам
    geo_df = df[["lat_num", "lng_num"]].copy().fillna(df[["lat_num", "lng_num"]].median())
    kmeans = KMeans(n_clusters=15, random_state=42, n_init=10)
    df["geo_cluster"] = kmeans.fit_predict(geo_df)
    df["geo_cluster"] = df["geo_cluster"].astype(str)

    # Район из адреса
    df["district_from_addr"] = df["address"].apply(extract_district_from_address)
    print(df["district_from_addr"].value_counts().head(10))

    # features_json
    features_list = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Parsing features_json"):
        fj = row.get("features_json")
        flags = {}
        if isinstance(fj, str) and fj.strip():
            try:
                data = json.loads(fj)
                if isinstance(data, dict):
                    for k, v in data.items():
                        kl = str(k).lower()
                        if isinstance(v, bool):
                            flags[f"feat_{kl}"] = 1 if v else 0
                        elif isinstance(v, (int, float)) and v in (0, 1):
                            flags[f"feat_{kl}"] = int(v)
                        elif isinstance(v, str) and v.strip().lower() in ["да", "есть", "yes", "true"]:
                            flags[f"feat_{kl}"] = 1
                        elif isinstance(v, str) and v.strip().lower() in ["нет", "no", "false"]:
                            flags[f"feat_{kl}"] = 0
                        else:
                            flags[f"feat_{kl}"] = 1
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, str):
                            key = item.strip().lower().replace(" ", "_")
                            flags[f"feat_{key}"] = 1
            except Exception:
                pass
        features_list.append(flags)

    feats_df = pd.DataFrame(features_list).fillna(0)
    df = pd.concat([df.reset_index(drop=True), feats_df.reset_index(drop=True)], axis=1)

    feat_cols_all = [c for c in df.columns if c.startswith("feat_")]
    if feat_cols_all:
        freq = (df[feat_cols_all].sum() / len(df)).sort_values(ascending=False)
        keep_feats = freq[freq >= 0.03].index.tolist()
        print(f"Keeping {len(keep_feats)} features")
        df = df.drop(columns=[c for c in feat_cols_all if c not in keep_feats])

    # Категории
    for c in ["district", "condition", "seller_type", "documents", "heating", "gas", "water", "electricity", "canalization", "furniture", "toilet", "security"]:
        if c in df.columns:
            df[f"{c}_cat"] = df[c].fillna("unknown").astype(str).str.strip().str.lower()

    df["district_clean"] = df["district_from_addr"]
    # если редкие - сгруппируем
    vc = df["district_clean"].value_counts()
    df["district_clean"] = df["district_clean"].apply(lambda x: x if vc.get(x, 0) >= 10 else "rare_district")

    out_path = DATA_PROCESSED / "apartments_bishkek_v3.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved: {out_path} ({len(df)} rows, {len(df.columns)} cols)")


if __name__ == "__main__":
    main()
