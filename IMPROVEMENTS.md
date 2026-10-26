# Обновлённые результаты

## Сравнение версий

| Модель | MAE (сом) | RMSE (сом) | MAPE | R² | Изменение |
|---|---|---|---|---|---|
| v1 (базовая) | 1 332 000 | 2 024 000 | 13.71% | 0.819 | - |
| v2 (+фичи) | 1 207 000 | 1 665 000 | 13.34% | 0.837 | **+1.8 п.п. по R²** |
| v3 (бленд + гео + target encoding) | **1 074 000** | **1 625 000** | **11.24%** | **0.871** | **+5.2 п.п. по R² от v1** |

## Что добавлено в v3
- **Парсинг адреса** - извлечение микрорайонов из `address` (раньше `district` был пустым)
- **Геокластеризация** (KMeans по lat/lng, 15 кластеров)
- **Target Encoding** для района с KFold (без утечки)
- **Полиномиальные фичи**: `area^2`, `age^2`, `floor_ratio^2`
- **Взаимодействия**: `total_area * rooms`, `total_area / (dist + 1)`
- **Бленд**: CatBoost (70%) + LightGBM (30%)
- **Больше фич из features_json** (порог 3% вместо 5%)

## Текущие модели
- `models/catboost_apartments_bishkek_v3.cbm`
- `models/lgbm_apartments_bishkek_v3.pkl`
- `models/ensemble_apartments_bishkek_v3_meta.json`
