# Веб-интерфейс для предсказания цены

## Запуск
```bash
cd /mnt/hdd/Projects/tools/houseAI
PYTHONPATH=/mnt/hdd/Projects/tools/houseAI/src python web/app.py
```

Открой http://127.0.0.1:8000

## Особенности
- Предсказание цены квартиры в Бишкеке
- Учитывает площадь, этаж, год постройки, координаты
- Использует финальный ансамбль CatBoost + LightGBM
