from pathlib import Path

ROOT = Path("/mnt/hdd/Projects/tools/houseAI")
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
SRC = ROOT / "src"
DB_PATH = ROOT / "housekg.db"

# Центр Бишкека для расчёта расстояния
BISHKEK_CENTER_LAT = 42.8746
BISHKEK_CENTER_LNG = 74.5698

# Целевая переменная
TARGET = "price_som"
TARGET_LOG = "price_som_log"

# Фильтры для квартир Бишкека
APT_MIN_AREA = 15
APT_MAX_AREA = 250
APT_MIN_PRICE = 1_000_000  # сом
APT_MAX_PRICE = 40_000_000  # сом
APT_MIN_PRICE_M2 = 10_000
APT_MAX_PRICE_M2 = 200_000
