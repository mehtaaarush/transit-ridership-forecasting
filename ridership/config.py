"""Project paths and modelling constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
DOCS_DATA_DIR = ROOT / "docs" / "data"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Sources (both public, no API key required)
RIDERSHIP_URL = "https://data.cityofchicago.org/resource/6iiy-9s97.csv"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
CHICAGO_LAT, CHICAGO_LON = 41.8781, -87.6298  # the Loop

# Modelling setup
MODEL_START = "2022-07-01"     # post-pandemic regime used for training
BACKTEST_START = "2025-09-01"  # final 12 months are forecast out-of-sample
HORIZON = 14                   # days ahead (a two-week service-planning window)
LEVEL_WINDOW = 28              # trailing days that define the ridership level (4 full weeks)
CALIBRATION_DAYS = 365         # inner hold-out (one full year of seasons) that sizes the intervals
INTERVAL = (0.10, 0.90)        # 80% prediction interval
RANDOM_STATE = 42
