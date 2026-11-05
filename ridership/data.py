"""Download and tidy the two raw sources: CTA daily boardings and Chicago weather."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from . import config

RIDERSHIP_FILE = config.RAW_DIR / "cta_daily_boardings.csv"
WEATHER_FILE = config.RAW_DIR / "chicago_weather_daily.csv"

_WEATHER_VARS = {
    "temperature_2m_mean": "temp_mean_f",
    "temperature_2m_max": "temp_max_f",
    "temperature_2m_min": "temp_min_f",
    "apparent_temperature_min": "feels_like_min_f",
    "precipitation_sum": "precip_in",
    "snowfall_sum": "snow_in",
}


def download_ridership(path: Path = RIDERSHIP_FILE, timeout: int = 90) -> Path:
    """Fetch the full CTA 'Daily Boarding Totals' table from the City of Chicago Socrata API."""
    params = {"$limit": 100_000, "$order": "service_date"}
    resp = requests.get(config.RIDERSHIP_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(resp.content)
    return path


def download_weather(start: str, end: str, path: Path = WEATHER_FILE, timeout: int = 120) -> Path:
    """Fetch daily Chicago weather (Open-Meteo historical archive, imperial units)."""
    params = {
        "latitude": config.CHICAGO_LAT,
        "longitude": config.CHICAGO_LON,
        "start_date": start,
        "end_date": end,
        "daily": ",".join(_WEATHER_VARS),
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch",
        "timezone": "America/Chicago",
    }
    resp = requests.get(config.WEATHER_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    daily = pd.DataFrame(resp.json()["daily"]).rename(columns={"time": "date", **_WEATHER_VARS})
    path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(path, index=False)
    return path


def load_ridership(path: Path = RIDERSHIP_FILE) -> pd.DataFrame:
    """Daily boardings indexed by date: day_type (W/A/U), bus, rail, total."""
    raw = pd.read_csv(path)
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(raw["service_date"]).dt.normalize(),
            "day_type": raw["day_type"].astype(str).str.strip(),
            "bus": pd.to_numeric(raw["bus"], errors="coerce"),
            "rail": pd.to_numeric(raw["rail_boardings"], errors="coerce"),
            "total": pd.to_numeric(raw["total_rides"], errors="coerce"),
        }
    )
    return clean_ridership(df)


def clean_ridership(df: pd.DataFrame) -> pd.DataFrame:
    """Resolve duplicate dates, enforce a continuous daily index and consistent totals."""
    df = df.dropna(subset=["date", "total"]).sort_values("date")
    # Defensive: if the source ever repeats a service date, keep the later (revised) record.
    df = df.drop_duplicates(subset=["date"], keep="last").set_index("date")
    # Totals are bus + rail; recompute so the components always reconcile.
    df["total"] = df["bus"].fillna(0) + df["rail"].fillna(0)
    full = pd.date_range(df.index.min(), df.index.max(), freq="D", name="date")
    return df.reindex(full)


def load_weather(path: Path = WEATHER_FILE) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    df[["precip_in", "snow_in"]] = df[["precip_in", "snow_in"]].fillna(0.0)
    return df


def ensure_raw_data(refresh: bool = False, horizon: int = config.HORIZON) -> None:
    """Download whatever is missing (or everything when refresh=True)."""
    if refresh or not RIDERSHIP_FILE.exists():
        download_ridership()
    rides = load_ridership()
    # Weather is also needed for the forecast window after the last published ridership day.
    end = rides.index.max() + pd.Timedelta(days=horizon)
    latest_archive = pd.Timestamp.today().normalize() - pd.Timedelta(days=6)
    end = min(end, latest_archive)
    if refresh or not WEATHER_FILE.exists() or load_weather().index.max() < end:
        download_weather(rides.index.min().strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
