"""Feature engineering.

Forecasts are framed as a *direct multi-horizon* problem: every row pairs a forecast origin T
(the last day with published ridership) with a target day d = T + h, h in 1..14. Origin
features only use ridership up to and including T, so the frame can be used for honest
backtests; target-day features are the calendar plus that day's weather.

The model predicts log(y_d / level_T), where level_T is the mean of the 28 days ending at T
(exactly four of each weekday). Working in ratios keeps the model valid while the overall
ridership level keeps drifting upward after the pandemic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .calendar_features import HOLIDAY_COLUMNS, holiday_table

WEATHER_COLUMNS = ["temp_mean_f", "feels_like_min_f", "precip_in", "snow_in", "snow_prev_in"]
MODEL_FEATURES = ["h", "dow", "doy", *HOLIDAY_COLUMNS, *WEATHER_COLUMNS, "dow_ratio_recent", "momentum"]


def build_daily(rides: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Ridership, weather and calendar on one daily index (weather may extend past ridership)."""
    end = max(rides.index.max(), weather.index.max())
    idx = pd.date_range(rides.index.min(), end, freq="D", name="date")
    df = pd.DataFrame(index=idx)
    df = df.join(rides[["day_type", "bus", "rail", "total"]]).join(weather)
    df = df.join(holiday_table(idx.min(), idx.max()))
    df["dow"] = idx.day_of_week
    df["doy"] = idx.day_of_year
    df["month"] = idx.month
    df["snow_prev_in"] = df["snow_in"].shift(1).fillna(0.0)
    return df


def origin_state(total: pd.Series, window: int = config.LEVEL_WINDOW) -> pd.DataFrame:
    """Quantities known at the end of each day t, computed from ridership <= t only."""
    level = total.rolling(window, min_periods=window).mean()
    same_dow = pd.concat([total.shift(7 * k) for k in range(4)], axis=1)
    return pd.DataFrame(
        {
            "level": level,
            "momentum": level / level.shift(window),
            "same_dow_last": total,
            "same_dow_mean4": same_dow.mean(axis=1, skipna=False),
            "same_dow_median4": same_dow.median(axis=1, skipna=False),
        }
    )


def make_supervised(
    daily: pd.DataFrame,
    origin_start,
    origin_end,
    horizon: int = config.HORIZON,
    require_target: bool = True,
) -> pd.DataFrame:
    """One row per (origin, horizon). Rows whose target day has no ridership are dropped
    unless require_target=False (used for the forward-looking forecast)."""
    state = origin_state(daily["total"])
    origins = daily.index[(daily.index >= pd.Timestamp(origin_start)) & (daily.index <= pd.Timestamp(origin_end))]
    target_cols = ["dow", "doy", "month", "day_type", *HOLIDAY_COLUMNS, *WEATHER_COLUMNS]
    frames = []
    for h in range(1, horizon + 1):
        target = origins + pd.Timedelta(days=h)
        # The most recent day <= origin that shares the target's weekday.
        anchor = target - pd.Timedelta(days=7 * int(np.ceil(h / 7)))
        st_origin = state.reindex(origins)
        st_anchor = state.reindex(anchor)
        frame = pd.DataFrame(
            {
                "origin": origins,
                "target_date": target,
                "h": h,
                "level": st_origin["level"].to_numpy(),
                "momentum": st_origin["momentum"].to_numpy(),
                "naive": st_anchor["same_dow_last"].to_numpy(),
                "mean4": st_anchor["same_dow_mean4"].to_numpy(),
                "dow_ratio_recent": st_anchor["same_dow_median4"].to_numpy() / st_origin["level"].to_numpy(),
            }
        )
        tgt = daily.reindex(target)[target_cols + ["total"]].reset_index(drop=True)
        frame = pd.concat([frame, tgt.rename(columns={"total": "y"})], axis=1)
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["level", "momentum", "dow_ratio_recent", "temp_mean_f"])
    if require_target:
        out = out.dropna(subset=["y"])
    return out.sort_values(["origin", "h"]).reset_index(drop=True)


def model_matrix(frame: pd.DataFrame) -> np.ndarray:
    return frame[MODEL_FEATURES].to_numpy(dtype=float)


def log_ratio_target(frame: pd.DataFrame) -> np.ndarray:
    return np.log(frame["y"].to_numpy(dtype=float) / frame["level"].to_numpy(dtype=float))
