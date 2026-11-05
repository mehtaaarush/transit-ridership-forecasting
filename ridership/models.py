"""Forecasters for daily ridership.

Every model exposes fit(frame) -> self and predict(frame) -> riders, where `frame` is the
direct multi-horizon table from features.make_supervised.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config
from .calendar_features import HOLIDAY_COLUMNS
from .features import log_ratio_target, model_matrix


class SeasonalNaive:
    """Ridership on the most recent same weekday available at the origin."""

    name = "Seasonal naive"
    key = "naive"

    def fit(self, frame: pd.DataFrame) -> "SeasonalNaive":
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return frame["naive"].to_numpy(dtype=float)


class WeekdayMean:
    """Mean of the last four same-weekday values: the usual planner's rule of thumb."""

    name = "4-week weekday mean"
    key = "mean4"

    def fit(self, frame: pd.DataFrame) -> "WeekdayMean":
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return frame["mean4"].to_numpy(dtype=float)


class RidgeForecaster:
    """Linear model on the log ratio with calendar dummies and weather terms."""

    name = "Ridge regression"
    key = "ridge"
    _categorical = ["dow", "month"]
    _numeric = [
        *HOLIDAY_COLUMNS, "h", "temp", "temp_sq", "freezing", "feels_like_min",
        "log_precip", "log_snow", "log_snow_prev", "log_dow_ratio", "log_momentum",
    ]

    def __init__(self) -> None:
        pre = ColumnTransformer(
            [
                ("cat", OneHotEncoder(handle_unknown="ignore"), self._categorical),
                ("num", StandardScaler(), self._numeric),
            ]
        )
        self.pipeline = make_pipeline(pre, RidgeCV(alphas=np.logspace(-2, 3, 16)))

    @staticmethod
    def _design(frame: pd.DataFrame) -> pd.DataFrame:
        temp = frame["temp_mean_f"].astype(float)
        return pd.DataFrame(
            {
                "dow": frame["dow"].astype(int),
                "month": frame["month"].astype(int),
                **{c: frame[c].astype(float) for c in HOLIDAY_COLUMNS},
                "h": frame["h"].astype(float),
                "temp": temp,
                "temp_sq": (temp - 55.0) ** 2,
                "freezing": (temp < 32).astype(float),
                "feels_like_min": frame["feels_like_min_f"].astype(float),
                "log_precip": np.log1p(frame["precip_in"].astype(float)),
                "log_snow": np.log1p(frame["snow_in"].astype(float)),
                "log_snow_prev": np.log1p(frame["snow_prev_in"].astype(float)),
                "log_dow_ratio": np.log(frame["dow_ratio_recent"].astype(float)),
                "log_momentum": np.log(frame["momentum"].astype(float)),
            }
        )

    def fit(self, frame: pd.DataFrame) -> "RidgeForecaster":
        self.pipeline.fit(self._design(frame), log_ratio_target(frame))
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return frame["level"].to_numpy(dtype=float) * np.exp(self.pipeline.predict(self._design(frame)))


class GradientBoostingForecaster:
    """Gradient-boosted trees on the log ratio; captures holiday x weekday and weather
    interactions that the linear model has to be told about."""

    name = "Gradient boosting"
    key = "gbm"

    def __init__(self, **params) -> None:
        defaults = dict(
            n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.8,
            min_samples_leaf=20, random_state=config.RANDOM_STATE,
        )
        defaults.update(params)
        self.model = GradientBoostingRegressor(**defaults)

    def fit(self, frame: pd.DataFrame) -> "GradientBoostingForecaster":
        self.model.fit(model_matrix(frame), log_ratio_target(frame))
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return frame["level"].to_numpy(dtype=float) * np.exp(self.model.predict(model_matrix(frame)))


MODEL_CLASSES = [SeasonalNaive, WeekdayMean, RidgeForecaster, GradientBoostingForecaster]


def residual_quantiles(frame: pd.DataFrame, pred: np.ndarray, quantiles=config.INTERVAL) -> dict[int, list[float]]:
    """Empirical quantiles of log(actual / forecast) per horizon, from held-out forecasts."""
    resid = np.log(frame["y"].to_numpy(dtype=float) / pred)
    out = {}
    for h, idx in frame.groupby("h").indices.items():
        out[int(h)] = [float(q) for q in np.quantile(resid[idx], quantiles)]
    return out


def apply_interval(frame: pd.DataFrame, pred: np.ndarray, quantiles: dict[int, list[float]]):
    lo_q = frame["h"].map(lambda h: quantiles[int(h)][0]).to_numpy(dtype=float)
    hi_q = frame["h"].map(lambda h: quantiles[int(h)][1]).to_numpy(dtype=float)
    return pred * np.exp(lo_q), pred * np.exp(hi_q)
