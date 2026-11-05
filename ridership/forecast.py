"""Production model: fit on all history, then forecast the 14 days after the last published day."""
from __future__ import annotations

import pandas as pd

from . import config
from .backtest import fit_with_intervals
from .features import make_supervised
from .models import GradientBoostingForecaster, apply_interval


def fit_final(daily: pd.DataFrame):
    """Returns (model, interval quantiles per horizon, forecast frame for origin = last published day)."""
    data_end = daily["total"].last_valid_index()
    rows = make_supervised(daily, config.MODEL_START, data_end)
    model, quantiles = fit_with_intervals(GradientBoostingForecaster, rows, data_end + pd.Timedelta(days=1))
    future = make_supervised(daily, data_end, data_end, require_target=False)
    pred = model.predict(future)
    lo, hi = apply_interval(future, pred, quantiles)
    return model, quantiles, future.assign(pred=pred, lo=lo, hi=hi)
