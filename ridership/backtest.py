"""Rolling-origin backtest: refit every month, forecast every origin in that month."""
from __future__ import annotations

import pandas as pd

from . import config
from .features import make_supervised
from .models import MODEL_CLASSES, apply_interval, residual_quantiles

KEEP_COLUMNS = ["origin", "target_date", "h", "y", "day_type", "dow", "hol_major"]


def fit_with_intervals(model_cls, rows: pd.DataFrame, cutoff: pd.Timestamp, calibration_days: int = config.CALIBRATION_DAYS):
    """Fit on targets before `cutoff` and size the prediction interval from an inner hold-out.

    The inner model only sees targets before cutoff - calibration_days and then forecasts the
    origins inside the calibration window, so the residual quantiles are out-of-sample.
    """
    calib_start = cutoff - pd.Timedelta(days=calibration_days)
    inner = rows[rows["target_date"] < calib_start]
    calib = rows[(rows["origin"] >= calib_start) & (rows["target_date"] < cutoff)]
    quantiles = residual_quantiles(calib, model_cls().fit(inner).predict(calib))
    model = model_cls().fit(rows[rows["target_date"] < cutoff])
    return model, quantiles


def run_backtest(daily: pd.DataFrame, model_classes=MODEL_CLASSES, start: str = config.BACKTEST_START) -> pd.DataFrame:
    """Long table of out-of-sample forecasts: one row per (model, origin, horizon)."""
    data_end = daily["total"].last_valid_index()
    rows = make_supervised(daily, config.MODEL_START, data_end)
    results = []
    for fold_start in pd.date_range(start, data_end, freq="MS"):
        fold_end = fold_start + pd.offsets.MonthBegin(1)
        test = rows[(rows["origin"] >= fold_start) & (rows["origin"] < fold_end)]
        if test.empty:
            continue
        for cls in model_classes:
            model, quantiles = fit_with_intervals(cls, rows, fold_start)
            pred = model.predict(test)
            lo, hi = apply_interval(test, pred, quantiles)
            results.append(test[KEEP_COLUMNS].assign(model=cls.key, pred=pred, lo=lo, hi=hi))
    return pd.concat(results, ignore_index=True)
