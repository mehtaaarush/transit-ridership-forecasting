"""Forecast accuracy metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def forecast_metrics(y, pred, lo=None, hi=None) -> dict[str, float]:
    """WAPE is the headline metric (total absolute error / total riders); MAPE, MAE, RMSE
    and bias are reported alongside, plus interval coverage when bounds are given."""
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = pred - y
    out = {
        "wape": float(np.abs(err).sum() / y.sum()),
        "mape": float(np.mean(np.abs(err) / y)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(err.sum() / y.sum()),
        "n": int(len(y)),
    }
    if lo is not None and hi is not None:
        lo = np.asarray(lo, dtype=float)
        hi = np.asarray(hi, dtype=float)
        out["coverage"] = float(np.mean((y >= lo) & (y <= hi)))
        out["interval_width"] = float(np.mean((hi - lo) / pred))
    return out


def summarize(preds: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Metrics per model (and optional extra grouping columns) from a long prediction table."""
    keys = ["model", *(by or [])]
    rows = []
    for key, g in preds.groupby(keys, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        lo = g["lo"] if "lo" in g and g["lo"].notna().all() else None
        hi = g["hi"] if "hi" in g and g["hi"].notna().all() else None
        rows.append({**dict(zip(keys, key)), **forecast_metrics(g["y"], g["pred"], lo, hi)})
    return pd.DataFrame(rows)
