"""Serialise results for the static dashboard (docs/data/*.json).

The gradient-boosting model is exported tree-by-tree so the browser can re-run it for the
what-if planner; `predict_exported` is the reference implementation the tests hold the
JavaScript port to.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from . import analysis, config
from .evaluate import summarize
from .features import MODEL_FEATURES

MODEL_NAMES = {
    "naive": "Seasonal naive",
    "mean4": "4-week weekday mean",
    "ridge": "Ridge regression",
    "gbm": "Gradient boosting",
}
SHOWN_HORIZONS = (1, 7, 14)


def export_gbm(model: GradientBoostingRegressor, feature_names: list[str]) -> dict:
    trees = []
    for est in model.estimators_[:, 0]:
        t = est.tree_
        trees.append(
            {
                "left": t.children_left.tolist(),
                "right": t.children_right.tolist(),
                "feature": t.feature.tolist(),
                "threshold": t.threshold.tolist(),
                "value": t.value[:, 0, 0].tolist(),
            }
        )
    return {
        "features": list(feature_names),
        "init": float(np.ravel(model.init_.constant_)[0]),
        "learning_rate": float(model.learning_rate),
        "trees": trees,
    }


def predict_exported(spec: dict, X: np.ndarray) -> np.ndarray:
    """Pure-Python traversal of the exported trees (inputs cast to float32, as scikit-learn does)."""
    X = np.asarray(X, dtype=np.float32)
    out = np.full(len(X), spec["init"], dtype=float)
    for tree in spec["trees"]:
        left, right, feat, thr, val = tree["left"], tree["right"], tree["feature"], tree["threshold"], tree["value"]
        for i, row in enumerate(X):
            node = 0
            while left[node] != -1:
                node = left[node] if row[feat[node]] <= thr[node] else right[node]
            out[i] += spec["learning_rate"] * val[node]
    return out


def _round(values, digits=0):
    arr = np.asarray(values, dtype=float)
    arr = np.round(arr, digits)
    return [None if np.isnan(v) else (int(v) if digits == 0 else float(v)) for v in arr]


def _iso(index) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.DatetimeIndex(index)]


def backtest_payload(preds: pd.DataFrame) -> dict:
    overall = summarize(preds)
    by_h = summarize(preds, by=["h"])
    by_type = summarize(preds, by=["day_type"])
    metrics = []
    for _, r in overall.iterrows():
        metrics.append(
            {
                "model": r["model"], "name": MODEL_NAMES[r["model"]],
                **{k: round(float(r[k]), 4) for k in ["wape", "mape", "bias", "coverage", "interval_width"]},
                "mae": round(float(r["mae"])), "rmse": round(float(r["rmse"])), "n": int(r["n"]),
            }
        )
    horizons = sorted(preds["h"].unique().tolist())
    series = {}
    for h in SHOWN_HORIZONS:
        sub = preds[preds["h"] == h]
        actual = sub[sub["model"] == "gbm"].set_index("target_date")["y"].sort_index()
        entry = {"date": _iso(actual.index), "actual": _round(actual.values), "models": {}}
        for key in MODEL_NAMES:
            m = sub[sub["model"] == key].set_index("target_date").reindex(actual.index)
            entry["models"][key] = {"pred": _round(m["pred"]), "lo": _round(m["lo"]), "hi": _round(m["hi"])}
        series[str(h)] = entry

    gbm7 = preds[(preds["model"] == "gbm") & (preds["h"] == 7)].copy()
    gbm7["ape"] = (gbm7["pred"] - gbm7["y"]).abs() / gbm7["y"]
    misses = []
    for _, r in gbm7.nlargest(10, "ape").iterrows():
        misses.append(
            {
                "date": r["target_date"].strftime("%Y-%m-%d"), "day_type": r["day_type"],
                "actual": int(r["y"]), "forecast": int(round(r["pred"])), "error": round(float(r["pred"] / r["y"] - 1), 4),
            }
        )
    return {
        "start": preds["target_date"].min().strftime("%Y-%m-%d"),
        "end": preds["target_date"].max().strftime("%Y-%m-%d"),
        "metrics": metrics,
        "by_horizon": {
            "h": horizons,
            **{k: _round(by_h[by_h["model"] == k].set_index("h")["wape"].reindex(horizons), 4) for k in MODEL_NAMES},
        },
        "by_day_type": [
            {"model": r["model"], "day_type": r["day_type"], "wape": round(float(r["wape"]), 4)} for _, r in by_type.iterrows()
        ],
        "series": series,
        "misses": misses,
    }


def forecast_payload(daily: pd.DataFrame, future: pd.DataFrame) -> dict:
    recent = daily["total"].dropna().iloc[-28:]
    days = []
    for _, r in future.iterrows():
        days.append(
            {
                "date": r["target_date"].strftime("%Y-%m-%d"), "h": int(r["h"]), "dow": int(r["dow"]),
                "pred": int(round(r["pred"])), "lo": int(round(r["lo"])), "hi": int(round(r["hi"])),
                "temp_mean_f": round(float(r["temp_mean_f"]), 1), "feels_like_min_f": round(float(r["feels_like_min_f"]), 1),
                "precip_in": round(float(r["precip_in"]), 3), "snow_in": round(float(r["snow_in"]), 2),
            }
        )
    return {
        "origin": future["origin"].iloc[0].strftime("%Y-%m-%d"),
        "level": round(float(future["level"].iloc[0])),
        "recent": {"date": _iso(recent.index), "actual": _round(recent.values)},
        "days": days,
    }


def dashboard_payload(daily: pd.DataFrame, preds: pd.DataFrame, future: pd.DataFrame) -> dict:
    hist = analysis.weekly_history(daily)
    rec = analysis.recovery_vs_2019(daily)
    grid = analysis.seasonality_grid(daily)
    annual = analysis.annual_totals(daily)
    return {
        "meta": {
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
            "data_start": daily["total"].first_valid_index().strftime("%Y-%m-%d"),
            "data_end": daily["total"].last_valid_index().strftime("%Y-%m-%d"),
            "model_start": config.MODEL_START,
            "backtest_start": config.BACKTEST_START,
            "horizon": config.HORIZON,
        },
        "kpis": analysis.headline_kpis(daily),
        "history": {"week": _iso(hist.index), "bus": _round(hist["bus"]), "rail": _round(hist["rail"])},
        "annual": {"year": annual.index.tolist(), "total": annual["total"].tolist()},
        "recovery": {"month": _iso(rec.index), **{c: _round(rec[c], 4) for c in rec.columns}},
        "seasonality": {
            "dow": grid.index.tolist(), "month": grid.columns.tolist(),
            "values": [_round(row) for row in grid.to_numpy()],
        },
        "weather": analysis.weather_effects(daily),
        "holidays": analysis.holiday_effects(daily),
        "backtest": backtest_payload(preds),
        "forecast": forecast_payload(daily, future),
    }


def model_payload(model, quantiles: dict, future: pd.DataFrame) -> dict:
    spec = export_gbm(model.model, MODEL_FEATURES)
    spec["interval_log_quantiles"] = {str(h): [round(q, 6) for q in v] for h, v in quantiles.items()}
    spec["origin"] = future["origin"].iloc[0].strftime("%Y-%m-%d")
    spec["level"] = float(future["level"].iloc[0])
    spec["rows"] = [
        {"date": r["target_date"].strftime("%Y-%m-%d"), "x": [float(r[f]) for f in MODEL_FEATURES]}
        for _, r in future.iterrows()
    ]
    return spec


def write_json(obj, path: Path, compact: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        if compact:
            json.dump(obj, fh, ensure_ascii=False, separators=(",", ":"))
        else:
            json.dump(obj, fh, ensure_ascii=False, indent=2)
    return path
