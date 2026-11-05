"""End-to-end pipeline.

    python -m ridership.pipeline            # use the committed data snapshot
    python -m ridership.pipeline --refresh  # re-download ridership + weather first

Steps: ingest -> features -> rolling backtest of four models -> final model + 14-day
forecast -> dashboard JSON (docs/data) -> metrics and figures (reports/).
"""
from __future__ import annotations

import argparse
import time

from . import config, data, export, features, figures
from .backtest import run_backtest
from .evaluate import summarize
from .forecast import fit_final


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refresh", action="store_true", help="re-download the raw data before running")
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    data.ensure_raw_data(refresh=args.refresh)
    daily = features.build_daily(data.load_ridership(), data.load_weather())
    print(f"[data] {daily['total'].first_valid_index():%Y-%m-%d} -> {daily['total'].last_valid_index():%Y-%m-%d}")

    preds = run_backtest(daily)
    overall = summarize(preds).set_index("model")
    print(f"[backtest] {len(preds):,} forecasts in {time.perf_counter() - t0:.0f}s")
    for key, row in overall.iterrows():
        print(f"    {export.MODEL_NAMES[key]:<22} WAPE {row['wape']:.2%}  MAPE {row['mape']:.2%}  coverage {row['coverage']:.0%}")

    model, quantiles, future = fit_final(daily)
    export.write_json(export.dashboard_payload(daily, preds, future), config.DOCS_DATA_DIR / "dashboard.json")
    export.write_json(export.model_payload(model, quantiles, future), config.DOCS_DATA_DIR / "model.json")

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summarize(preds).round(5).to_csv(config.REPORTS_DIR / "backtest_metrics.csv", index=False)
    summarize(preds, by=["h"]).round(5).to_csv(config.REPORTS_DIR / "backtest_metrics_by_horizon.csv", index=False)
    figures.make_all(daily, preds)
    print(f"[done] dashboard data, metrics and figures written in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
