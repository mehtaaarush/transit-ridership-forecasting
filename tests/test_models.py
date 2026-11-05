import numpy as np

from ridership.evaluate import forecast_metrics
from ridership.features import make_supervised
from ridership.models import (
    GradientBoostingForecaster,
    RidgeForecaster,
    SeasonalNaive,
    apply_interval,
    residual_quantiles,
)
from tests.conftest import make_daily


def test_seasonal_naive_is_exact_on_a_pure_weekly_pattern():
    daily = make_daily(noise=0.0)
    rows = make_supervised(daily, daily.index[60], daily.index[200])
    np.testing.assert_allclose(SeasonalNaive().predict(rows), rows["y"])


def test_learned_models_pick_up_a_weather_effect():
    daily = make_daily(n_days=420, snow_effect=0.3, seed=1)
    rows = make_supervised(daily, daily.index[60], daily.index[-15])
    train, test = rows[rows["origin"] < daily.index[330]], rows[rows["origin"] >= daily.index[345]]
    naive = forecast_metrics(test["y"], SeasonalNaive().predict(test))["wape"]
    ridge = forecast_metrics(test["y"], RidgeForecaster().fit(train).predict(test))["wape"]
    gbm = forecast_metrics(test["y"], GradientBoostingForecaster(n_estimators=150).fit(train).predict(test))["wape"]
    assert gbm < 0.5 * naive
    assert ridge < naive


def test_interval_sized_on_a_sample_covers_80_percent_of_it(daily):
    rows = make_supervised(daily, daily.index[60], daily.index[200])
    pred = SeasonalNaive().predict(rows)
    q = residual_quantiles(rows, pred)
    assert set(q) == set(range(1, 15))
    lo, hi = apply_interval(rows, pred, q)
    coverage = np.mean((rows["y"] >= lo) & (rows["y"] <= hi))
    assert 0.75 <= coverage <= 0.85
