import numpy as np
import pandas as pd
import pytest

from ridership.features import build_daily

WEEKLY_SHAPE = np.array([1.0, 1.05, 1.08, 1.06, 0.95, 0.6, 0.45])


def make_daily(n_days: int = 240, noise: float = 0.01, snow_effect: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """Synthetic ridership with a fixed weekly shape, optional noise and an optional snow penalty."""
    idx = pd.date_range("2024-01-01", periods=n_days, freq="D", name="date")
    rng = np.random.default_rng(seed)
    snow = np.where(rng.random(n_days) < 0.15, rng.uniform(1, 8, n_days), 0.0)
    total = 1_000_000 * WEEKLY_SHAPE[idx.day_of_week] * (1 + noise * rng.standard_normal(n_days))
    total = total * (1 - snow_effect * (snow > 0))
    rides = pd.DataFrame({"day_type": "W", "bus": total * 0.6, "rail": total * 0.4, "total": total}, index=idx)
    weather = pd.DataFrame(
        {
            "temp_mean_f": 50.0, "temp_max_f": 58.0, "temp_min_f": 42.0, "feels_like_min_f": 36.0,
            "precip_in": snow / 10, "snow_in": snow,
        },
        index=idx,
    )
    return build_daily(rides, weather)


@pytest.fixture
def daily():
    return make_daily()
