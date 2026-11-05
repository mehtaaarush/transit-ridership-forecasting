import pytest

from ridership.evaluate import forecast_metrics


def test_metrics_on_a_toy_example():
    m = forecast_metrics([100, 200], [110, 180], lo=[90, 150], hi=[120, 190])
    assert m["wape"] == pytest.approx(30 / 300)
    assert m["mape"] == pytest.approx(0.1)
    assert m["mae"] == pytest.approx(15)
    assert m["bias"] == pytest.approx(-10 / 300)
    assert m["coverage"] == pytest.approx(0.5)
    assert m["n"] == 2
