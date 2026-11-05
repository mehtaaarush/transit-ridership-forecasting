import numpy as np
import pandas as pd

from ridership.features import make_supervised, origin_state


def test_origin_features_ignore_the_future(daily):
    origin = daily.index[120]
    before = make_supervised(daily, origin, origin)
    tampered = daily.copy()
    tampered.loc[tampered.index > origin, "total"] *= 3
    after = make_supervised(tampered, origin, origin)
    cols = ["level", "momentum", "naive", "mean4", "dow_ratio_recent"]
    pd.testing.assert_frame_equal(before[cols], after[cols])
    assert not np.allclose(before["y"], after["y"])


def test_anchor_is_latest_same_weekday_at_or_before_origin(daily):
    origin = daily.index[120]
    rows = make_supervised(daily, origin, origin)
    assert rows["h"].tolist() == list(range(1, 15))
    for _, r in rows.iterrows():
        anchor = r["target_date"] - pd.Timedelta(days=7 * int(np.ceil(r["h"] / 7)))
        assert 0 <= (origin - anchor).days < 7
        assert anchor.day_of_week == r["target_date"].day_of_week
        assert r["naive"] == daily.loc[anchor, "total"]


def test_level_is_the_trailing_four_week_mean(daily):
    state = origin_state(daily["total"])
    t = daily.index[90]
    expected = daily.loc[t - pd.Timedelta(days=27): t, "total"].mean()
    assert np.isclose(state.loc[t, "level"], expected)


def test_forward_rows_are_kept_without_a_target(daily):
    end = daily.index[-15]
    daily.loc[daily.index > end, ["bus", "rail", "total"]] = np.nan
    assert len(make_supervised(daily, end, end)) == 0
    rows = make_supervised(daily, end, end, require_target=False)
    assert len(rows) == 14 and rows["y"].isna().all()
