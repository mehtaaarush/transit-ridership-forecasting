import pandas as pd

from ridership import data
from ridership.calendar_features import holiday_table, recurring_events


def test_major_holidays_match_cta_sunday_service_days():
    """On weekdays, CTA's own day_type 'U' (Sunday/holiday schedule) must coincide with hol_major."""
    rides = data.load_ridership().loc["2019-01-01":]
    flags = holiday_table(rides.index.min(), rides.index.max())
    weekday = rides.index.day_of_week < 5
    cta_holidays = set(rides.index[weekday & (rides["day_type"] == "U")])
    flagged = set(flags.index[(flags["hol_major"] == 1) & (flags.index.day_of_week < 5)])
    assert cta_holidays == flagged


def test_observed_holiday_and_weekend_flags():
    t = holiday_table("2026-07-01", "2026-07-06")
    assert t.loc["2026-07-03", "hol_major"] == 1  # Friday: observed Independence Day
    assert t.loc["2026-07-04", "hol_major"] == 0
    assert t.loc["2026-07-04", "hol_fixed_weekend"] == 1


def test_thanksgiving_period():
    t = holiday_table("2025-11-25", "2025-12-01")
    assert t.loc["2025-11-26", "hol_eve"] == 1
    assert t.loc["2025-11-27", "hol_major"] == 1
    assert t.loc["2025-11-28":"2025-11-30", "thanksgiving_weekend"].tolist() == [1, 1, 1]


def test_recurring_events_follow_scheduling_rules():
    events = set(recurring_events([2025, 2026]))
    expected = ["2025-03-15", "2025-06-29", "2025-10-12", "2026-03-14", "2026-06-28", "2026-10-11"]
    assert events == {pd.Timestamp(d) for d in expected}
