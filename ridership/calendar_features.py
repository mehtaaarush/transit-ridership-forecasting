"""Holiday, event and calendar flags for daily transit demand.

Everything here is known in advance, so it is safe to use for any forecast horizon.
"""
from __future__ import annotations

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

# CTA runs a Sunday schedule on these six holidays, using the federally observed date when
# the holiday falls on a weekend (verified against the `day_type` column in tests).
MAJOR_HOLIDAYS = frozenset(
    {"New Year's Day", "Memorial Day", "Independence Day", "Labor Day", "Thanksgiving Day", "Christmas Day"}
)

HOLIDAY_COLUMNS = [
    "hol_major", "hol_minor", "hol_eve", "thanksgiving_weekend",
    "hol_fixed_weekend", "winter_break", "major_event",
]


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> pd.Timestamp:
    first = pd.Timestamp(year=year, month=month, day=1)
    offset = (weekday - first.day_of_week) % 7
    return first + pd.Timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> pd.Timestamp:
    last = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    return last - pd.Timedelta(days=(last.day_of_week - weekday) % 7)


def recurring_events(years) -> pd.DatetimeIndex:
    """Large annual events with crowds that ride transit, by their published scheduling rule:
    Chicago Marathon (second Sunday of October), St. Patrick's Day parade (the Saturday before
    March 17) and the Pride Parade (last Sunday of June)."""
    days = []
    for y in years:
        days.append(_nth_weekday(y, 10, 6, 2))
        st_patricks = pd.Timestamp(year=y, month=3, day=17)
        days.append(st_patricks - pd.Timedelta(days=(st_patricks.day_of_week - 5) % 7 or 7))
        days.append(_last_weekday(y, 6, 6))
    return pd.DatetimeIndex(days)


def holiday_table(start, end) -> pd.DataFrame:
    """Daily flags between start and end (inclusive).

    hol_major             Sunday-schedule holidays (observed date)
    hol_minor             other federal holidays (MLK, Presidents, Juneteenth, Columbus, Veterans)
    hol_eve               Christmas Eve, New Year's Eve, the Wednesday before Thanksgiving
    thanksgiving_weekend  Friday to Sunday after Thanksgiving
    hol_fixed_weekend     Jan 1 / Jul 4 / Dec 25 when they fall on a weekend (observed elsewhere)
    winter_break          Dec 24 - Jan 2, when school and office travel both drop
    major_event           see recurring_events
    """
    idx = pd.date_range(start, end, freq="D", name="date")
    pad = pd.Timedelta(days=10)  # observed dates can fall in the neighbouring year
    names = USFederalHolidayCalendar().holidays(idx.min() - pad, idx.max() + pad, return_name=True)
    major = names.index[names.isin(MAJOR_HOLIDAYS)]
    minor = names.index[~names.isin(MAJOR_HOLIDAYS)]
    thanksgiving = names.index[names == "Thanksgiving Day"]

    out = pd.DataFrame(index=idx)
    out["hol_major"] = idx.isin(major).astype(int)
    out["hol_minor"] = idx.isin(minor).astype(int)
    eve = ((idx.month == 12) & idx.day.isin([24, 31])) | idx.isin(thanksgiving - pd.Timedelta(days=1))
    out["hol_eve"] = (eve & (out["hol_major"] == 0)).astype(int)
    weekend_after = thanksgiving.repeat(3) + pd.to_timedelta([1, 2, 3] * len(thanksgiving), unit="D")
    out["thanksgiving_weekend"] = idx.isin(weekend_after).astype(int)
    fixed = ((idx.month == 1) & (idx.day == 1)) | ((idx.month == 7) & (idx.day == 4)) | ((idx.month == 12) & (idx.day == 25))
    out["hol_fixed_weekend"] = (fixed & (idx.day_of_week >= 5)).astype(int)
    out["winter_break"] = (((idx.month == 12) & (idx.day >= 24)) | ((idx.month == 1) & (idx.day <= 2))).astype(int)
    out["major_event"] = idx.isin(recurring_events(range(idx.year.min(), idx.year.max() + 1))).astype(int)
    return out
