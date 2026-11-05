"""Descriptive analysis behind the dashboard: long-run trend, pandemic recovery,
weekly/seasonal pattern, and the effect of weather, holidays and events."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from .calendar_features import HOLIDAY_COLUMNS, MAJOR_HOLIDAYS, holiday_table, recurring_events

DAY_TYPES = {"W": "weekday", "A": "saturday", "U": "sunday_holiday"}
EVENT_LABELS = {10: "Chicago Marathon", 3: "St. Patrick's parade", 6: "Pride Parade"}
HOLIDAY_LABELS = {
    "Birthday of Martin Luther King, Jr.": "MLK Day",
    "Washington's Birthday": "Presidents Day",
    "Juneteenth National Independence Day": "Juneteenth",
}


def observed(daily: pd.DataFrame) -> pd.DataFrame:
    return daily[daily["total"].notna()]


def is_regular_day(daily: pd.DataFrame) -> pd.Series:
    return daily[HOLIDAY_COLUMNS].sum(axis=1) == 0


def weekly_history(daily: pd.DataFrame) -> pd.DataFrame:
    """Average daily boardings per Monday-Sunday week (complete weeks only), labelled by week start."""
    d = observed(daily)[["bus", "rail", "total"]]
    means = d.resample("W-SUN").mean()
    counts = d["total"].resample("W-SUN").count()
    out = means[counts == 7]
    out.index = out.index - pd.Timedelta(days=6)
    return out.round(0)


def recovery_vs_2019(daily: pd.DataFrame) -> pd.DataFrame:
    """Average daily boardings by month and service-day type, as a share of the same month in 2019."""
    d = observed(daily)
    monthly = d.groupby([d.index.to_period("M"), "day_type"])["total"].mean().unstack()
    base = monthly[monthly.index.year == 2019]
    base.index = base.index.month
    recent = monthly[monthly.index >= pd.Period("2020-01", "M")]
    ratio = recent / base.reindex(recent.index.month).to_numpy()
    ratio = ratio.rename(columns=DAY_TYPES)[list(DAY_TYPES.values())]
    ratio.index = ratio.index.to_timestamp()
    return ratio.round(4)


def seasonality_grid(daily: pd.DataFrame, months: int = 24) -> pd.DataFrame:
    """Mean boardings by weekday x calendar month over the last `months` months (holidays and events excluded)."""
    d = observed(daily)
    start = d.index.max() - pd.DateOffset(months=months) + pd.Timedelta(days=1)
    d = d[(d.index >= start) & is_regular_day(d)]
    return d.groupby(["dow", "month"])["total"].mean().unstack("month").round(0)


def expected_same_weekday(daily: pd.DataFrame) -> pd.Series:
    """'Typical' ridership for each day: the median of the same weekday 1-3 weeks either side,
    skipping neighbours that are holidays or events. Descriptive only (it looks ahead), used to
    isolate weather and holiday effects."""
    total = daily["total"].where(is_regular_day(daily))
    neighbours = pd.concat([total.shift(k) for k in (-21, -14, -7, 7, 14, 21)], axis=1)
    return neighbours.median(axis=1, skipna=True).where(neighbours.notna().sum(axis=1) >= 3)


def _band_effects(ratio: pd.Series, bands: pd.Series, order: list[str]) -> list[dict]:
    rows = []
    for label in order:
        r = ratio[bands == label].dropna()
        if len(r) >= 5:
            rows.append({"band": label, "effect": round(float(r.median() - 1), 4), "days": int(len(r))})
    return rows


def weather_effects(daily: pd.DataFrame, start: str = "2022-07-01") -> dict[str, list[dict]]:
    """Median % change versus a typical same-weekday day, by weather band (regular days only)."""
    d = observed(daily)
    d = d[d.index >= pd.Timestamp(start)]
    ratio = (d["total"] / expected_same_weekday(daily).reindex(d.index)).where(is_regular_day(d))

    temp_order = ["Below 10°F", "10–32°F", "32–50°F", "50–70°F", "70–85°F", "Above 85°F"]
    snow_order = ["No snow", "Up to 1 in", "1–3 in", "3–6 in", "6+ in"]
    rain_order = ["Dry", "Under 0.1 in", "0.1–0.5 in", "0.5+ in"]
    temp = pd.cut(d["temp_mean_f"], [-60, 10, 32, 50, 70, 85, 130], labels=temp_order).astype(str)
    snow = pd.cut(d["snow_in"], [-0.01, 0.0, 1, 3, 6, 100], labels=snow_order).astype(str)
    rain = pd.cut(d["precip_in"], [-0.01, 0.0, 0.1, 0.5, 20], labels=rain_order).astype(str)
    rain = rain.where(d["snow_in"] == 0)  # rain bands only on snow-free days
    return {
        "temperature": _band_effects(ratio, temp, temp_order),
        "snow": _band_effects(ratio, snow, snow_order),
        "rain": _band_effects(ratio, rain, rain_order),
    }


def special_days(start, end) -> pd.DataFrame:
    """Named holidays, eves and recurring events between start and end, with a kind label."""
    flags = holiday_table(start, end)
    names = USFederalHolidayCalendar().holidays(start, end, return_name=True)
    rows = {}
    for day, name in names.items():
        rows[day] = (HOLIDAY_LABELS.get(name, name), "major" if name in MAJOR_HOLIDAYS else "minor")
    for day in flags.index[flags["hol_eve"] == 1]:
        label = "Day before Thanksgiving" if day.month == 11 else ("Christmas Eve" if day.day == 24 else "New Year's Eve")
        rows[day] = (label, "eve")
    for day in flags.index[(flags["thanksgiving_weekend"] == 1) & (flags.index.day_of_week == 4)]:
        rows[day] = ("Day after Thanksgiving", "eve")
    for day in recurring_events(range(flags.index.year.min(), flags.index.year.max() + 1)):
        if flags.index.min() <= day <= flags.index.max():
            rows[day] = (EVENT_LABELS[day.month], "event")
    out = pd.DataFrame.from_dict(rows, orient="index", columns=["name", "kind"]).sort_index()
    out.index.name = "date"
    return out


def holiday_effects(daily: pd.DataFrame, start: str = "2022-07-01") -> list[dict]:
    """Average % change versus a typical same-weekday day for each holiday and recurring event."""
    d = observed(daily)
    d = d[d.index >= pd.Timestamp(start)]
    ratio = (d["total"] / expected_same_weekday(daily).reindex(d.index)).dropna()
    days = special_days(d.index.min(), d.index.max())
    days = days[days.index.isin(ratio.index)]
    rows = []
    for (name, kind), g in days.groupby(["name", "kind"]):
        r = ratio.reindex(g.index)
        rows.append({"name": name, "kind": kind, "effect": round(float(r.mean() - 1), 4), "days": int(len(r))})
    return sorted(rows, key=lambda x: x["effect"])


def headline_kpis(daily: pd.DataFrame) -> dict:
    d = observed(daily)
    end = d.index.max()
    last90 = d[d.index > end - pd.Timedelta(days=90)]
    prior90 = d[(d.index > end - pd.Timedelta(days=455)) & (d.index <= end - pd.Timedelta(days=365))]
    wk_now = last90.loc[last90["day_type"] == "W", "total"].mean()
    wk_prior = prior90.loc[prior90["day_type"] == "W", "total"].mean()
    last12 = d[d.index > end - pd.DateOffset(years=1)]
    prev12 = d[(d.index > end - pd.DateOffset(years=2)) & (d.index <= end - pd.DateOffset(years=1))]
    # The same 12 calendar months immediately before the pandemic.
    pre_end = end - pd.DateOffset(years=end.year - 2019)
    pre = d[(d.index > pre_end - pd.DateOffset(years=1)) & (d.index <= pre_end)]
    recent_by_type = last12.groupby("day_type")["total"].mean()
    pre_by_type = pre.groupby("day_type")["total"].mean()
    return {
        "data_end": end.strftime("%Y-%m-%d"),
        "weekday_avg_90d": round(float(wk_now)),
        "weekday_avg_90d_yoy": round(float(wk_now / wk_prior - 1), 4),
        "rides_12m": int(last12["total"].sum()),
        "rides_12m_yoy": round(float(last12["total"].sum() / prev12["total"].sum() - 1), 4),
        "recovery_12m_vs_2019": round(float(last12["total"].sum() / pre["total"].sum()), 4),
        "recovery_by_day_type": {DAY_TYPES[k]: round(float(recent_by_type[k] / pre_by_type[k]), 4) for k in DAY_TYPES},
        "bus_share_12m": round(float(last12["bus"].sum() / last12["total"].sum()), 4),
        "bus_share_2019": round(float(pre["bus"].sum() / pre["total"].sum()), 4),
        "rides_2019_12m": int(pre["total"].sum()),
        "peak_day_12m": {"date": last12["total"].idxmax().strftime("%Y-%m-%d"), "rides": int(last12["total"].max())},
    }


def annual_totals(daily: pd.DataFrame) -> pd.DataFrame:
    """Total boardings per complete calendar year."""
    d = observed(daily)
    counts = d.groupby(d.index.year)["total"].count()
    sums = d.groupby(d.index.year)[["bus", "rail", "total"]].sum()
    return sums[counts >= 365].astype(np.int64)
