"""Weather control for trip growth (scripts/26_weather_control.py).

The nowcast feature is an area's year-over-year trip growth on matched months, minus the city-year median. A wet or cold year moves
every area of a city together, and that common part is already removed by the city-year median. What could remain is a DIFFERENCE in
weather sensitivity: an area whose riders are mostly casual loses more trips in a rainy month than a commuter area does, and that
would look like weaker "activity". So:
  1. monthly_growth: each area's trip growth for calendar month m, year T vs T-1 (stations active in that month of both years).
  2. sensitivities: per area, regress monthly growth on the city's change in temperature, wet days and precipitation for that month.
     Fitted on all years: a sensitivity is a trait of the riders, and no income figure enters it.
  3. adjusted_growth: remove each month's weather effect from year-T trips, then rebuild the annual growth.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import panel as P

WX = ["d_temp", "d_wet", "d_precip"]          # change vs the same month a year earlier: mean temperature (C), wet days, precipitation (mm)
MIN_TRIPS = 200                               # area-months with fewer trips in either year are too noisy to fit on


def monthly_weather(daily: pd.DataFrame) -> pd.DataFrame:
    """city, ym, temp (mean C), wet (days with >= 1 mm), precip (mm), snow (cm)."""
    d = daily.assign(date=pd.to_datetime(daily["date"]))
    d["ym"] = d["date"].dt.year * 100 + d["date"].dt.month
    return d.groupby(["city", "ym"]).agg(temp=("temp_c", "mean"), wet=("precip_mm", lambda s: float((s >= 1.0).sum())),
                                         precip=("precip_mm", "sum"), snow=("snow_cm", "sum")).reset_index()


def weather_changes(monthly: pd.DataFrame) -> pd.DataFrame:
    """city, year, m, d_temp, d_wet, d_precip: this month minus the same month one year earlier."""
    w = monthly.assign(year=monthly["ym"] // 100, m=monthly["ym"] % 100)
    prev = w.assign(year=w["year"] + 1)[["city", "year", "m", "temp", "wet", "precip"]]
    j = w.merge(prev, on=["city", "year", "m"], suffixes=("", "_prev"))
    for c in ("temp", "wet", "precip"):
        j[f"d_{c}"] = j[c] - j[f"{c}_prev"]
    return j[["city", "year", "m"] + WX]


def monthly_growth(sm: pd.DataFrame, years) -> pd.DataFrame:
    """GEOID, year, m, trips_0, trips_1, g (log growth) on stations active in month m of both years."""
    rows = []
    for y in years:
        pairs = P.matched_pairs(sm, y)
        if len(pairs):
            t = pairs.groupby(["puma", "m"], as_index=False)[["dep_0", "dep_1"]].sum()
            rows.append(t.assign(year=y))
    out = pd.concat(rows, ignore_index=True).rename(columns={"puma": "GEOID", "dep_0": "trips_0", "dep_1": "trips_1"})
    out["g"] = np.log(out["trips_1"] / out["trips_0"])
    return out


def _ols(X, y, w=None):
    X = np.column_stack([np.ones(len(y)), X])
    sw = np.ones(len(y)) if w is None else np.sqrt(np.asarray(w, float))
    beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    res = y - X @ beta
    wt = sw ** 2
    r2 = 1 - (wt * res ** 2).sum() / (wt * (y - np.average(y, weights=wt)) ** 2).sum()
    return beta, float(r2)


def sensitivities(mg: pd.DataFrame) -> pd.DataFrame:
    """Per area: slopes of monthly trip growth on the weather changes (trip-weighted), with the fit's R2 and month count."""
    rows = []
    for geo, t in mg[(mg["trips_0"] >= MIN_TRIPS) & (mg["trips_1"] >= MIN_TRIPS)].dropna(subset=WX).groupby("GEOID"):
        if len(t) < 24:
            continue
        beta, r2 = _ols(t[WX].values, t["g"].values, (t["trips_0"] + t["trips_1"]).values / 2)
        rows.append({"GEOID": geo, "n_months": len(t), "r2": r2, **{f"b_{c}": b for c, b in zip(WX, beta[1:])}})
    return pd.DataFrame(rows)


def adjusted_growth(mg: pd.DataFrame, sens: pd.DataFrame) -> pd.DataFrame:
    """GEOID, year, g_raw, g_wx_adj (weather effect removed month by month), g_wx (the part attributed to weather)."""
    d = mg.merge(sens, on="GEOID", how="left").dropna(subset=WX)
    effect = sum(d[f"b_{c}"].fillna(0) * d[c] for c in WX)
    d["trips_1_adj"] = d["trips_1"] * np.exp(-effect)
    a = d.groupby(["GEOID", "year"])[["trips_0", "trips_1", "trips_1_adj"]].sum()
    out = pd.DataFrame({"g_raw": np.log(a["trips_1"] / a["trips_0"]), "g_wx_adj": np.log(a["trips_1_adj"] / a["trips_0"])})
    out["g_wx"] = out["g_raw"] - out["g_wx_adj"]
    return out.reset_index()


def explained(mg: pd.DataFrame) -> pd.DataFrame:
    """How much of monthly trip growth does weather explain? City-wide growth, then each area's growth relative to its city-month."""
    d = mg.dropna(subset=WX)
    d = d[(d["trips_0"] >= MIN_TRIPS) & (d["trips_1"] >= MIN_TRIPS)].copy()
    city = d.groupby(["city", "year", "m"]).agg(t0=("trips_0", "sum"), t1=("trips_1", "sum"), **{c: (c, "first") for c in WX}).reset_index()
    city["g"] = np.log(city["t1"] / city["t0"])
    beta, r2_city = _ols(city[WX].values, city["g"].values)
    d["g_rel"] = d["g"] - d.groupby(["city", "year", "m"])["g"].transform("median")
    sse = sst = 0.0
    for _, t in d.groupby("GEOID"):
        if len(t) < 24:
            continue
        b, _ = _ols(t[WX].values, t["g_rel"].values)
        res = t["g_rel"].values - np.column_stack([np.ones(len(t)), t[WX].values]) @ b
        sse, sst = sse + (res ** 2).sum(), sst + ((t["g_rel"] - t["g_rel"].mean()) ** 2).sum()
    return pd.DataFrame([
        {"what": "city-wide monthly trip growth", "n": len(city), "r2_weather": r2_city, "per_degree_c": beta[1], "per_wet_day": beta[2],
         "per_mm_precip": beta[3]},
        {"what": "area growth relative to its city-month (area-specific slopes)", "n": len(d), "r2_weather": 1 - sse / sst}])
