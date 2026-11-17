"""Whose trips are these? Split each area's matched-station trip growth by who is most likely riding (scripts/27_trip_purpose.py).

A bikeshare trip has no rider address, so the split uses time of day and direction, per area and year, on matched station-months:
  home_end  departures in the morning peak + arrivals in the evening peak  -> people who LIVE in the area (leave in the morning, come back)
  work_end  arrivals in the morning peak + departures in the evening peak  -> people who COME to the area to work
  offpeak   every other departure (midday, night, most of the weekend)
  weekend / weekday, member / casual departures
Peak hours are AM_HOURS and PM_HOURS in features.py; the peak counts include weekend days (the aggregates do not separate them), which
blurs the split a little. Growth is log(trips in year T / trips in year T-1) over station-months present in both years.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import panel as P

PARTS = {"home_end": "resident-like: AM departures + PM arrivals", "work_end": "commuter-inflow: AM arrivals + PM departures",
         "offpeak": "off-peak departures", "weekday": "weekday departures", "weekend": "weekend departures",
         "member": "member departures", "casual": "casual departures"}


def _parts(g: pd.DataFrame, s: int) -> pd.DataFrame:
    dep = g[f"dep_{s}"]
    return pd.DataFrame({"home_end": g[f"dep_am_{s}"] + g[f"arr_pm_{s}"], "work_end": g[f"arr_am_{s}"] + g[f"dep_pm_{s}"],
                         "offpeak": dep - g[f"dep_am_{s}"] - g[f"dep_pm_{s}"], "weekday": g[f"dep_weekday_{s}"],
                         "weekend": dep - g[f"dep_weekday_{s}"], "member": g[f"dep_member_{s}"], "casual": dep - g[f"dep_member_{s}"],
                         "all": dep})


def purpose_growth(sm: pd.DataFrame, years, min_matched: int = 3) -> pd.DataFrame:
    """One row per area-year: g_<part> = log growth of that kind of trip, and home_share = home_end / (home_end + work_end) in T-1."""
    rows = []
    for y in years:
        pairs = P.matched_pairs(sm, y)
        if pairs.empty:
            continue
        a = _parts(pairs, 0).groupby(pairs["puma"]).sum()
        b = _parts(pairs, 1).groupby(pairs["puma"]).sum()
        out = pd.DataFrame({f"g_{k}": np.log(b[k] / a[k].replace(0, np.nan)) for k in list(PARTS) + ["all"]})
        out["home_share"] = a["home_end"] / (a["home_end"] + a["work_end"])
        out.loc[pairs.groupby("puma")["station_id"].nunique() < min_matched, :] = np.nan
        rows.append(out.assign(year=y).rename_axis("GEOID").reset_index())
    return pd.concat(rows, ignore_index=True)
