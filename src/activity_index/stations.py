"""One clean table of station coordinates per city, shared by the tract study (06) and the PUMA study (12)."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config as cfg
from .features import haversine_km
from .trips import fill_start_coords, load_gbfs_stations, load_station_history

DEPOT = re.compile(r"\b(?:test|depot|warehouse|repair|cassette|map frame|mobile|workshop|temp|shop)\b", re.I)
MAX_KM_FROM_CENTER = 60


def station_table(city: str, names: pd.DataFrame) -> pd.DataFrame:
    """Coordinates for every station of one city, with the source of each coordinate.

    Order of preference: coordinates inside the trip files, station files bundled in other zips, the current
    GBFS feed. `names` is station_names.csv.gz (all cities)."""
    s = names[names["city"] == city].copy()
    df = pd.DataFrame({"start_station_id": s["station_id"], "start_station_name": s["name"],
                       "start_lat": s["lat"], "start_lon": s["lon"]})
    df["coord_source"] = np.where(df["start_lat"].notna(), "trip_file", None)
    zips = sorted((cfg.RAW / "trips" / city).glob("*.zip"))
    hist = load_station_history(zips) if zips else None
    try:
        gbfs = load_gbfs_stations(cfg.GBFS[city])
    except Exception as e:
        print(f"  [{city}] GBFS feed unavailable ({type(e).__name__}); continuing without it")
        gbfs = None
    for label, src in (("station_file", hist), ("gbfs", gbfs)):
        before = df["start_lat"].notna()
        df = fill_start_coords(df, src)
        df.loc[df["start_lat"].notna() & ~before, "coord_source"] = label
    out = df.rename(columns={"start_station_id": "station_id", "start_station_name": "name",
                             "start_lat": "lat", "start_lon": "lon"})
    out["city"] = city
    out["trips"] = s["n"].values
    return out


def clean_stations(city: str, names: pd.DataFrame):
    """station_table + drop stations without coordinates, depots/test stations, and stations far from the center.

    Returns (kept stations, report dict)."""
    t = station_table(city, names)
    n0, trips0 = len(t), t["trips"].sum()
    has = t["lat"].notna()
    lat0, lon0 = cfg.CITY_CENTERS[city]
    km = pd.Series(haversine_km(t["lat"], t["lon"], lat0, lon0), index=t.index)
    keep = has & (km <= MAX_KM_FROM_CENTER) & ~t["name"].fillna("").str.contains(DEPOT)
    report = {"city": city, "stations": n0, "with_coords": int(has.sum()),
              "trip_share_with_coords": t.loc[has, "trips"].sum() / trips0,
              "dropped_depot_or_far": int((has & ~keep).sum())}
    return t[keep].copy(), report
