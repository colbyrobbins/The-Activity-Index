"""Inputs of the nowcast area panel (needs geopandas): stations -> one key per physical station -> area, plus the summed ACS rows."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg
from .features import DEP_COLS, ARR_COLS
from .station_keys import canonical_keys
from .stations import clean_stations

ID = {"station_id": str, "start_id": str, "end_id": str}


def load_area_inputs():
    """Returns (station_month rows with `puma` = area id and `station_id` = canonical key, area ACS rows, area table, report).

    Needs scripts 05 --areas (station_*_areas), 20 (areas.csv, area_acs.csv, areas.gpkg)."""
    import geopandas as gpd

    areas = gpd.read_file(cfg.PROCESSED / "areas.gpkg")
    mem = pd.read_csv(cfg.PROCESSED / "areas.csv", dtype={"GEOID": str})
    city_of = mem.drop_duplicates("area_id").set_index("area_id")["city"]
    names = pd.read_csv(cfg.INTERIM / "station_names_areas.csv.gz", dtype=ID)
    sm = pd.read_csv(cfg.INTERIM / "station_month_areas.csv.gz", dtype=ID)

    parts, report = [], []
    for city in cfg.CITIES:
        if city not in set(names["city"]):
            print(f"[{city}] no trip aggregates in station_names_areas; run scripts/05_aggregate_trips.py --areas --cities {city}")
            continue
        t, rep = clean_stations(city, names)
        parts.append(t)
        report.append(rep)
    st = pd.concat(parts, ignore_index=True)
    act = sm[sm["dep"] > 0].groupby(["city", "station_id"])["ym"].agg(first_ym="min", last_ym="max").reset_index()
    st = st.merge(act, on=["city", "station_id"], how="inner")
    keys = canonical_keys(st[["city", "station_id", "lat", "lon", "first_ym", "last_ym"]], cfg.AREAS["link_radius_m"])
    st = st.merge(keys, on=["city", "station_id"])

    pts = gpd.GeoDataFrame(st, geometry=gpd.points_from_xy(st["lon"], st["lat"]), crs="EPSG:4326").to_crs(areas.crs)
    j = gpd.sjoin(pts, areas[["area_id", "geometry"]], predicate="within", how="left")
    j = j[~j.index.duplicated()]
    st["area_id"] = j["area_id"].values
    st = st[st["area_id"].notna() & (st["city"] == st["area_id"].map(city_of))]
    key_area = st.groupby("key")["area_id"].agg(lambda s: s.value_counts().index[0])          # a key lives in one area
    st["area_id"] = st["key"].map(key_area)

    sm = sm.merge(st[["city", "station_id", "key", "area_id"]], on=["city", "station_id"], how="inner")
    num = DEP_COLS + ARR_COLS
    sm = (sm.groupby(["area_id", "key", "ym"], as_index=False)[num].sum()
          .rename(columns={"area_id": "puma", "key": "station_id"}))

    rep = pd.DataFrame(report)
    link = st.groupby("city").agg(station_ids=("station_id", "size"), keys=("key", "nunique"), ids_linked_across_eras=("linked", "sum"))
    rep = rep.merge(link.reset_index(), on="city", how="left")

    a5 = areas.to_crs(5070)
    cent = a5.centroid.to_crs(4326)
    table = pd.DataFrame({"GEOID": areas["area_id"], "ALAND": a5.geometry.area.values, "clat": cent.y.values, "clon": cent.x.values,
                          "city": areas["area_id"].map(city_of).values})
    acs = pd.read_csv(cfg.PROCESSED / "area_acs.csv", dtype={"GEOID": str})
    acs["NAME"] = acs["GEOID"]
    return sm, acs, table, rep
