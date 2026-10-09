"""Load and join the inputs of the PUMA panel (needs geopandas): stations -> PUMAs, ACS 1-year tables, PUMA polygons."""
from __future__ import annotations

import pandas as pd

from . import config as cfg
from .stations import clean_stations

ID = {"station_id": str, "start_id": str, "end_id": str}


def stations_to_pumas(names: pd.DataFrame, pumas):
    """Clean station coordinates per city and assign each station to the PUMA that contains it."""
    import geopandas as gpd

    parts, report = [], []
    for city in cfg.CITIES:
        if city not in set(names["city"]):
            print(f"[{city}] no panel trip aggregates; run scripts/05_aggregate_trips.py --panel")
            continue
        t, rep = clean_stations(city, names)
        pts = gpd.GeoDataFrame(t, geometry=gpd.points_from_xy(t["lon"], t["lat"]), crs="EPSG:4326").to_crs(pumas.crs)
        j = gpd.sjoin(pts, pumas[pumas["city"] == city][["GEOID", "geometry"]], predicate="within", how="left")
        j = j[~j.index.duplicated()]
        rep["outside_pumas"] = int(j["GEOID"].isna().sum())
        report.append(rep)
        t["puma"] = j["GEOID"]
        parts.append(t.dropna(subset=["puma"])[["city", "station_id", "name", "puma", "lat", "lon"]])
    return pd.concat(parts, ignore_index=True), pd.DataFrame(report)


def load_panel_inputs():
    """Returns (station_month rows with a puma column, ACS 1-year rows, PUMA table with centroids, station report)."""
    import geopandas as gpd

    pumas = gpd.read_file(cfg.RAW / "pumas_2010.gpkg")
    cent = pumas.to_crs(5070).centroid.to_crs(4326)
    pumas["clat"], pumas["clon"] = cent.y.values, cent.x.values
    names = pd.read_csv(cfg.INTERIM / "station_names_panel.csv.gz", dtype=ID)
    sm = pd.read_csv(cfg.INTERIM / "station_month_panel.csv.gz", dtype=ID)
    acs = pd.read_csv(cfg.RAW / "acs1_puma.csv", dtype={"GEOID": str, "state": str})
    st, report = stations_to_pumas(names, pumas)
    sm = sm.merge(st[["city", "station_id", "puma"]], on=["city", "station_id"], how="inner")
    return sm, acs, pumas, report
