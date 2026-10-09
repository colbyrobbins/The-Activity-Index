"""Build the modelling table: one row per 2010 tract, with income targets, controls and bikeshare features.

Steps: give every station coordinates (trip file, then station files bundled in other zips, then the
current GBFS feed), drop depot/test stations, assign stations to 2010 tracts, compute station features
and aggregate them to tracts, then attach the ACS controls and the prior-window "momentum".

Needs: scripts 02, 04 and 05 done. Run from the repo root:  python scripts/06_build_tract_table.py
Writes (gitignored): data/processed/stations.csv and data/processed/tract_table.csv
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from activity_index import config as cfg  # noqa: E402
from activity_index.features import haversine_km, station_features, tract_features  # noqa: E402
from activity_index.stations import clean_stations  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
ID = {"station_id": str, "start_id": str, "end_id": str}


def main():
    cfg.PROCESSED.mkdir(parents=True, exist_ok=True)
    names = pd.read_csv(cfg.INTERIM / "station_names.csv.gz", dtype=ID)
    sm_all = pd.read_csv(cfg.INTERIM / "station_month.csv.gz", dtype=ID)
    od_all = pd.read_csv(cfg.INTERIM / "od_year.csv.gz", dtype=ID)
    tracts = gpd.read_file(cfg.RAW / f"tracts_{cfg.ACS_T0['label']}.gpkg")
    income = pd.read_csv(cfg.INTERIM / "tract_income.csv", dtype={"GEOID": str})

    stations, report = [], []
    for city in cfg.CITIES:
        if city not in set(names["city"]):
            print(f"[{city}] no aggregated trips; run scripts/05_aggregate_trips.py")
            continue
        t, rep = clean_stations(city, names)
        report.append(rep)
        pts = gpd.GeoDataFrame(t, geometry=gpd.points_from_xy(t["lon"], t["lat"]), crs="EPSG:4326").to_crs(tracts.crs)
        j = gpd.sjoin(pts, tracts[tracts["city"] == city][["GEOID", "geometry"]], predicate="within", how="left")
        j = j[~j.index.duplicated()]
        report[-1]["outside_tracts"] = int(j["GEOID"].isna().sum())
        t["tract"] = j["GEOID"]
        t = t.dropna(subset=["tract"])
        sm, od = sm_all[sm_all["city"] == city], od_all[od_all["city"] == city]
        sf = station_features(sm, od, t[["station_id", "lat", "lon"]], cfg.ACTIVITY_YEARS)
        stations.append(t.merge(sf, on="station_id", how="inner"))
    st = pd.concat(stations, ignore_index=True)
    st.to_csv(cfg.PROCESSED / "stations.csv", index=False)
    print("\nstation coordinates and tract assignment:")
    print(pd.DataFrame(report).round(3).to_string(index=False))

    tf = tract_features(st).rename(columns={"tract": "GEOID"})

    # ---- controls from the T0 ACS, density and distance to downtown ----
    a = pd.read_csv(cfg.RAW / f"acs_{cfg.ACS_T0['label']}.csv", dtype={"GEOID": str})
    ctr = tracts[["GEOID", "ALAND"]].copy()
    cent = tracts.to_crs(5070).centroid.to_crs(4326)
    ctr["clat"], ctr["clon"] = cent.y.values, cent.x.values
    a = a.merge(ctr, on="GEOID", how="left")
    c = pd.DataFrame({"GEOID": a["GEOID"]})
    c["edu_share_ba"] = (a["edu_bachelors"] + a["edu_masters"] + a["edu_professional"] + a["edu_doctorate"]) / a["edu_total"]
    c["renter_share"] = a["renter_households"] / a["households"]
    c["median_gross_rent"], c["median_home_value"], c["median_age"] = a["median_gross_rent"], a["median_home_value"], a["median_age"]
    with np.errstate(divide="ignore"):
        c["log_density"] = np.log(a["population"] / (a["ALAND"] / 1e6)).replace(-np.inf, np.nan)
    c["log_households"] = np.log(a["households"].where(a["households"] > 0))
    lat0 = a["city"].map(lambda k: cfg.CITY_CENTERS[k][0]); lon0 = a["city"].map(lambda k: cfg.CITY_CENTERS[k][1])
    c["km_to_center"] = haversine_km(a["clat"], a["clon"], lat0, lon0)
    c["clat"], c["clon"] = a["clat"], a["clon"]

    # ---- momentum: how far the tract's income rank moved BEFORE the study period ----
    pr_path = cfg.RAW / f"acs_{cfg.PRIOR_ACS['label']}.csv"
    if pr_path.exists():
        p = pd.read_csv(pr_path, dtype={"GEOID": str})
        p["mean_income_prior"] = (p["agg_household_income"] / p["households"]).where(p["households"] >= 50)
        p["income_rank_prior"] = p.groupby("city")["mean_income_prior"].rank(pct=True)
        c = c.merge(p[["GEOID", "income_rank_prior"]], on="GEOID", how="left")
    else:
        print(f"\n(no {pr_path.name}: run `python scripts/02_fetch_acs_tiger.py --prior-only` to enable the momentum baseline)")

    tab = (income.merge(c, on="GEOID", how="left").merge(tf, on="GEOID", how="left"))
    if "income_rank_prior" in tab:
        tab["momentum"] = tab["income_rank_t0"] - tab["income_rank_prior"]
    for col in ("n_stations", "n_mature", "trips_total"):
        tab[col] = tab[col].fillna(0)
    tab["in_scope"] = (tab["n_mature"] >= 1) & tab["rank_change"].notna() & tab["income_rank_t0"].notna()
    tab.to_csv(cfg.PROCESSED / "tract_table.csv", index=False)

    print(f"\ntract table: {len(tab)} tracts; with a rank change: {tab['rank_change'].notna().sum()}; "
          f"in scope (>=1 mature station): {int(tab['in_scope'].sum())}")
    print(tab.groupby("city").agg(tracts=("GEOID", "size"), with_station=("n_stations", lambda s: int((s > 0).sum())),
                                  in_scope=("in_scope", "sum")))
    print("\nwrote", cfg.PROCESSED / "tract_table.csv")


if __name__ == "__main__":
    main()
