"""PUMA time study, step 1: ACS 1-year tables and 2010 PUMA polygons.

A PUMA (public use microdata area) holds about 100,000 people. ACS 1-year estimates exist for every PUMA every year
(except 2020), which gives a yearly income series that tract data cannot. Years 2012-2019 all use the 2010 PUMA boundaries.

Outputs (gitignored):
    data/raw/acs1_puma.csv      every PUMA of the states in config.PUMA_STATES, years in config.PUMA["acs_years"]
    data/raw/pumas_2010.gpkg    PUMA polygons (GEOID, ALAND, city, geometry)

Needs the Census key in .env (same as script 02). Run from the repo root:  python scripts/11_fetch_puma.py
If the Census API says the PUMA geography is not available for the 1-year product, the message is printed as is
and nothing is written: tell me and the fetch can switch to the PUMS microdata API.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import geopandas as gpd  # noqa: E402
import pandas as pd  # noqa: E402

from activity_index import config as cfg  # noqa: E402
from activity_index.census import fetch_acs_puma, fetch_pumas  # noqa: E402


def main():
    cfg.RAW.mkdir(parents=True, exist_ok=True)
    parts = []
    for year in cfg.PUMA["acs_years"]:
        for state, city in cfg.PUMA_STATES.items():
            a = fetch_acs_puma(year, state, cfg.ACS_VARS)
            a["year"], a["state"], a["city"] = year, state, city
            parts.append(a)
        print(f"ACS 1-year {year}: {sum(len(p) for p in parts if p['year'].iloc[0] == year)} PUMAs")
    acs = pd.concat(parts, ignore_index=True)
    acs.to_csv(cfg.RAW / "acs1_puma.csv", index=False)

    geo = []
    for state, city in cfg.PUMA_STATES.items():
        g = fetch_pumas(cfg.PUMA["tiger_year"], state, cfg.RAW / "tiger")
        g["city"] = city
        geo.append(g)
    geo = gpd.GeoDataFrame(pd.concat(geo, ignore_index=True), crs=geo[0].crs)
    geo.to_file(cfg.RAW / "pumas_2010.gpkg", driver="GPKG")

    print(f"\nsaved {len(acs)} PUMA-year rows and {len(geo)} PUMA polygons")
    missing = set(acs["GEOID"]) - set(geo["GEOID"])
    print(f"ACS PUMAs without a polygon: {len(missing)} (should be 0; a mismatch means the PUMA vintage differs)")
    t = acs.assign(has_income=acs["agg_household_income"].notna()).pivot_table(
        index="city", columns="year", values="GEOID", aggfunc="count")
    print("\nPUMAs per city and year (whole states, before choosing the bikeshare ones):")
    print(t.to_string())
    print("\nshare of PUMA-years with income:", round(acs["agg_household_income"].notna().mean(), 3))


if __name__ == "__main__":
    main()
