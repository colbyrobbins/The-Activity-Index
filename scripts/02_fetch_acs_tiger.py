"""Fetch ACS tract tables and TIGER tract polygons for both windows of the active config.

Outputs (gitignored):
    data/raw/acs_<label>.csv       e.g. acs_2014-18.csv, acs_2019-23.csv   (column `city`)
    data/raw/tracts_<label>.gpkg   tract polygons in that window's tract vintage (2010 or 2020)
    data/raw/acs_<prior label>.csv  the window before T0 (e.g. acs_2009-13.csv), for the momentum baseline

Optional: set CENSUS_API_KEY (free) to avoid rate limits.
    python scripts/02_fetch_acs_tiger.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import geopandas as gpd  # noqa: E402
import pandas as pd  # noqa: E402

from activity_index import config as cfg  # noqa: E402
from activity_index.census import fetch_acs, fetch_tracts  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior-only", action="store_true", help="fetch only the pre-T0 ACS window")
    if ap.parse_args().prior_only:
        cfg.RAW.mkdir(parents=True, exist_ok=True)
        return fetch_prior()
    cfg.RAW.mkdir(parents=True, exist_ok=True)
    for win in (cfg.ACS_T0, cfg.ACS_T1):
        acs_parts, geo_parts = [], []
        for city, states in cfg.CITIES.items():
            for state, counties in states.items():
                print(f"{win['label']} {city}: state {state} counties {counties}")
                a = fetch_acs(win["year"], state, counties, cfg.ACS_VARS)
                a["city"] = city
                acs_parts.append(a)
                g = fetch_tracts(win["year"], state, counties, cfg.RAW / "tiger")
                g["city"] = city
                geo_parts.append(g)
        acs = pd.concat(acs_parts, ignore_index=True)
        geo = gpd.GeoDataFrame(pd.concat(geo_parts, ignore_index=True), crs=geo_parts[0].crs)
        acs.to_csv(cfg.RAW / f"acs_{win['label']}.csv", index=False)
        geo.to_file(cfg.RAW / f"tracts_{win['label']}.gpkg", driver="GPKG")
        print(f"  saved {len(acs)} ACS rows and {len(geo)} tract polygons "
              f"(tract vintage {win['tract_vintage']})")
        print(acs.groupby("city")["agg_household_income"].agg(["size", lambda s: s.notna().mean()])
              .rename(columns={"size": "tracts", "<lambda_0>": "income_nonnull"}).round(2))
    fetch_prior()


def fetch_prior():
    """ACS for the window before T0 (same 2010 tracts), used only for the momentum baseline."""
    pr = cfg.PRIOR_ACS
    parts = []
    for city, states in cfg.CITIES.items():
        for state, counties in states.items():
            print(f"{pr['label']} {city}: state {state} counties {counties}")
            a = fetch_acs(pr["year"], state, counties, cfg.ACS_VARS)
            a["city"] = city
            parts.append(a)
    acs = pd.concat(parts, ignore_index=True)
    acs.to_csv(cfg.RAW / f"acs_{pr['label']}.csv", index=False)
    print(f"  saved {len(acs)} ACS rows for the prior window {pr['label']}")


if __name__ == "__main__":
    main()
