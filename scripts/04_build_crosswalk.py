"""Put the 2019-23 ACS counts onto 2010 tracts, then build the income table both windows share.

Needs the outputs of 02_fetch_acs_tiger.py. Writes (gitignored):
    data/interim/xwalk_2020_to_2010.csv   area-share weights
    data/interim/tract_income.csv         one row per 2010 tract: both windows' mean household income
                                          and within-city percentile ranks, plus crosswalk quality columns

Run from the repo root:  python scripts/04_build_crosswalk.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import geopandas as gpd  # noqa: E402
import pandas as pd  # noqa: E402

from activity_index import config as cfg  # noqa: E402
from activity_index.crosswalk import apply_crosswalk, build_crosswalk, diagnostics  # noqa: E402

ADDITIVE = ["agg_household_income", "households", "renter_households", "population"]


def main():
    cfg.INTERIM.mkdir(parents=True, exist_ok=True)
    t0, t1 = cfg.ACS_T0["label"], cfg.ACS_T1["label"]
    geo0 = gpd.read_file(cfg.RAW / f"tracts_{t0}.gpkg")
    geo1 = gpd.read_file(cfg.RAW / f"tracts_{t1}.gpkg")
    acs0 = pd.read_csv(cfg.RAW / f"acs_{t0}.csv", dtype={"GEOID": str})
    acs1 = pd.read_csv(cfg.RAW / f"acs_{t1}.csv", dtype={"GEOID": str})

    xw = build_crosswalk(geo1, geo0)            # 2020 tracts -> 2010 tracts
    xw.to_csv(cfg.INTERIM / "xwalk_2020_to_2010.csv", index=False)
    on2010 = apply_crosswalk(acs1, xw, ADDITIVE)

    d = diagnostics(xw).rename(columns={"GEOID_dst": "GEOID"})
    t = (acs0[["GEOID", "city"] + ADDITIVE]
         .merge(on2010, on="GEOID", how="left", suffixes=("_t0", "_t1"))
         .merge(d, on="GEOID", how="left"))
    for s in ("t0", "t1"):
        t[f"mean_income_{s}"] = t[f"agg_household_income_{s}"] / t[f"households_{s}"]
        t.loc[t[f"households_{s}"] < 50, f"mean_income_{s}"] = pd.NA       # too few households to trust
        t[f"income_rank_{s}"] = t.groupby("city")[f"mean_income_{s}"].rank(pct=True)
    t["rank_change"] = t["income_rank_t1"] - t["income_rank_t0"]
    t.to_csv(cfg.INTERIM / "tract_income.csv", index=False)

    # ---- diagnostics: is the crosswalk believable? ----
    print(f"2010 tracts: {len(t)}; with a computed rank change: {t['rank_change'].notna().sum()}")
    print("\nmain checks (should be near 1.0):")
    for c in ("households", "agg_household_income"):
        by = pd.DataFrame({"2019-23 raw": acs1.groupby("city")[c].sum(),
                           "on 2010 tracts": on2010.merge(acs0[["GEOID", "city"]], on="GEOID")
                                                     .groupby("city")[c].sum()})
        print(f"\n{c} total, 2019-23 vs moved onto 2010 tracts:")
        print((by.assign(ratio=by["on 2010 tracts"] / by["2019-23 raw"])).round(3))
    print("\nshare of 2010 tracts that are essentially unchanged (one 2020 piece covers >=95%):")
    print((t.assign(unchanged=t["largest_piece"] >= 0.95).groupby("city")["unchanged"].mean()).round(2))
    print("\nrank_change summary by city:")
    print(t.groupby("city")["rank_change"].describe().round(3))
    print("\nwrote", cfg.INTERIM / "tract_income.csv")


if __name__ == "__main__":
    main()
