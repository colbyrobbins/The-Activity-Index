"""Nowcast step 1: build the areas that can be followed across the 2020 PUMA redraw, and their yearly income series. No trip downloads.

An area is a group of 2010 PUMAs that covers the same ground as a group of 2020 PUMAs (often one PUMA renumbered, sometimes a merge or
split). ACS income and household totals are additive, so each area's income is the sum over its member PUMAs of the vintage that matches
the year (2010 PUMAs through 2021, 2020 PUMAs from 2022). Only areas whose every 2010 PUMA is already in the bikeshare panel are kept,
so station coverage is complete.

Needs scripts 11 and 12 (acs1_puma.csv, pumas_2010.gpkg, puma_panel.csv) and a Census key (script 02). Run from the repo root:
    python scripts/20_build_areas.py
Writes data/raw/acs1_puma_post.csv (ACS 2021-2024), data/processed/areas.csv (membership), area_acs.csv (summed ACS), areas.gpkg (polygons).
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from activity_index import config as cfg  # noqa: E402
from activity_index import panel as P  # noqa: E402
from activity_index.census import fetch_acs_puma, fetch_pumas20  # noqa: E402
from activity_index.puma_geo import area_acs, area_members  # noqa: E402

warnings.filterwarnings("ignore")


def fetch_post_acs() -> pd.DataFrame:
    path = cfg.RAW / "acs1_puma_post.csv"
    if path.exists():
        return pd.read_csv(path, dtype={"GEOID": str, "state": str})
    parts = []
    for year in [y for y in cfg.AREAS["acs_old"] + cfg.AREAS["acs_new"] if y >= 2021]:
        for state, city in cfg.PUMA_STATES.items():
            a = fetch_acs_puma(year, state, cfg.ACS_VARS)
            a["year"], a["state"], a["city"] = year, state, city
            parts.append(a)
        print(f"ACS 1-year {year}: {sum(len(p) for p in parts if p['year'].iloc[0] == year)} PUMAs")
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(path, index=False)
    return out


def main():
    pd.set_option("display.width", 220)
    cfg.PROCESSED.mkdir(parents=True, exist_ok=True)
    old = gpd.read_file(cfg.RAW / "pumas_2010.gpkg")
    new = pd.concat([fetch_pumas20(s, cfg.RAW / "tiger").assign(city=c) for s, c in cfg.PUMA_STATES.items()], ignore_index=True)
    new = gpd.GeoDataFrame(new, crs=old.crs)
    old5, new5 = old.to_crs(5070), new.to_crs(5070)
    pieces = gpd.overlay(old5[["GEOID", "geometry"]].rename(columns={"GEOID": "old"}),
                         new5[["GEOID", "geometry"]].rename(columns={"GEOID": "new"}), how="intersection", keep_geom_type=True)
    pieces["area"] = pieces.geometry.area
    mem = area_members(pieces[["old", "new", "area"]], old5.set_index("GEOID").geometry.area, new5.set_index("GEOID").geometry.area)
    city = pd.concat([old[["GEOID", "city"]].assign(line="2010"), new[["GEOID", "city"]].assign(line="2020")])
    mem = mem.merge(city, on=["GEOID", "line"], how="left")

    panel = pd.read_csv(cfg.PROCESSED / "puma_panel.csv", dtype={"GEOID": str})
    in_panel = set(panel.loc[panel["usable"].fillna(False).astype(bool), "GEOID"])
    o = mem[mem["line"] == "2010"]
    keep = o.groupby("component").apply(lambda g: g["GEOID"].isin(in_panel).all() and g["GEOID"].isin(in_panel).any()
                                        and g["iou"].iloc[0] >= cfg.AREAS["min_iou"])
    keep = set(keep[keep].index)
    mem = mem[mem["component"].isin(keep)].copy()
    assert mem.groupby("component")["city"].nunique().max() == 1, "an area spans two cities"
    names = o[o["component"].isin(keep)].groupby("component").apply(lambda g: g["city"].iloc[0] + "_" + g["GEOID"].min())
    mem["area_id"] = mem["component"].map(names)
    mem.to_csv(cfg.PROCESSED / "areas.csv", index=False)

    poly = old[old["GEOID"].isin(mem.loc[mem["line"] == "2010", "GEOID"])].merge(
        mem.loc[mem["line"] == "2010", ["GEOID", "area_id"]], on="GEOID")
    poly.dissolve(by="area_id").reset_index()[["area_id", "geometry"]].to_file(cfg.PROCESSED / "areas.gpkg", driver="GPKG")

    acs_old = pd.read_csv(cfg.RAW / "acs1_puma.csv", dtype={"GEOID": str, "state": str})
    acs = pd.concat([acs_old, fetch_post_acs()], ignore_index=True)
    acs = acs[acs["year"].isin(cfg.AREAS["acs_old"] + cfg.AREAS["acs_new"])]
    aa = area_acs(acs, mem[["area_id", "line", "GEOID", "city"]])
    aa.to_csv(cfg.PROCESSED / "area_acs.csv", index=False)

    print(f"\nareas kept: {mem['area_id'].nunique()} (every 2010 member in the bikeshare panel, same ground to IoU >= {cfg.AREAS['min_iou']})")
    summ = mem.groupby(["city", "area_id"]).apply(lambda g: pd.Series({
        "pumas_2010": int((g["line"] == "2010").sum()), "pumas_2020": int((g["line"] == "2020").sum())})).reset_index()
    print(summ.groupby("city").agg(areas=("area_id", "size"), pumas_2010=("pumas_2010", "sum"), pumas_2020=("pumas_2020", "sum")).to_string())
    print("\nincome years present per area (areas x years):")
    print(aa.assign(x=1).pivot_table(index="city", columns="year", values="x", aggfunc="sum").fillna(0).astype(int).to_string())

    inc = P.add_changes(P.income_table(aa))
    t = inc[inc["year"].isin(cfg.AREAS["target_years"] + [2022])]
    print("\nyear-over-year income change of the summed areas (log points) - check that 2022 and 2023 look like other years:")
    print(t.groupby(["city", "year"])["dlog"].median().unstack().round(3).to_string())
    print("\nmedian standard error of the yearly change (survey noise; merged areas should be less noisy than single PUMAs):")
    print(t.groupby("year")["se_dlog"].median().round(3).to_string())
    big = t[t["dlog"].abs() > 0.25][["GEOID", "year", "dlog"]]
    print(f"\nlarge jumps (|change| > 0.25): {len(big)}")
    if len(big):
        print(big.round(3).to_string(index=False))
    print("\nwrote", cfg.PROCESSED / "areas.csv", "area_acs.csv", "areas.gpkg")


if __name__ == "__main__":
    main()
