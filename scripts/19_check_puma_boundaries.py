"""Cheap pre-flight (a few MB, no trips): which PUMAs keep the same boundary after the 2020 redraw, and does the ACS API serve 2021-2025?

Why: ACS 1-year uses 2010 PUMA boundaries through 2021 and 2020-based PUMAs from 2022. A PUMA can be followed across the break only if its
polygon is (nearly) unchanged, so this counts how many of our PUMAs survive before any large trip download.

Needs script 11 (data/raw/pumas_2010.gpkg) and, if present, script 12 (to count only PUMAs already in the panel).
    python scripts/19_check_puma_boundaries.py
Writes results/puma_boundary_check.csv (one row per 2010 PUMA with its overlap score).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from activity_index import config as cfg  # noqa: E402
from activity_index.census import fetch_acs_puma, fetch_pumas20  # noqa: E402
from activity_index.puma_geo import overlap_components  # noqa: E402

def iou(a, b) -> float:
    """Intersection over union of two polygons in an equal-area projection (1.0 = identical)."""
    u = a.union(b).area
    return float(a.intersection(b).area / u) if u > 0 else np.nan


def main():
    pd.set_option("display.width", 200)
    print("=== does the ACS 1-year API serve PUMA tables for these years? (New York state, one variable) ===")
    for y in (2019, 2021, 2022, 2023, 2024, 2025):
        try:
            d = fetch_acs_puma(y, "36", {"B25003_001E": "households"})
            print(f"{y}: OK, {len(d)} PUMAs, example codes {list(d['GEOID'].head(3))}")
        except Exception as e:
            print(f"{y}: not available ({type(e).__name__}: {str(e)[:120]})")

    old = gpd.read_file(cfg.RAW / "pumas_2010.gpkg")
    in_panel = None
    pp = cfg.PROCESSED / "puma_panel.csv"
    if pp.exists():
        p = pd.read_csv(pp, dtype={"GEOID": str})
        in_panel = set(p.loc[p["usable"].fillna(False).astype(bool), "GEOID"])
    new = pd.concat([fetch_pumas20(s, cfg.RAW / "tiger")[["GEOID", "geometry"]] for s in cfg.PUMA_STATES], ignore_index=True)
    new = gpd.GeoDataFrame(new, crs=old.crs)
    old, new = old.to_crs(5070), new.to_crs(5070)
    # (1) same code and shape; (2) shape-matched groups regardless of code (unions of PUMAs are followed by summing the additive aggregates)
    nmap = new.set_index("GEOID")["geometry"]
    rows = []
    for _, r in old.iterrows():
        g = nmap.get(r["GEOID"])
        rows.append({"GEOID": r["GEOID"], "city": r["city"], "same_code_exists": g is not None,
                     "iou_same_code": iou(r.geometry, g) if g is not None else np.nan})
    out = pd.DataFrame(rows)
    o2 = old[["GEOID", "geometry"]].rename(columns={"GEOID": "old"})
    n2 = new[["GEOID", "geometry"]].rename(columns={"GEOID": "new"})
    pieces = gpd.overlay(o2, n2, how="intersection", keep_geom_type=True)
    pieces["area"] = pieces.geometry.area
    comp = overlap_components(pieces[["old", "new", "area"]], old.set_index("GEOID").geometry.area, new.set_index("GEOID").geometry.area)
    out = out.merge(comp.rename(columns={"old": "GEOID"}), on="GEOID", how="left")
    out["in_panel"] = out["GEOID"].isin(in_panel) if in_panel is not None else np.nan
    members_in_panel = out.groupby("component")["in_panel"].transform("all") if in_panel is not None else np.nan
    out["all_members_in_panel"] = members_in_panel
    out["clean"] = out["iou"] >= 0.95
    out["kind"] = np.where(~out["clean"], "changed", np.where((out["n_old"] == 1) & (out["n_new"] == 1), "same area (maybe renumbered)", "split or merged"))
    out.to_csv(cfg.RESULTS / "puma_boundary_check.csv", index=False)

    print("\n=== by code only (the earlier check) ===")
    print(f"2010 PUMAs with a 2020 PUMA of the same code and IoU>=0.95: {int((out['iou_same_code'] >= 0.95).sum())} of {len(out)}")
    print("\n=== by shape, any code: is each 2010 PUMA (or group of them) the same ground as a 2020 PUMA (or group)? ===")
    k = out if in_panel is None else out[out["in_panel"] == True]  # noqa: E712
    print("only the PUMAs already usable in the bikeshare panel (the ones that matter):")
    print(pd.crosstab(k["city"], k["kind"], margins=True).to_string())
    clean = k[k["clean"]]
    print(f"\nfollowable areas among them: {clean['component'].nunique()} (PUMAs {len(clean)} of {len(k)}); "
          f"of these, every member PUMA is in the panel for {clean.loc[clean['all_members_in_panel'] == True, 'component'].nunique()}")
    print(clean.groupby("city")["component"].nunique().rename("followable areas per city").to_string())
    print("\nwrote", cfg.RESULTS / "puma_boundary_check.csv")


if __name__ == "__main__":
    main()
