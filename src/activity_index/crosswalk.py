"""Move ACS counts from 2020 census tracts onto 2010 tracts (area-weighted).

Why: 2014-18 ACS uses 2010 tracts, 2019-23 ACS uses 2020 tracts, and the study needs both on one set of
boundaries. Only ADDITIVE quantities (aggregate income, households, population, counts) may be moved this
way; medians and rates must be recomputed afterwards from the moved counts.

Method: intersect the two tract layers in an equal-area projection; the share of a 2020 tract's area that
falls in a 2010 tract is the share of its counts given to that 2010 tract. Area is a crude proxy for where
people live (water and parks distort it), so `diagnostics` reports how much of each 2010 tract is covered by
essentially one 2020 tract (those tracts are barely affected); the study can restrict to them as a check.
"""
from __future__ import annotations

import pandas as pd

EQUAL_AREA_CRS = "EPSG:5070"  # NAD83 / Conus Albers
MIN_SHARE = 0.01              # ignore slivers: intersections under 1% of the source tract's area


def build_crosswalk(src, dst, src_id: str = "GEOID", dst_id: str = "GEOID") -> pd.DataFrame:
    """Area-share crosswalk from `src` tracts (GeoDataFrame) to `dst` tracts.

    Returns columns [src_id, dst_id, weight, dst_cover]: `weight` is the share of the source tract's area
    in the destination tract (sums to 1 per source tract); `dst_cover` is the share of the destination
    tract's area made up by this source tract.
    """
    import geopandas as gpd

    s = src[[src_id, "geometry"]].rename(columns={src_id: "src"}).to_crs(EQUAL_AREA_CRS)
    d = dst[[dst_id, "geometry"]].rename(columns={dst_id: "dst"}).to_crs(EQUAL_AREA_CRS)
    s["src_area"] = s.geometry.area
    d["dst_area"] = d.geometry.area
    inter = gpd.overlay(s, d, how="intersection", keep_geom_type=True)
    inter["a"] = inter.geometry.area
    inter["weight"] = inter["a"] / inter["src_area"]
    inter = inter[inter["weight"] >= MIN_SHARE].copy()
    inter["weight"] = inter["weight"] / inter.groupby("src")["weight"].transform("sum")  # renormalise
    inter["dst_cover"] = inter["a"] / inter["dst_area"]
    return (inter[["src", "dst", "weight", "dst_cover"]]
            .rename(columns={"src": src_id, "dst": dst_id + "_dst" if dst_id == src_id else dst_id})
            .reset_index(drop=True))


def apply_crosswalk(values: pd.DataFrame, xw: pd.DataFrame, cols: list[str],
                    src_id: str = "GEOID", dst_id: str = "GEOID_dst") -> pd.DataFrame:
    """Sum `cols` of `values` (one row per source tract) onto destination tracts using the weights."""
    m = xw.merge(values[[src_id] + cols], on=src_id, how="inner")
    for c in cols:
        m[c] = m[c] * m["weight"]
    out = m.groupby(dst_id)[cols].sum(min_count=1)
    # a destination tract is only as good as its weakest piece: record how much source data was missing
    covered = xw.merge(values[[src_id, cols[0]]], on=src_id, how="left")
    covered["_ok"] = covered[cols[0]].notna() * covered["dst_cover"]
    out["share_with_data"] = covered.groupby(dst_id)["_ok"].sum() / covered.groupby(dst_id)["dst_cover"].sum()
    return out.reset_index().rename(columns={dst_id: "GEOID"})


def diagnostics(xw: pd.DataFrame, dst_id: str = "GEOID_dst") -> pd.DataFrame:
    """Per destination tract: number of source tracts and the largest single piece (1.0 = unchanged tract)."""
    g = xw.groupby(dst_id)
    return pd.DataFrame({"n_sources": g.size(), "largest_piece": g["dst_cover"].max(),
                         "total_cover": g["dst_cover"].sum()}).reset_index()


def apply_crosswalk_moe(values: pd.DataFrame, xw: pd.DataFrame, cols: list[str],
                        src_id: str = "GEOID", dst_id: str = "GEOID_dst") -> pd.DataFrame:
    """Margins of error for apportioned counts: root-sum-of-squares of weight x source MOE.

    Treats the source tracts as independent and ignores the extra uncertainty from using area as the
    weight, so it understates the true error a little.
    """
    m = xw.merge(values[[src_id] + cols], on=src_id, how="inner")
    for c in cols:
        m[c] = (m[c] * m["weight"]) ** 2
    out = m.groupby(dst_id)[cols].sum(min_count=1) ** 0.5
    return out.reset_index().rename(columns={dst_id: "GEOID"})
