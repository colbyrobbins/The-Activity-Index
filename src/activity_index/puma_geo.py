"""Follow PUMAs across the 2020 boundary redraw (ACS 1-year uses 2010 PUMAs through 2021 and 2020 PUMAs from 2022).

ACS income and household totals are additive, so a group of 2010 PUMAs that covers exactly the same ground as a group of 2020 PUMAs is a
valid area to follow through time: sum the aggregates on each side. This module finds those groups from the overlap table of the two
partitions: connected components of the 'old PUMA overlaps new PUMA' graph, ignoring slivers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LINE_BREAK = 2022      # first ACS 1-year that uses the 2020 PUMAs
ADDITIVE = ["agg_household_income", "households", "renter_households", "population", "edu_total", "edu_bachelors",
            "edu_masters", "edu_professional", "edu_doctorate"]
MOE = ["agg_household_income_moe", "households_moe"]
MEDIANS = ["median_gross_rent", "median_home_value", "median_age"]


def _group(pieces, old_area, new_area, min_share):
    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for o in old_area.index:
        find(("o", o))
    for n in new_area.index:
        find(("n", n))
    for r in pieces.itertuples(index=False):
        if r.area >= min_share * old_area[r.old] or r.area >= min_share * new_area[r.new]:
            parent[find(("o", r.old))] = find(("n", r.new))
    comp = {k: find(k) for k in list(parent)}
    ids = {c: i for i, c in enumerate(sorted(set(comp.values()), key=str))}
    old_c = pd.Series({o: ids[comp[("o", o)]] for o in old_area.index}, name="component")
    new_c = pd.Series({n: ids[comp[("n", n)]] for n in new_area.index}, name="component")
    p = pieces.assign(co=pieces["old"].map(old_c), cn=pieces["new"].map(new_c))
    inter = p[p["co"] == p["cn"]].groupby("co")["area"].sum()
    a_old = old_area.groupby(old_c).sum()
    a_new = new_area.groupby(new_c).sum()
    stats = pd.DataFrame({"n_old": old_c.value_counts(), "n_new": new_c.value_counts()}).fillna(0).astype(int)
    stats["iou"] = (inter / (a_old.add(a_new, fill_value=0) - inter)).reindex(stats.index)
    return old_c, new_c, stats


def overlap_components(pieces: pd.DataFrame, old_area: pd.Series, new_area: pd.Series, min_share: float = 0.03) -> pd.DataFrame:
    """Group 2010 and 2020 PUMAs into areas that match exactly (up to slivers).

    pieces:   one row per overlap with columns old, new, area (intersection area; any equal-area unit)
    old_area: area of each 2010 PUMA (index = id); new_area: same for 2020 PUMAs
    An old and a new PUMA are linked when their overlap is at least `min_share` of either one's area.
    Returns one row per old PUMA: old, component, n_old, n_new, iou (overlap of the old-union with the new-union; 1 = same ground).
    """
    old_c, _, stats = _group(pieces, old_area, new_area, min_share)
    return old_c.rename_axis("old").reset_index().merge(stats, left_on="component", right_index=True)


def area_members(pieces, old_area, new_area, min_share: float = 0.03) -> pd.DataFrame:
    """Long table of every PUMA of either vintage with its area: component, line ('2010'/'2020'), GEOID, n_old, n_new, iou."""
    old_c, new_c, stats = _group(pieces, old_area, new_area, min_share)
    rows = pd.concat([old_c.rename_axis("GEOID").reset_index().assign(line="2010"),
                      new_c.rename_axis("GEOID").reset_index().assign(line="2020")], ignore_index=True)
    return rows.merge(stats, left_on="component", right_index=True)


def area_acs(acs: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    """ACS 1-year rows summed over each area's member PUMAs, using the member vintage that matches the year.

    members: area_id, line, GEOID (+ city). Counts and aggregates add; margins of error add in quadrature; medians are household-weighted
    means of the member medians (an approximation). A year is dropped for an area when any member PUMA is missing from the table that year.
    Output keeps the ACS column names with GEOID = area_id, so income_table / add_changes / controls work unchanged."""
    a = acs.copy()
    a["line"] = np.where(a["year"] < LINE_BREAK, "2010", "2020")
    a = a.merge(members[["area_id", "line", "GEOID"]], on=["line", "GEOID"], how="inner")
    keys = [a["area_id"], a["year"]]
    out = a.groupby(["area_id", "year"])[ADDITIVE].sum(min_count=1)
    for c in MOE:
        out[c] = np.sqrt((a[c] ** 2).groupby(keys).sum(min_count=1))
    w = a["households"].where(a["households"] > 0)
    for c in MEDIANS:
        ok = a[c].notna() & w.notna()
        out[c] = (a[c].where(ok) * w.where(ok)).groupby(keys).sum(min_count=1) / w.where(ok).groupby(keys).sum(min_count=1)
    out["n_members"] = a.groupby(["area_id", "year"]).size()
    need = members.groupby(["area_id", "line"]).size().rename("n_expected").reset_index()
    out = out.reset_index()
    out["line"] = np.where(out["year"] < LINE_BREAK, "2010", "2020")
    out = out.merge(need, on=["area_id", "line"], how="left")
    out = out[out["n_members"] == out["n_expected"]].drop(columns=["line", "n_expected"])
    city = members.drop_duplicates("area_id").set_index("area_id")["city"]
    out["city"] = out["area_id"].map(city)
    return out.rename(columns={"area_id": "GEOID"}).reset_index(drop=True)
