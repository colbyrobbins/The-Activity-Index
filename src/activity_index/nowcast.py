"""Levels nowcast at tract level (scripts/16_levels_nowcast.py).

Question: using bikeshare data for 2015-2018, how well can we estimate each tract's income LEVEL in the 2014-18 ACS window,
(a) on its own, and (b) on top of everything that was already published before that window (the 2009-13 ACS)?

Aligned the way a nowcast would be: features and label cover the same period; every "free data" input is the
previous, non-overlapping window. A real operator would hold the 2013-17 release (it overlaps the target window by four years,
which makes persistence almost unbeatable), so this baseline is deliberately the stale-but-honest one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import modeling as M

TARGET = "income_rank_t0"
CONTROLS = M.CONTROLS            # same eight controls as the change study
GEO = ["log_density", "km_to_center"]   # geography, taken from the target-window table (it does not change)
LIFTS = [("N1", "N0"), ("N5", "N1"), ("N4", "N3")] + [(f"N5_{g}", "N1") for g in M.GROUPS]
WHAT = {"N0": "last published: 2009-13 income rank (+ square)",
        "N1": "N0 + free controls as published in 2009-13",
        "N2": "bikeshare only (10 features, within-city percentiles)",
        "N3": "free controls only, no income history",
        "N4": "free controls + bikeshare, no income history",
        "N5": "N1 + all 10 bikeshare features",
        **{f"N5_{g}": f"N1 + {g}" for g in M.GROUPS}}


def prior_controls(prior_acs: pd.DataFrame, geo: pd.DataFrame) -> pd.DataFrame:
    """Controls from the PRIOR ACS window as within-city percentiles (`<col>_pcr`), over every tract of the city.

    `geo` has GEOID, log_density, km_to_center (geography; copied from the target-window table)."""
    a = prior_acs
    c = pd.DataFrame({"GEOID": a["GEOID"], "city": a["city"]})
    c["edu_share_ba"] = (a["edu_bachelors"] + a["edu_masters"] + a["edu_professional"] + a["edu_doctorate"]) / a["edu_total"]
    c["renter_share"] = a["renter_households"] / a["households"]
    c["median_gross_rent"], c["median_home_value"], c["median_age"] = a["median_gross_rent"], a["median_home_value"], a["median_age"]
    c["log_households"] = np.log(a["households"].where(a["households"] > 0))
    c = c.merge(geo[["GEOID"] + GEO], on="GEOID", how="left")
    for col in CONTROLS:
        c[col + "_pcr"] = c.groupby("city")[col].rank(pct=True)
    return c[["GEOID"] + [col + "_pcr" for col in CONTROLS]]


def city_terms(df: pd.DataFrame):
    """City dummies, and city-by-PRIOR-rank terms (the target's own rank must not enter). Returns (df, dummies, rank_terms)."""
    out, dummies, rank_terms = df.copy(), [], []
    for c in sorted(df["city"].unique())[1:]:
        ind = (df["city"] == c).astype(float)
        out[f"city_{c}"] = ind
        out[f"city_{c}_x_prior"], out[f"city_{c}_x_prior2"] = ind * df["income_rank_prior"], ind * df["income_rank_prior"] ** 2
        dummies.append(f"city_{c}")
        rank_terms += [f"city_{c}_x_prior", f"city_{c}_x_prior2"]
    return out, dummies, rank_terms


def feature_sets(dummies: list[str], rank_terms: list[str]) -> dict:
    ctrl = [c + "_pcr" for c in CONTROLS]
    groups = {g: [c + "_cr" for c in cols] for g, cols in M.GROUPS.items()}
    bike = [c for cols in groups.values() for c in cols]
    last = ["income_rank_prior", "income_rank_prior_sq"] + dummies + rank_terms
    sets = {"N0": last, "N1": last + ctrl, "N2": bike + dummies, "N3": ctrl + dummies, "N4": ctrl + bike + dummies,
            "N5": last + ctrl + bike}
    for g, cols in groups.items():
        sets[f"N5_{g}"] = last + ctrl + cols
    return sets


def score(y, pred) -> dict:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    return {"spearman": M.spearman(y, pred), "r2": M.r2(y, pred), "mae_pct_points": float(np.abs(y - pred).mean() * 100)}
