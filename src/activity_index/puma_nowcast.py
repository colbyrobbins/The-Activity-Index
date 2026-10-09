"""PUMA nowcast, time-series design (scripts/17_puma_nowcast.py).

Question: in year T, can the trailing-12-month bikeshare record of a PUMA tell us its year-T income before the ACS publishes it,
beyond what the PUMA's own published history already says?

Rows are PUMA-years. Label: ACS 1-year log mean household income of year T (relative to the city's median PUMA that year).
No static cross-sectional descriptors (density, distance to center, size) are used: they do not change over time, so in a time
study they only re-describe the PUMA. What carries information over time is each PUMA's own record:
  own history    income of T-1 (last published), income change of T-1 (momentum)
  free census    education, renters, rent, home value, age as published for T-1 (slow-moving, same release as income)
  bikeshare      year-T trailing-12-month level of each rate, its year-over-year change, and its deviation from the PUMA's own
                 average over PRIOR years only (expanding mean; the test year never enters its own baseline)
Sets (rungs):
  T0 last published | T1 T0 + momentum | T2 T1 + free census | T3 bikeshare only
  T4 T1 + bikeshare (the nowcast question) | T5 T2 + bikeshare | T4_<group> one bikeshare group at a time
Validation: forward in time (train on earlier years, test year Y) is the headline; PUMA-held-out folds as a secondary check.
Level and change are the same prediction once last year's income is in the model, so the model predicts the change vs last
published directly (level = prev + change) and the headline gain is the change in Spearman of that change.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import panel as P
from . import panel_modeling as PM
from .modeling import fit_weighted, make_model, spearman, r2

TARGET = "log_income"
TARGET_REL = "log_income_rel"
CENSUS = ["edu_share_ba", "renter_share", "log_rent", "log_home_value", "median_age"]   # slow-moving, time-varying (no static geography)
# year-T trailing-12-month level of each rate = starting level + this year's change
LEVEL_T = {"lvlT_log_trips": "usage", "lvlT_casual_share": "rider_mix", "lvlT_round_share": "rider_mix",
           "lvlT_net_am": "commute_role", "lvlT_net_pm": "commute_role"}
DEV = {c: "dev_" + c[len("lvlT_"):] for c in LEVEL_T}            # deviation from the PUMA's own prior-year mean
GROUPS = {g: list(c) for g, c in PM.GROUPS.items()}
for _col, _g in LEVEL_T.items():
    GROUPS[_g] = GROUPS[_g] + [_col, DEV[_col]]
BIKE_ALL = [c for cols in GROUPS.values() for c in cols]
LIFTS = [("T1", "T0"), ("T2", "T1"), ("T4", "T1"), ("T5", "T2")] + [(f"T4_{g}", "T1") for g in GROUPS]
WHAT = {"T0": "last published: income T-1 (+ square)", "T1": "T0 + last year's income change",
        "T2": "T1 + free census (as of T-1)", "T3": "bikeshare only (no income history)",
        "T4": "T1 + all bikeshare", "T5": "T2 + all bikeshare", **{f"T4_{g}": f"T1 + {g}" for g in GROUPS}}


def add_year_t_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Year-T level of each rate = starting level + change (trips: log trips per matched station-month in year T)."""
    out = df.copy()
    out["lvlT_log_trips"] = np.log(out["trips_1"] / out["matched_station_months"])
    for f in ("casual_share", "round_share", "net_am", "net_pm"):
        out[f"lvlT_{f}"] = out[f"lvl_{f}"] + out[f"d_{f}"]
    return out


def add_own_history(df: pd.DataFrame) -> pd.DataFrame:
    """dev_* = year-T level minus the mean of the same PUMA's EARLIER years (NaN in its first year). Needs rows from every year."""
    out = df.sort_values(["GEOID", "year"]).copy()
    for lvl, dev in DEV.items():
        prior_mean = out.groupby("GEOID")[lvl].transform(lambda s: s.shift(1).expanding().mean())
        out[dev] = out[lvl] - prior_mean
    return out


def prepare(panel: pd.DataFrame) -> pd.DataFrame:
    """Usable PUMA-years with a prior-year baseline, the level target, and every feature relative to the city-year median."""
    d = add_own_history(add_year_t_levels(panel))
    d = d[d["usable"].fillna(False).astype(bool) & d[list(DEV.values())].notna().all(axis=1)].reset_index(drop=True)
    base = ["log_income", "log_income_prev", "dlog_lag"] + CENSUS + BIKE_ALL
    d = d.drop(columns=[c + "_rel" for c in base if c + "_rel" in d])
    d = P.relative_to_group(d, base).rename(columns={"log_income_rel": TARGET_REL})
    d["log_income_prev_rel_sq"] = d["log_income_prev_rel"] ** 2
    d["change_rel"] = d[TARGET_REL] - d["log_income_prev_rel"]
    return d


def feature_sets() -> dict:
    census = [c + "_rel" for c in CENSUS]
    groups = {g: [c + "_rel" for c in cols] for g, cols in GROUPS.items()}
    bike = [c for cols in groups.values() for c in cols]
    t0 = ["log_income_prev_rel", "log_income_prev_rel_sq"]
    t1 = t0 + ["dlog_lag_rel"]
    sets = {"T0": t0, "T1": t1, "T2": t1 + census, "T3": bike, "T4": t1 + bike, "T5": t1 + census + bike}
    for g, cols in groups.items():
        sets[f"T4_{g}"] = t1 + cols
    return sets


def oof_predictions(df, cols, scheme) -> np.ndarray:
    """Out-of-sample predicted CHANGE vs last published (level = prev + change). Predicting the change directly keeps a
    shrinkage term in last year's income from leaking into the score of the added features."""
    X, y = df[cols], df["change_rel"].values
    pred = np.full(len(df), np.nan)
    for tr, te in PM.splits_for(df, scheme, min_train_years=1):
        pred[te] = fit_weighted(make_model("ridge"), X.iloc[tr], y[tr]).predict(X.iloc[te])
    return pred


def score(df, pc) -> dict:
    """change_spearman / change_r2: does the model rank and size who moved vs last published. level_spearman / level_r2: year-T
    income level vs city. mae_pct: typical error of the level in % of income."""
    ok = ~np.isnan(pc)
    d, c = df[ok], pc[ok]
    y, prev = d[TARGET_REL].values, d["log_income_prev_rel"].values
    return {"n": int(ok.sum()), "change_spearman": spearman(d["change_rel"].values, c), "change_r2": r2(d["change_rel"].values, c),
            "level_spearman": spearman(y, prev + c), "level_r2": r2(y, prev + c), "mae_pct": float(np.abs(y - prev - c).mean() * 100)}


def _gain(df, p_new, p_old, n_boot, seed=0):
    ok = ~np.isnan(p_new) & ~np.isnan(p_old)
    d, a, b = df[ok], p_new[ok], p_old[ok]
    y = d["change_rel"].values
    est = spearman(y, a) - spearman(y, b)
    ids = d["GEOID"].values
    uniq = np.unique(ids)
    idx = {g: np.where(ids == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        pick = np.concatenate([idx[g] for g in rng.choice(uniq, len(uniq))])
        out.append(spearman(y[pick], a[pick]) - spearman(y[pick], b[pick]))
    lo, hi = np.nanpercentile(out, [2.5, 97.5])
    return est, lo, hi


def run(df: pd.DataFrame, schemes=("forward", "puma"), n_boot: int = 500):
    """Returns (metrics per scheme and set, gains in change_spearman with PUMA-cluster bootstrap intervals)."""
    df = df.reset_index(drop=True)
    sets = feature_sets()
    preds, rows = {}, []
    for s in schemes:
        for name, cols in sets.items():
            p = oof_predictions(df, cols, s)
            preds[(s, name)] = p
            rows.append({"cv": s, "set": name, "features": WHAT[name], **score(df, p)})
    gains = []
    for s in schemes:
        for new, old in LIFTS:
            e, lo, hi = _gain(df, preds[(s, new)], preds[(s, old)], n_boot)
            gains.append({"cv": s, "comparison": f"{new} vs {old}", "d_change_spearman": e, "lo": lo, "hi": hi})
    return pd.DataFrame(rows), pd.DataFrame(gains)
