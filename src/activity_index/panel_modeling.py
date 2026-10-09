"""Baselines for the PUMA panel (scripts/14_puma_baselines.py).

Same ladder as the tract study, adapted to a yearly outcome:
  B0  income level relative to the city at the start of the year (+ square)        -> mean reversion only
  B1  B0 + free controls (education, renters, rent, home value, age, density, ...)  -> what free census data says
  B2  B1 + last year's income change                                                -> momentum
  ALT_<group> / ALT_all  B2 + a group of year-over-year bikeshare changes
All features are measured relative to the city's median PUMA in the same year (`_rel` columns).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .modeling import fit_weighted, make_model, r2, spearman
from .panel import noise_ceiling, puma_weights

SEED = 0
TARGET = "dlog_rel"
CONTROLS = ["edu_share_ba", "renter_share", "log_rent", "log_home_value", "median_age", "log_density",
            "km_to_center", "log_households"]
GROUPS = {
    "usage": ["d_log_trips", "station_growth", "new_station_share"],
    "rider_mix": ["d_casual_share", "d_weekend_share", "d_round_share"],
    "commute_role": ["d_net_am", "d_net_pm"],
    "trip_character": ["d_log_duration"],
}
BIKE_ALL = [c for cols in GROUPS.values() for c in cols]
LIFTS = [("B1", "B0"), ("B2", "B1")] + [(f"ALT_{g}", "B2") for g in GROUPS] + [("ALT_all", "B2")]
WHAT = {"B0": "income level vs city (+ square)", "B1": "B0 + 8 free controls", "B2": "B1 + last year's income change",
        "ALT_usage": "B2 + usage (3)", "ALT_rider_mix": "B2 + rider mix (3)", "ALT_commute_role": "B2 + commute role (2)",
        "ALT_trip_character": "B2 + trip duration (1)", "ALT_all": "B2 + all 9 bikeshare changes"}


def relative_columns() -> list[str]:
    """Columns that script 12 must turn into `<col>_rel` (minus the city-year median)."""
    return ["log_income_prev", "dlog_lag", "dlog"] + CONTROLS + BIKE_ALL


def feature_sets(df: pd.DataFrame) -> dict:
    b0 = ["log_income_prev_rel", "log_income_prev_rel_sq"]
    b1 = b0 + [c + "_rel" for c in CONTROLS]
    sets = {"B0": b0, "B1": b1, "B2": b1 + ["dlog_lag_rel"]}
    for g, cols in GROUPS.items():
        sets[f"ALT_{g}"] = sets["B2"] + [c + "_rel" for c in cols]
    sets["ALT_all"] = sets["B2"] + [c + "_rel" for c in BIKE_ALL]
    return sets


def group_folds(groups, n_splits: int = 5, seed: int = SEED):
    """Deterministic grouped k-fold: every group (PUMA) lands in exactly one test fold. Returns (train, test) index arrays."""
    groups = np.asarray(groups)
    ids = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    fold_of = {g: i % n_splits for i, g in enumerate(ids)}
    fold = np.array([fold_of[g] for g in groups])
    return [(np.where(fold != k)[0], np.where(fold == k)[0]) for k in range(min(n_splits, len(ids)))]


def splits_for(df: pd.DataFrame, scheme: str, min_train_years: int = 2):
    """Validation schemes. Rows never in a test fold get no prediction (NaN).
      puma:    5 folds, each PUMA entirely in one fold (the model never sees a PUMA it is tested on)
      loco:    leave one city out
      forward: test year Y is predicted from years before Y only (needs `min_train_years` earlier years)
    """
    if scheme == "puma":
        return group_folds(df["GEOID"].values)
    if scheme == "loco":
        return [(np.where(df["city"] != c)[0], np.where(df["city"] == c)[0]) for c in sorted(df["city"].unique())]
    if scheme == "forward":
        years = sorted(df["year"].unique())
        return [(np.where(df["year"] < y)[0], np.where(df["year"] == y)[0]) for y in years[min_train_years:]]
    raise ValueError(scheme)


def oof_predictions(df, cols, scheme, weights=None, kind: str = "ridge"):
    X, y = df[cols], df[TARGET].values
    pred = np.full(len(df), np.nan)
    for tr, te in splits_for(df, scheme):
        m = fit_weighted(make_model(kind), X.iloc[tr], y[tr], None if weights is None else np.asarray(weights)[tr])
        pred[te] = m.predict(X.iloc[te])
    return pred


def score(y, pred) -> dict:
    ok = ~np.isnan(pred)
    y, pred = np.asarray(y)[ok], np.asarray(pred)[ok]
    return {"n": int(ok.sum()), "spearman": spearman(y, pred), "r2": r2(y, pred),
            "mae": float(np.abs(y - pred).mean())}


def cluster_bootstrap_gain(y, p_new, p_old, clusters, n_boot: int = 500, seed: int = SEED):
    """95% interval for the change in Spearman (new minus old) when whole PUMAs are resampled.

    Rows of one PUMA are correlated across years, so resampling rows would make intervals too narrow."""
    y, p_new, p_old, clusters = map(np.asarray, (y, p_new, p_old, clusters))
    ok = ~np.isnan(p_new) & ~np.isnan(p_old)
    y, p_new, p_old, clusters = y[ok], p_new[ok], p_old[ok], clusters[ok]
    ids = np.unique(clusters)
    idx = {g: np.where(clusters == g)[0] for g in ids}
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = np.concatenate([idx[g] for g in rng.choice(ids, len(ids))])
        out[b] = spearman(y[pick], p_new[pick]) - spearman(y[pick], p_old[pick])
    est = spearman(y, p_new) - spearman(y, p_old)
    lo, hi = np.percentile(out, [2.5, 97.5])
    return est, lo, hi


def cluster_corr(x, y, clusters, n_boot: int = 500, seed: int = SEED):
    """Spearman correlation with a 95% interval from resampling whole clusters (PUMAs). Returns (rho, lo, hi, n)."""
    x, y, clusters = pd.Series(x).reset_index(drop=True), pd.Series(y).reset_index(drop=True), pd.Series(clusters).reset_index(drop=True)
    ok = x.notna() & y.notna()
    x, y, clusters = x[ok].values, y[ok].values, clusters[ok].values
    if len(x) < 15:
        return np.nan, np.nan, np.nan, len(x)
    ids = np.unique(clusters)
    idx = {g: np.where(clusters == g)[0] for g in ids}
    rng = np.random.default_rng(seed)
    b = []
    for _ in range(n_boot):
        pick = np.concatenate([idx[g] for g in rng.choice(ids, len(ids))])
        b.append(spearman(x[pick], y[pick]))
    lo, hi = np.nanpercentile(b, [2.5, 97.5])
    return spearman(x, y), lo, hi, len(x)


def run_ladder(df: pd.DataFrame, schemes=("puma", "loco", "forward"), n_boot: int = 500):
    """The whole baseline ladder on a panel table: returns (summary, per-scheme metrics).

    Ridge, inverse-variance weights from se_dlog. The summary has one row per baseline with Spearman and R2 under each
    scheme and the gain over the `compared_to` baseline (95% interval from resampling PUMAs)."""
    df = df.reset_index(drop=True)
    w = puma_weights(df["se_dlog"], df[TARGET]).values
    y = df[TARGET].values
    sets = feature_sets(df)
    preds, mrows = {}, []
    for scheme in schemes:
        for name, cols in sets.items():
            p = oof_predictions(df, cols, scheme, w)
            preds[(scheme, name)] = p
            mrows.append({"cv": scheme, "set": name, **score(y, p)})
    met = pd.DataFrame(mrows)
    rows = []
    for name in sets:
        ref = dict(LIFTS).get(name)
        r = {"baseline": name, "features": WHAT[name], "compared_to": ref or ""}
        for scheme in schemes:
            m = met[(met["cv"] == scheme) & (met["set"] == name)].iloc[0]
            r[f"n_{scheme}"], r[f"spearman_{scheme}"], r[f"r2_{scheme}"] = m["n"], m["spearman"], m["r2"]
            if ref:
                g, lo, hi = cluster_bootstrap_gain(y, preds[(scheme, name)], preds[(scheme, ref)], df["GEOID"], n_boot)
                r[f"gain_{scheme}"], r[f"gain_{scheme}_lo"], r[f"gain_{scheme}_hi"] = g, lo, hi
        rows.append(r)
    return pd.DataFrame(rows), met
