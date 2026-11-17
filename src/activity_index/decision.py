"""From nowcast to a decision: which areas of a city should a retail or real-estate team look at first? (scripts/28_decision_screen.py)

The ACS 1-year income for year T is published in September of T+1. Bikeshare trips for year T are complete in January of T+1, so the
nowcast is available about eight months earlier. A site-selection or acquisitions team does not need the dollar figure to the last
thousand; it needs to know which areas are pulling ahead of their city. So the screen is scored as a ranking inside each city-year:
  called_up  the model expects the area's income to grow faster than the city's median area
  top pick   the area with the highest expected growth in its city that year
All predictions are forward in time (the test year is never seen), the same ones script 22 scores in dollars.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import area_nowcast as AN
from . import panel_modeling as PM
from .modeling import fit_weighted, make_model


def scored(df: pd.DataFrame, cols: list[str] | None, name: str, direct: str | None = None) -> pd.DataFrame:
    """Scored area-years with `pred_rel` (expected growth minus the city-year median of expectations) for one screen.

    cols: model features (forward predictions). direct: rank by this column itself instead of a model (e.g. raw trip growth)."""
    if direct is None:
        p = AN.forward_predictions(df, cols)
    else:
        p = np.where(np.isnan(AN.forward_predictions(df, [])), np.nan, df[direct].values)
    d = df[~np.isnan(p)].copy()
    d["pred"] = p[~np.isnan(p)]
    g = d.groupby(["city", "year"])
    d["pred_rel"] = d["pred"] - g["pred"].transform("median")
    d["actual_rel"] = d["dlog_rel"]
    d["actual_rel_dollars"] = d["income_prev"] * (np.exp(d["dlog_rel"]) - 1)        # income gained vs growing at the city's median pace
    d["rank_pred"] = g["pred"].rank(ascending=False, method="first")
    d["n_city_year"] = g["pred"].transform("size")
    d["actual_pct"] = g["actual_rel"].rank(pct=True)
    return d.assign(screen=name)


def metrics(d: pd.DataFrame) -> dict:
    called = d[d["pred_rel"] != 0]                                   # the median area of an odd-sized city-year gets no call
    top = d[d["rank_pred"] == 1]
    third = np.ceil(d["n_city_year"] / 3)
    hi, lo = d[d["rank_pred"] <= third], d[d["rank_pred"] > d["n_city_year"] - third]
    rho = d.groupby(["city", "year"]).apply(lambda t: t["pred"].rank().corr(t["actual_rel"].rank()))
    return {"screen": d["screen"].iloc[0], "area_years": len(d), "city_years": int(d.groupby(["city", "year"]).ngroups),
            "direction_accuracy": float((np.sign(called["pred_rel"]) == np.sign(called["actual_rel"])).mean()),
            "rank_corr_within_city_year": float(rho.mean()),
            "top_pick_in_top_half": float((top["actual_pct"] > 0.5).mean()),
            "top_pick_extra_income": float(top["actual_rel_dollars"].mean()),
            "top_third_minus_bottom_third": float(hi["actual_rel_dollars"].mean() - lo["actual_rel_dollars"].mean())}


def bootstrap(d: pd.DataFrame, keys: list[str], n: int = 1000, seed: int = 0) -> dict:
    """95% interval for each metric in `keys`, resampling whole city-years (the unit a ranking is made in)."""
    groups = [t for _, t in d.groupby(["city", "year"])]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        m = metrics(pd.concat([groups[i].assign(year=groups[i]["year"].iloc[0] + 10000 * j) for j, i in enumerate(pick)]))   # copies stay separate
        draws.append([m[k] for k in keys])
    lo, hi = np.percentile(np.array(draws), [2.5, 97.5], axis=0)
    return {k: (float(a), float(b)) for k, a, b in zip(keys, lo, hi)}


def signal_direction(df: pd.DataFrame, feature: str = "d_log_trips_rel") -> pd.DataFrame:
    """Which way does trip growth point? One row per forward fold: the fitted slope of income growth on trip growth (both relative to
    the city-year, momentum in the model), plus the plain and momentum-adjusted correlations on all rows.
    slope -0.3 means: an area whose trips grew 10% faster than its city is expected to see income grow about 3% slower."""
    cols = AN.feature_sets()["B1"] + [feature]
    rows = []
    for tr, te in PM.splits_for(df, "forward", min_train_years=1):
        trn = df.iloc[tr]
        m = fit_weighted(make_model("ridge"), trn[cols], trn["dlog_rel"].values)
        scale = float(np.nanstd(trn[feature].values))
        rows.append({"what": f"model slope, test year {int(df.iloc[te]['year'].iloc[0])}", "n": len(tr),
                     "value": float(m.steps[-1][1].coef_[-1]) / scale})

    def resid(y, x):
        X = np.column_stack([np.ones(len(x)), x])
        return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    y, x, mom = df["dlog_rel"].values, df[feature].values, df["dlog_lag_rel"].values
    rows.append({"what": "correlation with income growth, all rows", "n": len(df), "value": float(np.corrcoef(x, y)[0, 1])})
    rows.append({"what": "same, after removing income momentum", "n": len(df), "value": float(np.corrcoef(resid(x, mom), resid(y, mom))[0, 1])})
    for c, t in df.groupby("city"):
        rows.append({"what": f"correlation with income growth, {c}", "n": len(t), "value": float(t[feature].corr(t["dlog_rel"]))})
    return pd.DataFrame(rows)
