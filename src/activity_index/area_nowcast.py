"""Dollar nowcast on the area panel (scripts/22_area_nowcast.py).

Target: the area's mean household income in DOLLARS for year T. Success = how close the predicted dollar average is to the posted one.
Rows: one area in one target year (2017-2019, 2023, 2024). Validation is forward in time only (a test year is predicted from earlier
years); the recent test years (2023, 2024) are the nowcast years that matter.

Models predict the change in log income vs the last published year, relative to the city's median change that year (dlog_rel), then
  predicted income = income_(T-1) x exp(predicted relative change + city trend)
where the city trend is the mean of that city's yearly median change over the TRAINING years (known at nowcast time). Nobody knows the
city-wide shift of year T in advance, so errors are reported twice: raw, and after removing each city-year's average error
('shift-free', what remains if city growth were known - the part an area-level signal like bikeshare can affect).
Rungs:  B0 last published x city trend | B1 + the area's last income change | F_<group>, F_all  B1 + year-T bikeshare features
(level, change vs T-1, deviation from the area's own earlier years; all relative to the city-year median).
Also reported: the sampling-noise floor, the dollar error a perfect model would still show because the posting itself is a survey estimate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import panel as P
from . import panel_modeling as PM
from . import puma_nowcast as PN
from .modeling import fit_weighted, make_model

GROUPS = PN.GROUPS
BIKE_ALL = PN.BIKE_ALL
TWO = ["d_log_trips", "dev_log_trips"]     # matched-station trip growth, and this year's trip level vs the area's own earlier years
LIFTS = [("B1", "B0"), ("B2", "B1"), ("B2g", "B1"), ("B2l", "B1")] + [(f"F_{g}", "B1") for g in GROUPS] + [("F_all", "B1"), ("F_all", "B2")]
WHAT = {"B0": "last posting x city trend", "B1": "B0 + area's last income change",
        "B2": "B1 + trip growth + trip level vs own history",
        "B2g": "B1 + trip growth only", "B2l": "B1 + trip level vs own history only",
        **{f"F_{g}": f"B1 + {g}" for g in GROUPS}, "F_all": "B1 + all bikeshare"}
NOISE_MAE = 0.7979          # E|N(0,1)|: mean absolute survey error is this times the standard error


def prepare(panel: pd.DataFrame, target_years) -> pd.DataFrame:
    """Target-year rows with every field, plus `<col>_rel` (minus the city-year median) for the change, momentum and bike features."""
    d = PN.add_own_history(PN.add_year_t_levels(panel))
    need = ["mean_income", "se_log_income", "log_income_prev", "dlog", "dlog_lag"] + BIKE_ALL
    d = d[d["year"].isin(list(target_years))]
    d = d[d[need].notna().all(axis=1)].reset_index(drop=True)
    d["income_prev"] = np.exp(d["log_income_prev"])
    return P.relative_to_group(d, ["dlog", "dlog_lag"] + BIKE_ALL)


def feature_sets() -> dict:
    mom = ["dlog_lag_rel"]
    sets = {"B0": [], "B1": mom, "B2": mom + [c + "_rel" for c in TWO],
            "B2g": mom + [TWO[0] + "_rel"], "B2l": mom + [TWO[1] + "_rel"]}
    groups = {g: [c + "_rel" for c in cols] for g, cols in GROUPS.items()}
    for g, cols in groups.items():
        sets[f"F_{g}"] = mom + cols
    sets["F_all"] = mom + [c for cols in groups.values() for c in cols]
    return sets


def city_trend(train: pd.DataFrame) -> pd.Series:
    """Mean over the training years of each city's yearly median log change."""
    return train.groupby(["city", "year"])["dlog"].median().groupby("city").mean()


def forward_predictions(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Predicted log change vs last posting (relative change + city trend) for every row that has earlier years to train on."""
    pred = np.full(len(df), np.nan)
    for tr, te in PM.splits_for(df, "forward", min_train_years=1):
        trn = df.iloc[tr]
        trend = df.iloc[te]["city"].map(city_trend(trn)).values
        if cols:
            m = fit_weighted(make_model("ridge"), trn[cols], trn["dlog_rel"].values)
            rel = m.predict(df.iloc[te][cols])
        else:
            rel = np.zeros(len(te))
        pred[te] = rel + trend
    return pred


def score(df: pd.DataFrame, dlog_pred) -> dict:
    ok = ~np.isnan(dlog_pred)
    d, p = df[ok], dlog_pred[ok]
    y, prev = d["mean_income"].values, d["income_prev"].values
    pred = prev * np.exp(p)
    err = pred - y
    adj = err - pd.Series(err, index=d.index).groupby([d["city"], d["year"]]).transform("mean").values
    floor = NOISE_MAE * float((d["se_log_income"] * d["mean_income"]).mean())
    return {"n": int(ok.sum()), "mae_dollars": float(np.abs(err).mean()), "median_ape_pct": float(np.median(np.abs(err) / y) * 100),
            "bias_dollars": float(err.mean()), "mae_shift_free": float(np.abs(adj).mean()), "noise_floor_mae": floor}


def _abs_err(df, dlog_pred):
    return np.abs(df["income_prev"].values * np.exp(dlog_pred) - df["mean_income"].values)


def cluster_gain(df, p_new, p_old, n_boot=500, seed=0):
    """Change in mean absolute dollar error (new minus old; negative = better) with a 95% interval from resampling whole areas."""
    ok = ~np.isnan(p_new) & ~np.isnan(p_old)
    d, a, b = df[ok].reset_index(drop=True), p_new[ok], p_old[ok]
    diff = _abs_err(d, a) - _abs_err(d, b)
    ids = d["GEOID"].values
    uniq = np.unique(ids)
    idx = {g: np.where(ids == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    boots = [diff[np.concatenate([idx[g] for g in rng.choice(uniq, len(uniq))])].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


def run(df: pd.DataFrame, n_boot: int = 500):
    """Returns (metrics by sample and rung, gains with intervals). Samples: 'all test years' and 'recent (2023-2024)'."""
    df = df.reset_index(drop=True)
    sets = feature_sets()
    preds = {name: forward_predictions(df, cols) for name, cols in sets.items()}
    samples = {"all_test_years": np.ones(len(df), bool), "recent_2023_24": (df["year"] >= 2023).values}
    rows, gains = [], []
    for sname, mask in samples.items():
        sub = df[mask].reset_index(drop=True)
        for name in sets:
            rows.append({"sample": sname, "set": name, "features": WHAT[name], **score(sub, preds[name][mask])})
        for new, old in LIFTS:
            g, lo, hi = cluster_gain(sub, preds[new][mask], preds[old][mask], n_boot)
            gains.append({"sample": sname, "comparison": f"{new} vs {old}", "d_mae_dollars": g, "lo": lo, "hi": hi})
    return pd.DataFrame(rows), pd.DataFrame(gains)
