"""Stress tests for the one bikeshare feature that helped the dollar nowcast: matched-station trip growth (B2g vs B1).

Used by scripts/24_placebo.py. Everything reuses the forward validation and dollar scoring of area_nowcast, so numbers are comparable.
  gain = mean absolute dollar error with the feature minus without it (negative = the feature helped).
Placebos keep the model, the sample and the feature's distribution, and break only the link being tested:
  across areas : each city-year's trip-growth values are dealt out to the wrong areas (is it THIS area's growth that matters?)
  across years : each area's trip-growth values are dealt out to the wrong years (is it THIS year's growth, or just which area it is?)
If the real gain is not clearly better than the placebo gains, the improvement is what a meaningless feature produces by chance.
  best of all  : trip growth was picked AFTER seeing that it was the best of the bikeshare features. To be fair to chance, each draw
                 hands every area another area's whole bikeshare record (same city-year) and keeps the best single feature's gain.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import area_nowcast as AN

FEATURE = "d_log_trips_rel"


def mae(df: pd.DataFrame, pred: np.ndarray) -> float:
    ok = ~np.isnan(pred)
    return float(AN._abs_err(df[ok], pred[ok]).mean())


def gain(df: pd.DataFrame, new_cols: list[str], old_cols: list[str]) -> float:
    return mae(df, AN.forward_predictions(df, new_cols)) - mae(df, AN.forward_predictions(df, old_cols))


def shuffled(df: pd.DataFrame, col: str, within: list[str], rng) -> pd.DataFrame:
    """Copy of df with `col` permuted inside each group of `within`."""
    out = df.copy()
    out[col] = df.groupby(within)[col].transform(lambda s: rng.permutation(s.values))
    return out


def placebo(df: pd.DataFrame, within: list[str], n: int = 2000, seed: int = 0, feature: str = FEATURE) -> np.ndarray:
    """Gains (feature rung vs B1) from n placebo datasets where `feature` is shuffled inside `within` groups."""
    sets = AN.feature_sets()
    old, new = sets["B1"], sets["B1"] + [feature]
    base = mae(df, AN.forward_predictions(df, old))            # B1 does not use the feature, so it is the same in every draw
    rng = np.random.default_rng(seed)
    return np.array([mae(d, AN.forward_predictions(d, new)) - base for d in (shuffled(df, feature, within, rng) for _ in range(n))])


def single_feature_gains(df: pd.DataFrame, features: list[str]) -> pd.Series:
    """Gain of B1 + one feature vs B1, for each feature."""
    old = AN.feature_sets()["B1"]
    base = mae(df, AN.forward_predictions(df, old))
    return pd.Series({f: mae(df, AN.forward_predictions(df, old + [f])) - base for f in features})


def best_of_placebo(df: pd.DataFrame, features: list[str], n: int = 500, seed: int = 0) -> np.ndarray:
    """Best (most negative) single-feature gain in n datasets where whole bikeshare records are swapped between areas of a city-year."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        d = df.copy()
        for _, idx in df.groupby(["city", "year"]).groups.items():
            d.loc[idx, features] = df.loc[rng.permutation(np.asarray(idx)), features].values
        out.append(single_feature_gains(d, features).min())
    return np.array(out)


def placebo_row(name: str, real: float, draws: np.ndarray) -> dict:
    """p = share of placebo draws that helped at least as much as the real feature (with the +1 correction)."""
    return {"placebo": name, "real_gain": real, "placebo_mean": float(draws.mean()), "placebo_p05": float(np.percentile(draws, 5)),
            "placebo_p95": float(np.percentile(draws, 95)), "n_draws": len(draws),
            "p_value": float((np.sum(draws <= real) + 1) / (len(draws) + 1))}


def by_group(df: pd.DataFrame, new_cols, old_cols, col: str) -> pd.DataFrame:
    """Gain inside each value of `col` (city or year), from the full-sample forward predictions."""
    p_new, p_old = AN.forward_predictions(df, new_cols), AN.forward_predictions(df, old_cols)
    ok = ~np.isnan(p_new)
    d = df[ok].copy()
    d["miss_old"], d["miss_new"] = AN._abs_err(d, p_old[ok]), AN._abs_err(d, p_new[ok])
    g = d.groupby(col).agg(n=("miss_old", "size"), mae_b1=("miss_old", "mean"), mae_b2g=("miss_new", "mean"),
                           helped_share=("miss_new", lambda s: float((s < d.loc[s.index, "miss_old"]).mean())))
    g["gain"] = g["mae_b2g"] - g["mae_b1"]
    return g.reset_index().rename(columns={col: "group"}).assign(by=col)


def leave_one_city_out(df: pd.DataFrame, new_cols, old_cols, n_boot: int = 2000) -> pd.DataFrame:
    """Refit and rescore without each city in turn: does the gain depend on one city?"""
    rows = []
    for city in [None] + sorted(df["city"].unique()):
        d = (df if city is None else df[df["city"] != city]).reset_index(drop=True)
        p_new, p_old = AN.forward_predictions(d, new_cols), AN.forward_predictions(d, old_cols)
        g, lo, hi = AN.cluster_gain(d, p_new, p_old, n_boot)
        rows.append({"dropped": city or "none", "areas": d["GEOID"].nunique(), "n": int((~np.isnan(p_new)).sum()),
                     "mae_b1": mae(d, p_old), "mae_b2g": mae(d, p_new), "gain": g, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)
