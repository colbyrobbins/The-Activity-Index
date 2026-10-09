"""Models, validation and metrics for the noise-aware baselines (scripts/10_baselines.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd

CONTROLS = ["edu_share_ba", "renter_share", "median_gross_rent", "median_home_value", "median_age",
            "log_density", "km_to_center", "log_households"]
# Bikeshare features in four groups fixed in advance from the spec (usage, rider mix, commute role, trip
# character), so the question is "does a kind of signal help", not "which of 10 columns got lucky".
GROUPS = {
    "usage": ["trips_per_station_month", "log_growth"],
    "rider_mix": ["casual_share", "weekend_share", "round_share"],
    "commute_role": ["net_am", "net_pm"],
    "trip_character": ["mean_duration_min", "mean_km", "dest_norm_entropy"],
}
BLOCK_KM, SEED = 3.0, 0
LIFTS = [("B1", "B0"), ("B2", "B1")] + [(f"ALT_{g}", "B2") for g in GROUPS] + [("ALT_all", "B2")]


def noise_weights(se: pd.Series, observed: pd.Series) -> pd.Series:
    """Inverse-variance weights for the rank-change target: 1 / (signal variance + noise variance).

    The signal variance is what is left of the observed variance after removing the average noise
    variance (floored so weights stay finite). Weights are scaled to mean 1.
    """
    noise_var = se.astype(float) ** 2
    tau2 = max(1e-3, float(observed.var()) - float(noise_var.mean()))
    w = 1.0 / (tau2 + noise_var)
    return w / w.mean()


def add_city_terms(df: pd.DataFrame):
    """City dummies and city-by-initial-rank terms (city-specific mean reversion). Returns (df, columns)."""
    out, cols = df.copy(), []
    cities = sorted(df["city"].unique())
    for c in cities[1:]:                     # first city is the reference level
        ind = (df["city"] == c).astype(float)
        out[f"city_{c}"], out[f"city_{c}_x_rank"], out[f"city_{c}_x_rank2"] = ind, ind * df["income_rank_t0"], ind * df["income_rank_t0"] ** 2
        cols += [f"city_{c}", f"city_{c}_x_rank", f"city_{c}_x_rank2"]
    return out, cols


BIKE_ALL = [c for cols in GROUPS.values() for c in cols]


def city_relative(tab: pd.DataFrame) -> pd.DataFrame:
    """Add `<col>_cr` columns: each feature as a within-city percentile (0 to 1), so cities of different size
    and cost level become comparable. Computed on `tab` = every tract in the study area, no target involved:
    controls are ranked among all tracts of their city, bikeshare features among the tracts that have an
    established station (the population they describe). Initial rank and momentum are already city-relative."""
    out = tab.copy()
    for col in CONTROLS:
        out[f"{col}_cr"] = out.groupby("city")[col].rank(pct=True)
    ref = out["n_mature"] >= 1
    for col in BIKE_ALL:
        out[f"{col}_cr"] = np.nan
        out.loc[ref, f"{col}_cr"] = out[ref].groupby("city")[col].rank(pct=True)
    return out


def feature_sets(df: pd.DataFrame, extra: list[str] | None = None, relative: bool = False) -> dict:
    """Nested feature sets. relative=True swaps controls and bikeshare features for their `_cr` versions."""
    extra = extra or []
    sfx = "_cr" if relative else ""
    controls = [c + sfx for c in CONTROLS]
    groups = {g: [c + sfx for c in cols] for g, cols in GROUPS.items()}
    b0 = ["income_rank_t0", "income_rank_t0_sq"] + extra
    sets = {"B0": b0, "B1": b0 + controls}
    sets["B2"] = sets["B1"] + (["momentum"] if "momentum" in df and df["momentum"].notna().mean() > 0.5 else [])
    for g, cols in groups.items():
        sets[f"ALT_{g}"] = sets["B2"] + cols
    sets["ALT_all"] = sets["B2"] + [c for cols in groups.values() for c in cols]
    return sets


def make_model(kind: str):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    imp = SimpleImputer(strategy="median")
    if kind == "ridge":
        return make_pipeline(imp, StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30)))
    import lightgbm as lgb
    return make_pipeline(imp, lgb.LGBMRegressor(n_estimators=250, learning_rate=0.03, num_leaves=6, min_child_samples=25,
                                                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=10.0,
                                                random_state=SEED, verbose=-1))


def fit_weighted(model, X, y, w=None):
    """Fit a pipeline from make_model, routing sample weights to its final step."""
    final = model.steps[-1][0]
    if w is None:
        return model.fit(X, y)
    return model.fit(X, y, **{f"{final}__sample_weight": np.asarray(w)})


def spatial_groups(df: pd.DataFrame) -> np.ndarray:
    lat0 = np.radians(df["clat"])
    x, y = df["clon"] * 111.32 * np.cos(lat0), df["clat"] * 110.57
    return (df["city"] + "_" + np.floor(x / BLOCK_KM).astype(int).astype(str) + "_" + np.floor(y / BLOCK_KM).astype(int).astype(str)).values


def splits_for(df: pd.DataFrame, scheme: str):
    if scheme == "loco":
        return [(np.where(df["city"] != c)[0], np.where(df["city"] == c)[0]) for c in sorted(df["city"].unique())]
    from sklearn.model_selection import GroupKFold
    return list(GroupKFold(n_splits=5).split(df, groups=spatial_groups(df)))


def oof_predictions(df, cols, kind, scheme, weights=None, target="rank_change"):
    X, y = df[cols], df[target].values
    pred = np.full(len(df), np.nan)
    for tr, te in splits_for(df, scheme):
        m = fit_weighted(make_model(kind), X.iloc[tr], y[tr], None if weights is None else np.asarray(weights)[tr])
        pred[te] = m.predict(X.iloc[te])
    return pred


def spearman(a, b) -> float:
    return pd.Series(np.asarray(a)).rank().corr(pd.Series(np.asarray(b)).rank())


def r2(y, p, w=None) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    ybar = np.average(y, weights=w)
    return 1 - np.sum(w * (y - p) ** 2) / np.sum(w * (y - ybar) ** 2)


def precision_at(score, truth, k=20) -> float:
    top = np.argsort(-np.asarray(score))[:k]
    return float(np.asarray(truth)[top].mean()) if len(top) else np.nan


def metric_rows(df, pred, weights, lownoise, label) -> list[dict]:
    from sklearn.metrics import average_precision_score
    cut = df.groupby("city")["rank_change"].transform(lambda s: s.quantile(0.75))
    top = (df["rank_change"] >= cut).values
    y = df["rank_change"].values
    rows = []
    for scope in ["ALL"] + sorted(df["city"].unique()):
        m = np.ones(len(df), bool) if scope == "ALL" else (df["city"] == scope).values
        r = {**label, "scope": scope, "n": int(m.sum()), "spearman": spearman(y[m], pred[m]), "r2": r2(y[m], pred[m]),
             "r2_weighted": r2(y[m], pred[m], np.asarray(weights)[m]),
             "mae": float(np.abs(y[m] - pred[m]).mean()),
             "pr_auc": float(average_precision_score(top[m], pred[m])), "base_rate": float(top[m].mean())}
        ml = m & lownoise
        r["spearman_lownoise"] = spearman(y[ml], pred[ml]) if ml.sum() > 20 else np.nan
        if scope == "ALL":
            r["p_at_20"] = float(np.mean([precision_at(pred[(df["city"] == c).values], top[(df["city"] == c).values])
                                          for c in df["city"].unique()]))
        else:
            r["p_at_20"] = precision_at(pred[m], top[m])
        rows.append(r)
    return rows


def bootstrap_lift(y, p_new, p_old, n_boot=500, seed=SEED) -> np.ndarray:
    """Bootstrap changes in Spearman and R2 (new minus old) over tracts; returns array (n_boot, 2)."""
    rng = np.random.default_rng(seed)
    y, p_new, p_old = map(np.asarray, (y, p_new, p_old))
    out = np.empty((n_boot, 2))
    for b in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        out[b] = (spearman(y[i], p_new[i]) - spearman(y[i], p_old[i]), r2(y[i], p_new[i]) - r2(y[i], p_old[i]))
    return out
