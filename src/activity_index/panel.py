"""PUMA-year panel: matched-month year-over-year bikeshare features and year-over-year income change.

Design (docs/puma_study.md): the outcome is an annual average (ACS 1-year pools interviews over all twelve months),
so features are year-over-year changes built from *matched months*: a station-month counts only if the same station
had departures in the same calendar month of both years. Seasonality and weather then cancel exactly, and a station
that opened or closed between the two years cannot create a fake trend.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import ARR_COLS, DEP_COLS, haversine_km
from .noise import ratio_se

VALUE_COLS = DEP_COLS + ARR_COLS
FEATURES = ["d_log_trips", "station_growth", "new_station_share",
            "d_casual_share", "d_weekend_share", "d_round_share", "d_net_am", "d_net_pm", "d_log_duration"]
LEVELS = ["lvl_trips_per_station_month", "lvl_casual_share", "lvl_round_share", "lvl_net_am", "lvl_net_pm"]


def matched_pairs(sm: pd.DataFrame, year: int, months=None, span: int = 1) -> pd.DataFrame:
    """Station-months with departures in the same calendar month of `year - span` and `year`.

    `sm` needs puma, station_id, ym and the count columns. Columns come back suffixed _0 (year - span) and _1 (year)."""
    sm = sm.assign(y=sm["ym"] // 100, m=sm["ym"] % 100)
    if months is not None:
        sm = sm[sm["m"].isin(list(months))]
    act = sm[sm["dep"] > 0]
    keep = ["puma", "station_id", "m"] + VALUE_COLS
    a = act.loc[act["y"] == year - span, keep]
    b = act.loc[act["y"] == year, keep]
    return a.merge(b, on=["puma", "station_id", "m"], suffixes=("_0", "_1"))


def _station_features(g: pd.DataFrame, s: int) -> pd.DataFrame:
    d = g[f"dep_{s}"].replace(0, np.nan)
    out = pd.DataFrame({
        "casual_share": 1 - g[f"dep_member_{s}"] / d,
        "weekend_share": 1 - g[f"dep_weekday_{s}"] / d,
        "round_share": g[f"dep_round_{s}"] / d,
        "mean_duration": g[f"dur_sum_{s}"] / d}, index=g.index)
    for part in ("am", "pm"):
        tot = (g[f"dep_{part}_{s}"] + g[f"arr_{part}_{s}"]).replace(0, np.nan)
        out[f"net_{part}"] = (g[f"dep_{part}_{s}"] - g[f"arr_{part}_{s}"]) / tot     # > 0: mostly a departure point
    return out


def _wmean(values: pd.Series, w: pd.Series, groups: pd.Series) -> pd.Series:
    ok = values.notna() & (w > 0)
    num = (values * w).where(ok).groupby(groups).sum()
    den = w.where(ok).groupby(groups).sum()
    return num / den.replace(0, np.nan)


def puma_year_features(sm: pd.DataFrame, year: int, months=None, min_matched: int = 3, span: int = 1) -> pd.DataFrame:
    """One row per PUMA: bikeshare change from year - span to year (span=1: year over year) on matched station-months.

    `months` limits both years to those calendar months (e.g. range(1, 7) = what is known by the end of June).
    Features are NaN where fewer than `min_matched` stations are matched. Columns:
      d_log_trips, d_casual_share, d_weekend_share, d_round_share, d_net_am, d_net_pm, d_log_duration:
          change on matched station-months (shares are trip-weighted means over stations, with the same weights
          in both years); d_log_trips is log(trips_t / trips_t-1).
      station_growth, new_station_share: system expansion (log change in active stations; share of this year's
          trips from stations with no trips last year). Controls for "more stations", not part of the matched set.
      lvl_*: the same quantities in year - 1 (the starting level)."""
    pairs = matched_pairs(sm, year, months, span)
    cols = []
    if len(pairs):
        st = pairs.groupby(["puma", "station_id"], as_index=False)[[f"{c}_{s}" for c in VALUE_COLS for s in (0, 1)]].sum()
        w = (st["dep_0"] + st["dep_1"]) / 2
        f0, f1 = _station_features(st, 0), _station_features(st, 1)
        grp = st["puma"]
        res = pd.DataFrame({"n_matched": st.groupby("puma")["station_id"].nunique(),
                            "matched_station_months": pairs.groupby("puma").size(),
                            "trips_0": st.groupby("puma")["dep_0"].sum(), "trips_1": st.groupby("puma")["dep_1"].sum()})
        res["d_log_trips"] = np.log(res["trips_1"] / res["trips_0"])
        res["lvl_trips_per_station_month"] = res["trips_0"] / res["matched_station_months"]
        for f in ("casual_share", "weekend_share", "round_share", "net_am", "net_pm"):
            a, b = _wmean(f0[f], w, grp), _wmean(f1[f], w, grp)
            res[f"d_{f}"], res[f"lvl_{f}"] = b - a, a
        a, b = _wmean(f0["mean_duration"], w, grp), _wmean(f1["mean_duration"], w, grp)
        res["d_log_duration"] = np.log(b / a)
        thin = res["n_matched"] < min_matched
        cols = [c for c in res.columns if c.startswith(("d_", "lvl_"))]
        res.loc[thin, cols] = np.nan
    else:
        res = pd.DataFrame(columns=["n_matched", "matched_station_months", "trips_0", "trips_1"]
                           + [f for f in FEATURES if f.startswith("d_")] + LEVELS, dtype=float)
        res.index.name = "puma"

    # system expansion on all active stations (not only matched ones), same months filter
    s2 = sm.assign(y=sm["ym"] // 100, m=sm["ym"] % 100)
    if months is not None:
        s2 = s2[s2["m"].isin(list(months))]
    act = s2[s2["dep"] > 0]
    prev = act.loc[act["y"] == year - span, ["puma", "station_id"]].drop_duplicates().assign(was=1)
    cur = act.loc[act["y"] == year].merge(prev, on=["puma", "station_id"], how="left")
    cur["was"] = cur["was"].notna()
    exp = pd.DataFrame({"n_active_0": prev.groupby("puma").size(), "n_active_1": cur.groupby("puma")["station_id"].nunique(),
                        "trips_new_stations": cur.loc[~cur["was"]].groupby("puma")["dep"].sum(),
                        "trips_all": cur.groupby("puma")["dep"].sum()})
    exp["trips_new_stations"] = exp["trips_new_stations"].fillna(0)
    exp["station_growth"] = np.log(exp["n_active_1"] / exp["n_active_0"])
    exp["new_station_share"] = exp["trips_new_stations"] / exp["trips_all"]
    out = res.join(exp[["n_active_0", "n_active_1", "station_growth", "new_station_share"]], how="outer")
    out["year"] = year
    return out.reset_index().rename(columns={"index": "puma"})


def income_table(acs: pd.DataFrame) -> pd.DataFrame:
    """Mean household income, its log and the standard error of the log, per (GEOID, year).

    `acs` needs GEOID, year, agg_household_income, households and their margins of error (_moe columns)."""
    d = acs.copy()
    hh = d["households"].where(d["households"] > 0)
    d["mean_income"] = d["agg_household_income"] / hh
    se = ratio_se(d["agg_household_income"], d["agg_household_income_moe"], hh, d["households_moe"])
    d["log_income"] = np.log(d["mean_income"].where(d["mean_income"] > 0))
    d["se_log_income"] = se / d["mean_income"]
    return d


def add_changes(inc: pd.DataFrame) -> pd.DataFrame:
    """Add dlog (log income change from year - 1 to year), its SE, and the same quantities one year earlier.

    Only consecutive years are used (the 2020 gap never creates a two-year change). The two years come from
    independent samples, so the SEs add in quadrature."""
    base = inc[["GEOID", "year", "log_income", "se_log_income"]]
    prev = base.assign(year=base["year"] + 1).rename(columns={"log_income": "log_income_prev", "se_log_income": "se_prev"})
    d = inc.merge(prev, on=["GEOID", "year"], how="left")
    d["dlog"] = d["log_income"] - d["log_income_prev"]
    d["se_dlog"] = np.sqrt(d["se_log_income"] ** 2 + d["se_prev"] ** 2)
    lag = d[["GEOID", "year", "dlog", "se_dlog"]]
    lag = lag.assign(year=lag["year"] + 1).rename(columns={"dlog": "dlog_lag", "se_dlog": "se_dlog_lag"})
    return d.merge(lag, on=["GEOID", "year"], how="left")


def add_span_changes(inc: pd.DataFrame, span: int) -> pd.DataFrame:
    """Multi-year version of add_changes: dlog is the change in log income from year - span to year.

    Same column names as add_changes so everything downstream is unchanged:
      log_income_prev  income level at the START of the window (year - span)
      dlog, se_dlog    the change over the window and its standard error (the two years are independent samples)
      dlog_lag         momentum: change from year - 2*span to year - span - 1. It stops one year BEFORE the window starts, so it
                       shares no survey year with dlog (a shared year would make last period's noise "predict" this period's).
    span=1 has no such gap window (dlog_lag is NaN); use add_changes for year over year."""
    L = inc.pivot(index="GEOID", columns="year", values="log_income")
    S = inc.pivot(index="GEOID", columns="year", values="se_log_income")

    def col(T, y):
        return T[y].values if y in T.columns else np.full(len(T), np.nan)

    parts = []
    for t in sorted(L.columns):
        s0, a, b = t - span, t - span - 1, t - 2 * span
        d = pd.DataFrame({"GEOID": L.index, "year": t})
        d["log_income_prev"] = col(L, s0)
        d["dlog"] = col(L, t) - col(L, s0)
        d["se_dlog"] = np.sqrt(col(S, t) ** 2 + col(S, s0) ** 2)
        if a > b:
            d["dlog_lag"] = col(L, a) - col(L, b)
            d["se_dlog_lag"] = np.sqrt(col(S, a) ** 2 + col(S, b) ** 2)
        else:
            d["dlog_lag"], d["se_dlog_lag"] = np.nan, np.nan
        parts.append(d)
    return inc.merge(pd.concat(parts, ignore_index=True), on=["GEOID", "year"], how="left")


def relative_to_group(df: pd.DataFrame, cols: list[str], by=("city", "year"), suffix: str = "_rel") -> pd.DataFrame:
    """Add `<col>_rel` = value minus the median of its (city, year) group, which removes city-wide shocks
    (a recession, system-wide expansion) so cities and years are comparable."""
    out = df.copy()
    med = df.groupby(list(by))[cols].transform("median")
    for c in cols:
        out[c + suffix] = df[c] - med[c]
    return out


def puma_weights(se, y) -> pd.Series:
    """Inverse-variance weights 1 / (signal variance + noise variance), scaled to mean 1.

    The signal variance is the observed variance minus the average noise variance, floored at 10% of the
    observed variance so weights stay finite."""
    se, y = pd.Series(se, dtype=float), pd.Series(y, dtype=float)
    noise_var = se ** 2
    var_y = float(y.var())
    tau2 = max(0.1 * var_y, var_y - float(noise_var.mean()))
    w = 1.0 / (tau2 + noise_var)
    return w / w.mean()


def noise_ceiling(y, se) -> dict:
    """Share of the variance of y that is sampling noise, and the best achievable R2 and correlation."""
    y, se = pd.Series(y, dtype=float), pd.Series(se, dtype=float)
    share = float((se ** 2).mean() / y.var())
    share = min(share, 1.0)
    return {"n": int(len(y)), "var_observed": float(y.var()), "var_noise": float((se ** 2).mean()),
            "noise_share": share, "max_r2": 1 - share, "max_corr": float(np.sqrt(1 - share))}


CONTROLS = ["edu_share_ba", "renter_share", "log_rent", "log_home_value", "median_age", "log_density", "km_to_center", "log_households"]
BIKE_ALL = ["d_log_trips", "station_growth", "new_station_share", "d_casual_share", "d_weekend_share", "d_round_share",
            "d_net_am", "d_net_pm", "d_log_duration"]


def controls_from_acs(acs: pd.DataFrame, pumas: pd.DataFrame, centers: dict, lag: int = 1) -> pd.DataFrame:
    """Controls measured `lag` years before the outcome year (labelled with the outcome year so they merge onto it).

    `pumas` has GEOID, ALAND (m2), clat, clon; `centers` maps city -> (lat, lon)."""
    a = acs.merge(pumas[["GEOID", "ALAND", "clat", "clon"]], on="GEOID", how="left")
    c = pd.DataFrame({"GEOID": a["GEOID"], "year": a["year"] + lag})
    c["edu_share_ba"] = (a["edu_bachelors"] + a["edu_masters"] + a["edu_professional"] + a["edu_doctorate"]) / a["edu_total"]
    c["renter_share"] = a["renter_households"] / a["households"]
    c["log_rent"], c["log_home_value"] = np.log(a["median_gross_rent"]), np.log(a["median_home_value"])
    c["median_age"] = a["median_age"]
    with np.errstate(divide="ignore"):
        c["log_density"] = np.log(a["population"] / (a["ALAND"] / 1e6)).replace(-np.inf, np.nan)
        c["log_households"] = np.log(a["households"].where(a["households"] > 0))
    lat0 = a["city"].map(lambda k: centers[k][0])
    lon0 = a["city"].map(lambda k: centers[k][1])
    c["km_to_center"] = haversine_km(a["clat"], a["clon"], lat0, lon0)
    return c


def build_panel(sm: pd.DataFrame, acs: pd.DataFrame, pumas: pd.DataFrame, params: dict, centers: dict,
                relative_cols: list[str], min_group: int = 4, span: int = 1, require_controls: bool = True):
    """The whole PUMA-year table from station_month rows (with a `puma` column), ACS 1-year rows and PUMA geometry facts.

    span=1: year-over-year (outcome years 2016-2019). span=h>=2: the outcome and the bikeshare features both cover the
    h years ending in the row's year, controls are measured at the window start, and momentum is the gap-window change
    (see add_span_changes). Rows exist for end years whose window starts inside the bike data (e.g. h=3: 2018, 2019).

    Returns (panel, first-half panel, mature-station counts). `params` is config.PUMA."""
    first, last = params["bike_years"][0], params["bike_years"][-1]
    act = sm[sm["dep"] > 0].assign(year=lambda d: d["ym"] // 100)
    in_first = set(map(tuple, act.loc[act["year"] == first, ["puma", "station_id"]].drop_duplicates().values))
    in_last = set(map(tuple, act.loc[act["year"] == last, ["puma", "station_id"]].drop_duplicates().values))
    mature = pd.DataFrame(sorted(in_first & in_last), columns=["puma", "station_id"])
    n_mature = mature.groupby("puma").size().rename("n_mature_stations")
    scope = n_mature[n_mature >= params["min_mature_stations"]].index
    sm = sm[sm["puma"].isin(scope)]

    years = [y for y in params["bike_years"] if y - span >= first]

    def feats(months):
        f = pd.concat([puma_year_features(sm, y, months, params["min_matched_stations"], span) for y in years],
                      ignore_index=True)
        return f.rename(columns={"puma": "GEOID"})
    full, h1 = feats(None), feats(range(1, 7))

    inc0 = income_table(acs)
    inc = add_changes(inc0) if span == 1 else add_span_changes(inc0, span)
    keep = ["GEOID", "year", "NAME", "city", "mean_income", "log_income", "se_log_income", "log_income_prev", "dlog", "se_dlog", "dlog_lag",
            "se_dlog_lag", "households"]
    panel = full.merge(inc[keep], on=["GEOID", "year"], how="left")
    panel = panel.merge(controls_from_acs(acs, pumas, centers, lag=span), on=["GEOID", "year"], how="left")
    panel = panel.merge(n_mature.rename_axis("GEOID").reset_index(), on="GEOID", how="left")
    panel = panel.merge(pumas[["GEOID", "ALAND"]], on="GEOID", how="left")
    panel["stations_per_km2"] = panel["n_active_1"] / (panel["ALAND"] / 1e6)

    need = ["dlog", "dlog_lag", "log_income_prev"] + BIKE_ALL + (CONTROLS if require_controls else [])
    panel["valid"] = panel[need].notna().all(axis=1)
    panel["group_n"] = panel.groupby(["city", "year"])["valid"].transform("sum")
    panel["usable"] = panel["valid"] & (panel["group_n"] >= min_group)

    def add_rel(df, cols):
        v = df[df["usable"].fillna(False).astype(bool)]
        rel = relative_to_group(v, cols)[[c + "_rel" for c in cols]]
        return df.join(rel)

    panel = add_rel(panel, relative_cols)
    panel["log_income_prev_rel_sq"] = panel["log_income_prev_rel"] ** 2
    h1 = h1.merge(panel[["GEOID", "year", "city", "usable", "dlog"]], on=["GEOID", "year"], how="left")
    h1 = add_rel(h1, BIKE_ALL)
    return panel, h1, n_mature
