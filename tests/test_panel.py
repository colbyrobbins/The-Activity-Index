import numpy as np
import pandas as pd

from activity_index import panel as P
from activity_index.features import ARR_COLS, DEP_COLS


def _sm():
    rows = []

    def add(st, year, month, dep, member):
        r = {c: 0.0 for c in DEP_COLS + ARR_COLS}
        r.update(dep=dep, dep_member=member, dep_weekday=dep)
        rows.append({"puma": "P1", "station_id": st, "ym": year * 100 + month, **r})

    for m in range(1, 13):
        add("A", 2015, m, 100, 60)
        add("B", 2015, m, 50, 25)
        add("A", 2016, m, 120 if m < 12 else 0, 60 if m < 12 else 0)   # A has no departures in Dec 2016
        add("B", 2016, m, 50, 25)
        add("C", 2016, m, 30, 15)                                        # C opens in 2016
    return pd.DataFrame(rows)


def test_matched_months_ignore_new_stations_and_missing_months():
    f = P.puma_year_features(_sm(), 2016, min_matched=1).set_index("puma").loc["P1"]
    assert f["n_matched"] == 2 and f["matched_station_months"] == 11 + 12
    assert np.isclose(f["d_log_trips"], np.log((120 * 11 + 50 * 12) / (100 * 11 + 50 * 12)))
    w_a, w_b = (1100 + 1320) / 2, 600
    expected = 0.5 - (w_a * 0.4 + w_b * 0.5) / (w_a + w_b)
    assert np.isclose(f["d_casual_share"], expected)
    assert np.isclose(f["station_growth"], np.log(3 / 2))
    assert np.isclose(f["new_station_share"], 360 / (120 * 11 + 600 + 360))


def test_months_filter_limits_both_years():
    f = P.puma_year_features(_sm(), 2016, months=range(1, 7), min_matched=1).set_index("puma").loc["P1"]
    assert np.isclose(f["d_log_trips"], np.log((720 + 300) / (600 + 300)))


def test_thin_puma_years_get_nan_features():
    f = P.puma_year_features(_sm(), 2016, min_matched=3).set_index("puma").loc["P1"]
    assert np.isnan(f["d_log_trips"]) and f["n_matched"] == 2


def test_year_without_a_previous_year_returns_no_matches():
    f = P.puma_year_features(_sm(), 2015, min_matched=1)
    assert f.empty or f["d_log_trips"].isna().all()


def _acs():
    return pd.DataFrame({"GEOID": ["x"] * 4 + ["y"] * 3, "year": [2014, 2015, 2016, 2018, 2015, 2016, 2017],
                         "agg_household_income": [1e9, 1.1e9, 1.21e9, 1.3e9, 2e9, 2e9, 2e9],
                         "households": [1e4] * 7, "agg_household_income_moe": [2e7] * 7, "households_moe": [300] * 7})


def test_changes_use_consecutive_years_only_and_se_adds_in_quadrature():
    d = P.add_changes(P.income_table(_acs())).set_index(["GEOID", "year"])
    assert np.isclose(d.loc[("x", 2015), "dlog"], np.log(1.1))
    assert np.isclose(d.loc[("x", 2016), "dlog"], np.log(1.1))
    assert np.isclose(d.loc[("x", 2016), "dlog_lag"], np.log(1.1))
    assert np.isnan(d.loc[("x", 2018), "dlog"])                       # 2017 is missing: no two-year change
    assert np.isclose(d.loc[("y", 2016), "dlog"], 0.0)
    se = d.loc[("x", 2016), ["se_log_income", "se_prev"]]
    assert np.isclose(d.loc[("x", 2016), "se_dlog"], np.sqrt((se ** 2).sum()))
    assert 0 < d.loc[("x", 2016), "se_dlog"] < 0.05


def test_relative_to_group_removes_city_year_median():
    df = pd.DataFrame({"city": ["a", "a", "a", "b"], "year": [1, 1, 1, 1], "v": [1.0, 2.0, 9.0, 5.0]})
    out = P.relative_to_group(df, ["v"])
    assert out["v_rel"].tolist() == [-1.0, 0.0, 7.0, 0.0]


def test_weights_downweight_noisy_rows_and_average_one():
    y = pd.Series(np.random.default_rng(0).normal(0, 0.03, 200))
    w = P.puma_weights(pd.Series([0.005, 0.02, 0.08]), y)
    assert np.isclose(w.mean(), 1) and w.iloc[0] > w.iloc[1] > w.iloc[2] > 0


def test_noise_ceiling_bounds():
    y = pd.Series(np.random.default_rng(1).normal(0, 0.03, 500))
    c = P.noise_ceiling(y, pd.Series(np.full(500, 0.015)))
    assert 0 < c["noise_share"] < 1 and np.isclose(c["max_r2"], 1 - c["noise_share"])
    assert P.noise_ceiling(y, pd.Series(np.full(500, 1.0)))["noise_share"] == 1.0


# ---- panel_modeling ----
from activity_index import panel_modeling as PM  # noqa: E402


def _panel(n_puma=12, years=(2016, 2017, 2018, 2019)):
    rng = np.random.default_rng(0)
    rows = [{"GEOID": f"p{i}", "city": "abc"[i % 3], "year": y} for i in range(n_puma) for y in years]
    return pd.DataFrame(rows).assign(dlog_rel=lambda d: rng.normal(0, 0.03, len(d)))


def test_group_folds_keep_every_puma_in_one_fold():
    d = _panel()
    folds = PM.group_folds(d["GEOID"].values, 5)
    seen = []
    for tr, te in folds:
        assert not set(d["GEOID"].iloc[tr]) & set(d["GEOID"].iloc[te])
        seen += list(te)
    assert sorted(seen) == list(range(len(d)))


def test_forward_scheme_trains_only_on_earlier_years():
    d = _panel()
    folds = PM.splits_for(d, "forward")
    assert len(folds) == 2
    for tr, te in folds:
        assert d["year"].iloc[tr].max() < d["year"].iloc[te].min()
    assert sorted(d["year"].iloc[folds[0][1]].unique()) == [2018]


def test_loco_holds_out_each_city():
    d = _panel()
    for tr, te in PM.splits_for(d, "loco"):
        assert len(set(d["city"].iloc[te])) == 1 and not set(d["city"].iloc[tr]) & set(d["city"].iloc[te])


def test_feature_sets_nest_and_groups_cover_nine_bike_changes():
    s = PM.feature_sets(None)
    assert set(s["B0"]) < set(s["B1"]) < set(s["B2"]) < set(s["ALT_all"])
    assert len(PM.BIKE_ALL) == len(set(PM.BIKE_ALL)) == 9
    assert all(set(s["B2"]) < set(s[f"ALT_{g}"]) < set(s["ALT_all"]) for g in PM.GROUPS)


def test_oof_predictions_and_cluster_bootstrap_run_end_to_end():
    d = _panel(20)
    rng = np.random.default_rng(1)
    d["x_rel"] = rng.normal(size=len(d))
    d[PM.TARGET] = 0.5 * d["x_rel"] + rng.normal(0, 0.5, len(d))
    p = PM.oof_predictions(d, ["x_rel"], "puma")
    q = PM.oof_predictions(d, ["x_rel"], "forward")
    assert not np.isnan(p).any() and np.isnan(q).sum() == (d["year"] <= 2017).sum()
    assert PM.score(d[PM.TARGET], p)["spearman"] > 0.3
    junk = rng.normal(size=len(d))
    est, lo, hi = PM.cluster_bootstrap_gain(d[PM.TARGET], p, junk, d["GEOID"], n_boot=100)
    assert lo < est < hi and est > 0


# ---- build_panel on synthetic data ----
from activity_index import config as cfg  # noqa: E402


def synthetic_inputs(seed=0, per_city=8, stations=7):
    rng = np.random.default_rng(seed)
    cities = {"nyc": "36", "chicago": "17", "dc": "11"}
    pumas, acs_rows, sm_rows = [], [], []
    for city, st in cities.items():
        for i in range(per_city):
            gid = f"{st}{i:05d}"
            clat, clon = cfg.CITY_CENTERS[city][0] + 0.01 * i, cfg.CITY_CENTERS[city][1]
            pumas.append({"GEOID": gid, "ALAND": 2e7, "clat": clat, "clon": clon, "city": city})
            lvl = rng.normal(11.3, 0.3)
            bike_trend = rng.normal(0, 0.1, 5)               # yearly log growth in trips, per PUMA
            for y in range(2012, 2020):
                lvl += rng.normal(0.03, 0.01) + (0.1 * bike_trend[y - 2015] if y >= 2016 else 0)
                hh = 40000 + rng.integers(-500, 500)
                acs_rows.append({"GEOID": gid, "year": y, "NAME": f"PUMA {gid}", "city": city,
                                 "agg_household_income": np.exp(lvl) * hh, "households": hh,
                                 "agg_household_income_moe": np.exp(lvl) * hh * 0.02, "households_moe": 800,
                                 "renter_households": hh * 0.5, "population": 100000, "edu_total": 70000,
                                 "edu_bachelors": 15000, "edu_masters": 8000, "edu_professional": 1000, "edu_doctorate": 500,
                                 "median_gross_rent": 1500 + 20 * i, "median_home_value": 400000, "median_age": 35})
            for s in range(stations):
                level = np.exp(rng.normal(5, 0.3))
                for y in range(2015, 2020):
                    level *= np.exp(bike_trend[y - 2015])
                    for m in range(1, 13):
                        dep = level * (1 + 0.8 * np.sin((m - 4) / 12 * 2 * np.pi))
                        r = {c: 0.0 for c in DEP_COLS + ARR_COLS}
                        r.update(dep=float(np.round(dep)), dep_member=float(np.round(0.7 * dep)), dep_weekday=float(np.round(0.7 * dep)),
                                 dep_am=float(np.round(0.1 * dep)), arr_am=float(np.round(0.08 * dep)), dep_pm=float(np.round(0.1 * dep)),
                                 arr_pm=float(np.round(0.12 * dep)), dep_round=float(np.round(0.05 * dep)), dur_sum=float(dep * 12),
                                 arr=float(np.round(dep)))
                        sm_rows.append({"puma": gid, "station_id": f"{gid}-{s}", "city": city, "ym": y * 100 + m, **r})
    return pd.DataFrame(sm_rows), pd.DataFrame(acs_rows), pd.DataFrame(pumas)


def test_build_panel_end_to_end_on_synthetic_data():
    sm, acs, pumas = synthetic_inputs()
    pan, h1, nm = P.build_panel(sm, acs, pumas, cfg.PUMA, cfg.CITY_CENTERS, PM.relative_columns())
    assert len(pan) == 24 * 4 and pan["usable"].all()
    # relative columns are centred on the (city, year) median
    assert np.allclose(pan.groupby(["city", "year"])["dlog_rel"].median(), 0)
    assert np.allclose(pan["log_income_prev_rel_sq"], pan["log_income_prev_rel"] ** 2)
    # the planted link (income growth follows bike growth) is visible
    assert pan["d_log_trips"].corr(pan["dlog"]) > 0.3
    # first-half features exist for the same rows and agree in sign with the full-year ones
    assert len(h1) == len(pan) and h1["d_log_trips_rel"].corr(pan["d_log_trips_rel"]) > 0.5
    # seasonality cancels: matched-month growth is not contaminated by the sine wave
    assert abs(pan["d_log_trips"].mean()) < 0.1
    assert set(PM.feature_sets(pan)["ALT_all"]) <= set(pan.columns)
    assert PM.CONTROLS == P.CONTROLS and PM.BIKE_ALL == P.BIKE_ALL


def test_puma_with_too_few_mature_stations_is_out_of_scope():
    sm, acs, pumas = synthetic_inputs()
    drop = sm[(sm["puma"] == "3600000") & (~sm["station_id"].str.endswith(("-0", "-1")))].index
    sm = sm.drop(drop)                                   # leaves 2 stations, below the minimum of 5
    pan, _, nm = P.build_panel(sm, acs, pumas, cfg.PUMA, cfg.CITY_CENTERS, PM.relative_columns())
    assert "3600000" not in set(pan["GEOID"])


# ---- multi-year variant ----
def test_span_matches_months_across_the_gap_years():
    sm = _sm()
    sm = sm[sm["ym"] // 100 == 2015].copy()
    later = sm.assign(ym=sm["ym"] + 300, dep=sm["dep"] * 2, dep_member=sm["dep_member"] * 2, dep_weekday=sm["dep_weekday"] * 2)   # 2018
    both = pd.concat([sm, later], ignore_index=True)
    f = P.puma_year_features(both, 2018, min_matched=1, span=3).set_index("puma").loc["P1"]
    assert np.isclose(f["d_log_trips"], np.log(2)) and f["n_matched"] == 2
    assert P.puma_year_features(both, 2018, min_matched=1, span=1)["d_log_trips"].isna().all()   # no 2017 data: nothing to match


def test_span_changes_use_a_momentum_window_that_shares_no_year_with_the_outcome():
    acs = pd.DataFrame([{"GEOID": "x", "year": y, "agg_household_income": np.exp(0.1 * (y - 2012)) * 1e9, "households": 1e4,
                         "agg_household_income_moe": 1e7, "households_moe": 100} for y in range(2012, 2020)])
    inc = P.income_table(acs)
    d = P.add_span_changes(inc, 3).set_index(["GEOID", "year"])
    row = d.loc[("x", 2018)]
    assert np.isclose(row["dlog"], 0.3) and np.isclose(row["log_income_prev"], inc.loc[inc["year"] == 2015, "log_income"].iloc[0])
    assert np.isclose(row["dlog_lag"], 0.2)                       # 2012 -> 2014: ends before the window (2015-2018) starts
    se = inc.set_index("year")["se_log_income"]
    assert np.isclose(row["se_dlog"], np.sqrt(se[2018] ** 2 + se[2015] ** 2))
    assert np.isnan(d.loc[("x", 2014), "dlog_lag"])               # not enough history
    assert np.isnan(P.add_span_changes(inc, 1).set_index(["GEOID", "year"]).loc[("x", 2018), "dlog_lag"])


def test_build_panel_multiyear_keeps_only_windows_starting_inside_the_bike_data():
    sm, acs, pumas = synthetic_inputs()
    pan3, _, _ = P.build_panel(sm, acs, pumas, cfg.PUMA, cfg.CITY_CENTERS, PM.relative_columns(), span=3)
    assert sorted(pan3["year"].unique()) == [2018, 2019] and pan3["usable"].all() and len(pan3) == 48
    assert np.allclose(pan3.groupby(["city", "year"])["dlog_rel"].median(), 0)
    assert pan3["d_log_trips"].corr(pan3["dlog"]) > 0.3
    pan2, _, _ = P.build_panel(sm, acs, pumas, cfg.PUMA, cfg.CITY_CENTERS, PM.relative_columns(), span=2)
    assert sorted(pan2["year"].unique()) == [2017, 2018, 2019]
    # a longer window carries more real change, so its noise share is smaller
    pan1, _, _ = P.build_panel(sm, acs, pumas, cfg.PUMA, cfg.CITY_CENTERS, PM.relative_columns())
    n = lambda p: P.noise_ceiling(p["dlog_rel"], p["se_dlog"])["noise_share"]
    assert n(pan3) < n(pan1)


def test_run_ladder_returns_one_row_per_baseline_with_gains():
    sm, acs, pumas = synthetic_inputs()
    pan, _, _ = P.build_panel(sm, acs, pumas, cfg.PUMA, cfg.CITY_CENTERS, PM.relative_columns(), span=2)
    summ, met = PM.run_ladder(pan, ("puma", "loco"), n_boot=30)
    assert list(summ["baseline"]) == list(PM.feature_sets(pan))
    assert summ.loc[summ["baseline"] == "ALT_usage", "gain_puma"].notna().all() and summ.loc[0, "compared_to"] == ""
    assert summ.loc[summ["baseline"] == "ALT_usage", "spearman_puma"].iloc[0] > summ.loc[summ["baseline"] == "B2", "spearman_puma"].iloc[0]


def test_cluster_corr_detects_a_real_relation():
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(30), 4)
    x = rng.normal(size=len(g))
    rho, lo, hi, n = PM.cluster_corr(x, x + rng.normal(0, 0.5, len(g)), g, 100)
    assert n == 120 and lo > 0.3 and lo < rho < hi


def test_build_panel_across_the_covid_gap_uses_only_consecutive_years():
    sm, acs, pumas = synthetic_inputs()
    acs = acs[acs["year"].between(2015, 2019)]
    # add 2021 (income only), 2022, 2023, 2024 (income and trips) as copies of 2019 with growth
    extra_acs, extra_sm = [], []
    for yr, g in ((2021, 1.05), (2022, 1.08), (2023, 1.12), (2024, 1.16)):
        e = acs[acs["year"] == 2019].copy()
        e["year"], e["agg_household_income"] = yr, e["agg_household_income"] * g
        extra_acs.append(e)
        if yr >= 2022:
            s = sm[sm["ym"] // 100 == 2019].copy()
            s["ym"] = s["ym"] + (yr - 2019) * 100
            extra_sm.append(s)
    acs, sm = pd.concat([acs] + extra_acs, ignore_index=True), pd.concat([sm] + extra_sm, ignore_index=True)
    params = {"bike_years": list(range(2015, 2020)) + [2022, 2023, 2024], "min_mature_stations": 3, "min_matched_stations": 3}
    pan, _, _ = P.build_panel(sm, acs, pumas, params, cfg.CITY_CENTERS, PM.relative_columns(), min_group=3)
    use = pan[pan["usable"]]
    assert sorted(use["year"].unique()) == [2017, 2018, 2019, 2023, 2024]           # 2016 lacks a lagged change, 2022 lacks 2021 trips
    y22 = pan[pan["year"] == 2022]
    assert y22["d_log_trips"].isna().all() and y22["dlog"].notna().all()           # income 2021->2022 exists, bike change does not
    y23 = pan[pan["year"] == 2023]
    assert np.allclose(y23["dlog"], np.log(1.12 / 1.08)) and np.allclose(y23["dlog_lag"], np.log(1.08 / 1.05))
    assert np.allclose(y23["d_log_trips"], 0, atol=1e-9)                            # 2023 trips are copies of 2022 trips
