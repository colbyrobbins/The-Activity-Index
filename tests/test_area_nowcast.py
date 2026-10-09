import numpy as np
import pandas as pd

from activity_index import area_nowcast as AN
from activity_index import panel_modeling as PM

YEARS = [2016, 2017, 2018, 2019, 2023, 2024]


def _panel(signal=0.0, n_areas=10, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for city in ("a", "b"):
        shock = {y: rng.normal(0.04, 0.03) for y in YEARS}
        for i in range(n_areas):
            lvl = rng.normal(11.3, 0.3)
            prev_d = 0.04
            for y in YEARS:
                r = {"GEOID": f"{city}{i}", "city": city, "year": y, "usable": True, "households": 40000}
                for f in ("casual_share", "round_share", "net_am", "net_pm"):
                    r[f"lvl_{f}"], r[f"d_{f}"] = 0.3 + rng.normal(0, .02), rng.normal(0, .02)
                for c in PM.BIKE_ALL:
                    r.setdefault(c, rng.normal())
                r["trips_1"], r["matched_station_months"] = 1000 * (1 + i / 5), 60
                d = shock[y] + signal * r["d_log_trips"] * 0.05 + rng.normal(0, 0.02)
                r.update(log_income_prev=lvl, dlog=d, dlog_lag=prev_d, log_income=lvl + d, mean_income=float(np.exp(lvl + d)),
                         se_log_income=0.03)
                lvl, prev_d = lvl + d, d
                rows.append(r)
    return pd.DataFrame(rows)


def test_b0_is_last_posting_times_city_trend():
    d = AN.prepare(_panel(), YEARS)
    p = AN.forward_predictions(d, [])
    last = d["year"] == d["year"].max()
    trend = AN.city_trend(d[d["year"] < d["year"].max()])
    expect = d.loc[last, "city"].map(trend).values
    assert np.allclose(p[last.values], expect) and np.isnan(p[(d["year"] == d["year"].min()).values]).all()


def test_noise_floor_and_metrics():
    d = AN.prepare(_panel(), YEARS)
    p = AN.forward_predictions(d, [])
    s = AN.score(d, p)
    ok = ~np.isnan(p)
    assert np.isclose(s["noise_floor_mae"], 0.7979 * (d.loc[ok, "se_log_income"] * d.loc[ok, "mean_income"]).mean())
    assert s["mae_shift_free"] <= s["mae_dollars"] + 1e-9 and s["n"] == ok.sum()


def test_bikeshare_helps_when_it_carries_signal_and_not_when_it_does_not():
    gains_sig, gains_null = [], []
    for seed in range(4):
        for signal, store in ((6.0, gains_sig), (0.0, gains_null)):
            d = AN.prepare(_panel(signal=signal, n_areas=12, seed=seed), YEARS)
            _, g = AN.run(d, n_boot=20)
            store.append(g[(g["sample"] == "all_test_years") & (g["comparison"] == "F_all vs B1")]["d_mae_dollars"].iloc[0])
    assert np.mean(gains_sig) < np.mean(gains_null) and np.mean(gains_sig) < 0


def test_two_feature_rung_is_a_small_baseline():
    assert set(AN.TWO) <= set(AN.BIKE_ALL)
    sets = AN.feature_sets()
    assert sets["B2"] == ["dlog_lag_rel", "d_log_trips_rel", "dev_log_trips_rel"]
    d = AN.prepare(_panel(), YEARS)
    _, g = AN.run(d, n_boot=10)
    assert {"B2 vs B1", "F_all vs B2"} <= set(g["comparison"])


def test_single_feature_rungs_and_comparisons():
    sets = AN.feature_sets()
    assert sets["B2g"] == ["dlog_lag_rel", "d_log_trips_rel"] and sets["B2l"] == ["dlog_lag_rel", "dev_log_trips_rel"]
    d = AN.prepare(_panel(), YEARS)
    _, g = AN.run(d, n_boot=10)
    assert {"B2g vs B1", "B2l vs B1"} <= set(g["comparison"])
