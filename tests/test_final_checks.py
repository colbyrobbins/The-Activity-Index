"""Tests for the final-phase checks: placebo (robustness), weather adjustment, trip purpose and the decision screen."""
import numpy as np
import pandas as pd

from activity_index import area_nowcast as AN
from activity_index import decision as D
from activity_index import robustness as R
from activity_index import trip_purpose as TP
from activity_index import weather as W
from test_area_nowcast import YEARS, _panel


def test_shuffle_keeps_each_groups_values():
    d = AN.prepare(_panel(), YEARS)
    s = R.shuffled(d, R.FEATURE, ["city", "year"], np.random.default_rng(0))
    for _, idx in d.groupby(["city", "year"]).groups.items():
        assert np.allclose(sorted(d.loc[idx, R.FEATURE]), sorted(s.loc[idx, R.FEATURE]))
    assert not np.allclose(d[R.FEATURE], s[R.FEATURE])


def test_placebo_separates_a_planted_signal_from_noise():
    def p_value(signal):
        d = AN.prepare(_panel(signal=signal, n_areas=12), YEARS).reset_index(drop=True)
        sets = AN.feature_sets()
        real = R.gain(d, sets["B2g"], sets["B1"])
        return R.placebo_row("x", real, R.placebo(d, ["city", "year"], n=60, seed=1))["p_value"]
    assert p_value(1.0) < 0.05 < p_value(0.0)


def test_weather_adjustment_removes_a_planted_weather_effect():
    days = pd.date_range("2015-01-01", "2017-12-31")
    rng = np.random.default_rng(0)
    daily = pd.DataFrame({"city": "a", "date": days.strftime("%Y-%m-%d"), "temp_c": rng.normal(10, 5, len(days)),
                          "precip_mm": np.where(rng.random(len(days)) < 0.3, 5.0, 0.0), "snow_cm": 0.0})
    wx = W.weather_changes(W.monthly_weather(daily))
    assert set(wx["year"]) == {2016, 2017} and len(wx) == 24
    rows = []
    for geo, beta in (("a1", 0.05), ("a2", 0.0)):                      # a1 reacts to temperature, a2 does not; no true growth in either
        for r in wx.itertuples():
            rows.append({"GEOID": geo, "city": "a", "year": r.year, "m": r.m, "trips_0": 1000.0,
                         "trips_1": 1000.0 * np.exp(beta * r.d_temp), **{c: getattr(r, c) for c in W.WX}})
    mg = pd.DataFrame(rows)
    mg["g"] = np.log(mg["trips_1"] / mg["trips_0"])
    sens = W.sensitivities(mg).set_index("GEOID")
    assert abs(sens.loc["a1", "b_d_temp"] - 0.05) < 1e-6 and abs(sens.loc["a2", "b_d_temp"]) < 1e-6
    adj = W.adjusted_growth(mg, sens.reset_index())
    assert np.allclose(adj["g_wx_adj"], 0, atol=1e-6)


def test_trip_purpose_split_counts_home_and_work_ends():
    def row(ym, dep_am, arr_am):
        return {"puma": "p", "station_id": "s", "ym": ym, "dep": 100.0, "dep_member": 80.0, "dep_weekday": 70.0, "dep_am": dep_am,
                "dep_pm": 10.0, "dep_round": 1.0, "dur_sum": 900.0, "arr": 100.0, "arr_am": arr_am, "arr_pm": 10.0}
    sm = pd.DataFrame([row(201606, 10.0, 10.0), row(201706, 20.0, 10.0)])       # morning departures double, morning arrivals flat
    g = TP.purpose_growth(sm, [2017], min_matched=1).iloc[0]
    assert np.isclose(g["g_home_end"], np.log(30 / 20)) and np.isclose(g["g_work_end"], 0) and np.isclose(g["home_share"], 0.5)


def test_decision_metrics_reward_a_perfect_ranking_and_punish_a_reversed_one():
    d = AN.prepare(_panel(), YEARS).reset_index(drop=True)
    good = D.metrics(D.scored(d.assign(x=d["dlog_rel"]), None, "good", direct="x"))
    bad = D.metrics(D.scored(d.assign(x=-d["dlog_rel"]), None, "bad", direct="x"))
    assert good["direction_accuracy"] == 1 and good["rank_corr_within_city_year"] > 0.99 and good["top_pick_in_top_half"] == 1
    assert bad["direction_accuracy"] == 0 and bad["top_third_minus_bottom_third"] < 0 < good["top_third_minus_bottom_third"]
