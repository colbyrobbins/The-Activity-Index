import numpy as np
import pandas as pd

from activity_index import panel_modeling as PM
from activity_index import puma_nowcast as PN


def _panel(n_puma=12, years=range(2014, 2022), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_puma):
        base = rng.normal(11, .3)
        for k, y in enumerate(years):
            r = {"GEOID": f"p{i}", "city": "a" if i % 2 else "b", "year": y, "usable": True,
                 "log_income": base + .03 * k + rng.normal(0, .02), "log_income_prev": base + .03 * (k - 1), "dlog_lag": .03 + rng.normal(0, .01),
                 "trips_1": 1000 * np.exp(.05 * k) * (1 + i / 10), "matched_station_months": 60}
            for f in ("casual_share", "round_share", "net_am", "net_pm"):
                r[f"lvl_{f}"] = .3 + .01 * k + rng.normal(0, .01)
                r[f"d_{f}"] = rng.normal(0, .01)
            for c in PM.BIKE_ALL + PN.CENSUS:
                r.setdefault(c, rng.normal())
            rows.append(r)
    return pd.DataFrame(rows)


def test_own_history_uses_only_earlier_years():
    d = PN.add_own_history(PN.add_year_t_levels(_panel()))
    g = d[d["GEOID"] == "p0"].sort_values("year")
    lv = g["lvlT_casual_share"].values
    assert np.isnan(g["dev_casual_share"].iloc[0])
    assert np.isclose(g["dev_casual_share"].iloc[3], lv[3] - lv[:3].mean())
    # changing the LAST year must not change any earlier deviation
    p2 = _panel(); p2.loc[(p2.GEOID == "p0") & (p2.year == 2021), "lvl_casual_share"] += 5
    g2 = PN.add_own_history(PN.add_year_t_levels(p2)).query("GEOID == 'p0'").sort_values("year")
    assert np.allclose(g["dev_casual_share"].values[:-1], g2["dev_casual_share"].values[:-1], equal_nan=True)


def test_no_static_geography_and_target_never_a_feature():
    sets = PN.feature_sets()
    for cols in sets.values():
        assert not any(c.startswith(("log_density", "km_to_center", "log_households")) for c in cols)
        assert PN.TARGET_REL not in cols and "log_income_rel" not in cols
    assert set(sets["T0"]) < set(sets["T1"]) < set(sets["T2"]) < set(sets["T5"])
    assert not set(sets["T3"]) & {"log_income_prev_rel", "dlog_lag_rel"}


def test_prepare_drops_first_year_and_run_scores_forward():
    d = PN.prepare(_panel())
    assert d["year"].min() == 2015 and d[PN.TARGET_REL].notna().all()
    met, gains = PN.run(d, n_boot=20)
    f = met[met["cv"] == "forward"]
    assert set(f["set"]) == set(PN.WHAT) and (f["n"] > 0).all()
    assert {"T4 vs T1", "T5 vs T2"} <= set(gains["comparison"])


def test_noise_features_do_not_show_a_gain():
    g = []
    for seed in range(8):
        d = PN.prepare(_panel(n_puma=30, seed=seed))
        _, gains = PN.run(d, schemes=("forward",), n_boot=5)
        g.append(gains.set_index("comparison").loc["T4 vs T1", "d_change_spearman"])
    assert abs(np.mean(g)) < 0.06
