import numpy as np
import pandas as pd

from activity_index import modeling as M


def _df(n=60):
    rng = np.random.default_rng(0)
    d = pd.DataFrame({"city": np.where(np.arange(n) % 2 == 0, "a", "b"), "income_rank_t0": rng.random(n),
                      "momentum": rng.normal(0, .1, n)})
    for c in M.CONTROLS + sum(M.GROUPS.values(), []):
        d[c] = rng.normal(size=n)
    d["income_rank_t0_sq"] = d["income_rank_t0"] ** 2
    return d


def test_noise_weights_downweight_noisy_tracts_and_average_one():
    se = pd.Series([0.02, 0.05, 0.30])
    obs = pd.Series(np.random.default_rng(1).normal(0, 0.12, 500))
    w = M.noise_weights(se, obs)
    assert np.isclose(w.mean(), 1.0) and w.iloc[0] > w.iloc[1] > w.iloc[2] > 0


def test_feature_sets_nest_and_groups_are_disjoint():
    d = _df()
    s = M.feature_sets(d)
    assert set(s["B0"]) < set(s["B1"]) < set(s["B2"])
    allg = [c for cols in M.GROUPS.values() for c in cols]
    assert len(allg) == len(set(allg)) == 10
    assert set(s["ALT_all"]) == set(s["B2"]) | set(allg)
    assert all(set(s["B2"]) < set(s[f"ALT_{g}"]) for g in M.GROUPS)


def test_city_terms_use_first_city_as_reference():
    d, cols = M.add_city_terms(_df())
    assert cols == ["city_b", "city_b_x_rank", "city_b_x_rank2"]
    assert (d.loc[d["city"] == "a", cols] == 0).all().all()
    assert np.allclose(d.loc[d["city"] == "b", "city_b_x_rank"], d.loc[d["city"] == "b", "income_rank_t0"])


def test_weighted_r2_reduces_to_plain_r2_with_equal_weights():
    y, p = np.array([1.0, 2, 3, 4]), np.array([1.1, 1.9, 3.2, 3.8])
    assert np.isclose(M.r2(y, p), M.r2(y, p, np.ones(4)))
    assert M.r2(y, y) == 1.0


def test_city_relative_makes_cities_comparable_and_uses_mature_tracts_only():
    d = _df()
    d["n_mature"] = np.where(np.arange(len(d)) % 3 == 0, 0, 2)
    d["trips_per_station_month"] = np.where(d["city"] == "a", 1000.0, 10.0) * (1 + d["income_rank_t0"])  # scale differs by city
    out = M.city_relative(d)
    for c in ("a", "b"):
        m = out["city"] == c
        assert np.isclose(out.loc[m, "edu_share_ba_cr"].max(), 1.0)
        ref = m & (out["n_mature"] >= 1)
        assert np.isclose(out.loc[ref, "trips_per_station_month_cr"].max(), 1.0)
        assert out.loc[m & (out["n_mature"] == 0), "trips_per_station_month_cr"].isna().all()
    # same within-city ordering gives the same percentile range despite a 100x scale gap
    assert np.isclose(out.loc[out["city"] == "a", "trips_per_station_month_cr"].min(),
                      out.loc[out["city"] == "b", "trips_per_station_month_cr"].min(), atol=0.05)


def test_feature_sets_relative_swaps_in_cr_columns_but_keeps_rank_terms():
    d = M.city_relative(_df().assign(n_mature=2))
    s = M.feature_sets(d, relative=True)
    assert all(c.endswith("_cr") for c in s["ALT_all"] if c not in ("income_rank_t0", "income_rank_t0_sq", "momentum"))
    assert "income_rank_t0" in s["B0"] and "momentum" in s["B2"]
    assert set(s["B0"]) < set(s["B1"]) < set(s["B2"]) and not set(s["ALT_all"]) & set(M.BIKE_ALL)
    assert set(s["ALT_all"]) <= set(d.columns)
