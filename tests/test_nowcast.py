import numpy as np
import pandas as pd

from activity_index import modeling as M
from activity_index import nowcast as N


def _prior():
    n = 20
    rng = np.random.default_rng(0)
    rows = []
    for city in ("a", "b"):
        rows.append(pd.DataFrame({
            "GEOID": [f"{city}{i}" for i in range(n)], "city": city,
            "edu_bachelors": rng.integers(10, 100, n), "edu_masters": 5, "edu_professional": 1, "edu_doctorate": 1,
            "edu_total": 300, "renter_households": rng.integers(10, 200, n), "households": 400,
            "median_gross_rent": np.arange(n) * 10 + 500, "median_home_value": np.arange(n) * 1e4 + 1e5,
            "median_age": rng.uniform(25, 50, n)}))
    return pd.concat(rows, ignore_index=True)


def _geo(prior):
    return pd.DataFrame({"GEOID": prior["GEOID"], "log_density": np.arange(len(prior)) * 0.1, "km_to_center": np.arange(len(prior)) * 0.5})


def test_prior_controls_are_within_city_percentiles():
    p = _prior()
    pc = N.prior_controls(p, _geo(p)).set_index("GEOID")
    assert set(pc.columns) == {c + "_pcr" for c in M.CONTROLS}
    assert np.isclose(pc.loc["a19", "median_gross_rent_pcr"], 1.0) and np.isclose(pc.loc["b19", "median_gross_rent_pcr"], 1.0)
    assert np.isclose(pc.loc["b0", "median_gross_rent_pcr"], 1 / 20)


def test_city_terms_use_prior_rank_never_the_target():
    df = pd.DataFrame({"city": ["a", "a", "b", "b"], "income_rank_prior": [.1, .9, .2, .8], "income_rank_t0": [.3, .7, .6, .4]})
    out, dummies, rank_terms = N.city_terms(df)
    assert dummies == ["city_b"] and rank_terms == ["city_b_x_prior", "city_b_x_prior2"]
    assert np.allclose(out["city_b_x_prior"], [0, 0, .2, .8])
    assert np.allclose(out["city_b_x_prior2"], [0, 0, .04, .64])


def test_feature_sets_nest_and_keep_the_target_out():
    s = N.feature_sets(["city_b"], ["city_b_x_prior"])
    assert set(s["N0"]) < set(s["N1"]) < set(s["N5"])
    assert set(s["N4"]) == set(s["N3"]) | {c + "_cr" for c in M.BIKE_ALL}
    assert not any("income_rank_prior" in c for c in s["N2"] + s["N3"] + s["N4"])
    for cols in s.values():
        assert "income_rank_t0" not in cols and "rank_change" not in cols
    assert set(s) >= {f"N5_{g}" for g in M.GROUPS}


def test_score_perfect_and_mae_in_percentile_points():
    y = np.array([.1, .5, .9])
    s = N.score(y, y)
    assert np.isclose(s["spearman"], 1) and np.isclose(s["r2"], 1) and np.isclose(s["mae_pct_points"], 0)
    assert np.isclose(N.score(y, y + .1)["mae_pct_points"], 10)
