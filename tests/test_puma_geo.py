import pandas as pd

from activity_index.puma_geo import overlap_components


def test_identical_renamed_split_and_merged_pumas():
    old = pd.Series({"a": 100., "b": 100., "c": 100., "d": 100., "e": 100.})
    new = pd.Series({"A": 100., "B1": 50., "B2": 50., "CD": 200., "E": 90.})
    pieces = pd.DataFrame([("a", "A", 100), ("b", "B1", 50), ("b", "B2", 50), ("c", "CD", 100), ("d", "CD", 100), ("e", "E", 90),
                           ("e", "A", 1)], columns=["old", "new", "area"])           # 'e' also leaks a sliver into A
    r = overlap_components(pieces, old, new).set_index("old")
    assert r.loc["a", "iou"] > 0.99 and r.loc["a", "n_old"] == 1 and r.loc["a", "n_new"] == 1      # renamed, same ground
    assert r.loc["b", "n_new"] == 2 and r.loc["b", "iou"] > 0.99                                   # split in two: follow the union
    assert r.loc["c", "component"] == r.loc["d", "component"] and r.loc["c", "n_old"] == 2          # merged: follow the pair
    assert r.loc["e", "component"] != r.loc["a", "component"]                                      # a sliver does not chain areas
    assert 0.85 < r.loc["e", "iou"] < 0.95                                                         # shrank by 10%: not clean


def test_unmatched_old_puma_has_its_own_component():
    r = overlap_components(pd.DataFrame(columns=["old", "new", "area"]), pd.Series({"x": 1.}), pd.Series({"y": 1.})).set_index("old")
    assert r.loc["x", "n_new"] == 0 or pd.isna(r.loc["x", "iou"]) or r.loc["x", "iou"] == 0


def test_area_members_lists_both_vintages():
    from activity_index.puma_geo import area_members
    old = pd.Series({"a": 100., "b": 100.}); new = pd.Series({"X": 200.})
    pieces = pd.DataFrame([("a", "X", 100), ("b", "X", 100)], columns=["old", "new", "area"])
    m = area_members(pieces, old, new)
    assert set(m["line"]) == {"2010", "2020"} and m["component"].nunique() == 1 and len(m) == 3


def test_area_acs_sums_the_right_vintage_and_drops_incomplete_years():
    import numpy as np
    from activity_index.puma_geo import area_acs
    members = pd.DataFrame({"area_id": ["A1", "A1", "A1"], "line": ["2010", "2010", "2020"], "GEOID": ["a", "b", "X"], "city": "nyc"})
    base = dict(renter_households=0, population=0, edu_total=0, edu_bachelors=0, edu_masters=0, edu_professional=0, edu_doctorate=0)
    rows = [dict(GEOID="a", year=2019, agg_household_income=100., households=10, agg_household_income_moe=3., households_moe=1.,
                 median_gross_rent=1000., median_home_value=1., median_age=30., **base),
            dict(GEOID="b", year=2019, agg_household_income=300., households=30, agg_household_income_moe=4., households_moe=2.,
                 median_gross_rent=2000., median_home_value=1., median_age=40., **base),
            dict(GEOID="a", year=2018, agg_household_income=1., households=1, agg_household_income_moe=1., households_moe=1.,
                 median_gross_rent=1., median_home_value=1., median_age=1., **base),             # b missing in 2018 -> incomplete
            dict(GEOID="X", year=2023, agg_household_income=500., households=50, agg_household_income_moe=5., households_moe=1.,
                 median_gross_rent=1500., median_home_value=1., median_age=35., **base)]
    out = area_acs(pd.DataFrame(rows), members).set_index("year")
    assert set(out.index) == {2019, 2023}
    assert out.loc[2019, "agg_household_income"] == 400 and out.loc[2019, "households"] == 40
    assert np.isclose(out.loc[2019, "agg_household_income_moe"], 5.0)                      # sqrt(3^2 + 4^2)
    assert np.isclose(out.loc[2019, "median_gross_rent"], (1000 * 10 + 2000 * 30) / 40)     # household-weighted
    assert out.loc[2023, "agg_household_income"] == 500 and out.loc[2023, "city"] == "nyc"
