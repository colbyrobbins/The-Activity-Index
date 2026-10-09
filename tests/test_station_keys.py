import pandas as pd

from activity_index.station_keys import canonical_keys


def _st(rows):
    return pd.DataFrame(rows, columns=["city", "station_id", "lat", "lon", "first_ym", "last_ym"])


def test_new_id_at_the_same_spot_inherits_the_old_station():
    k = canonical_keys(_st([("b", "116", 42.3500, -71.0600, 201501, 201912), ("b", "A32000", 42.35003, -71.06002, 202201, 202412)]))
    assert k["key"].nunique() == 1 and k["linked"].all()


def test_stations_active_together_are_never_merged():
    k = canonical_keys(_st([("b", "1", 42.35, -71.06, 201501, 202412), ("b", "2", 42.35003, -71.06002, 201801, 202412)]))
    assert k["key"].nunique() == 2 and not k["linked"].any()


def test_two_new_stations_cannot_both_inherit_one_old_station():
    k = canonical_keys(_st([("b", "old", 42.35000, -71.0600, 201501, 201912),
                            ("b", "new_near", 42.35002, -71.0600, 202201, 202412),
                            ("b", "new_far", 42.35040, -71.0600, 202201, 202412)])).set_index("station_id")
    assert k.loc["old", "key"] == k.loc["new_near", "key"] != k.loc["new_far", "key"]


def test_far_apart_stations_and_other_cities_stay_separate():
    k = canonical_keys(_st([("b", "1", 42.35, -71.06, 201501, 201912), ("b", "2", 42.36, -71.06, 202201, 202412),
                            ("c", "1", 42.35, -71.06, 202201, 202412)]))
    assert k["key"].nunique() == 3
