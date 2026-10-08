import numpy as np
import pandas as pd

from activity_index import features as F
from activity_index.trips import CANONICAL


def _trips(rows):
    df = pd.DataFrame(rows, columns=["start_time", "end_time", "s", "e", "user"])
    out = pd.DataFrame({c: np.nan for c in CANONICAL}, index=df.index)
    out["start_time"] = pd.to_datetime(df["start_time"]); out["end_time"] = pd.to_datetime(df["end_time"])
    out["start_station_id"], out["end_station_id"] = df["s"].astype("string"), df["e"].astype("string")
    out["start_station_name"] = ("Stn " + df["s"]).astype("string")
    out["user_type"] = df["user"].astype("string")
    out["start_lat"] = 40.0; out["start_lon"] = -74.0
    return out


ROWS = [
    ("2015-06-01 08:00", "2015-06-01 08:10", "A", "B", "Subscriber"),   # Monday, AM peak, member
    ("2015-06-01 17:00", "2015-06-01 17:20", "B", "A", "Customer"),     # PM peak, casual
    ("2015-06-06 12:00", "2015-06-06 12:30", "A", "A", "Customer"),     # Saturday round trip
    ("2015-06-02 08:00", "2015-06-02 08:00", "A", "B", "Subscriber"),   # zero length: dropped
    ("2014-12-31 08:00", "2014-12-31 08:10", "A", "B", "Subscriber"),   # outside activity years: dropped
    ("2015-06-03 08:00", "2015-06-03 15:00", "A", "B", "Subscriber"),   # 7 hours: dropped
]


def test_aggregate_counts_filters_and_flags():
    r = F.aggregate_trips(_trips(ROWS), years=[2015])
    assert r["qc"]["rows"] == 6 and r["qc"]["rows_valid"] == 3
    sm = r["station_month"].set_index(["ym", "station_id"])
    a = sm.loc[(201506, "A")]
    assert a["dep"] == 2 and a["dep_member"] == 1 and a["dep_weekday"] == 1 and a["dep_am"] == 1
    assert a["dep_round"] == 1 and a["dur_sum"] == 40
    assert a["arr"] == 2 and a["arr_pm"] == 1            # arrivals at A: from B (17:20) and the round trip
    b = sm.loc[(201506, "B")]
    assert b["dep"] == 1 and b["arr"] == 1 and b["arr_am"] == 1
    assert r["od_year"].set_index(["start_id", "end_id"]).loc[("A", "B"), "n"] == 1


def test_combine_parts_sums_split_months():
    p = F.aggregate_trips(_trips(ROWS[:1]), [2015]); q = F.aggregate_trips(_trips(ROWS[1:3]), [2015])
    c = F.combine_parts([p, q])
    assert c["station_month"].set_index("station_id").loc["A", "dep"].sum() == 2
    assert c["qc"]["rows"].sum() == 3


def test_station_and_tract_features():
    # two years of data for A and B (mature); C appears only in the last year (not mature)
    rows = []
    for y, n in ((2015, 2), (2018, 4)):
        for i in range(n):
            rows.append((f"{y}-06-0{i + 1} 08:00", f"{y}-06-0{i + 1} 08:10", "A", "B", "Subscriber"))
        rows.append((f"{y}-06-09 17:00", f"{y}-06-09 17:10", "B", "A", "Customer"))
    rows.append(("2018-06-10 08:00", "2018-06-10 08:10", "C", "A", "Customer"))
    r = F.aggregate_trips(_trips(rows), years=range(2015, 2019))
    coords = pd.DataFrame({"station_id": pd.array(["A", "B", "C"], dtype="string"),
                           "lat": [40.0, 40.01, 40.02], "lon": [-74.0, -74.0, -74.0]})
    sf = F.station_features(r["station_month"], r["od_year"], coords, range(2015, 2019)).set_index("station_id")
    assert bool(sf.loc["A", "mature"]) and bool(sf.loc["B", "mature"]) and not bool(sf.loc["C", "mature"])
    assert sf.loc["A", "rate_first"] == 2 and sf.loc["A", "rate_last"] == 4
    assert np.isclose(sf.loc["A", "log_growth"], np.log(2))
    assert np.isclose(sf.loc["A", "mean_km"], 1.11, atol=0.02)          # 0.01 degrees of latitude
    sf = sf.reset_index(); sf["tract"] = "T1"
    t = F.tract_features(sf).iloc[0]
    assert t["n_stations"] == 3 and t["n_mature"] == 2
    assert np.isclose(t["log_growth"], np.log((4 + 1) / (2 + 1)))       # A and B pooled
    assert 0 <= t["member_share"] <= 1 and np.isclose(t["casual_share"], 1 - t["member_share"])


def test_normalize_handles_divvy_2017_and_2018q1_headers():
    from activity_index.trips import normalize_trips
    d17 = pd.DataFrame({"trip_id": [1], "start_time": ["3/31/2017 23:59:07"], "end_time": ["4/1/2017 00:13:24"],
                        "from_station_id": [66], "from_station_name": ["A"], "to_station_id": [171],
                        "to_station_name": ["B"], "usertype": ["Subscriber"]})
    n = normalize_trips(d17)
    assert n["end_time"].notna().all() and n["end_station_id"].iloc[0] == "171"
    d18 = pd.DataFrame({"01 - Rental Details Rental ID": [1],
                        "01 - Rental Details Local Start Time": ["2018-01-01 00:12:00"],
                        "01 - Rental Details Local End Time": ["2018-01-01 00:17:23"],
                        "03 - Rental Start Station ID": [69], "03 - Rental Start Station Name": ["Damen"],
                        "02 - Rental End Station ID": [159], "02 - Rental End Station Name": ["Claremont"],
                        "User Type": ["Subscriber"]})
    n = normalize_trips(d18)
    assert n["start_time"].notna().all() and n["end_time"].notna().all()
    assert n["start_station_id"].iloc[0] == "69" and n["end_station_name"].iloc[0] == "Claremont"
