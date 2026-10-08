"""Offline tests for duplicate-file handling, unmatched-station reporting, and the Census client."""
import zipfile

import pandas as pd

from activity_index import census
from activity_index import trips as T

HEADER = ("tripduration,starttime,stoptime,start station id,start station name,start station latitude,"
          "start station longitude,end station id,end station name,end station latitude,"
          "end station longitude,bikeid,usertype,birth year,gender\n")
ROW_A = '600,2018-01-01 00:00:05,2018-01-01 00:10:05,72,W 52 St,40.76,-73.99,79,Franklin St,40.71,-74.00,1,Subscriber,1985,1\n'
ROW_B = '300,2018-01-01 01:00:00,2018-01-01 01:05:00,79,Franklin St,40.71,-74.00,72,W 52 St,40.76,-73.99,2,Customer,,0\n'
ROW_C = '200,2018-01-01 02:00:00,2018-01-01 02:05:00,999,Retired Stn,,,72,W 52 St,40.76,-73.99,3,Customer,,0\n'


def _zip(path, files):
    with zipfile.ZipFile(path, "w") as z:
        for name, text in files.items():
            z.writestr(name, text)
    return path


def _citibike_like(path):
    """Mirrors the real 2018 layout: monthly CSV + split parts + same parts again in a month folder."""
    full = HEADER + ROW_A + ROW_B
    part1, part2 = HEADER + ROW_A, HEADER + ROW_B
    return _zip(path, {
        "2018-citibike-tripdata/201801-citibike-tripdata.csv": full,
        "2018-citibike-tripdata/201801-citibike-tripdata_1.csv": part1,
        "2018-citibike-tripdata/201801-citibike-tripdata_2.csv": part2,
        "2018-citibike-tripdata/1_January/201801-citibike-tripdata_1.csv": part1,
        "2018-citibike-tripdata/1_January/201801-citibike-tripdata_2.csv": part2,
        "2018-citibike-tripdata/.DS_Store": "",
        "__MACOSX/2018-citibike-tripdata/._201801-citibike-tripdata.csv": "junk",
    })


def test_select_csvs_prefers_full_monthly_file():
    names = [
        "y/201801-x.csv", "y/201801-x_1.csv", "y/201801-x_2.csv",
        "y/1_January/201801-x_1.csv", "y/1_January/201801-x_2.csv",
        "y/201802-x.csv",
    ]
    assert T.select_csvs(names) == ["y/201801-x.csv", "y/201802-x.csv"]


def test_select_csvs_keeps_all_parts_when_no_full_file():
    names = ["y/201805-x_1.csv", "y/201805-x_2.csv", "y/5_May/201805-x_1.csv", "y/5_May/201805-x_2.csv"]
    assert T.select_csvs(names) == ["y/201805-x_1.csv", "y/201805-x_2.csv"]


def test_select_csvs_leaves_unrelated_files_alone():
    names = ["Divvy_Trips_2015-Q1.csv", "Divvy_Trips_2015-Q2.csv"]
    assert T.select_csvs(names) == names


def test_duplicate_months_are_not_double_counted(tmp_path):
    z = _citibike_like(tmp_path / "2018-citibike-tripdata.zip")
    trips, _ = T.read_trip_zip(z)
    assert len(trips) == 2                      # not 6 (full + parts + month-folder parts)
    res = T.analyze_zip(z)
    assert res["rows"] == 2
    assert res["csvs_skipped_as_duplicates"] == 4 and res["csvs_in_zip"] == 5


def test_sampling_limits_files_and_rows(tmp_path):
    z = _zip(tmp_path / "s.zip", {
        "201801-x.csv": HEADER + ROW_A * 5, "201802-x.csv": HEADER + ROW_B * 5, "201803-x.csv": HEADER + ROW_A * 5})
    res = T.analyze_zip(z, nrows=3, max_files=2)
    assert res["rows"] == 6 and res["csvs_read"] == 2 and res["sampled"] is True


def test_top_unmatched_lists_busiest_missing_stations(tmp_path):
    z = _zip(tmp_path / "u.zip", {"201801-x.csv": HEADER + ROW_A + ROW_C * 3})
    res = T.analyze_zip(z)
    assert res["coords_final"] == 0.25
    assert res["top_unmatched_stations"][0] == {"id": "999", "name": "Retired Stn", "trips": 3}


def test_acs_url_uses_percent20_like_census_docs():
    url = census.acs_url(2018, "36", ["061", "047"], {"B19025_001E": "agg"}, key="K")
    assert "in=state:36%20county:061,047" in url
    assert "for=tract:*" in url and "get=NAME,B19025_001E" in url and url.endswith("key=K")


def test_fetch_acs_explains_non_json_response():
    class FakeResp:
        status_code, url, text = 200, "https://x", "<html>Service unavailable</html>"
        headers = {"content-type": "text/html"}

        def json(self):
            raise ValueError("no json")

    real = census.requests.get
    census.requests.get = lambda *a, **k: FakeResp()
    try:
        census.fetch_acs(2018, "36", ["061"], {"B19025_001E": "agg"})
        raise AssertionError("should have raised")
    except RuntimeError as e:
        msg = str(e)
    finally:
        census.requests.get = real
    assert "HTTP 200" in msg and "text/html" in msg and "Service unavailable" in msg


def test_fetch_acs_explains_missing_key():
    class FakeResp:
        status_code, url, headers = 200, "https://api.census.gov/data/missing_key.html", {"content-type": "text/html"}
        text = "<html><head><title>Missing Key</title></head></html>"

        def json(self):
            raise ValueError("no json")

    import os
    saved = os.environ.pop("CENSUS_API_KEY", None)
    real = census.requests.get
    census.requests.get = lambda *a, **k: FakeResp()
    try:
        census.fetch_acs(2018, "36", ["061"], {"B19025_001E": "agg"})
        raise AssertionError("should have raised")
    except RuntimeError as e:
        msg = str(e)
    finally:
        census.requests.get = real
        if saved is not None:
            os.environ["CENSUS_API_KEY"] = saved
    assert "CENSUS_API_KEY" in msg and "key_signup" in msg


def test_fetch_acs_parses_good_response_and_sentinels():
    class FakeResp:
        status_code, url, text, headers = 200, "https://x", "", {}

        def json(self):
            return [["NAME", "B19025_001E", "state", "county", "tract"],
                    ["Tract 1", "1000000", "36", "061", "000100"],
                    ["Tract 2", "-666666666", "36", "061", "000200"]]

    real = census.requests.get
    census.requests.get = lambda *a, **k: FakeResp()
    try:
        df = census.fetch_acs(2018, "36", ["061"], {"B19025_001E": "agg_household_income"})
    finally:
        census.requests.get = real
    assert df["GEOID"].tolist() == ["36061000100", "36061000200"]
    assert df["agg_household_income"].iloc[0] == 1_000_000 and pd.isna(df["agg_household_income"].iloc[1])


def test_station_history_pools_files_across_zips_and_fills_unbundled_zip(tmp_path):
    stn_a = "id,name,latitude,longitude\n35,Streeter Dr & Grand Ave,41.892,-87.612\n"
    stn_b = "id,name,latitude,longitude\n35,Streeter Dr & Grand Ave,41.8923,-87.6117\n76,Lake Shore Dr & Monroe St,41.881,-87.617\n"
    hdr = "trip_id,starttime,stoptime,bikeid,tripduration,from_station_id,from_station_name,to_station_id,to_station_name,usertype\n"
    row = "1,7/1/2015 00:01,7/1/2015 00:10,5,540,{sid},{nm},76,Lake Shore Dr & Monroe St,Subscriber\n"
    z1 = _zip(tmp_path / "Divvy_2015_Q1Q2.zip", {"Divvy_Stations_2015.csv": stn_a})
    z2 = _zip(tmp_path / "Divvy_2016_Q1Q2.zip", {"Divvy_Stations_2016.csv": stn_b})
    z3 = _zip(tmp_path / "Divvy_2015_Q3Q4.zip",
              {"Divvy_Trips_2015_07.csv": hdr + row.format(sid=35, nm="Streeter Dr & Illinois St") * 2
               + row.format(sid=76, nm="Lake Shore Dr & Monroe St")})
    hist = T.load_station_history([z1, z2, z3])
    assert len(hist) == 2 and hist.set_index("station_id").loc["35", "lat"] == 41.8923   # later file wins
    res = T.analyze_zip(z3, history=hist)
    assert res["coords_in_file"] == 0 and res["coords_after_history"] == 1.0   # matched by id despite rename
