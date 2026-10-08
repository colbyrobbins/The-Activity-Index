"""Offline tests for trip parsing, using small mock files in each system's expected format.

The real files are not available here, so these encode our *assumptions* about the formats.
If real files disagree with one of them, fix the code and update the mock.
"""
import importlib
import os
import zipfile

import numpy as np
import pandas as pd

from activity_index import config as cfg
from activity_index import trips as T


def _zip(path, files):
    with zipfile.ZipFile(path, "w") as z:
        for name, text in files.items():
            z.writestr(name, text)
    return path


NYC = (
    "tripduration,starttime,stoptime,start station id,start station name,start station latitude,"
    "start station longitude,end station id,end station name,end station latitude,"
    "end station longitude,bikeid,usertype,birth year,gender\n"
    '600,2018-07-01 00:00:05,2018-07-01 00:10:05,72,W 52 St & 11 Ave,40.7673,-73.9939,'
    '79,Franklin St & W Broadway,40.7191,-74.0067,100,Subscriber,1985,1\n'
    '300,7/1/2018 01:00:00,7/1/2018 01:05:00,79,Franklin St & W Broadway,40.7191,-74.0067,'
    '72,W 52 St & 11 Ave,40.7673,-73.9939,101,Customer,,0\n'
)

DIVVY_TRIPS = (
    "trip_id,starttime,stoptime,bikeid,tripduration,from_station_id,from_station_name,"
    "to_station_id,to_station_name,usertype,gender,birthyear\n"
    "1,2018-07-01 00:01:00,2018-07-01 00:11:00,5,600,91,Clinton St & Washington Blvd,"
    "77,Clinton St & Madison St,Subscriber,Male,1990\n"
    "2,2018-07-01 00:02:00,2018-07-01 00:12:00,6,600,77,Clinton St & Madison St,"
    "91,Clinton St & Washington Blvd,Customer,,\n"
)
DIVVY_STATIONS = (
    "id,name,latitude,longitude,dpcapacity,online_date\n"
    "91,Clinton St & Washington Blvd,41.8833,-87.6412,23,6/28/2013\n"
    "77,Clinton St & Madison St,41.8821,-87.6411,19,6/28/2013\n"
)

CABI = (
    "Duration,Start date,End date,Start station number,Start station,End station number,"
    "End station,Bike number,Member type\n"
    "300,2018-07-01 00:00:00,2018-07-01 00:05:00,31200,Massachusetts Ave & Dupont Circle NW,"
    "31201,15th & P St NW,W1,Member\n"
    "400,2018-07-01 00:01:00,2018-07-01 00:08:00,31201,15th & P St NW,"
    "31200,Massachusetts Ave & Dupont Circle NW,W2,Casual\n"
)

BLUEBIKES = (
    "tripduration,starttime,stoptime,start station id,start station name,end station id,"
    "end station name,bikeid,usertype\n"
    "500,2018-07-01 00:00:00,2018-07-01 00:08:20,22,South Station - 700 Atlantic Ave,"
    "36,Boylston St at Fairfield St,B1,Subscriber\n"
)


def test_nyc_old_format_has_coords_and_mixed_times():
    df = T.normalize_trips(pd.read_csv(pd.io.common.StringIO(NYC)))
    assert list(df.columns) == T.CANONICAL
    assert df["start_lat"].notna().all() and df["start_lon"].notna().all()
    assert df["start_time"].notna().all()          # ISO and m/d/Y both parsed
    assert df["start_station_id"].tolist() == ["72", "79"]
    assert T.coord_coverage(df) == 1.0


def test_divvy_uses_bundled_station_file(tmp_path):
    z = _zip(tmp_path / "Divvy_Trips_2018_Q3.zip",
             {"Divvy_Trips_2018_Q3.csv": DIVVY_TRIPS, "Divvy_Stations_2018.csv": DIVVY_STATIONS})
    res = T.analyze_zip(z)
    assert res["rows"] == 2
    assert res["coords_in_file"] == 0.0
    assert res["coords_after_bundled_stations"] == 1.0
    assert res["time_parse_rate"] == 1.0


def test_cabi_fills_coords_from_gbfs_by_short_name(tmp_path):
    z = _zip(tmp_path / "201807-capitalbikeshare-tripdata.zip", {"201807-cabi.csv": CABI})
    gbfs = pd.DataFrame({
        "station_id": ["uuid-1", "uuid-2"], "short_name": ["31200", "31201"],
        "name": ["Mass Ave & Dupont", "15th & P"], "lat": [38.9101, 38.9098], "lon": [-77.0444, -77.0342]})
    res = T.analyze_zip(z, gbfs_stations=gbfs)
    assert res["coords_in_file"] == 0.0 and res["coords_final"] == 1.0
    assert res["start_stations"] == 2


def test_bluebikes_falls_back_to_name_match():
    df = T.normalize_trips(pd.read_csv(pd.io.common.StringIO(BLUEBIKES)))
    gbfs = pd.DataFrame({"station_id": ["x"], "short_name": ["A32000"],
                         "name": ["south station - 700 atlantic ave"], "lat": [42.35], "lon": [-71.055]})
    out = T.fill_start_coords(df, gbfs)
    assert T.coord_coverage(out) == 1.0


def test_unmatched_stations_stay_missing():
    df = T.normalize_trips(pd.read_csv(pd.io.common.StringIO(CABI)))
    gbfs = pd.DataFrame({"station_id": ["z"], "short_name": ["99999"], "name": ["Elsewhere"],
                         "lat": [1.0], "lon": [2.0]})
    assert T.coord_coverage(T.fill_start_coords(df, gbfs)) == 0.0


def test_parse_listing_handles_namespace_and_truncation():
    xml = (b'<?xml version="1.0"?><ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
           b'<IsTruncated>true</IsTruncated>'
           b'<Contents><Key>201801-hubway-tripdata.zip</Key><Size>123</Size></Contents>'
           b'<Contents><Key>201802-hubway-tripdata.zip</Key><Size>456</Size></Contents>'
           b'</ListBucketResult>')
    items, truncated = T.parse_listing(xml)
    assert items == [("201801-hubway-tripdata.zip", 123), ("201802-hubway-tripdata.zip", 456)]
    assert truncated is True


def test_zip_skips_macosx_junk(tmp_path):
    z = _zip(tmp_path / "x.zip", {"__MACOSX/._a.csv": "garbage", "a.csv": BLUEBIKES})
    trips, stations = T.read_trip_zip(z)
    assert len(trips) == 1 and stations.empty


def test_window_selection_via_env():
    try:
        os.environ["ACTIVITY_WINDOW"] = "robustness"
        c = importlib.reload(cfg)
        assert c.ACS_T0["label"] == "2015-19" and c.ACS_T1["label"] == "2020-24"
        assert list(c.ACTIVITY_YEARS) == [2016, 2017, 2018, 2019]
    finally:
        os.environ.pop("ACTIVITY_WINDOW", None)
        c = importlib.reload(cfg)
    assert c.ACS_T0["label"] == "2014-18" and c.ACS_T1["label"] == "2019-23"
    # T1 data must sit on 2020 tracts, T0 on 2010 tracts, and the windows must not overlap
    for w in c.WINDOWS.values():
        assert w["t0"]["tract_vintage"] == 2010 and w["t1"]["tract_vintage"] == 2020
        assert w["t1"]["year"] - 4 > w["t0"]["year"]
        assert max(w["activity_years"]) <= w["t0"]["year"]
