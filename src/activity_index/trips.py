"""Find, download, and normalize bikeshare trip files from the public S3 buckets.

Each system uses different column names (and some omit station coordinates), so everything is
mapped to one canonical schema. Station coordinates come from, in order: the trip file itself,
station files bundled in the zip, then the system's current GBFS feed.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

CANONICAL = [
    "start_time", "end_time",
    "start_station_id", "start_station_name", "start_lat", "start_lon",
    "end_station_id", "end_station_name", "end_lat", "end_lon",
    "user_type",
]

# Cleaned (lowercase, underscores) source column names, in priority order.
ALIASES = {
    "start_time": ["starttime", "start_time", "started_at", "start_date", "01_rental_details_local_start_time"],
    "end_time": ["stoptime", "stop_time", "end_time", "ended_at", "end_date", "01_rental_details_local_end_time"],
    "start_station_id": ["start_station_id", "from_station_id", "start_station_number", "03_rental_start_station_id"],
    "start_station_name": ["start_station_name", "from_station_name", "start_station", "03_rental_start_station_name"],
    "start_lat": ["start_station_latitude", "start_lat"],
    "start_lon": ["start_station_longitude", "start_lng", "start_lon"],
    "end_station_id": ["end_station_id", "to_station_id", "end_station_number", "02_rental_end_station_id"],
    "end_station_name": ["end_station_name", "to_station_name", "end_station", "02_rental_end_station_name"],
    "end_lat": ["end_station_latitude", "end_lat"],
    "end_lon": ["end_station_longitude", "end_lng", "end_lon"],
    "user_type": ["usertype", "user_type", "member_type", "member_casual"],
}
STATION_ALIASES = {
    "station_id": ["id", "station_id", "terminal", "terminalname", "short_name"],
    "name": ["name", "station_name", "stationname"],
    "lat": ["latitude", "lat"],
    "lon": ["longitude", "lon", "lng"],
}


def _clean(col) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(col).strip().lower()).strip("_")


def parse_times(s: pd.Series) -> pd.Series:
    """Parse timestamps; files mix formats, so re-parse any failures with format='mixed'."""
    t = pd.to_datetime(s, errors="coerce")
    bad = t.isna() & s.notna()
    if bad.any():
        t[bad] = pd.to_datetime(s[bad], errors="coerce", format="mixed")
    return t


def _id_series(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


def normalize_trips(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_clean(c) for c in df.columns]
    out = pd.DataFrame(index=df.index)
    for canon, names in ALIASES.items():
        src = next((n for n in names if n in df.columns), None)
        out[canon] = df[src] if src else np.nan
    for c in ("start_time", "end_time"):
        out[c] = parse_times(out[c])
    for c in ("start_lat", "start_lon", "end_lat", "end_lon"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in ("start_station_id", "end_station_id"):
        out[c] = _id_series(out[c])
    for c in ("start_station_name", "end_station_name", "user_type"):
        out[c] = out[c].astype("string").str.strip()
    return out[CANONICAL]


def normalize_stations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_clean(c) for c in df.columns]
    out = pd.DataFrame(index=df.index)
    for canon, names in STATION_ALIASES.items():
        src = next((n for n in names if n in df.columns), None)
        out[canon] = df[src] if src else np.nan
    out["station_id"] = _id_series(out["station_id"])
    out["name"] = out["name"].astype("string").str.strip()
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    return out.dropna(subset=["lat", "lon"]).reset_index(drop=True)


# ---- S3 listing and download ------------------------------------------------------------

def parse_listing(content: bytes):
    """Parse one page of an S3 ListObjects XML response -> ([(key, size)], is_truncated)."""
    root = ET.fromstring(content)
    items, truncated = [], False
    for el in root.iter():
        tag = el.tag.rsplit("}", 1)[-1]
        if tag == "Contents":
            key = size = None
            for ch in el:
                t = ch.tag.rsplit("}", 1)[-1]
                if t == "Key":
                    key = ch.text
                elif t == "Size":
                    size = int(ch.text)
            items.append((key, size))
        elif tag == "IsTruncated":
            truncated = (el.text or "").strip().lower() == "true"
    return items, truncated


def list_keys(bucket_url: str, prefix: str, timeout: int = 60):
    """All (key, size_bytes) in a public S3 bucket whose key starts with prefix."""
    keys, marker = [], ""
    while True:
        r = requests.get(bucket_url, params={"prefix": prefix, "marker": marker, "max-keys": 1000},
                         timeout=timeout)
        r.raise_for_status()
        page, truncated = parse_listing(r.content)
        keys += page
        if not truncated or not page:
            return keys
        marker = page[-1][0]


def trip_keys(city_bucket: str, prefix: str):
    """Zip files only (skips any non-trip index files)."""
    return [(k, s) for k, s in list_keys(city_bucket, prefix) if k.lower().endswith(".zip")]


def download(url: str, dest, chunk: int = 1 << 20) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for part in r.iter_content(chunk):
                f.write(part)
    tmp.replace(dest)
    return dest


# ---- Reading zips -------------------------------------------------------------------------

_SPLIT_PART = re.compile(r"_\d+\.csv$", re.I)


def select_csvs(names):
    """Pick one copy of each month's data from a zip's trip CSV names.

    Citi Bike's annual zips repeat every month: a full monthly CSV in the year folder, split parts
    (..._1.csv, ..._2.csv) beside it, and the same split parts again inside month subfolders.
    Reading everything double- or triple-counts trips (the 2018 zip yields ~2x rows). Rule: group by
    the leading YYYYMM (or the whole file name when there is none), prefer an unsplit file at the
    shallowest folder depth, otherwise take the split parts at the shallowest depth.
    """
    groups = {}
    for n in names:
        m = re.match(r"(\d{6})", Path(n).name)
        groups.setdefault(m.group(1) if m else Path(n).name, []).append(n)
    keep = []
    for files in groups.values():
        unsplit = [f for f in files if not _SPLIT_PART.search(f)]
        pool = unsplit or files
        depth = min(f.count("/") for f in pool)
        pool = sorted(f for f in pool if f.count("/") == depth)
        keep += pool[:1] if unsplit else pool
    return sorted(keep)


def plan_zip(zip_path) -> dict:
    """Which CSVs in a zip to read: {'trip_files', 'station_files', 'skipped' (duplicate copies)}."""
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in sorted(z.namelist())
                 if not n.endswith("/") and "__MACOSX" not in n and Path(n).name.lower().endswith(".csv")]
    stations = [n for n in names if "station" in Path(n).name.lower()]
    trips_all = [n for n in names if n not in stations]
    trips = select_csvs(trips_all)
    return {"trip_files": trips, "station_files": stations,
            "skipped": [n for n in trips_all if n not in trips]}


def read_trip_zip(zip_path, nrows: int | None = None, max_files: int | None = None):
    """Read a trip zip -> (trips in canonical schema, normalized station files bundled in the zip).

    nrows limits rows per CSV and max_files limits how many trip CSVs are read (for quick samples).
    Duplicate copies of the same month are skipped (see select_csvs).
    """
    plan = plan_zip(zip_path)
    trip_files = plan["trip_files"][:max_files] if max_files else plan["trip_files"]
    trips, stations = [], []
    with zipfile.ZipFile(zip_path) as z:
        for name in trip_files:
            with z.open(name) as f:
                df = pd.read_csv(f, low_memory=False, encoding_errors="replace", nrows=nrows)
            trips.append(normalize_trips(df))
        for name in plan["station_files"]:
            with z.open(name) as f:
                stations.append(normalize_stations(pd.read_csv(f, low_memory=False, encoding_errors="replace")))
    trips_df = pd.concat(trips, ignore_index=True) if trips else pd.DataFrame(columns=CANONICAL)
    stations_df = (pd.concat(stations, ignore_index=True).drop_duplicates()
                   if stations else pd.DataFrame(columns=list(STATION_ALIASES)))
    return trips_df, stations_df


def iter_trip_csvs(zip_path, nrows: int | None = None):
    """Yield (csv_name, trips in canonical schema) one CSV at a time, so big zips never sit in memory."""
    plan = plan_zip(zip_path)
    with zipfile.ZipFile(zip_path) as z:
        for name in plan["trip_files"]:
            with z.open(name) as f:
                df = pd.read_csv(f, low_memory=False, encoding_errors="replace", nrows=nrows)
            yield name, normalize_trips(df)


# ---- Station coordinates --------------------------------------------------------------------

def load_gbfs_stations(gbfs_url: str, timeout: int = 60) -> pd.DataFrame:
    """Current stations with coordinates from a GBFS feed (station_id, short_name, name, lat, lon)."""
    root = requests.get(gbfs_url, timeout=timeout)
    root.raise_for_status()
    data = root.json()["data"]
    feeds = (data.get("en") or next(iter(data.values())))["feeds"]
    info_url = next(f["url"] for f in feeds if f["name"] == "station_information")
    r = requests.get(info_url, timeout=timeout)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["data"]["stations"])
    for c in ("station_id", "short_name", "name"):
        if c not in df:
            df[c] = np.nan
    out = df[["station_id", "short_name", "name", "lat", "lon"]].copy()
    for c in ("station_id", "short_name"):
        out[c] = _id_series(out[c])
    out["name"] = out["name"].astype("string").str.strip()
    return out.dropna(subset=["lat", "lon"]).reset_index(drop=True)


def load_station_history(zip_paths) -> pd.DataFrame:
    """Station coordinates collected from the station files bundled in many trip zips.

    Divvy ships a station file inside some zips (2015 Q1Q2, 2016, 2017) but not others (2015 Q3Q4, 2018).
    Pooling them lets every zip use every year's list. Zips are read in name order; where the same
    station id appears more than once, the latest file wins.
    """
    frames = []
    for zp in sorted(map(str, zip_paths)):
        plan = plan_zip(zp)
        with zipfile.ZipFile(zp) as z:
            for name in plan["station_files"]:
                with z.open(name) as f:
                    frames.append(normalize_stations(pd.read_csv(f, low_memory=False, encoding_errors="replace")))
    cols = list(STATION_ALIASES)
    if not frames:
        return pd.DataFrame(columns=cols)
    df = pd.concat(frames, ignore_index=True)
    has_id = df["station_id"].notna()
    return pd.concat([df[has_id].drop_duplicates("station_id", keep="last"), df[~has_id]],
                     ignore_index=True)


def fill_start_coords(trips: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Fill missing start_lat/start_lon by matching station id, then station name."""
    if stations is None or stations.empty:
        return trips
    by_id, by_name = {}, {}
    for r in stations.itertuples(index=False):
        for key in (getattr(r, "station_id", None), getattr(r, "short_name", None)):
            if pd.notna(key):
                by_id.setdefault(str(key).strip(), (r.lat, r.lon))
        nm = getattr(r, "name", None)
        if pd.notna(nm):
            by_name.setdefault(str(nm).strip().casefold(), (r.lat, r.lon))

    keys = trips[["start_station_id", "start_station_name"]].drop_duplicates()
    lookup = {}
    for sid, nm in keys.itertuples(index=False):
        hit = by_id.get(str(sid).strip()) if pd.notna(sid) else None
        if hit is None and pd.notna(nm):
            hit = by_name.get(str(nm).strip().casefold())
        lookup[(None if pd.isna(sid) else str(sid), None if pd.isna(nm) else str(nm))] = hit

    pair = list(zip(trips["start_station_id"].astype(object).where(trips["start_station_id"].notna(), None),
                    trips["start_station_name"].astype(object).where(trips["start_station_name"].notna(), None)))
    hits = [lookup.get((None if a is None else str(a), None if b is None else str(b))) for a, b in pair]
    lat = pd.Series([h[0] if h else np.nan for h in hits], index=trips.index)
    lon = pd.Series([h[1] if h else np.nan for h in hits], index=trips.index)
    out = trips.copy()
    out["start_lat"] = out["start_lat"].fillna(lat)
    out["start_lon"] = out["start_lon"].fillna(lon)
    return out


def coord_coverage(trips: pd.DataFrame) -> float:
    """Share of trips whose start station has coordinates."""
    if len(trips) == 0:
        return 0.0
    return float((trips["start_lat"].notna() & trips["start_lon"].notna()).mean())


def top_unmatched(trips: pd.DataFrame, n: int = 10) -> list[dict]:
    """Start stations still lacking coordinates, ranked by trip count (what to look up next)."""
    miss = trips[trips["start_lat"].isna() | trips["start_lon"].isna()]
    if miss.empty:
        return []
    counts = (miss.groupby(["start_station_id", "start_station_name"], dropna=False)
              .size().sort_values(ascending=False).head(n))
    return [{"id": None if pd.isna(i) else str(i), "name": None if pd.isna(nm) else str(nm), "trips": int(c)}
            for (i, nm), c in counts.items()]


def analyze_zip(zip_path, gbfs_stations: pd.DataFrame | None = None,
                nrows: int | None = None, max_files: int | None = None,
                history: pd.DataFrame | None = None) -> dict:
    """Summarize one trip zip: files, time range, columns found, and station-coordinate coverage."""
    plan = plan_zip(zip_path)
    trips, bundled = read_trip_zip(zip_path, nrows=nrows, max_files=max_files)
    res = {
        "csvs_in_zip": len(plan["trip_files"]) + len(plan["skipped"]),
        "csvs_skipped_as_duplicates": len(plan["skipped"]),
        "csvs_read": min(len(plan["trip_files"]), max_files) if max_files else len(plan["trip_files"]),
        "rows": int(len(trips)),
        "sampled": bool(nrows or max_files),
        "start_min": str(trips["start_time"].min()) if len(trips) else None,
        "start_max": str(trips["start_time"].max()) if len(trips) else None,
        "time_parse_rate": float(trips["start_time"].notna().mean()) if len(trips) else 0.0,
        "start_stations": int(trips["start_station_id"].nunique()),
        "columns_found": [c for c in CANONICAL if len(trips) and trips[c].notna().any()],
        "coords_in_file": coord_coverage(trips),
    }
    trips = fill_start_coords(trips, bundled)
    res["coords_after_bundled_stations"] = coord_coverage(trips)
    trips = fill_start_coords(trips, history)
    res["coords_after_history"] = coord_coverage(trips)
    trips = fill_start_coords(trips, gbfs_stations)
    res["coords_final"] = coord_coverage(trips)
    res["top_unmatched_stations"] = top_unmatched(trips)
    return res
