"""Pre-flight check before extending the panel to 2014 and 2020-2024: do the newer trip files parse, and can stations be matched over time?

For each city it (1) lists the trip files and sizes for the extra years (a listing only, nothing is downloaded), then (2) downloads ONE
July file per sample year (2019, 2021, 2024), reads the first 200k rows and reports: which columns were found, how many trips have a
station id, what the ids look like, and (the key question) how many stations in the later year can be matched to the earlier year
(a) by the same id, (b) by location (nearest earlier station within 100 m).
If (a) is low and (b) is high, station ids changed and the panel must match stations by location.

    python scripts/18_check_new_years.py                 # all cities (NYC July files are a few hundred MB each)
    python scripts/18_check_new_years.py --cities dc boston chicago
    python scripts/18_check_new_years.py --listing-only
Writes results/new_years_check.csv. Downloads are cached in data/raw/_check/.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from activity_index import config as cfg  # noqa: E402
from activity_index.trips import city_year_keys, download, read_trip_zip  # noqa: E402

LIST_YEARS = [2014, 2020, 2021, 2022, 2023, 2024]
SAMPLE_YEARS = [2019, 2021, 2024]
MATCH_M = 100.0


def keys_for(city, year):
    return city_year_keys(cfg.TRIP_BUCKETS[city], [cfg.TRIP_PREFIXES[city]] + cfg.TRIP_PREFIXES_ALT.get(city, []), year,
                          skip_prefix=("JC",) if city == "nyc" else ())


def station_table(trips: pd.DataFrame) -> pd.DataFrame:
    """One row per start station id: median coordinates and trip count."""
    t = trips[trips["start_station_id"].notna() & trips["start_lat"].notna() & trips["start_lon"].notna()].copy()
    t["start_lat"], t["start_lon"] = pd.to_numeric(t["start_lat"], errors="coerce"), pd.to_numeric(t["start_lon"], errors="coerce")
    g = t.groupby("start_station_id").agg(lat=("start_lat", "median"), lon=("start_lon", "median"), trips=("start_lat", "size"))
    return g.reset_index().rename(columns={"start_station_id": "station_id"})


def match_rates(early: pd.DataFrame, late: pd.DataFrame, metres: float = MATCH_M) -> dict:
    """Share of the later year's stations (weighted by trips) that appear in the earlier year by id, and by location."""
    if early.empty or late.empty:
        return {"same_id": np.nan, "within_m": np.nan}
    same = late["station_id"].isin(set(early["station_id"]))
    la1, lo1 = np.radians(late["lat"].values)[:, None], np.radians(late["lon"].values)[:, None]
    la2, lo2 = np.radians(early["lat"].values)[None, :], np.radians(early["lon"].values)[None, :]
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    d = 2 * 6371000 * np.arcsin(np.sqrt(a))
    near = d.min(axis=1) <= metres
    w = late["trips"].values
    return {"same_id": float((w * same).sum() / w.sum()), "within_m": float((w * near).sum() / w.sum())}


def sample_file(city, year):
    ks = keys_for(city, year)
    if not ks:
        return None
    hint = [f"{year}07", f"{year}_Q3", f"{year}-07"]
    for h in hint:
        for k, s in ks:
            if h in k:
                return k, s
    return min(ks, key=lambda x: x[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="+", default=list(cfg.CITIES), choices=list(cfg.CITIES))
    ap.add_argument("--listing-only", action="store_true")
    args = ap.parse_args()
    pd.set_option("display.width", 220)
    rows = []

    print("=== files available (listing only) ===")
    for city in args.cities:
        for y in LIST_YEARS:
            try:
                ks = keys_for(city, y)
                print(f"{city:8} {y}: {len(ks):3} files, {sum(s for _, s in ks) / 1e6:8.0f} MB" + ("" if ks else "   <-- NO FILES: prefix needs fixing"))
            except Exception as e:
                print(f"{city:8} {y}: listing failed: {type(e).__name__}: {e}")
    if args.listing_only:
        return

    print("\n=== sample parse and station matching ===")
    for city in args.cities:
        tables = {}
        for y in SAMPLE_YEARS:
            try:
                sf = sample_file(city, y)
                if sf is None:
                    print(f"{city} {y}: no files"); continue
                key, size = sf
                dest = cfg.RAW / "_check" / city / key
                if not dest.exists():
                    print(f"  downloading {city} {key} ({size / 1e6:.0f} MB)")
                    download(f"{cfg.TRIP_BUCKETS[city].rstrip('/')}/{key}", dest)
                trips, _ = read_trip_zip(dest, nrows=200_000, max_files=1)
                st = station_table(trips)
                tables[y] = st
                found = [c for c in trips.columns if trips[c].notna().any()]
                row = {"city": city, "year": y, "file": key, "rows": len(trips), "stations": len(st),
                       "start_id_present": float(trips["start_station_id"].notna().mean()),
                       "start_coords_present": float(trips["start_lat"].notna().mean()),
                       "user_types": ",".join(map(str, trips["user_type"].dropna().unique()[:4])),
                       "id_examples": ",".join(map(str, st["station_id"].head(4))), "columns_found": len(found)}
                if y != SAMPLE_YEARS[0]:
                    prev = tables.get(SAMPLE_YEARS[SAMPLE_YEARS.index(y) - 1])
                    if prev is not None:
                        row.update({f"vs_prev_sample_{k}": v for k, v in match_rates(prev, st).items()})
                rows.append(row)
            except Exception as e:
                print(f"{city} {y}: FAILED {type(e).__name__}: {e}")
    out = pd.DataFrame(rows)
    if len(out):
        out.to_csv(cfg.RESULTS / "new_years_check.csv", index=False)
        print(out.drop(columns=["file"]).round(3).to_string(index=False))
        print("\nvs_prev_sample_*: share of this year's trips from stations found in the earlier sample year (2021 vs 2019, 2024 vs 2021)")
        print("  same_id  = by identical station id;  within_m = by an earlier station within 100 m")
    print("\nwrote", cfg.RESULTS / "new_years_check.csv")


if __name__ == "__main__":
    main()
