"""Aggregate the raw trip zips into compact tables (no GIS libraries needed).

For every trip CSV: keep trips that start in the activity years and last 1-180 minutes, then count by
station and month (departures and arrivals, member share, weekday, AM/PM peak, round trips, duration),
by origin-destination pair and year, and record one row of data-quality counts.

Each zip is cached in data/interim/agg/<city>/<zip name>/ so an interrupted run resumes where it stopped.

Outputs (gitignored), all with a `city` column:
    data/interim/station_month.csv.gz   data/interim/od_year.csv.gz
    data/interim/station_names.csv.gz   data/interim/trip_qc.csv

Run from the repo root:  python scripts/05_aggregate_trips.py            (all cities)
                         python scripts/05_aggregate_trips.py --cities boston dc
                         python scripts/05_aggregate_trips.py --areas     (nowcast: trip years in config.AREAS["bike_years"], *_areas outputs)
                         python scripts/05_aggregate_trips.py --panel     (PUMA study: trip years in config.PUMA["bike_years"];
                                                                           separate cache and *_panel outputs, so the tract study is untouched)
NYC is the slow part (about 55M trips); expect roughly 10-25 minutes in total.
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", category=FutureWarning)

from activity_index import config as cfg  # noqa: E402
from activity_index.features import aggregate_trips, combine_parts  # noqa: E402
from activity_index.trips import iter_trip_csvs  # noqa: E402

TABLES = ("station_month", "od_year", "names", "qc")
ID_COLS = {"station_id": str, "start_id": str, "end_id": str}


def read_cached(d: Path) -> dict:
    return {t: pd.read_csv(d / f"{t}.csv.gz", dtype=ID_COLS) for t in TABLES}


def process_zip(zp: Path, cache: Path, years) -> dict:
    if (cache / "DONE").exists():
        cached = read_cached(cache)
        if cached["qc"]["rows_valid"].sum() > 0:     # a cache with zero valid trips came from a parsing bug: redo
            return cached
    parts = []
    for name, trips in iter_trip_csvs(zp):
        parts.append(aggregate_trips(trips, years, cfg.MIN_TRIP_MIN, cfg.MAX_TRIP_MIN))
        del trips
    res = combine_parts(parts) if parts else None
    if res is None:
        return {}
    cache.mkdir(parents=True, exist_ok=True)
    for t in TABLES:
        res[t].to_csv(cache / f"{t}.csv.gz", index=False)
    (cache / "DONE").write_text("ok")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="+", default=list(cfg.CITIES), choices=list(cfg.CITIES))
    ap.add_argument("--panel", action="store_true", help="aggregate the PUMA-study trip years into *_panel outputs")
    ap.add_argument("--areas", action="store_true", help="nowcast years (config.AREAS['bike_years']) into *_areas outputs; reuses the "
                                                          "agg_panel cache, so 2015-2019 zips are not read again")
    args = ap.parse_args()
    cfg.INTERIM.mkdir(parents=True, exist_ok=True)
    years = cfg.AREAS["bike_years"] if args.areas else cfg.PUMA["bike_years"] if args.panel else list(cfg.ACTIVITY_YEARS)
    sfx = "_areas" if args.areas else "_panel" if args.panel else ""
    agg_dir = cfg.INTERIM / ("agg_panel" if (args.panel or args.areas) else "agg")

    per_city = {t: [] for t in TABLES}
    for city in args.cities:
        zips = sorted((cfg.RAW / "trips" / city).glob("*.zip"))
        print(f"[{city}] {len(zips)} zips")
        parts, used = [], []
        for i, zp in enumerate(zips, 1):
            t0 = time.time()
            res = process_zip(zp, agg_dir / city / zp.stem, years)
            if res:
                parts.append(res)
                used.append(zp)
                q = res["qc"]
                print(f"  {i:>3}/{len(zips)} {zp.name:42} rows {int(q['rows'].sum()):>10,} "
                      f"valid {int(q['rows_valid'].sum()):>10,}  ({time.time() - t0:.0f}s)"
                      + ("   <-- NO VALID TRIPS: check column names/timestamps" if q["rows_valid"].sum() == 0 else ""))
        if not parts:
            continue
        # qc is kept per zip (not summed) so the EDA can show data quality by file
        for p, zp in zip(parts, used):
            p["qc"] = p["qc"].assign(file=zp.name)
        comb = combine_parts(parts)
        comb["qc"] = pd.concat([p["qc"] for p in parts], ignore_index=True)
        for t in TABLES:
            per_city[t].append(comb[t].assign(city=city))

    names = {"station_month": f"station_month{sfx}.csv.gz", "od_year": f"od_year{sfx}.csv.gz",
             "names": f"station_names{sfx}.csv.gz", "qc": f"trip_qc{sfx}.csv"}
    for t, fn in names.items():
        if not per_city[t]:
            continue
        new = pd.concat(per_city[t], ignore_index=True)
        path = cfg.INTERIM / fn
        if path.exists() and set(args.cities) != set(cfg.CITIES):   # partial run: keep other cities' rows
            old = pd.read_csv(path, dtype=ID_COLS)
            new = pd.concat([old[~old["city"].isin(args.cities)], new], ignore_index=True)
        new.to_csv(path, index=False)
        print(f"wrote {path}  ({len(new):,} rows)")

    sm = pd.read_csv(cfg.INTERIM / f"station_month{sfx}.csv.gz", dtype=ID_COLS)
    print("\ntrips counted by city and year (valid trips in the activity window):")
    print(sm.assign(year=sm["ym"] // 100).pivot_table(index="city", columns="year", values="dep",
                                                      aggfunc="sum").round(0).astype("Int64"))
    for city in sm["city"].unique():
        have = set((sm.loc[sm["city"] == city, "ym"] // 100).unique())
        missing = [y for y in years if y not in have]
        if missing:
            print(f"WARNING: {city} has no trips for {missing}")


if __name__ == "__main__":
    main()
