"""Download all trip files for the activity years of the active window (see config.WINDOWS).

Files land in data/raw/trips/<city>/ (gitignored; the licenses forbid redistributing raw data).
Existing files with the right size are skipped, so it is safe to re-run. This can be 1 GB+ in total:
use --dry-run first to see sizes.

    python scripts/01_download_trips.py --dry-run
    python scripts/01_download_trips.py --cities chicago boston
    python scripts/01_download_trips.py --years 2016 2017
    python scripts/01_download_trips.py --years 2022 2023 2024 --cities dc boston chicago   # nowcast years
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from activity_index import config as cfg  # noqa: E402
from activity_index.trips import city_year_keys, download  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="+", default=list(cfg.CITIES), choices=list(cfg.CITIES))
    ap.add_argument("--years", nargs="+", type=int, default=list(cfg.ACTIVITY_YEARS))
    ap.add_argument("--dry-run", action="store_true", help="list files and sizes, download nothing")
    args = ap.parse_args()

    plan = []
    for city in args.cities:
        for year in args.years:
            keys = city_year_keys(cfg.TRIP_BUCKETS[city], [cfg.TRIP_PREFIXES[city]] + cfg.TRIP_PREFIXES_ALT.get(city, []), year,
                                  skip_prefix=("JC",) if city == "nyc" else ())
            if not keys:
                print(f"WARNING: no files for {city} {year}")
            plan += [(city, year, k, s or 0) for k, s in keys]

    total = sum(p[3] for p in plan)
    print(f"{len(plan)} files, {total / 1e9:.2f} GB total")
    for city in args.cities:
        mine = [p for p in plan if p[0] == city]
        print(f"  {city:8} {len(mine):3} files  {sum(p[3] for p in mine) / 1e6:9.0f} MB")
    if args.dry_run:
        return

    for i, (city, year, key, size) in enumerate(plan, 1):
        dest = cfg.RAW / "trips" / city / key
        if dest.exists() and dest.stat().st_size == size:
            continue
        print(f"[{i}/{len(plan)}] {city} {key} ({size / 1e6:.0f} MB)")
        download(f"{cfg.TRIP_BUCKETS[city].rstrip('/')}/{key}", dest)
        if size and dest.stat().st_size != size:
            print(f"  WARNING: {key} is {dest.stat().st_size} bytes, expected {size}; delete it and re-run")
    print("done")


if __name__ == "__main__":
    main()
