"""PUMA time study, step 3: build the PUMA-year panel (one row per PUMA and year).

Needs: script 11 (ACS + polygons), and the panel trip aggregates (python scripts/05_aggregate_trips.py --panel).

Each row is one PUMA in one year t (2016-2019):
  outcome   dlog = change in log mean household income from t-1 to t (ACS 1-year); dlog_rel = minus the city's median that year
  bikeshare year-over-year changes on matched station-months (see panel.py): trips, rider mix, commute role, duration,
            and system expansion (stations added)
  controls  ACS levels at t-1 (education, renters, rent, home value, age, density, distance to center, households)
  momentum  last year's income change
A PUMA is in scope if at least config.PUMA["min_mature_stations"] stations had departures in the first AND last bike year.
Every model column also exists as `<col>_rel` = value minus the median of its (city, year) group.

Writes: data/processed/puma_panel.csv, puma_panel_h1.csv (bike features using only January-June), pumas.csv.
Run from the repo root:  python scripts/12_build_puma_panel.py
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from activity_index import config as cfg  # noqa: E402
from activity_index import panel as P  # noqa: E402
from activity_index import panel_modeling as PM  # noqa: E402
from activity_index.puma_inputs import load_panel_inputs  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    cfg.PROCESSED.mkdir(parents=True, exist_ok=True)
    pc = cfg.PUMA
    sm, acs, pumas, rep = load_panel_inputs()
    print("station coordinates and PUMA assignment:")
    print(rep.round(3).to_string(index=False))
    panel, h1, n_mature = P.build_panel(sm, acs, pumas, pc, cfg.CITY_CENTERS, PM.relative_columns())
    print(f"\nPUMAs with >= {pc['min_mature_stations']} stations active in both {pc['bike_years'][0]} and {pc['bike_years'][-1]}: "
          f"{int((n_mature >= pc['min_mature_stations']).sum())}")
    panel.to_csv(cfg.PROCESSED / "puma_panel.csv", index=False)
    h1.to_csv(cfg.PROCESSED / "puma_panel_h1.csv", index=False)
    pumas_out = (panel.groupby("GEOID").agg(name=("NAME", "first"), city=("city", "first"),
                                            n_mature_stations=("n_mature_stations", "first"), rows=("usable", "sum"))
                 .reset_index())
    pumas_out.to_csv(cfg.PROCESSED / "pumas.csv", index=False)

    print(f"\npanel: {len(panel)} PUMA-year rows, {int(panel['valid'].sum())} with every field, {int(panel['usable'].sum())} usable "
          "(city-year group of at least 4 PUMAs)")
    print(panel.groupby("city").agg(pumas=("GEOID", "nunique"), rows=("GEOID", "size"), usable=("usable", "sum"),
                                    median_matched_stations=("n_matched", "median")).to_string())
    need = ["dlog", "dlog_lag", "log_income_prev"] + P.BIKE_ALL + P.CONTROLS
    miss = panel[need].isna().sum()
    print("\nmissing values among the fields a row needs:")
    print(miss[miss > 0].to_string() if (miss > 0).any() else "none")
    print("\nwrote", cfg.PROCESSED / "puma_panel.csv", "puma_panel_h1.csv", "pumas.csv")


if __name__ == "__main__":
    main()
