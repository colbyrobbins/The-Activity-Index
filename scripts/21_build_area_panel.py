"""Nowcast step 3 (data): the area-year panel - bikeshare features and income for each area and year.

Needs: script 20 (areas), script 05 --areas (trip aggregates for config.AREAS["bike_years"]; run script 01 for 2022-2024 first).
Stations are given one key per physical station across the 2019/2022 change of station ids (src/activity_index/station_keys.py), then
assigned to the area that contains them. Each row is one area in one year t:
  bikeshare  year-over-year changes on matched station-months (panel.py), plus year-t levels
  income     ACS 1-year mean household income summed over the area's PUMAs; dlog = log change from t-1; log_income_prev = t-1
Rows exist for 2016-2019, 2022, 2023, 2024; `is_target` marks the agreed target years (2022 is a baseline year only, 2021 and 2020 are
not used). Writes data/processed/area_panel.csv. Run from the repo root:  python scripts/21_build_area_panel.py
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from activity_index import config as cfg  # noqa: E402
from activity_index import panel as P  # noqa: E402
from activity_index import panel_modeling as PM  # noqa: E402
from activity_index.area_inputs import load_area_inputs  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    pd.set_option("display.width", 220)
    A = cfg.AREAS
    sm, acs, areas, rep = load_area_inputs()
    print("stations (ids linked across the id change are counted in ids_linked_across_eras):")
    print(rep.round(3).to_string(index=False))
    params = {"bike_years": A["bike_years"], "min_mature_stations": A["min_mature_stations"],
              "min_matched_stations": A["min_matched_stations"]}
    panel, h1, n_mature = P.build_panel(sm, acs, areas, params, cfg.CITY_CENTERS, PM.relative_columns(), min_group=A["min_group"],
                                    require_controls=False)       # the free-census rung is dropped, so census gaps must not drop rows
    panel["is_target"] = panel["year"].isin(A["target_years"])
    panel.to_csv(cfg.PROCESSED / "area_panel.csv", index=False)

    print(f"\nareas with >= {A['min_mature_stations']} stations active in both {A['bike_years'][0]} and {A['bike_years'][-1]}: "
          f"{int((n_mature >= A['min_mature_stations']).sum())} of {areas['GEOID'].nunique()}")
    t = panel[panel["is_target"]]
    print(f"\narea-year rows in target years: {len(t)}, with every field: {int(t['valid'].sum())}, usable: {int(t['usable'].sum())}")
    print("usable rows by city and year:")
    print(t.pivot_table(index="city", columns="year", values="usable", aggfunc="sum").fillna(0).astype(int).to_string())
    print("\nmedian matched stations per area (a thin area-year gives noisy features):")
    print(t.pivot_table(index="city", columns="year", values="n_matched", aggfunc="median").round(0).to_string())
    need = ["dlog", "dlog_lag", "log_income_prev"] + P.BIKE_ALL
    miss = t[need].isna().sum()
    print("\nmissing values among the fields a target-year row needs:")
    print(miss[miss > 0].to_string() if (miss > 0).any() else "none")
    print("\nwrote", cfg.PROCESSED / "area_panel.csv")


if __name__ == "__main__":
    main()
