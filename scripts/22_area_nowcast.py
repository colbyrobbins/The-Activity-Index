"""Nowcast: predict each area's mean household income in DOLLARS for year T from its history and the year-T bikeshare record.

Needs script 21 (data/processed/area_panel.csv). Rows are area-years in config.AREAS["target_years"] (2017-2019, 2023, 2024).
Validation is forward in time: a test year is predicted from earlier years only. Compare rungs on the dollar error:
  B0  last posting x the city's usual growth          B1  B0 + the area's last income change
  B2g / B2l  B1 + only the trip-growth / only the trip-level-vs-history feature
  B2  B1 + two bikeshare features (matched-station trip growth, trip level vs the area's own earlier years)
  F_<group>, F_all  B1 + year-T bikeshare (level, change vs T-1, deviation from the area's own earlier years)
Columns: mae_dollars = average absolute miss; median_ape_pct = typical miss as % of income; mae_shift_free = the miss after removing each
city-year's average error (what remains if city-wide growth were known); noise_floor_mae = the miss a perfect model would still show
because the posted figure is a survey estimate. Gains: change in mae_dollars vs the compared rung (negative = better), 95% interval
from resampling whole areas.
Run from the repo root:  python scripts/22_area_nowcast.py      Writes results/area_nowcast_summary.csv and area_nowcast_gains.csv.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from activity_index import area_nowcast as AN  # noqa: E402
from activity_index import config as cfg  # noqa: E402

warnings.filterwarnings("ignore")


def main():
    pd.set_option("display.width", 240)
    pan = pd.read_csv(cfg.PROCESSED / "area_panel.csv", dtype={"GEOID": str})
    d = AN.prepare(pan, cfg.AREAS["target_years"])
    print(f"area-year rows {len(d)}: {d['city'].value_counts().to_dict()}; areas {d['GEOID'].nunique()}; "
          f"rows by year {d['year'].value_counts().sort_index().to_dict()}")
    met, gains = AN.run(d)
    # Robustness: 2023's momentum is the COVID-recovery change (2021->2022). Drop 2023 rows and re-run (2024 is still predicted
    # from its 2023 posting, but 2023 is neither trained on nor scored).
    met2, gains2 = AN.run(d[d["year"] != 2023], n_boot=500)
    met2 = met2[met2["sample"] == "recent_2023_24"].assign(sample="2024_only_without_2023_rows")
    gains2 = gains2[gains2["sample"] == "recent_2023_24"].assign(sample="2024_only_without_2023_rows")
    met, gains = pd.concat([met, met2], ignore_index=True), pd.concat([gains, gains2], ignore_index=True)
    out = cfg.RESULTS
    met.to_csv(out / "area_nowcast_summary.csv", index=False)
    gains.to_csv(out / "area_nowcast_gains.csv", index=False)
    for s in met["sample"].unique():
        m = met[met["sample"] == s].set_index("set").loc[list(AN.WHAT)]
        print(f"\n=== {s}: {int(m['n'].iloc[0])} area-years scored (income in dollars) ===")
        print(m[["features", "mae_dollars", "median_ape_pct", "mae_shift_free", "noise_floor_mae", "bias_dollars"]].round(1).to_string())
    print("\nchange in average dollar error vs the compared rung (negative = better), 95% interval from resampling areas:")
    print(gains.round(0).to_string(index=False))
    print("\nwrote", out / "area_nowcast_summary.csv", "area_nowcast_gains.csv")


if __name__ == "__main__":
    main()
