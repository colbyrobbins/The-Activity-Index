"""Where does trip growth help? Per-area and per-city change in dollar error (B2g vs B1), plus a stability check.

Needs scripts 21 (area_panel.csv). Uses the same forward predictions as script 22.
  gain = mean |miss| with trip growth minus mean |miss| without it (negative = trip growth helped), per area, over its scored years.
Stability check: do the areas that gained in the early test years (2018-2019) also gain in the recent ones (2023-2024)?
Spearman correlation near 0 = the spread across areas is mostly noise; clearly positive = some areas really do respond more.
Each area has only ~2 scored years per half, so read single areas as noisy; the correlation and the city rows are the safer read.
Run from the repo root:  python scripts/23_area_gains.py      Writes results/area_gains.csv.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from activity_index import area_nowcast as AN  # noqa: E402
from activity_index import config as cfg  # noqa: E402

warnings.filterwarnings("ignore")


def main():
    pd.set_option("display.width", 220)
    pan = pd.read_csv(cfg.PROCESSED / "area_panel.csv", dtype={"GEOID": str})
    d = AN.prepare(pan, cfg.AREAS["target_years"]).reset_index(drop=True)
    sets = AN.feature_sets()
    p_old, p_new = AN.forward_predictions(d, sets["B1"]), AN.forward_predictions(d, sets["B2g"])
    ok = ~np.isnan(p_old) & ~np.isnan(p_new)
    d = d[ok].reset_index(drop=True)
    d["miss_b1"], d["miss_b2g"] = AN._abs_err(d, p_old[ok]), AN._abs_err(d, p_new[ok])
    d["gain"] = d["miss_b2g"] - d["miss_b1"]
    d["half"] = np.where(d["year"] <= 2019, "early_2018_19", "recent_2023_24")

    area = d.groupby(["city", "GEOID"]).agg(n=("gain", "size"), gain=("gain", "mean"), miss_b1=("miss_b1", "mean")).reset_index()
    halves = d.pivot_table(index="GEOID", columns="half", values="gain", aggfunc="mean")
    area = area.merge(halves, left_on="GEOID", right_index=True).sort_values("gain")
    area.to_csv(cfg.RESULTS / "area_gains.csv", index=False)

    print("per area: change in average dollar miss from adding trip growth (negative = helped)")
    print(area.round(0).to_string(index=False))
    print(f"\nareas where trip growth helped overall: {(area['gain'] < 0).sum()} of {len(area)}")
    city = d.groupby("city").agg(n=("gain", "size"), areas=("GEOID", "nunique"), gain=("gain", "mean"),
                                  helped_share=("gain", lambda s: (s < 0).mean()))
    print("\nper city:")
    print(city.round(2).to_string())
    rho = area["early_2018_19"].corr(area["recent_2023_24"], method="spearman")
    print(f"\nstability: Spearman correlation of an area's early gain with its recent gain = {rho:.2f} (n = {len(area)} areas)")
    print("near 0: which areas gain is mostly noise; clearly positive: the same areas keep responding")
    print("wrote", cfg.RESULTS / "area_gains.csv")


if __name__ == "__main__":
    main()
