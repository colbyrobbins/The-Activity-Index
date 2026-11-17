"""Does the trip-growth gain ($8,495 -> $7,995 average miss, B2g vs B1) survive stress tests?

Needs script 21 (data/processed/area_panel.csv). Same forward validation and dollar error as script 22.
  1. Interval: the gain with a 95% interval from resampling whole areas (2,000 draws).
  2. Placebo across areas: trip growth is shuffled between areas of the same city-year. Tests whether THIS area's growth matters.
  3. Placebo across years: each area's trip growth is shuffled between its own years. Tests whether THIS year's growth matters.
     p-value = share of placebo draws that helped at least as much as the real feature. Small (< 0.05) = unlikely to be chance.
  4. Placebo best-of-all: trip growth was chosen as the best of all bikeshare features, so compare it with the best feature found
     when areas are handed another area's whole bikeshare record (corrects for having looked at every feature first).
  5. Where the gain comes from: by city, by test year, and with each city left out in turn.
Run from the repo root:  python scripts/24_placebo.py [--draws 2000]
Writes results/robustness/placebo.csv, placebo_draws.csv, single_feature_gains.csv, gain_by_group.csv, leave_one_city_out.csv and fig_placebo.png.
"""
import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from activity_index import area_nowcast as AN  # noqa: E402
from activity_index import config as cfg  # noqa: E402
from activity_index import robustness as R  # noqa: E402

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=2000)
    a = ap.parse_args()
    pd.set_option("display.width", 220)
    out = cfg.RESULTS / "robustness"
    out.mkdir(parents=True, exist_ok=True)
    pan = pd.read_csv(cfg.PROCESSED / "area_panel.csv", dtype={"GEOID": str})
    d = AN.prepare(pan, cfg.AREAS["target_years"]).reset_index(drop=True)
    sets = AN.feature_sets()
    old, new = sets["B1"], sets["B2g"]

    p_old, p_new = AN.forward_predictions(d, old), AN.forward_predictions(d, new)
    real, lo, hi = AN.cluster_gain(d, p_new, p_old, n_boot=2000)
    print(f"B1 ${R.mae(d, p_old):,.0f} -> B2g ${R.mae(d, p_new):,.0f}: gain ${real:,.0f} (95% interval ${lo:,.0f} to ${hi:,.0f}), "
          f"{int((~np.isnan(p_new)).sum())} area-years, {d['GEOID'].nunique()} areas")

    draws = {"across_areas (same city-year)": R.placebo(d, ["city", "year"], a.draws, seed=1),
             "across_years (same area)": R.placebo(d, ["GEOID"], a.draws, seed=2)}
    tab = pd.DataFrame([R.placebo_row(k, real, v) for k, v in draws.items()])
    feats = [c + "_rel" for c in AN.BIKE_ALL]
    single = R.single_feature_gains(d, feats).sort_values()
    single.rename("gain").rename_axis("feature").reset_index().to_csv(out / "single_feature_gains.csv", index=False)
    best = R.best_of_placebo(d, feats, max(a.draws // 4, 100), seed=3)
    tab = pd.concat([tab, pd.DataFrame([R.placebo_row(f"best of {len(feats)} features (records swapped)", float(single.min()), best)])],
                    ignore_index=True)
    print(f"\neach bikeshare feature alone on top of B1 (best: {single.index[0]}):")
    print(single.round(0).to_string())
    tab.to_csv(out / "placebo.csv", index=False)
    pd.DataFrame(draws).to_csv(out / "placebo_draws.csv", index=False)
    print("\nplacebo tests (gain in dollars; negative = helped):")
    print(tab.round(3).to_string(index=False))

    groups = pd.concat([R.by_group(d, new, old, "city"), R.by_group(d, new, old, "year")], ignore_index=True)
    groups.to_csv(out / "gain_by_group.csv", index=False)
    print("\nwhere the gain comes from:")
    print(groups.round(2).to_string(index=False))
    loco = R.leave_one_city_out(d, new, old)
    loco.to_csv(out / "leave_one_city_out.csv", index=False)
    print("\nrefit without each city:")
    print(loco.round(0).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, (name, v) in zip(axes, draws.items()):
        ax.hist(v, bins=40, color="#9aa5b1")
        ax.axvline(real, color="#c0392b", lw=2, label=f"real trip growth: ${real:,.0f}")
        ax.axvline(0, color="black", lw=0.8, ls=":")
        p = tab.loc[tab["placebo"] == name, "p_value"].iloc[0]
        ax.set_title(f"shuffled {name}\np = {p:.3f}", fontsize=10)
        ax.set_xlabel("change in average dollar miss vs B1 (negative = helped)")
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("placebo draws")
    fig.tight_layout()
    fig.savefig(out / "fig_placebo.png", dpi=160)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
