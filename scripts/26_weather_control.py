"""Is the trip-growth signal a weather effect? (mid-semester feedback: control for rainfall, temperature and seasonality)

Needs scripts 21 (area_panel.csv, data/interim/area_station_month.csv.gz) and 25 (data/raw/weather_daily.csv).
  1. How much weather moves usage: monthly trip growth vs the change in temperature, wet days and precipitation, city-wide and
     for each area relative to its city (the second is the only part that can reach the model, see activity_index/weather.py).
  2. Weather-adjusted trip growth: remove each area's fitted weather effect month by month, rebuild the annual growth, re-run the
     nowcast (B1 + adjusted growth), and also add the removed weather part as its own feature.
  3. Seasonality: trip growth from the warm months only (Apr-Oct) and from the cold months only (Nov-Mar).
Same forward validation and dollar error as script 22; negative gain = helped.
Run from the repo root:  python scripts/26_weather_control.py
Writes results/robustness/weather_explained.csv, weather_sensitivity.csv, weather_gains.csv and fig_weather.png.
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
from activity_index import panel as P  # noqa: E402
from activity_index import robustness as R  # noqa: E402
from activity_index import weather as W  # noqa: E402

warnings.filterwarnings("ignore")
WARM, COLD = [4, 5, 6, 7, 8, 9, 10], [11, 12, 1, 2, 3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weather", default=str(cfg.RAW / "weather_daily.csv"))
    a = ap.parse_args()
    pd.set_option("display.width", 220)
    out = cfg.RESULTS / "robustness"
    out.mkdir(parents=True, exist_ok=True)
    pan = pd.read_csv(cfg.PROCESSED / "area_panel.csv", dtype={"GEOID": str})
    sm = pd.read_csv(cfg.INTERIM / "area_station_month.csv.gz", dtype={"station_id": str, "puma": str})
    city_of = sm.drop_duplicates("puma").set_index("puma")["city"]
    wx = W.weather_changes(W.monthly_weather(pd.read_csv(a.weather)))
    years = [y for y in cfg.AREAS["bike_years"] if y - 1 in cfg.AREAS["bike_years"]]

    mg = W.monthly_growth(sm, years)
    mg["city"] = mg["GEOID"].map(city_of)
    mg = mg.merge(wx, on=["city", "year", "m"], how="left")
    expl = W.explained(mg)
    expl.to_csv(out / "weather_explained.csv", index=False)
    print("1. how much of monthly trip growth weather explains:")
    print(expl.round(4).to_string(index=False))

    sens = W.sensitivities(mg)
    sens.to_csv(out / "weather_sensitivity.csv", index=False)
    print(f"\nper-area weather sensitivity ({len(sens)} areas): median R2 {sens['r2'].median():.2f}; per degree C "
          f"{sens['b_d_temp'].median():+.3f} (range {sens['b_d_temp'].min():+.3f} to {sens['b_d_temp'].max():+.3f}); per wet day "
          f"{sens['b_d_wet'].median():+.3f} (range {sens['b_d_wet'].min():+.3f} to {sens['b_d_wet'].max():+.3f})")

    d = AN.prepare(pan, cfg.AREAS["target_years"]).reset_index(drop=True)
    d = d.merge(W.adjusted_growth(mg, sens), on=["GEOID", "year"], how="left")
    seas = []
    for name, months in (("g_warm", WARM), ("g_cold", COLD)):
        f = pd.concat([P.puma_year_features(sm, y, months, cfg.AREAS["min_matched_stations"]) for y in sorted(d["year"].unique())])
        seas.append(f.rename(columns={"puma": "GEOID", "d_log_trips": name})[["GEOID", "year", name]])
    d = d.merge(seas[0], on=["GEOID", "year"], how="left").merge(seas[1], on=["GEOID", "year"], how="left")
    d = P.relative_to_group(d, ["g_raw", "g_wx_adj", "g_wx", "g_warm", "g_cold"])
    print(f"\n{len(d)} area-years; annual growth rebuilt from months vs the panel: max abs difference "
          f"{(d['g_raw'] - d['d_log_trips']).abs().max():.4f}; correlation of adjusted with raw relative growth "
          f"{d['g_wx_adj_rel'].corr(d['d_log_trips_rel']):.3f}; sd of the weather part {d['g_wx_rel'].std():.4f} "
          f"vs sd of relative growth {d['d_log_trips_rel'].std():.4f}")

    sets = AN.feature_sets()
    b1, b2g = AN.forward_predictions(d, sets["B1"]), AN.forward_predictions(d, sets["B2g"])
    variants = {"trip growth (B2g)": sets["B2g"], "weather-adjusted trip growth": sets["B1"] + ["g_wx_adj_rel"],
                "trip growth + weather part": sets["B2g"] + ["g_wx_rel"], "weather part only": sets["B1"] + ["g_wx_rel"],
                "warm-month growth (Apr-Oct)": sets["B1"] + ["g_warm_rel"], "cold-month growth (Nov-Mar)": sets["B1"] + ["g_cold_rel"]}
    rows = []
    for name, cols in variants.items():
        p = AN.forward_predictions(d, cols)
        gain, lo, hi = AN.cluster_gain(d, p, b1, n_boot=2000)
        rows.append({"model": "B1 + " + name, "mae_dollars": R.mae(d, p), "gain_vs_b1": gain, "lo": lo, "hi": hi,
                     "vs_b2g": R.mae(d, p) - R.mae(d, b2g)})
    gains = pd.DataFrame(rows)
    gains.to_csv(out / "weather_gains.csv", index=False)
    print(f"\n2-3. B1 ${R.mae(d, b1):,.0f}; the nowcast with each version of trip growth:")
    print(gains.round(1).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    c = mg.dropna(subset=W.WX).groupby(["city", "year", "m"]).agg(t0=("trips_0", "sum"), t1=("trips_1", "sum"),
                                                                 d_temp=("d_temp", "first")).reset_index()
    axes[0].scatter(c["d_temp"], np.log(c["t1"] / c["t0"]), s=10, color="#4c78a8", alpha=0.7)
    axes[0].set_xlabel("change in monthly mean temperature vs a year earlier (C)")
    axes[0].set_ylabel("city-wide trip growth (log)")
    axes[0].set_title(f"Weather moves a whole city (R2 = {expl['r2_weather'].iloc[0]:.2f})", fontsize=10)
    axes[1].scatter(d["d_log_trips_rel"], d["g_wx_adj_rel"], s=14, color="#c0392b", alpha=0.7)
    lim = float(np.nanmax(np.abs(d[["d_log_trips_rel", "g_wx_adj_rel"]].values))) * 1.05
    axes[1].plot([-lim, lim], [-lim, lim], color="black", lw=0.8, ls=":")
    axes[1].set_xlabel("area trip growth relative to its city (the model feature)")
    axes[1].set_ylabel("same, weather-adjusted")
    axes[1].set_title("...but barely changes an area's growth relative to its city", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_weather.png", dpi=160)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
