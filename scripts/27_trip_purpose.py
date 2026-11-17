"""Do bikeshare trips reflect residents' activity or commuting into the area? (mid-semester feedback)

Needs script 21 (area_panel.csv and data/interim/area_station_month.csv.gz). Same forward validation and dollar error as script 22.
  1. What the trips are: share of peak trips at the home end vs the work end, per city and for each area.
  2. Which growth carries the signal: B1 + growth of one kind of trip (resident-like, commuter-inflow, off-peak, weekday, weekend,
     member, casual), each against B1 and against all-trip growth (B2g).
  3. Where the all-trip gain lands: residential areas (home-end share above the city-year median) vs job centers.
Run from the repo root:  python scripts/27_trip_purpose.py
Writes results/robustness/trip_purpose_mix.csv, trip_purpose_gains.csv, trip_purpose_by_area_type.csv and fig_trip_purpose.png.
"""
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
from activity_index import trip_purpose as TP  # noqa: E402

warnings.filterwarnings("ignore")


def main():
    pd.set_option("display.width", 220)
    out = cfg.RESULTS / "robustness"
    out.mkdir(parents=True, exist_ok=True)
    pan = pd.read_csv(cfg.PROCESSED / "area_panel.csv", dtype={"GEOID": str})
    sm = pd.read_csv(cfg.INTERIM / "area_station_month.csv.gz", dtype={"station_id": str, "puma": str})
    d = AN.prepare(pan, cfg.AREAS["target_years"]).reset_index(drop=True)
    g = TP.purpose_growth(sm, sorted(d["year"].unique()), cfg.AREAS["min_matched_stations"])
    d = d.merge(g, on=["GEOID", "year"], how="left")
    gcols = [c for c in g.columns if c.startswith("g_")]
    d = P.relative_to_group(d, gcols + ["home_share"])
    print(f"{len(d)} area-years; check: all-trip growth rebuilt here vs the panel, max abs difference "
          f"{(d['g_all'] - d['d_log_trips']).abs().max():.4f}")

    mix = d.groupby("city").agg(areas=("GEOID", "nunique"), home_share_mean=("home_share", "mean"),
                                home_share_min=("home_share", "min"), home_share_max=("home_share", "max")).reset_index()
    peak = sm.groupby("city")[["dep", "dep_am", "dep_pm", "dep_weekday", "dep_member"]].sum()
    mix = mix.merge(pd.DataFrame({"peak_share_of_departures": (peak["dep_am"] + peak["dep_pm"]) / peak["dep"],
                                  "weekday_share": peak["dep_weekday"] / peak["dep"], "member_share": peak["dep_member"] / peak["dep"]})
                    .reset_index(), on="city")
    mix.to_csv(out / "trip_purpose_mix.csv", index=False)
    print("\nwhat the trips are (home_share = home-end share of peak trips; 0.5 = balanced):")
    print(mix.round(3).to_string(index=False))

    sets = AN.feature_sets()
    b1 = AN.forward_predictions(d, sets["B1"])
    b2g = AN.forward_predictions(d, sets["B2g"])
    rows = []
    for k in ["all"] + list(TP.PARTS):
        p = AN.forward_predictions(d, sets["B1"] + [f"g_{k}_rel"])
        gain, lo, hi = AN.cluster_gain(d, p, b1, n_boot=2000)
        vs_all = R.mae(d, p) - R.mae(d, b2g)
        rows.append({"trips": k, "what": TP.PARTS.get(k, "all departures (= B2g)"), "mae_dollars": R.mae(d, p), "gain_vs_b1": gain,
                     "lo": lo, "hi": hi, "vs_all_trip_growth": vs_all,
                     "corr_with_all_trip_growth": float(d[f"g_{k}_rel"].corr(d["g_all_rel"]))})
    p = AN.forward_predictions(d, sets["B1"] + ["g_home_end_rel", "g_work_end_rel"])
    gain, lo, hi = AN.cluster_gain(d, p, b1, n_boot=2000)
    rows.append({"trips": "home_end + work_end", "what": "both peak directions as two features", "mae_dollars": R.mae(d, p),
                 "gain_vs_b1": gain, "lo": lo, "hi": hi, "vs_all_trip_growth": R.mae(d, p) - R.mae(d, b2g)})
    gains = pd.DataFrame(rows)
    gains.to_csv(out / "trip_purpose_gains.csv", index=False)
    print(f"\nB1 ${R.mae(d, b1):,.0f}. B1 + growth of one kind of trip (negative gain = helped):")
    print(gains.round(2).to_string(index=False))

    ok = ~np.isnan(b2g)
    s = d[ok].copy()
    s["miss_b1"], s["miss_b2g"] = AN._abs_err(s, b1[ok]), AN._abs_err(s, b2g[ok])
    s["area_type"] = np.where(s["home_share_rel"] > 0, "more residential (home-end share above city median)", "more job-center")
    typ = s.groupby("area_type").agg(n=("miss_b1", "size"), home_share=("home_share", "mean"), mae_b1=("miss_b1", "mean"),
                                     mae_b2g=("miss_b2g", "mean")).reset_index()
    typ["gain"] = typ["mae_b2g"] - typ["mae_b1"]
    typ.to_csv(out / "trip_purpose_by_area_type.csv", index=False)
    print("\nwhere the all-trip gain lands:")
    print(typ.round(2).to_string(index=False))

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    t = gains[gains["trips"] != "home_end + work_end"].iloc[::-1]
    ax.barh(t["trips"], t["gain_vs_b1"], xerr=[t["gain_vs_b1"] - t["lo"], t["hi"] - t["gain_vs_b1"]], color="#4c78a8", ecolor="#555")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("change in average dollar miss vs B1 (negative = helped), 95% interval")
    ax.set_title("Which kind of trip growth carries the signal?", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig_trip_purpose.png", dpi=160)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
