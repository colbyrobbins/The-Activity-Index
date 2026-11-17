"""Turn the nowcast into a site-screening decision and score it as one.

Needs script 21 (area_panel.csv). Uses the forward predictions of script 22 (B1 = income history only, B2g = B1 + trip growth).
Question a retail or real-estate team would ask in January, eight months before the ACS posts last year's incomes:
"which areas of this city are pulling ahead?"  Three screens are compared on the same 80 area-years:
  income history only (B1) | income history + trip growth (B2g) | trip growth alone (no model: rank areas by their trip growth)
Scores: direction accuracy (ahead of / behind the city median), rank correlation inside a city-year, how the top pick did, and the
income gap between the top third and bottom third of each city's ranking. 95% intervals resample whole city-years.
Also reports which way the signal points (the fitted slope of income growth on trip growth in every forward fold), and whether it is
only standing in for the area's income level (B1 + level, with and without trip growth).
Also writes the screen itself: one row per area and year with the last posted income, the nowcast, the call and what happened.
Run from the repo root:  python scripts/28_decision_screen.py
Writes results/decision/decision_metrics.csv, signal_direction.csv, level_control.csv and screen.csv.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from activity_index import area_nowcast as AN  # noqa: E402
from activity_index import config as cfg  # noqa: E402
from activity_index import decision as D  # noqa: E402
from activity_index import panel as P  # noqa: E402
from activity_index import robustness as R  # noqa: E402

warnings.filterwarnings("ignore")
KEYS = ["direction_accuracy", "rank_corr_within_city_year", "top_pick_in_top_half", "top_pick_extra_income", "top_third_minus_bottom_third"]


def main():
    pd.set_option("display.width", 240)
    out = cfg.RESULTS / "decision"
    out.mkdir(parents=True, exist_ok=True)
    pan = pd.read_csv(cfg.PROCESSED / "area_panel.csv", dtype={"GEOID": str})
    d = AN.prepare(pan, cfg.AREAS["target_years"]).reset_index(drop=True)
    sets = AN.feature_sets()
    screens = {"income history only (B1)": D.scored(d, sets["B1"], "income history only (B1)"),
               "income history + trip growth (B2g)": D.scored(d, sets["B2g"], "income history + trip growth (B2g)"),
               "trip growth alone (no model)": D.scored(d, None, "trip growth alone (no model)", direct="d_log_trips_rel")}
    rows = []
    for name, s in screens.items():
        m = D.metrics(s)
        for k, (lo, hi) in D.bootstrap(s, KEYS).items():
            m[k + "_lo"], m[k + "_hi"] = lo, hi
        rows.append(m)
    met = pd.DataFrame(rows)
    met.to_csv(out / "decision_metrics.csv", index=False)
    print("the screen scored as a decision (coin flip: direction 0.50, rank correlation 0, top pick in top half 0.50, dollars 0):")
    for k in KEYS:
        print(f"\n{k}")
        for _, r in met.iterrows():
            f = (lambda v: f"${v:,.0f}") if "income" in k or "third" in k else (lambda v: f"{v:.2f}")
            print(f"  {r['screen']:<38} {f(r[k]):>9}   (95% interval {f(r[k + '_lo'])} to {f(r[k + '_hi'])})")

    direction = D.signal_direction(d)
    direction.to_csv(out / "signal_direction.csv", index=False)
    print("\nwhich way trip growth points (negative = faster trip growth goes with SLOWER income growth):")
    print(direction.round(3).to_string(index=False))
    d2 = P.relative_to_group(d, ["log_income_prev"]) if "log_income_prev_rel" not in d else d
    lvl = sets["B1"] + ["log_income_prev_rel"]
    p_b1, p_l, p_lt = (AN.forward_predictions(d2, c) for c in (sets["B1"], lvl, lvl + ["d_log_trips_rel"]))
    g, lo, hi = AN.cluster_gain(d2, p_lt, p_l, n_boot=2000)
    ctl = pd.DataFrame([{"model": "B1 + income level", "mae_dollars": R.mae(d2, p_l)},
                        {"model": "B1 + income level + trip growth", "mae_dollars": R.mae(d2, p_lt), "gain_from_trip_growth": g, "lo": lo, "hi": hi}])
    ctl.to_csv(out / "level_control.csv", index=False)
    print(f"\nis trip growth standing in for how rich the area already is? (B1 ${R.mae(d2, p_b1):,.0f})")
    print(ctl.round(0).to_string(index=False))

    b1, b2 = screens["income history only (B1)"], screens["income history + trip growth (B2g)"]
    tab = b2[["city", "GEOID", "year", "income_prev", "mean_income", "se_log_income", "d_log_trips", "d_log_trips_rel", "pred", "pred_rel",
              "rank_pred", "n_city_year", "actual_rel", "actual_pct"]].copy()
    tab["nowcast_income"] = tab["income_prev"] * np.exp(tab["pred"])
    tab["nowcast_income_b1"] = (b1["income_prev"] * np.exp(b1["pred"])).values
    tab["posting_margin_90"] = 1.645 * tab["se_log_income"] * tab["mean_income"]
    tab["call"] = np.where(tab["pred_rel"] > 0, "ahead of city", np.where(tab["pred_rel"] < 0, "behind city", "at city median"))
    tab["outcome"] = np.where(tab["actual_rel"] > 0, "ahead of city", np.where(tab["actual_rel"] < 0, "behind city", "at city median"))
    tab["call_correct"] = np.where(tab["call"] == "at city median", "", (tab["call"] == tab["outcome"]).astype(str))
    tab = tab.rename(columns={"income_prev": "last_posted_income", "mean_income": "posted_income", "d_log_trips": "trip_growth",
                              "d_log_trips_rel": "trip_growth_vs_city", "pred": "expected_growth", "pred_rel": "expected_growth_vs_city",
                              "actual_rel": "actual_growth_vs_city"}).sort_values(["year", "city", "rank_pred"])
    tab.to_csv(out / "screen.csv", index=False)
    last = tab[tab["year"] == tab["year"].max()]
    print(f"\nthe {int(last['year'].iloc[0])} screen (made from {int(last['year'].iloc[0]) - 1} postings and {int(last['year'].iloc[0])} trips):")
    print(last[["city", "GEOID", "rank_pred", "last_posted_income", "nowcast_income", "posted_income", "trip_growth_vs_city", "call",
                "outcome"]].round({"last_posted_income": 0, "nowcast_income": 0, "posted_income": 0, "trip_growth_vs_city": 3})
          .to_string(index=False))
    print("\nwrote", out)


if __name__ == "__main__":
    main()
