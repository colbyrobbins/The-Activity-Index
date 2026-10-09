"""PUMA time study, variant: multi-year outcome (income change over h years instead of one year).

Why: one-year changes in ACS 1-year income are about two-thirds sampling noise (scripts 13-14). Real change adds up over
a longer window while the survey error of the two end years does not, so the noise share should fall. The cost is fewer rows
(windows must start inside the bike data, 2015 on) and overlapping windows (the cluster bootstrap resamples whole PUMAs for that).

For each horizon h in config.PUMA["horizons"]:
  outcome      dlog_h = log mean income in year t minus year t-h, minus the city's median (end years 2015+h ... 2019)
  features     the same nine bikeshare changes, matched station-months in year t vs year t-h
  controls     measured at the window start (t-h)
  momentum     change from t-2h to t-h-1: it ends one year BEFORE the window starts, so it shares no survey year with the
               outcome and cannot "predict" the noise of the start year (the artifact that inflated B2 in the one-year panel)
  validation   puma (5 folds) and loco; no forward scheme, because overlapping windows would leak the future into training

Needs script 12's inputs (11 and `05 --panel`). Run from the repo root:  python scripts/15_puma_multiyear.py
Writes data/processed/puma_panel_h<h>.csv, results/puma/multiyear_summary_h<h>.csv and multiyear_comparison.csv.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from activity_index import config as cfg  # noqa: E402
from activity_index import panel as P  # noqa: E402
from activity_index import panel_modeling as PM  # noqa: E402
from activity_index.puma_inputs import load_panel_inputs  # noqa: E402

warnings.filterwarnings("ignore")
SCHEMES = ("puma", "loco")
N_BOOT = 500


def one_year_reference():
    path = cfg.PROCESSED / "puma_panel.csv"
    if not path.exists():
        return None
    u = pd.read_csv(path, dtype={"GEOID": str})
    u = u[u["usable"]]
    return {"horizon": 1, **P.noise_ceiling(u["dlog_rel"], u["se_dlog"]), "pumas": u["GEOID"].nunique()}


def main():
    out = cfg.RESULTS / "puma"
    out.mkdir(parents=True, exist_ok=True)
    sm, acs, pumas, _ = load_panel_inputs()
    pd.set_option("display.width", 240)
    comp = [r for r in [one_year_reference()] if r]
    for h in cfg.PUMA["horizons"]:
        panel, h1, _ = P.build_panel(sm, acs, pumas, cfg.PUMA, cfg.CITY_CENTERS, PM.relative_columns(), span=h)
        panel.to_csv(cfg.PROCESSED / f"puma_panel_h{h}.csv", index=False)
        df = panel[panel["usable"]].sort_values(["year", "GEOID"]).reset_index(drop=True)
        print(f"\n================ horizon {h} years ================")
        print(f"usable rows {len(df)} from {df['GEOID'].nunique()} PUMAs; end years {sorted(df['year'].unique())}; "
              f"by city {df['city'].value_counts().to_dict()}")
        nz = P.noise_ceiling(df[PM.TARGET], df["se_dlog"])
        by_city = {c: round(P.noise_ceiling(g[PM.TARGET], g["se_dlog"])["noise_share"], 2) for c, g in df.groupby("city")}
        print(f"noise share of the outcome {nz['noise_share']:.2f} (best possible R2 {nz['max_r2']:.2f}, correlation {nz['max_corr']:.2f}); "
              f"by city {by_city}; SD of dlog_rel {df[PM.TARGET].std():.3f}, median SE {df['se_dlog'].median():.3f}")
        comp.append({"horizon": h, **nz, "pumas": df["GEOID"].nunique()})

        mom = PM.cluster_corr(df["dlog_lag_rel"], df[PM.TARGET], df["GEOID"], 200)
        lvl = PM.cluster_corr(df["log_income_prev_rel"], df[PM.TARGET], df["GEOID"], 200)
        print(f"Spearman with the outcome: start level {lvl[0]:.3f} [{lvl[1]:.3f}, {lvl[2]:.3f}], "
              f"gap momentum {mom[0]:.3f} [{mom[1]:.3f}, {mom[2]:.3f}]")
        rows = []
        for f in PM.BIKE_ALL:
            r = PM.cluster_corr(df[f + "_rel"], df[PM.TARGET], df["GEOID"], 300)
            rows.append({"feature": f, "rho": r[0], "lo": r[1], "hi": r[2], "n": r[3]})
        print("\nbikeshare changes vs the outcome (Spearman, PUMA-cluster 95% interval):")
        print(pd.DataFrame(rows).round(3).to_string(index=False))

        summ, met = PM.run_ladder(df, SCHEMES, N_BOOT)
        summ.to_csv(out / f"multiyear_summary_h{h}.csv", index=False)
        print(f"\n=== baseline ladder, horizon {h}: ridge, noise-weighted fit, pooled out-of-sample ===")
        print(summ[["baseline", "compared_to"] + [f"{m}_{s}" for s in SCHEMES for m in ("spearman", "r2")]].round(3).to_string(index=False))
        print(summ[["baseline", "compared_to"] + [f"gain_{s}{x}" for s in SCHEMES for x in ("", "_lo", "_hi")]].dropna().round(3).to_string(index=False))

    cmp_df = pd.DataFrame(comp)[["horizon", "pumas", "n", "var_observed", "var_noise", "noise_share", "max_r2", "max_corr"]]
    cmp_df.to_csv(out / "multiyear_comparison.csv", index=False)
    print("\n=== does a longer window cut the noise? ===")
    print(cmp_df.round(3).to_string(index=False))
    print("\nwrote", out)


if __name__ == "__main__":
    main()
