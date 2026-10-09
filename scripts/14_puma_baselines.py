"""PUMA time study, step 5: the baseline ladder on the PUMA-year panel (no tuning, no final models).

Target: dlog_rel, the change in log mean household income from year-1 to year minus the city's median that year.
Baselines (see panel_modeling.py): B0 level vs city; B1 + free controls; B2 + last year's change; ALT_* + one group of
year-over-year bikeshare changes; ALT_all + all nine. Ridge regression, fitted with inverse-variance weights from the ACS
margins of error. Three validations, each predicting rows the model never trained on:
  puma     5 folds, a PUMA is never in train and test together
  loco     leave one city out
  forward  test year Y predicted from earlier years only (2018 and 2019 are scored)
Gains vs the compared baseline come with a 95% interval from resampling whole PUMAs.

Needs script 12. Run from the repo root:  python scripts/14_puma_baselines.py
Writes results/puma/baseline_summary.csv (the one table to read) and baseline_metrics.csv.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from activity_index import config as cfg  # noqa: E402
from activity_index import panel as P  # noqa: E402
from activity_index import panel_modeling as PM  # noqa: E402

warnings.filterwarnings("ignore")
SCHEMES = ("puma", "loco", "forward")
N_BOOT = 500


def main():
    out = cfg.RESULTS / "puma"
    out.mkdir(parents=True, exist_ok=True)
    pan = pd.read_csv(cfg.PROCESSED / "puma_panel.csv", dtype={"GEOID": str})
    df = pan[pan["usable"]].sort_values(["year", "GEOID"]).reset_index(drop=True)
    print(f"usable rows {len(df)}: {df['city'].value_counts().to_dict()}; PUMAs {df['GEOID'].nunique()}; years {sorted(df['year'].unique())}")
    nz = P.noise_ceiling(df[PM.TARGET], df["se_dlog"])
    print(f"noise ceiling for this target: noise share {nz['noise_share']:.2f}, best possible R2 {nz['max_r2']:.2f}, "
          f"best possible correlation {nz['max_corr']:.2f}")
    summ, met = PM.run_ladder(df, SCHEMES, N_BOOT)
    met.to_csv(out / "baseline_metrics.csv", index=False)
    summ.to_csv(out / "baseline_summary.csv", index=False)

    pd.set_option("display.width", 240)
    print("\n=== ridge, noise-weighted fit, pooled out-of-sample ===")
    print("spearman = rank correlation of predicted and actual change; r2 = share of variance explained; "
          "gain = change in Spearman vs compared_to, with 95% interval from resampling PUMAs")
    cols = ["baseline", "compared_to"] + [f"{m}_{s}" for s in SCHEMES for m in ("spearman", "r2")]
    print(summ[cols].round(3).to_string(index=False))
    print("\nrows scored:", {s: int(summ[f"n_{s}"].iloc[0]) for s in SCHEMES})
    gcols = ["baseline", "compared_to"] + [f"gain_{s}{x}" for s in SCHEMES for x in ("", "_lo", "_hi")]
    print(summ[gcols].dropna().round(3).to_string(index=False))
    print("\nwrote", out / "baseline_summary.csv")


if __name__ == "__main__":
    main()
