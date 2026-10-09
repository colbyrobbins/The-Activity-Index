"""PUMA nowcast (time-series design): from the year-T trailing-12-month bikeshare record, estimate a PUMA's year-T income.

Rows are PUMA-years. Features: own published history (income T-1, last change), slow-moving free census as of T-1, and bikeshare
(year-T level, year-over-year change, deviation from the PUMA's own earlier years). No static geography. Label: ACS 1-year log
mean household income of year T, relative to the city's median PUMA. See src/activity_index/puma_nowcast.py.
Validation (each predicts rows the model never trained on):
  forward  year Y predicted from earlier years only (the headline; the last year is a true nowcast)
  puma     5 folds, a PUMA never in train and test together
Needs script 12. Run from the repo root:  python scripts/17_puma_nowcast.py
Writes results/puma/nowcast_summary.csv and nowcast_gains.csv.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from activity_index import config as cfg  # noqa: E402
from activity_index import puma_nowcast as PN  # noqa: E402

warnings.filterwarnings("ignore")


def main():
    pd.set_option("display.width", 240)
    out = cfg.RESULTS / "puma"
    out.mkdir(parents=True, exist_ok=True)
    pan = pd.read_csv(cfg.PROCESSED / "puma_panel.csv", dtype={"GEOID": str})
    df = PN.prepare(pan)
    print(f"usable PUMA-years {len(df)}: {df['city'].value_counts().to_dict()}; PUMAs {df['GEOID'].nunique()}; "
          f"years {sorted(df['year'].unique())}")
    print(f"persistence: Spearman(income T-1, income T), relative to city = "
          f"{df['log_income_prev_rel'].corr(df[PN.TARGET_REL], method='spearman'):.3f}")
    met, gains = PN.run(df)
    met.to_csv(out / "nowcast_summary.csv", index=False)
    gains.to_csv(out / "nowcast_gains.csv", index=False)
    for s in ("forward", "puma"):
        m = met[met["cv"] == s].set_index("set").loc[list(PN.WHAT)]
        print(f"\n=== {s} validation, {int(m['n'].iloc[0])} PUMA-years scored ===")
        print("change_*: ranks/sizes who moved vs last published; level_*: year-T level vs city; mae_pct: typical level error in % of income")
        print(m[["features", "change_spearman", "change_r2", "level_spearman", "mae_pct"]].round(3).to_string())
    print("\ngain in change_spearman, 95% interval from resampling whole PUMAs:")
    print(gains.round(3).to_string(index=False))
    print("\nwrote", out)


if __name__ == "__main__":
    main()
