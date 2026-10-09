"""Levels nowcast (tract level): how well does 2015-2018 bikeshare use estimate a tract's 2014-18 income rank?

This is the "nowcasting" framing: features and label cover the same period, and the question is what bikeshare adds to
what was already published. Targets the income LEVEL (within-city percentile rank of mean household income, 2014-18).

Competitors, all predicting income_rank_t0 for tracts the model never trained on:
  N0  last published: the 2009-13 income rank (+ square)                  <- the number a business would already have
  N1  N0 + the free controls as published in 2009-13
  N2  bikeshare only
  N3  free controls only (no income history)
  N4  free controls + bikeshare (no income history)
  N5  N1 + all bikeshare (and N5_<group> for each group)                    <- the nowcast: does bikeshare add to N1?
Validation: spatial blocks (5 folds, city terms) and leave-one-city-out. Ridge, no weights (level noise is small).
The last-published baseline is the previous NON-overlapping ACS window, so it is a stale baseline: a real operator would hold a
fresher, overlapping release, and the bar for bikeshare would be higher still.

Needs scripts 06 and 10 (model_table.csv) and the 2009-13 ACS file (script 02 --prior-only).
Run from the repo root:  python scripts/16_levels_nowcast.py
Writes results/nowcast_summary.csv (all four cities) and nowcast_summary_no_boston.csv.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from activity_index import config as cfg  # noqa: E402
from activity_index import modeling as M  # noqa: E402
from activity_index import nowcast as N  # noqa: E402

warnings.filterwarnings("ignore")
N_BOOT = 500


def load():
    mt = pd.read_csv(cfg.PROCESSED / "model_table.csv", dtype={"GEOID": str})
    tt = pd.read_csv(cfg.PROCESSED / "tract_table.csv", dtype={"GEOID": str})
    prior = pd.read_csv(cfg.RAW / f"acs_{cfg.PRIOR_ACS['label']}.csv", dtype={"GEOID": str})
    pc = N.prior_controls(prior, tt)
    df = mt.merge(pc, on="GEOID", how="left")
    df = df[df["income_rank_t0"].notna() & df["income_rank_prior"].notna()].reset_index(drop=True)
    df["income_rank_prior_sq"] = df["income_rank_prior"] ** 2
    return df


def run(df, name):
    y = df[N.TARGET].values
    store, rows = {}, []
    for scheme in ("spatial", "loco"):
        d, dummies, rank_terms = N.city_terms(df) if scheme == "spatial" else (df.assign(), [], [])
        if scheme == "spatial":
            sets = N.feature_sets(dummies, rank_terms)
        else:
            sets = N.feature_sets([], [])
        for sname, cols in sets.items():
            p = M.oof_predictions(d, cols, "ridge", scheme, None, target=N.TARGET)
            store[(scheme, sname)] = p
            rows.append({"sample": name, "cv": scheme, "set": sname, "features": N.WHAT[sname], **N.score(y, p)})
    met = pd.DataFrame(rows)
    lifts = []
    for scheme in ("spatial", "loco"):
        for new, old in N.LIFTS:
            b = M.bootstrap_lift(y, store[(scheme, new)], store[(scheme, old)], N_BOOT)
            lo, hi = np.percentile(b, [2.5, 97.5], axis=0)
            lifts.append({"sample": name, "cv": scheme, "comparison": f"{new} vs {old}",
                          "d_spearman": M.spearman(y, store[(scheme, new)]) - M.spearman(y, store[(scheme, old)]),
                          "lo": lo[0], "hi": hi[0],
                          "d_r2": M.r2(y, store[(scheme, new)]) - M.r2(y, store[(scheme, old)]), "r2_lo": lo[1], "r2_hi": hi[1]})
    return met, pd.DataFrame(lifts)


def main():
    pd.set_option("display.width", 240)
    df = load()
    print(f"tracts with a 2014-18 rank and a 2009-13 rank: {len(df)} {df['city'].value_counts().to_dict()}")
    print(f"how persistent is income already? Spearman(2009-13 rank, 2014-18 rank) = "
          f"{M.spearman(df['income_rank_prior'], df['income_rank_t0']):.3f}")
    for name, d in {"all_cities": df, "no_boston": df[df["city"] != "boston"].reset_index(drop=True)}.items():
        met, lift = run(d, name)
        wide = met.pivot(index="set", columns="cv", values=["spearman", "r2", "mae_pct_points"])
        wide.columns = [f"{a}_{b}" for a, b in wide.columns]
        t = wide.reset_index().assign(features=lambda x: x["set"].map(N.WHAT))
        order = list(N.WHAT)
        t = t.set_index("set").loc[order].reset_index()
        t.to_csv(cfg.RESULTS / ("nowcast_summary.csv" if name == "all_cities" else "nowcast_summary_no_boston.csv"), index=False)
        lift.to_csv(cfg.RESULTS / f"nowcast_lift_{name}.csv", index=False)
        print(f"\n=== {name}: {len(d)} tracts, ridge, out-of-sample (target = 2014-18 income rank) ===")
        print("spearman / r2 on the rank scale; mae in percentile points (a tract off by 10 points would count 10)")
        print(t[["set", "features", "spearman_spatial", "r2_spatial", "mae_pct_points_spatial", "spearman_loco", "r2_loco",
                 "mae_pct_points_loco"]].round(3).to_string(index=False))
        print("\ngain in Spearman, 95% bootstrap interval:")
        print(lift[["cv", "comparison", "d_spearman", "lo", "hi", "d_r2"]].round(3).to_string(index=False))
    print("\nwrote", cfg.RESULTS)


if __name__ == "__main__":
    main()
