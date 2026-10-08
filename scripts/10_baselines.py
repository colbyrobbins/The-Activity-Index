"""Noise-aware baseline ladder (B0, B1, B2, bikeshare feature groups) with one comparison table.

Design:
  * training is weighted by 1 / (signal variance + noise variance) of each tract's rank change (script 09);
  * tracts whose income estimate is very imprecise (coefficient of variation above CV_MAX in either
    window) are dropped from the main sample and reported;
  * the spatial-block validation also learns city-specific mean reversion (city dummies and
    city-by-initial-rank terms); leave-one-city-out cannot, because the held-out city is unseen;
  * the bikeshare features enter as four groups fixed in advance (usage, rider mix, commute role, trip
    character), each tested on top of the best free-data baseline B2, and all together;
  * the linear model is ridge (supports sample weights); trees are a small LightGBM;
  * Boston is excluded from the headline sample (92% of its rank change is noise); the four-city run is a sensitivity check;
  * Spearman is also reported on the lower-noise half of tracts.

Run after scripts 06 and 09, from the repo root:  python scripts/10_baselines.py
Writes data/processed/model_table.csv (the table future models start from), results/baseline_summary.csv
(the one table that compares every baseline, Boston excluded), baseline_summary_with_boston.csv (same with Boston) and the full detail in results/baselines_metrics.csv and
results/baselines_lift.csv. All controls and bikeshare features are within-city percentiles (see modeling.city_relative).
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from activity_index import config as cfg  # noqa: E402
from activity_index import modeling as M  # noqa: E402

warnings.filterwarnings("ignore")
CV_MAX = 0.3
EXCLUDED = {"boston": "92% of its rank change is sampling noise (noise ceiling, script 09): max correlation 0.29 vs 0.61-0.68 elsewhere"}
N_BOOT = 500


def load():
    tab = M.city_relative(pd.read_csv(cfg.PROCESSED / "tract_table.csv", dtype={"GEOID": str}))
    noise = pd.read_csv(cfg.PROCESSED / "tract_noise.csv", dtype={"GEOID": str})[["GEOID", "rank_change_se", "cv0", "cv1"]]
    df = tab[tab["in_scope"] & tab["rank_change"].notna()].merge(noise, on="GEOID", how="left")
    n0 = len(df)
    imprecise = (df["cv0"] > CV_MAX) | (df["cv1"] > CV_MAX) | df["rank_change_se"].isna()
    print(f"in-scope tracts {n0}; dropped {int(imprecise.sum())} with income CV above {CV_MAX} (or no SE): "
          f"{df.loc[imprecise, 'city'].value_counts().to_dict()}")
    df = df[~imprecise].reset_index(drop=True)
    df["income_rank_t0_sq"] = df["income_rank_t0"] ** 2
    df["weight"] = M.noise_weights(df["rank_change_se"], df["rank_change"]).values
    df["low_noise"] = df["rank_change_se"] <= df["rank_change_se"].median()
    df["use_in_models"] = ~df["city"].isin(EXCLUDED)      # Boston stays in the table for reference, not in the headline models
    df.to_csv(cfg.PROCESSED / "model_table.csv", index=False)       # the table future models start from
    print("wrote", cfg.PROCESSED / "model_table.csv")
    return df


def run_sample(name, df):
    """All schemes, models and feature sets on one sample. Returns metric rows, lift rows, predictions."""
    w = M.noise_weights(df["rank_change_se"], df["rank_change"]).values
    lownoise = (df["rank_change_se"] <= df["rank_change_se"].median()).values
    y = df["rank_change"].values
    mrows, lrows, store = [], [], {}
    for scheme in ("loco", "spatial"):
        d, extra = (M.add_city_terms(df) if scheme == "spatial" else (df, []))
        sets = M.feature_sets(d, extra, True)
        for kind in ("ridge", "lgbm"):
            for sname, cols in sets.items():
                for weighted in (False, True):
                    p = M.oof_predictions(d, cols, kind, scheme, w if weighted else None)
                    key = (scheme, kind, sname, weighted)
                    store[key] = p
                    mrows += M.metric_rows(d, p, w, lownoise, {"sample": name, "cv": scheme, "model": kind,
                                                               "set": sname, "weighted_fit": weighted})
            for new, old in M.LIFTS:
                b = M.bootstrap_lift(y, store[(scheme, kind, new, True)], store[(scheme, kind, old, True)], N_BOOT)
                lo, hi = np.percentile(b, [2.5, 97.5], axis=0)
                lrows.append({"sample": name, "cv": scheme, "model": kind, "comparison": f"{new} vs {old}",
                              "d_spearman": M.spearman(y, store[(scheme, kind, new, True)]) - M.spearman(y, store[(scheme, kind, old, True)]),
                              "lo": lo[0], "hi": hi[0],
                              "d_r2": M.r2(y, store[(scheme, kind, new, True)]) - M.r2(y, store[(scheme, kind, old, True)]),
                              "r2_lo": lo[1], "r2_hi": hi[1]})
    return mrows, lrows


STEP = {"B0": None, "B1": "B0", "B2": "B1", "ALT_usage": "B2", "ALT_rider_mix": "B2", "ALT_commute_role": "B2",
        "ALT_trip_character": "B2", "ALT_all": "B2"}
WHAT = {"B0": "initial rank + rank squared", "B1": "B0 + 8 free-data controls", "B2": "B1 + 2009-13 to 2014-18 momentum",
        "ALT_usage": "B2 + usage (2)", "ALT_rider_mix": "B2 + rider mix (3)", "ALT_commute_role": "B2 + commute role (2)",
        "ALT_trip_character": "B2 + trip character (3)", "ALT_all": "B2 + all 10 bikeshare features"}


def summary(met, lift, name="no_boston"):
    """One row per baseline: ridge, noise-weighted fit, pooled out-of-sample, under both validations."""
    m = met[(met["sample"] == name) & (met["scope"] == "ALL") & (met["model"] == "ridge") & met["weighted_fit"]]
    lf = lift[(lift["sample"] == name) & (lift["model"] == "ridge")]
    rows = []
    for s, ref in STEP.items():
        r = {"baseline": s, "features": WHAT[s], "compared_to": ref or ""}
        for cv in ("loco", "spatial"):
            x = m[(m["cv"] == cv) & (m["set"] == s)].iloc[0]
            r[f"spearman_{cv}"], r[f"r2_{cv}"] = x["spearman"], x["r2"]
            if ref:
                d = lf[(lf["cv"] == cv) & (lf["comparison"] == f"{s} vs {ref}")].iloc[0]
                r[f"gain_{cv}"], r[f"gain_{cv}_lo"], r[f"gain_{cv}_hi"] = d["d_spearman"], d["lo"], d["hi"]
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    cfg.RESULTS.mkdir(exist_ok=True)
    df = load()
    # headline sample excludes Boston (see EXCLUDED); the four-city run is kept as a sensitivity check
    samples = {"no_boston": df[df["use_in_models"]].reset_index(drop=True), "all_cities": df}
    M_, L_ = [], []
    for name, d in samples.items():
        print(f"sample {name}: {len(d)} tracts {d['city'].value_counts().to_dict()}")
        m, l = run_sample(name, d)
        M_ += m; L_ += l
    met, lift = pd.DataFrame(M_), pd.DataFrame(L_)
    met.to_csv(cfg.RESULTS / "baselines_metrics.csv", index=False)
    lift.to_csv(cfg.RESULTS / "baselines_lift.csv", index=False)
    pd.set_option("display.width", 220)
    for name in samples:
        t = summary(met, lift, name)
        t.to_csv(cfg.RESULTS / ("baseline_summary.csv" if name == "no_boston" else "baseline_summary_with_boston.csv"), index=False)
        print(f"\n=== {'HEADLINE' if name == 'no_boston' else 'SENSITIVITY (with Boston)'} / {name}: ridge, noise-weighted fit, pooled out-of-sample ===")
        print("spearman = rank correlation of predicted and actual rank change; r2 = share of variance explained; "
              "gain = change in Spearman vs the 'compared_to' baseline, with 95% bootstrap interval")
        print(t[["baseline", "compared_to", "spearman_loco", "r2_loco", "spearman_spatial", "r2_spatial"]].round(3).to_string(index=False))
        print(t[["baseline", "compared_to", "gain_loco", "gain_loco_lo", "gain_loco_hi", "gain_spatial", "gain_spatial_lo",
                 "gain_spatial_hi"]].dropna().round(3).to_string(index=False))
    print(f"\nfiles written to {cfg.RESULTS}")


if __name__ == "__main__":
    main()
