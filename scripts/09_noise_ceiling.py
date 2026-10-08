"""How much of the target is sampling noise? Noise ceiling for predicting income-rank change.

Uses the ACS margins of error: converts them to standard errors of each tract's mean household income
(both windows, the 2020 window moved onto 2010 tracts), redraws incomes many times, re-ranks within city,
and measures how much the rank change moves from sampling noise alone.

Needs scripts 02 (re-run it once so the ACS files include margins of error), 04, and 06. Writes
data/processed/tract_noise.csv and results/noise_ceiling.csv. Run from the repo root:
    python scripts/09_noise_ceiling.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from activity_index import config as cfg  # noqa: E402
from activity_index.crosswalk import apply_crosswalk, apply_crosswalk_moe  # noqa: E402
from activity_index.noise import ceiling, ratio_se, simulate  # noqa: E402

N_DRAWS = 300


def main():
    t0, t1 = cfg.ACS_T0["label"], cfg.ACS_T1["label"]
    a0 = pd.read_csv(cfg.RAW / f"acs_{t0}.csv", dtype={"GEOID": str})
    a1 = pd.read_csv(cfg.RAW / f"acs_{t1}.csv", dtype={"GEOID": str})
    for a in (a0, a1):
        if "agg_household_income_moe" not in a:
            sys.exit("ACS files have no margins of error: run `python scripts/02_fetch_acs_tiger.py` once more.")
    xw = pd.read_csv(cfg.INTERIM / "xwalk_2020_to_2010.csv", dtype={"GEOID": str, "GEOID_dst": str})
    tab = pd.read_csv(cfg.PROCESSED / "tract_table.csv", dtype={"GEOID": str})

    # T1 on 2010 tracts: apportion the estimates (as in script 04) and the MOEs (root-sum-of-squares)
    cols = ["agg_household_income", "households"]
    est1 = apply_crosswalk(a1, xw, cols).set_index("GEOID")
    m1 = a1[["GEOID", "agg_household_income_moe", "households_moe"]].rename(
        columns={"agg_household_income_moe": "agg_household_income", "households_moe": "households"})
    moe1 = apply_crosswalk_moe(m1, xw, cols).set_index("GEOID").reindex(est1.index)
    se1 = ratio_se(est1["agg_household_income"], moe1["agg_household_income"], est1["households"], moe1["households"])

    d = a0[["GEOID", "city", "agg_household_income", "households", "agg_household_income_moe", "households_moe"]].copy()
    d["mean0"] = d["agg_household_income"] / d["households"]
    d["se0"] = ratio_se(d["agg_household_income"], d["agg_household_income_moe"], d["households"], d["households_moe"])
    d = d.set_index("GEOID")
    d["mean1"] = (est1["agg_household_income"] / est1["households"]).reindex(d.index)
    d["se1"] = se1.reindex(d.index)
    # same exclusion as the target: tracts with fewer than 50 households are not ranked
    d.loc[d["households"] < 50, ["mean0", "se0"]] = np.nan
    d.loc[est1["households"].reindex(d.index) < 50, ["mean1", "se1"]] = np.nan
    d = d.reset_index()

    d = d.merge(tab[["GEOID", "rank_change", "in_scope", "income_rank_t0"]], on="GEOID", how="left")
    scope = d["in_scope"].fillna(False).values.astype(bool)
    sim = simulate(d[["city", "mean0", "se0", "mean1", "se1"]], n_draws=N_DRAWS, scope=scope)
    d["rank_change_se"] = sim["noise_sd"].values
    d["z"] = d["rank_change"] / d["rank_change_se"]
    d["cv0"], d["cv1"] = d["se0"] / d["mean0"], d["se1"] / d["mean1"]
    d[["GEOID", "city", "in_scope", "rank_change", "rank_change_se", "z", "cv0", "cv1"]].to_csv(
        cfg.PROCESSED / "tract_noise.csv", index=False)

    sc = d[d["in_scope"].fillna(False) & d["rank_change"].notna()]
    rows = []
    for scope_name, g in [("ALL in-scope", sc)] + [(c, sc[sc["city"] == c]) for c in sorted(sc["city"].unique())]:
        c = ceiling(g["rank_change"], g["rank_change_se"])
        c.update({"scope": scope_name, "median_cv_t0": g["cv0"].median(), "median_cv_t1": g["cv1"].median(),
                  "median_rank_change_se": g["rank_change_se"].median(),
                  "share_moves_beyond_2se": (g["z"].abs() > 2).mean()})
        rows.append(c)
    res = pd.DataFrame(rows).set_index("scope")
    res.to_csv(cfg.RESULTS / "noise_ceiling.csv")

    pd.set_option("display.width", 200)
    print("income precision (coefficient of variation = SE / mean income), in-scope tracts, median:")
    print(res[["median_cv_t0", "median_cv_t1"]].round(3).to_string())
    print("\nnoise in the target (rank change; observed SD for reference):")
    print(res[["n", "var_observed", "var_noise", "noise_share", "max_r2", "max_corr", "median_rank_change_se",
               "share_moves_beyond_2se"]].round(3).to_string())
    print("\nSpearman(initial rank, rank change) produced by noise ALONE (no true change), vs observed:")
    obs = {c: pd.Series(g["income_rank_t0"]).rank().corr(pd.Series(g["rank_change"]).rank())
           for c, g in sc.groupby("city")}
    print(pd.DataFrame({"noise_only": sim["pure_noise_rho"], "observed": obs}).round(3).to_string())
    print("\nlargest observed moves and how many standard errors they are:")
    print(sc.reindex(sc["rank_change"].abs().sort_values(ascending=False).index)[
        ["GEOID", "city", "rank_change", "rank_change_se", "z", "cv0", "cv1"]].head(12).round(3).to_string(index=False))
    print("\nobserved SD of rank change by noise tercile (is the spread bigger where data are noisier?):")
    sc = sc.assign(tercile=pd.qcut(sc["rank_change_se"], 3, labels=["low noise", "mid", "high noise"]))
    print(sc.groupby("tercile", observed=True)["rank_change"].agg(["size", "std"]).round(3).to_string())
    print("\nwrote", cfg.PROCESSED / "tract_noise.csv", "and", cfg.RESULTS / "noise_ceiling.csv")


if __name__ == "__main__":
    main()
