"""PUMA time study, step 4: EDA before any model.

Answers, in numbers and three figures:
  1. How big is the panel (PUMAs and rows per city and year)?
  2. How noisy is the yearly income change (ACS 1-year margins of error), and what is the best any model could do?
  3. Is there mean reversion and momentum in the yearly change?
  4. Month-to-month or year-to-year features? How much of month-to-month variation is just the calendar (seasonality),
     and does a half-year version of the feature (January-June only) carry the same signal as the full year?
  5. Which bikeshare changes move with the income change (Spearman, PUMA-cluster bootstrap interval)?

Needs script 12 (and the panel aggregates for the seasonality check). Run from the repo root:  python scripts/13_puma_eda.py
Writes results/puma/eda_summary.md, correlations.csv and fig1-fig3.
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

from activity_index import config as cfg  # noqa: E402
from activity_index import panel as P  # noqa: E402
from activity_index import panel_modeling as PM  # noqa: E402
from activity_index.modeling import spearman  # noqa: E402
from activity_index.panel_modeling import cluster_corr  # noqa: E402

warnings.filterwarnings("ignore")
OUT = cfg.RESULTS / "puma"
CITY_COLORS = {"nyc": "#1f77b4", "chicago": "#d62728", "dc": "#2ca02c", "boston": "#9467bd"}


def md(df, floatfmt="{:.3f}"):
    df = df.reset_index()
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) and not pd.isna(v) else str(v)
                                       for v in r) + " |")
    return "\n".join(lines)


def seasonality(sm):
    """Share of the within-year variation in monthly system trips that the calendar month alone explains."""
    rows = []
    for city, g in sm.groupby("city"):
        m = g.groupby("ym")["dep"].sum().reset_index()
        m["year"], m["month"] = m["ym"] // 100, m["ym"] % 100
        m["ly"] = np.log(m["dep"])
        within = m["ly"] - m.groupby("year")["ly"].transform("mean")
        seas = within.groupby(m["month"]).transform("mean")
        rows.append({"city": city, "months": len(m), "calendar_share_of_within_year_variance": 1 - ((within - seas) ** 2).sum() / (within ** 2).sum(),
                     "summer_vs_winter_ratio": float(np.exp(m.loc[m["month"].isin([6, 7, 8]), "ly"].mean() - m.loc[m["month"].isin([12, 1, 2]), "ly"].mean()))})
    return pd.DataFrame(rows).set_index("city")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pan = pd.read_csv(cfg.PROCESSED / "puma_panel.csv", dtype={"GEOID": str})
    h1 = pd.read_csv(cfg.PROCESSED / "puma_panel_h1.csv", dtype={"GEOID": str})
    u = pan[pan["usable"]].reset_index(drop=True)
    uh = h1[h1["usable"].astype(bool)].reset_index(drop=True)
    sm_path = cfg.INTERIM / "station_month_panel.csv.gz"
    md_out = ["# PUMA time study: EDA (numbers only)\n",
              f"Outcome: change in log mean household income from year-1 to year (ACS 1-year), years {cfg.PUMA['outcome_years']}; "
              "`dlog_rel` subtracts the city's median that year.\n"]

    # 1. coverage
    cov = u.groupby(["city", "year"]).agg(pumas=("GEOID", "nunique")).unstack("year")
    cov.columns = [int(c[1]) for c in cov.columns]
    cov["total_rows"] = u.groupby("city").size()
    ms = u.groupby("city").agg(median_mature_stations=("n_mature_stations", "median"), median_stations_per_km2=("stations_per_km2", "median"))
    md_out += ["## 1. Panel size (usable PUMA-year rows)\n", md(cov.join(ms)), f"\nTotal usable rows: {len(u)}, PUMAs: {u['GEOID'].nunique()}.\n"]

    # 2. noise
    c_all = P.noise_ceiling(u["dlog_rel"], u["se_dlog"])
    rows = [{"scope": "ALL", **c_all}] + [{"scope": c, **P.noise_ceiling(g["dlog_rel"], g["se_dlog"])} for c, g in u.groupby("city")]
    nz = pd.DataFrame(rows).set_index("scope")
    sd = u.groupby("city")["dlog_rel"].std()
    md_out += ["## 2. Noise in the yearly income change\n", md(nz[["n", "var_observed", "var_noise", "noise_share", "max_r2", "max_corr"]]),
               "\nObserved SD of dlog_rel by city: " + ", ".join(f"{k} {v:.3f}" for k, v in sd.items())
               + f"; median standard error of dlog: {u['se_dlog'].median():.3f}.\n"]

    # 3. mean reversion and momentum
    mr = pd.DataFrame({
        "level vs city -> change": [cluster_corr(u["log_income_prev_rel"], u["dlog_rel"], u["GEOID"])[0]] +
        [spearman(g["log_income_prev_rel"], g["dlog_rel"]) for _, g in u.groupby("city")],
        "last year's change -> change": [cluster_corr(u["dlog_lag_rel"], u["dlog_rel"], u["GEOID"])[0]] +
        [spearman(g["dlog_lag_rel"], g["dlog_rel"]) for _, g in u.groupby("city")]}, index=pd.Index(["ALL"] + sorted(u["city"].unique()), name="scope"))
    md_out += ["## 3. Mean reversion and momentum (Spearman)\n", md(mr),
               "\nA negative momentum correlation is the signature of noise: a PUMA whose income was measured high last year looks like it grew, then falls back.\n"]

    # 4. month-to-month or year-to-year?
    if sm_path.exists():
        seas = seasonality(pd.read_csv(sm_path, dtype={"station_id": str}))
        md_out += ["## 4. Why year-over-year on matched months\n", "Share of within-year variation in monthly system trips explained by the calendar month alone:\n",
                   md(seas)]
    rows = []
    for f in PM.BIKE_ALL:
        a = cluster_corr(u[f + "_rel"], u["dlog_rel"], u["GEOID"])
        b = cluster_corr(uh[f + "_rel"], uh["dlog"] - uh.groupby(["city", "year"])["dlog"].transform("median"), uh["GEOID"]) if len(uh) else (np.nan,) * 4
        rows.append({"feature": f, "full_year_rho": a[0], "lo": a[1], "hi": a[2], "n": a[3], "jan_jun_rho": b[0], "jj_lo": b[1], "jj_hi": b[2]})
    corr = pd.DataFrame(rows).set_index("feature")
    corr.to_csv(OUT / "correlations.csv")
    both = corr[["full_year_rho", "jan_jun_rho"]].dropna()
    agree = float(np.corrcoef(both["full_year_rho"], both["jan_jun_rho"])[0, 1]) if len(both) > 2 else np.nan
    md_out += ["\n## 5. Bikeshare changes vs the income change (Spearman with PUMA-cluster 95% interval)\n", md(corr),
               f"\nCorrelation between the full-year and January-June columns across features: {agree:.2f}. "
               "If the half-year version tracks the full-year one, a forecast can be issued by July.\n"]
    (OUT / "eda_summary.md").write_text("\n".join(md_out))

    # figures
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for c, g in u.groupby("city"):
        ax[0].hist(g["dlog_rel"], bins=20, alpha=0.55, label=f"{c} (n={len(g)})", color=CITY_COLORS.get(c))
    ax[0].set(title="Yearly income change vs city median", xlabel="dlog_rel (log points)", ylabel="PUMA-years")
    ax[0].legend()
    ax[1].scatter(u["se_dlog"], u["dlog_rel"].abs(), s=14, alpha=0.6, color="#555")
    ax[1].plot([0, u["se_dlog"].max()], [0, u["se_dlog"].max()], color="#d62728", lw=1)
    ax[1].set(title="Size of the move vs its standard error", xlabel="standard error of dlog", ylabel="|dlog_rel|")
    fig.tight_layout(); fig.savefig(OUT / "fig1_outcome_noise.png", dpi=140); plt.close(fig)

    if sm_path.exists():
        sm = pd.read_csv(sm_path, dtype={"station_id": str})
        fig, ax = plt.subplots(figsize=(7, 4))
        for city, g in sm.groupby("city"):
            m = g.assign(month=g["ym"] % 100).groupby("month")["dep"].sum()
            ax.plot(m.index, m / m.mean(), marker="o", label=city, color=CITY_COLORS.get(city))
        ax.set(title="Trips by calendar month (1 = average month)", xlabel="month", ylabel="relative trips")
        ax.legend(); fig.tight_layout(); fig.savefig(OUT / "fig2_seasonality.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    y = np.arange(len(corr))
    ax.barh(y - 0.2, corr["full_year_rho"], height=0.36, label="full year", color="#1f77b4")
    ax.barh(y + 0.2, corr["jan_jun_rho"], height=0.36, label="Jan-Jun only", color="#ff7f0e")
    ax.errorbar(corr["full_year_rho"], y - 0.2, xerr=[corr["full_year_rho"] - corr["lo"], corr["hi"] - corr["full_year_rho"]], fmt="none", ecolor="k", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(corr.index); ax.axvline(0, color="k", lw=0.8)
    ax.set(title="Spearman with yearly income change (vs city median)", xlabel="rho"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "fig3_feature_correlations.png", dpi=140); plt.close(fig)

    print("\n".join(md_out))
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
