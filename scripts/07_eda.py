"""Exploratory analysis before any modelling. Writes figures and a numbers-only summary.

Reads data/interim (station_month, trip_qc) and data/processed (stations, tract_table).
Writes results/eda/*.png and results/eda_summary.md (tables only; the interpretation lives in
docs/eda_findings.md). Run from the repo root:  python scripts/07_eda.py
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

warnings.filterwarnings("ignore")
OUT = cfg.RESULTS / "eda"
ID = {"station_id": str}

# categorical slots 1-4 of the reference palette, fixed city order (never re-assigned by rank)
CITY_ORDER = ["nyc", "chicago", "boston", "dc"]
CITY_COLOR = {"nyc": "#2a78d6", "chicago": "#eb6834", "boston": "#1baf7a", "dc": "#eda100"}
CITY_LABEL = {"nyc": "NYC", "chicago": "Chicago", "boston": "Boston", "dc": "DC"}
BLUE, ORANGE, INK, MUTED, GRID, SURFACE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

BIKE = ["trips_per_station_month", "log_growth", "casual_share", "weekend_share", "net_am", "net_pm",
        "round_share", "mean_duration_min", "mean_km", "dest_norm_entropy"]
CONTROLS = ["edu_share_ba", "renter_share", "median_gross_rent", "median_home_value", "median_age",
            "log_density", "km_to_center", "log_households"]
NICE = {"trips_per_station_month": "Trips per station-month", "log_growth": "Usage growth (log, first to last yr)",
        "casual_share": "Casual-rider share", "weekend_share": "Weekend share", "net_am": "AM net flow (+ = origin)",
        "net_pm": "PM net flow (+ = origin)", "round_share": "Round-trip share", "mean_duration_min": "Mean trip duration",
        "mean_km": "Mean trip distance", "dest_norm_entropy": "Destination diversity", "edu_share_ba": "Bachelor's+ share",
        "renter_share": "Renter share", "median_gross_rent": "Median rent", "median_home_value": "Median home value",
        "median_age": "Median age", "log_density": "Population density (log)", "km_to_center": "Distance to downtown",
        "log_households": "Households (log)", "momentum": "Prior rank momentum"}


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlesize": 11,
        "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "font.size": 9, "lines.linewidth": 2, "legend.frameon": False, "figure.dpi": 130,
    })


def md_table(df, floatfmt="{:,.3f}"):
    df = df.reset_index() if df.index.name or isinstance(df.index, pd.MultiIndex) else df
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    rows = []
    for r in df.itertuples(index=False):
        rows.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) and pd.notna(v)
                                      else ("" if pd.isna(v) else str(v)) for v in r) + " |")
    return head + "\n".join(rows) + "\n"


def residualize(y, x, city):
    """Residuals of y after removing city means and a quadratic in initial income rank x."""
    d = pd.get_dummies(city, drop_first=True).astype(float).values
    X = np.column_stack([np.ones(len(y)), x, x ** 2, d])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return y - X @ beta


def spearman(a, b):
    return pd.Series(a).rank().corr(pd.Series(b).rank())


def corr_table(d, feats, target="rank_change", n_boot=300, seed=0):
    """Spearman correlation with the target, raw and after removing initial rank and city (bootstrap CIs)."""
    rng = np.random.default_rng(seed)
    rows = []
    for f in feats:
        s = d[[f, target, "income_rank_t0", "city"]].dropna()
        if len(s) < 50:
            continue
        def stat(s):
            raw = spearman(s[f], s[target])
            adj = spearman(residualize(s[f].rank().values.astype(float), s["income_rank_t0"].values, s["city"].values),
                           residualize(s[target].values, s["income_rank_t0"].values, s["city"].values))
            return raw, adj
        raw, adj = stat(s)
        boots = np.array([stat(s.iloc[rng.integers(0, len(s), len(s))]) for _ in range(n_boot)])
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
        rows.append({"feature": f, "n": len(s), "rho_raw": raw, "raw_lo": lo[0], "raw_hi": hi[0],
                     "rho_adj": adj, "adj_lo": lo[1], "adj_hi": hi[1]})
    return pd.DataFrame(rows)


def line_by_city(ax, wide, ylabel=None, end_labels=True):
    for c in CITY_ORDER:
        if c in wide.columns:
            ax.plot(wide.index, wide[c], color=CITY_COLOR[c], label=CITY_LABEL[c])
            if end_labels:
                ax.annotate(CITY_LABEL[c], (wide.index[-1], wide[c].iloc[-1]), xytext=(5, 0),
                            textcoords="offset points", color=MUTED, va="center", fontsize=8)
    ax.legend(loc="upper left", ncol=4)
    if ylabel:
        ax.set_ylabel(ylabel)


def main():
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    sm = pd.read_csv(cfg.INTERIM / "station_month.csv.gz", dtype=ID)
    qc = pd.read_csv(cfg.INTERIM / "trip_qc.csv")
    tab = pd.read_csv(cfg.PROCESSED / "tract_table.csv", dtype={"GEOID": str})
    st = pd.read_csv(cfg.PROCESSED / "stations.csv", dtype=ID)
    sm["year"], sm["month"] = sm["ym"] // 100, sm["ym"] % 100
    md = ["# EDA summary (numbers only)\n",
          f"Window {cfg.WINDOW}: ACS {cfg.ACS_T0['label']} to {cfg.ACS_T1['label']}, trips {min(cfg.ACTIVITY_YEARS)}-{max(cfg.ACTIVITY_YEARS)}.\n"]

    # ---- 1. data volume and quality ----
    q = qc.groupby("city").agg(rows=("rows", "sum"), with_times=("rows_with_times", "sum"), valid=("rows_valid", "sum"))
    q["valid_share"] = q["valid"] / q["rows"]
    yearly = sm.pivot_table(index="city", columns="year", values="dep", aggfunc="sum")
    md += ["\n## 1. Trips counted\n", md_table(q.join(yearly).reset_index(), "{:,.2f}"),
           "Valid = start in activity years, lasting 1-180 minutes. Rows outside those years are in the raw files but not counted.\n"]
    coords = st.groupby("city").agg(stations_in_tracts=("station_id", "size"),
                                    from_trip_file=("coord_source", lambda s: (s == "trip_file").mean()),
                                    from_station_file=("coord_source", lambda s: (s == "station_file").mean()),
                                    from_gbfs=("coord_source", lambda s: (s == "gbfs").mean()))
    md += ["\n## 2. Stations and where their coordinates came from\n", md_table(coords.reset_index())]

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    line_by_city(ax, (yearly.T / 1e6), "Trips (millions)")
    ax.set_xticks(list(yearly.columns)); ax.set_title("Annual valid trips by system")
    fig.tight_layout(); fig.savefig(OUT / "fig1_trips_by_year.png"); plt.close(fig)

    # ---- 2. seasonality and system growth ----
    monthly = sm.groupby(["city", "year", "month"])["dep"].sum().reset_index()
    monthly["t"] = monthly["year"] + (monthly["month"] - 1) / 12
    idx = monthly.pivot_table(index="t", columns="city", values="dep")
    idx = idx / idx.mean()
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    line_by_city(ax, idx, "Monthly trips / city average", end_labels=False)
    ax.set_xticks(sorted({int(v) for v in idx.index})); ax.set_xticklabels(sorted({int(v) for v in idx.index}))
    ax.set_title("Monthly trips relative to each city's average")
    fig.tight_layout(); fig.savefig(OUT / "fig2_seasonality.png"); plt.close(fig)
    peak = idx.groupby((idx.index % 1 * 12).round().astype(int)).mean()
    md += ["\n## 3. Seasonality (monthly trips / city mean, averaged over years; month 0 = January)\n", md_table(peak.round(2).reset_index())]

    act = sm[sm["dep"] > 0].groupby(["city", "year"])["station_id"].nunique().unstack(0)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    line_by_city(ax, act, "Stations with at least one trip")
    ax.set_xticks(list(act.index)); ax.set_title("Stations with trips, by year")
    fig.tight_layout(); fig.savefig(OUT / "fig3_station_growth.png"); plt.close(fig)
    mat = st.groupby("city").agg(stations=("station_id", "size"), mature=("mature", "mean"), median_trips_per_month=("trips_per_month", "median"))
    md += ["\n## 4. Active stations per year\n", md_table(act.reset_index()), "\n## 5. Stations in tracts and share that are mature\n", md_table(mat.reset_index())]

    # ---- 3. coverage: which tracts are in scope ----
    d = tab[tab["rank_change"].notna()].copy()
    d["scope"] = np.where(d["in_scope"], "in scope", "no mature station")
    cov = d.groupby("city").agg(tracts=("GEOID", "size"), in_scope=("in_scope", "sum"))
    cov["share"] = cov["in_scope"] / cov["tracts"]
    md += ["\n## 6. Tract coverage (tracts with a computable rank change)\n", md_table(cov.reset_index())]
    num = ["income_rank_t0", "edu_share_ba", "renter_share", "median_gross_rent", "log_density", "km_to_center", "median_age"]
    comp = d.groupby("scope")[num].mean().T
    sd = d[num].std()
    comp["std_diff"] = (comp.get("in scope") - comp.get("no mature station")) / sd
    md += ["\n## 7. In-scope vs out-of-scope tracts (means; std_diff = difference in SD units)\n", md_table(comp.reset_index().rename(columns={"index": "variable"}))]
    s = comp["std_diff"].drop("income_rank_t0", errors="ignore").sort_values()
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.barh([NICE.get(k, k) for k in s.index], s.values, color=BLUE, height=0.55)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlabel("In-scope minus out-of-scope tracts (SD units)")
    ax.set_title("In-scope vs out-of-scope tracts, by characteristic")
    fig.tight_layout(); fig.savefig(OUT / "fig4_selection.png"); plt.close(fig)

    # ---- 4. the target ----
    sc = d[d["in_scope"]].copy()
    desc = sc.groupby("city")["rank_change"].describe().round(3)
    md += ["\n## 8. Target: rank change, in-scope tracts\n", md_table(desc.reset_index()),
           f"\nPooled in-scope tracts: {len(sc)}. Share moving more than 0.10 in either direction: {(sc['rank_change'].abs() > 0.10).mean():.2f}\n"]
    fig, axes = plt.subplots(1, 4, figsize=(9.5, 2.8), sharex=True, sharey=True)
    for ax, c in zip(axes, CITY_ORDER):
        v = sc.loc[sc["city"] == c, "rank_change"]
        ax.hist(v, bins=30, color=BLUE, edgecolor=SURFACE, linewidth=0.8)
        ax.set_title(f"{CITY_LABEL[c]} (n={len(v)})", fontsize=9); ax.set_xlabel("Rank change")
    axes[0].set_ylabel("Tracts"); fig.tight_layout(); fig.savefig(OUT / "fig5_target.png"); plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(9.5, 2.9), sharex=True, sharey=True)
    rev = []
    for ax, c in zip(axes, CITY_ORDER):
        v = sc[sc["city"] == c]
        ax.scatter(v["income_rank_t0"], v["rank_change"], s=7, color=BLUE, alpha=0.35, linewidths=0)
        bins = pd.cut(v["income_rank_t0"], np.linspace(0, 1, 11))
        m = v.groupby(bins, observed=True).agg(x=("income_rank_t0", "mean"), y=("rank_change", "mean"))
        ax.plot(m["x"], m["y"], color=ORANGE, lw=2); ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(CITY_LABEL[c], fontsize=9); ax.set_xlabel("Initial income rank")
        rev.append({"city": c, "spearman_rank_t0_vs_change": spearman(v["income_rank_t0"], v["rank_change"])})
    axes[0].set_ylabel("Rank change"); fig.suptitle("Rank change vs initial income rank (orange = decile means)", x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "fig6_mean_reversion.png"); plt.close(fig)
    md += ["\n## 9. Mean reversion: Spearman(initial rank, rank change)\n", md_table(pd.DataFrame(rev))]
    if "momentum" in sc:
        mm = sc[["momentum", "rank_change"]].dropna()
        md += [f"\nSpearman(prior momentum, rank change), pooled: {spearman(mm['momentum'], mm['rank_change']):.3f}\n"]

    # ---- 5. bikeshare features vs target and vs income level ----
    miss = sc[BIKE].isna().mean().round(3).rename("share_missing").reset_index().rename(columns={"index": "feature"})
    md += ["\n## 10. Bikeshare feature summary (in-scope tracts)\n", md_table(sc[BIKE].describe().T.reset_index().rename(columns={"index": "feature"})),
           "\nMissing share:\n", md_table(miss)]
    level = pd.DataFrame({f: {"spearman_with_income_rank_t0": spearman(sc[[f, "income_rank_t0"]].dropna()[f], sc[[f, "income_rank_t0"]].dropna()["income_rank_t0"])} for f in BIKE}).T
    md += ["\n## 11. Cross-section: do bikeshare features track income LEVEL? (Spearman with initial rank)\n", md_table(level.reset_index().rename(columns={"index": "feature"}))]

    feats = BIKE + CONTROLS + (["momentum"] if "momentum" in sc else [])
    ct = corr_table(sc, feats)
    ct["label"] = ct["feature"].map(NICE)
    md += ["\n## 12. Correlation with rank change: raw and after removing initial rank (quadratic) and city\n",
           md_table(ct[["feature", "n", "rho_raw", "raw_lo", "raw_hi", "rho_adj", "adj_lo", "adj_hi"]]),
           "\nBootstrap 95% intervals, 300 resamples of tracts. rho_adj is the number that matters for the project.\n"]
    ct = ct.sort_values("rho_adj")
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    y = np.arange(len(ct))
    ax.barh(y - 0.2, ct["rho_raw"], height=0.36, color=BLUE, label="Raw")
    ax.barh(y + 0.2, ct["rho_adj"], height=0.36, color=ORANGE, label="After removing initial rank and city")
    ax.errorbar(ct["rho_adj"], y + 0.2, xerr=[ct["rho_adj"] - ct["adj_lo"], ct["adj_hi"] - ct["rho_adj"]], fmt="none", ecolor=MUTED, elinewidth=1, capsize=2)
    ax.set_yticks(y); ax.set_yticklabels(ct["label"]); ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlabel("Spearman correlation with rank change"); ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=2)
    ax.set_title("What predicts income-rank change? (in-scope tracts)")
    fig.tight_layout(); fig.savefig(OUT / "fig7_correlations.png"); plt.close(fig)

    # ---- 6. quartile view of the headline features ----
    qrows = []
    for f in ("trips_per_station_month", "log_growth", "casual_share", "net_am"):
        s2 = sc[[f, "rank_change", "city"]].dropna().copy()
        s2["q"] = s2.groupby("city")[f].transform(lambda x: pd.qcut(x.rank(method="first"), 4, labels=False) + 1)
        for qn, g in s2.groupby("q"):
            qrows.append({"feature": f, "quartile_within_city": int(qn), "n": len(g), "mean_rank_change": g["rank_change"].mean(),
                          "share_top_quartile_movers": (g["rank_change"] >= sc["rank_change"].quantile(0.75)).mean()})
    md += ["\n## 13. Mean rank change by within-city quartile of each feature\n", md_table(pd.DataFrame(qrows))]

    (cfg.RESULTS / "eda_summary.md").write_text("".join(md), encoding="utf-8")
    ct.to_csv(OUT / "correlations.csv", index=False)
    print("\n".join(md))
    print(f"\nfigures in {OUT}; summary in {cfg.RESULTS / 'eda_summary.md'}")


if __name__ == "__main__":
    main()
