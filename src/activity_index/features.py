"""Turn normalized trips into station-month counts, origin-destination counts and tract features.

Everything here is plain pandas, so the heavy trip pass (scripts/05) needs no GIS libraries.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MEMBER_TYPES = {"subscriber", "member"}
AM_HOURS, PM_HOURS = (7, 8, 9), (16, 17, 18)

DEP_COLS = ["dep", "dep_member", "dep_weekday", "dep_am", "dep_pm", "dep_round", "dur_sum"]
ARR_COLS = ["arr", "arr_am", "arr_pm"]


def aggregate_trips(t: pd.DataFrame, years, min_min: float = 1, max_min: float = 180) -> dict:
    """Aggregate one batch of canonical trips.

    Returns {'station_month', 'od_year', 'names', 'qc'}:
      station_month: city-free table keyed (ym, station_id) with departure and arrival counts
      od_year:       (year, start_id, end_id, n) trip counts
      names:         per start station: name, median in-file coordinates, trips
      qc:            one-row dict of row counts, to report data quality
    Counts include only trips that start in `years` and last between min_min and max_min minutes.
    """
    qc = {"rows": int(len(t)), "rows_with_times": 0, "rows_valid": 0, "start_min": None, "start_max": None}
    t = t[t["start_time"].notna() & t["end_time"].notna()]
    qc["rows_with_times"] = int(len(t))
    if len(t):
        qc["start_min"], qc["start_max"] = str(t["start_time"].min()), str(t["start_time"].max())
    dur = (t["end_time"] - t["start_time"]).dt.total_seconds() / 60.0
    t = t.assign(dur=dur)
    t = t[(t["dur"] >= min_min) & (t["dur"] <= max_min) & t["start_time"].dt.year.isin(list(years))
          & t["start_station_id"].notna()]
    qc["rows_valid"] = int(len(t))
    empty = {"station_month": pd.DataFrame(columns=["ym", "station_id"] + DEP_COLS + ARR_COLS),
             "od_year": pd.DataFrame(columns=["year", "start_id", "end_id", "n"]),
             "names": pd.DataFrame(columns=["station_id", "name", "lat", "lon", "n"]), "qc": qc}
    if t.empty:
        return empty

    st, en = t["start_time"], t["end_time"]
    f = pd.DataFrame({
        "ym": (st.dt.year * 100 + st.dt.month).astype("int32"),
        "s": t["start_station_id"].astype("string"),
        "e": t["end_station_id"].astype("string"),
        "member": t["user_type"].astype("string").str.lower().isin(MEMBER_TYPES),
        "weekday": st.dt.dayofweek < 5,
        "am": st.dt.hour.isin(AM_HOURS),
        "pm": st.dt.hour.isin(PM_HOURS),
        "dur": t["dur"].astype("float32"),
        "one": np.int8(1),
    })
    f["round"] = f["s"] == f["e"]
    dep = f.groupby(["ym", "s"]).agg(dep=("one", "sum"), dep_member=("member", "sum"),
                                     dep_weekday=("weekday", "sum"), dep_am=("am", "sum"),
                                     dep_pm=("pm", "sum"), dep_round=("round", "sum"),
                                     dur_sum=("dur", "sum")).reset_index().rename(columns={"s": "station_id"})

    a = pd.DataFrame({"ym": (en.dt.year * 100 + en.dt.month).astype("int32"), "e": f["e"], "one": np.int8(1),
                      "am": en.dt.hour.isin(AM_HOURS), "pm": en.dt.hour.isin(PM_HOURS)})
    a = a[a["e"].notna() & a["ym"].between(f["ym"].min(), f["ym"].max() + 1)]
    arr = (a.groupby(["ym", "e"]).agg(arr=("one", "sum"), arr_am=("am", "sum"), arr_pm=("pm", "sum"))
           .reset_index().rename(columns={"e": "station_id"}))
    sm = dep.merge(arr, on=["ym", "station_id"], how="outer")
    num = DEP_COLS + ARR_COLS
    sm[num] = sm[num].fillna(0)

    od = (f.assign(year=f["ym"] // 100).dropna(subset=["e"]).groupby(["year", "s", "e"]).size()
          .rename("n").reset_index().rename(columns={"s": "start_id", "e": "end_id"}))

    g = t.groupby("start_station_id")
    names = pd.DataFrame({
        "name": g["start_station_name"].agg(lambda s: s.dropna().iloc[0] if s.notna().any() else pd.NA),
        "lat": g["start_lat"].median(), "lon": g["start_lon"].median(), "n": g.size()}).reset_index()
    names = names.rename(columns={"start_station_id": "station_id"})
    return {"station_month": sm, "od_year": od, "names": names, "qc": qc}


def combine_parts(parts: list[dict]) -> dict:
    """Sum the per-file aggregates (a month can span several files, e.g. split CSV parts)."""
    sm = pd.concat([p["station_month"] for p in parts], ignore_index=True)
    sm = sm.groupby(["ym", "station_id"], as_index=False).sum(numeric_only=True)
    od = pd.concat([p["od_year"] for p in parts], ignore_index=True)
    od = od.groupby(["year", "start_id", "end_id"], as_index=False)["n"].sum()
    nm = pd.concat([p["names"] for p in parts], ignore_index=True)
    nm = nm.sort_values("n", ascending=False)
    names = nm.groupby("station_id").agg(name=("name", "first"), lat=("lat", "median"),
                                         lon=("lon", "median"), n=("n", "sum")).reset_index()
    qc = pd.concat([p["qc"] if isinstance(p["qc"], pd.DataFrame) else pd.DataFrame([p["qc"]]) for p in parts],
                   ignore_index=True)
    return {"station_month": sm, "od_year": od, "names": names, "qc": qc}


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    h = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(h))


def mature_stations(sm: pd.DataFrame, years) -> pd.DataFrame:
    """Per station: active months in the first and last activity year and whether it is 'mature'
    (active in both), so growth in usage is not confounded with the system adding stations."""
    first, last = min(years), max(years)
    sm = sm.assign(year=sm["ym"] // 100)
    act = sm[sm["dep"] > 0].groupby(["station_id", "year"])["ym"].nunique().unstack(fill_value=0)
    for y in (first, last):
        if y not in act.columns:
            act[y] = 0
    out = pd.DataFrame({"months_first": act[first], "months_last": act[last]})
    out["mature"] = (out["months_first"] > 0) & (out["months_last"] > 0)
    return out.reset_index()


def station_features(sm: pd.DataFrame, od: pd.DataFrame, coords: pd.DataFrame, years) -> pd.DataFrame:
    """Per-station features over the activity window. `coords` has station_id, lat, lon.

    All rates are per active station-month (months with at least one departure), so a station that
    opened late is not penalised for months it did not exist.
    """
    first, last = min(years), max(years)
    sm = sm.assign(year=sm["ym"] // 100)
    pooled = sm.groupby("station_id")[DEP_COLS + ARR_COLS].sum()
    active = sm[sm["dep"] > 0].groupby("station_id")["ym"].nunique().rename("active_months")
    yr = sm[sm["dep"] > 0].groupby(["station_id", "year"]).agg(trips=("dep", "sum"),
                                                              months=("ym", "nunique")).reset_index()
    rate = yr.assign(rate=yr["trips"] / yr["months"]).pivot(index="station_id", columns="year", values="rate")
    out = pooled.join(active).join(mature_stations(sm, years).set_index("station_id"))
    rate = rate.reindex(columns=[first, last])
    out["rate_first"] = rate[first]
    out["rate_last"] = rate[last]
    out["trips_per_month"] = out["dep"] / out["active_months"]
    out["log_growth"] = np.log(out["rate_last"] / out["rate_first"])
    d = out["dep"].replace(0, np.nan)
    out["member_share"] = out["dep_member"] / d
    out["weekday_share"] = out["dep_weekday"] / d
    out["round_share"] = out["dep_round"] / d
    out["mean_duration_min"] = out["dur_sum"] / d
    for part, col_d, col_a in (("am", "dep_am", "arr_am"), ("pm", "dep_pm", "arr_pm")):
        tot = (out[col_d] + out[col_a]).replace(0, np.nan)
        out[f"net_{part}"] = (out[col_d] - out[col_a]) / tot      # >0: mostly a departure point in that peak

    # origin-destination features
    o = od.merge(coords.rename(columns={"station_id": "start_id", "lat": "slat", "lon": "slon"}), on="start_id")
    o = o.merge(coords.rename(columns={"station_id": "end_id", "lat": "elat", "lon": "elon"}), on="end_id")
    o = o[o["start_id"] != o["end_id"]]
    o["km"] = haversine_km(o["slat"], o["slon"], o["elat"], o["elon"])
    o["w"] = o["n"] * o["km"]
    gi = o.groupby("start_id")
    out["mean_km"] = (gi["w"].sum() / gi["n"].sum()).reindex(out.index)
    share = o["n"] / gi["n"].transform("sum")
    o["ent"] = -share * np.log(share)
    n_dest = gi["end_id"].nunique()
    out["dest_entropy"] = gi["ent"].sum().reindex(out.index)
    out["dest_norm_entropy"] = (out["dest_entropy"] / np.log(n_dest.reindex(out.index))).where(n_dest.reindex(out.index) > 1)
    out.index.name = "station_id"
    return out.reset_index()


def weighted_mean(df: pd.DataFrame, cols: list[str], weight: str) -> pd.Series:
    w = df[weight]
    return pd.Series({c: np.average(df[c].dropna(), weights=w[df[c].notna()]) if df[c].notna().any() and w[df[c].notna()].sum() > 0 else np.nan
                      for c in cols})


def tract_features(stations: pd.DataFrame) -> pd.DataFrame:
    """Aggregate station features (rows have a `tract` column) to one row per tract.

    Mature stations only for the rate/growth features; every share is weighted by the station's trips.
    """
    rows = []
    for tract, g in stations.groupby("tract"):
        m = g[g["mature"]]
        r = {"tract": tract, "n_stations": len(g), "n_mature": len(m), "trips_total": g["dep"].sum()}
        if len(m):
            r["trips_per_station_month"] = m["dep"].sum() / m["active_months"].sum()
            r["log_growth"] = np.log(m["rate_last"].sum() / m["rate_first"].sum())
            r.update(weighted_mean(m, ["member_share", "weekday_share", "round_share", "mean_duration_min",
                                       "net_am", "net_pm", "mean_km", "dest_norm_entropy"], "dep"))
            r["casual_share"] = 1 - r["member_share"]
            r["weekend_share"] = 1 - r["weekday_share"]
        rows.append(r)
    return pd.DataFrame(rows)
