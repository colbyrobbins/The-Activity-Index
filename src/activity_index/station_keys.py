"""One key per physical station across years, even when the bikeshare operator changed its station ids.

Citi Bike and Bluebikes re-issued station ids after 2019 (e.g. Boston '116' became 'A32000'), so 'same station id in both years' fails
and the matched-station features would see every station as new. A retired id and a new id are treated as one station when they are
close (default 75 m), were never active in the same month, and are each other's best available predecessor/successor. Matching is
greedy by distance and one-to-one, so two new stations next to one old station cannot both inherit its history.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371000.0 * np.arcsin(np.sqrt(a))


def canonical_keys(st: pd.DataFrame, radius_m: float = 75.0) -> pd.DataFrame:
    """st: city, station_id, lat, lon, first_ym, last_ym (one row per station id). Returns city, station_id, key, linked (id was linked to
    an older/newer id)."""
    parts = []
    for city, g in st.groupby("city"):
        g = g.reset_index(drop=True)
        n = len(g)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        lat, lon = g["lat"].values, g["lon"].values
        first, last = g["first_ym"].values, g["last_ym"].values
        d = _haversine_m(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
        cand = []
        for i, j in zip(*np.where(np.triu(d <= radius_m, k=1))):
            if last[i] < first[j]:
                cand.append((d[i, j], i, j))          # i retired before j started
            elif last[j] < first[i]:
                cand.append((d[i, j], j, i))
        has_succ, has_pred, linked = set(), set(), set()
        for _, a, b in sorted(cand):
            if a not in has_succ and b not in has_pred:
                has_succ.add(a); has_pred.add(b); linked |= {a, b}
                parent[find(a)] = find(b)
        ids = g["station_id"].astype(str).values
        root_name = {}
        for i in range(n):
            r = find(i)
            root_name[r] = min(root_name.get(r, ids[i]), ids[i])
        parts.append(pd.DataFrame({"city": city, "station_id": ids, "key": [f"{city}:{root_name[find(i)]}" for i in range(n)],
                                   "linked": [i in linked for i in range(n)]}))
    return pd.concat(parts, ignore_index=True)
