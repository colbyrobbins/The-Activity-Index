"""Build the demo page: the area screen a site-selection or acquisitions team would open (one self-contained HTML file).

Needs scripts 20 (areas.gpkg, areas.csv, area_acs.csv) and 28 (results/decision/screen.csv, decision_metrics.csv).
Pick a city and a year: the map and table show each area's last posted income, the nowcast made before the ACS posting, and the call
(pulling ahead of / falling behind its city). "Reveal the ACS posting" shows what was published months later.
No GIS library needed: the area polygons are read straight from the GeoPackage (SQLite + WKB).
Run from the repo root:  python scripts/29_build_demo.py      Writes demo/activity_index_demo.html (open it in a browser).
"""
import json
import re
import sqlite3
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from activity_index import config as cfg  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CITY_NAMES = {"nyc": "New York", "chicago": "Chicago", "boston": "Boston", "dc": "Washington DC"}


def read_rings(blob: bytes, max_points: int = 160) -> list:
    """Outer and inner rings of a GeoPackage MULTIPOLYGON blob as [[lon, lat], ...], thinned to about max_points each."""
    env = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[(blob[3] >> 1) & 7]
    b, o, out = blob[8 + env:], 0, []

    def rd(n=1):
        nonlocal o
        bo = "<" if b[o] == 1 else ">"
        vals = struct.unpack_from(bo + "I" * n, b, o + 1)
        o += 1 + 4 * n
        return bo, vals
    _, (typ, npoly) = rd(2)
    assert typ == 6, f"expected MULTIPOLYGON, got WKB type {typ}"
    for _ in range(npoly):
        bo, (_, nrings) = rd(2)
        for _ in range(nrings):
            (n,) = struct.unpack_from(bo + "I", b, o)
            pts = np.frombuffer(b, dtype=bo + "f8", count=2 * n, offset=o + 4).reshape(n, 2)
            o += 4 + 16 * n
            if n >= 40:                                            # drop slivers and tiny islands
                out.append(np.round(pts[:: max(1, n // max_points)], 5).tolist())
    return out


def short_name(names: list[str]) -> str:
    """Readable area name from its 2020 PUMA names, e.g. 'NYC-Manhattan Community District 1 & 2--Financial District ... PUMA, New York'."""
    parts = []
    for n in names:
        n = re.sub(r" PUMA.*$", "", n)
        n = re.sub(r"^NYC-(\w[\w ]*?) Community Districts? [\d, &]+--", r"\1: ", n)
        n = re.sub(r"^Chicago City \(([^)]*)\)--", r"\1: ", n)
        n = re.sub(r"^Boston City--", "", n)
        parts.append(n.strip())
    parts = list(dict.fromkeys(parts))
    if len(parts) > 3 and all(x.startswith("District of Columbia") for x in parts):
        return "District of Columbia (excluding West)"
    heads = {x.split(": ")[0] for x in parts if ": " in x}
    if len(parts) > 1 and len(heads) == 1 and all(": " in x for x in parts):          # "Manhattan: A; Manhattan: B" -> "Manhattan: A; B"
        return heads.pop() + ": " + "; ".join(x.split(": ", 1)[1] for x in parts)
    return "; ".join(parts)


def main():
    screen = pd.read_csv(cfg.RESULTS / "decision" / "screen.csv", dtype={"GEOID": str})
    met = pd.read_csv(cfg.RESULTS / "decision" / "decision_metrics.csv").set_index("screen")
    mem = pd.read_csv(cfg.PROCESSED / "areas.csv", dtype=str)
    puma = pd.read_csv(cfg.RAW / "acs1_puma_post.csv", dtype=str)
    puma = puma[puma["year"].astype(int) >= 2022].drop_duplicates("GEOID").set_index("GEOID")["NAME"]      # 2020-vintage PUMA names only
    names = {a: short_name([puma.get(g, g) for g in t.loc[t["line"] == "2020", "GEOID"]]) for a, t in mem.groupby("area_id")}
    acs = pd.read_csv(cfg.PROCESSED / "area_acs.csv", dtype={"GEOID": str})
    acs["income"] = acs["agg_household_income"] / acs["households"]
    con = sqlite3.connect(str(cfg.PROCESSED / "areas.gpkg"))
    geo = {aid: read_rings(g) for g, aid in con.execute("select geom, area_id from areas")}

    areas = {}
    for aid in sorted(screen["GEOID"].unique()):
        s = screen[screen["GEOID"] == aid]
        areas[aid] = {"city": s["city"].iloc[0], "name": names.get(aid, aid), "rings": geo[aid],
                      "history": [[int(y), round(v)] for y, v in acs[acs["GEOID"] == aid].sort_values("year")[["year", "income"]].values],
                      "rows": {int(r.year): {"last": round(r.last_posted_income), "now": round(r.nowcast_income), "b1": round(r.nowcast_income_b1),
                                             "posted": round(r.posted_income), "margin": round(r.posting_margin_90),
                                             "trips": round(float(r.trip_growth), 4), "tripsRel": round(float(r.trip_growth_vs_city), 4),
                                             "rank": int(r.rank_pred), "n": int(r.n_city_year), "call": r.call, "outcome": r.outcome}
                                for r in s.itertuples()}}
    b1, b2 = met.loc["income history only (B1)"], met.loc["income history + trip growth (B2g)"]
    data = {"areas": areas, "cities": CITY_NAMES, "years": sorted(int(y) for y in screen["year"].unique()),
            "metrics": {"dir": [float(b1["direction_accuracy"]), float(b2["direction_accuracy"])],
                        "spread": [float(b1["top_third_minus_bottom_third"]), float(b2["top_third_minus_bottom_third"])],
                        "n": int(b2["area_years"])}}
    html = (ROOT / "demo" / "template.html").read_text(encoding="utf-8").replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
    out = ROOT / "demo" / "activity_index_demo.html"
    out.write_text(html, encoding="utf-8")
    print("wrote", out, f"({len(html) / 1024:.0f} KB, {len(areas)} areas)")
    for a, v in areas.items():
        print(f"  {a:<18} {v['name'][:90]}")


if __name__ == "__main__":
    main()
