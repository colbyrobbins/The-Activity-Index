"""ACS tract data (Census API) and TIGER tract polygons."""
from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

try:  # optional: read CENSUS_API_KEY from a git-ignored .env file in the repo root
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

USER_AGENT ="the-activity-index/0.1 (student research project)"


def acs_url(year: int, state: str, counties: list[str], variables: dict, key: str | None = None) -> str:
    """Request URL in the form the Census docs use (spaces as %20, ':' ',' '*' left readable)."""
    params = {
        "get": "NAME," + ",".join(variables),
        "for": "tract:*",
        "in": f"state:{state} county:{','.join(counties)}",
    }
    if key:
        params["key"] = key
    query = "&".join(f"{k}={quote(v, safe=':,*')}" for k, v in params.items())
    return f"https://api.census.gov/data/{year}/acs/acs5?{query}"


def fetch_acs(year: int, state: str, counties: list[str], variables: dict, timeout: int = 60) -> pd.DataFrame:
    """ACS 5-year tract table for the given counties. `variables` maps API name -> column name."""
    url = acs_url(year, state, counties, variables, os.environ.get("CENSUS_API_KEY"))
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    head = r.text[:600]
    if "missing_key" in r.url or "Missing Key" in head or "Invalid Key" in head or "invalid_key" in r.url:
        sent = bool(os.environ.get("CENSUS_API_KEY"))
        if not sent:
            raise RuntimeError(
                "No CENSUS_API_KEY found. Get a free key at https://api.census.gov/data/key_signup.html and put "
                "CENSUS_API_KEY=<key> in the repo-root .env file (or set $env:CENSUS_API_KEY).")
        raise RuntimeError(
            "Census rejected the key that was sent (key length "
            f"{len(os.environ['CENSUS_API_KEY'])}). Click the activation link in the key email, then retry; "
            "if it still fails, request a new key. Response page: " + r.url)
    try:
        rows = r.json()
    except ValueError:
        raise RuntimeError(
            f"Census API did not return JSON (HTTP {r.status_code}, "
            f"content-type {r.headers.get('content-type')!r}) for {r.url}\n"
            f"response body starts: {r.text[:300]!r}") from None
    if r.status_code != 200:
        raise RuntimeError(f"Census API HTTP {r.status_code} for {r.url}: {str(rows)[:300]}")
    df = pd.DataFrame(rows[1:], columns=rows[0])
    df["GEOID"] = df["state"] + df["county"] + df["tract"]
    for v in variables:
        df[v] = pd.to_numeric(df[v], errors="coerce")
        df.loc[df[v] < -1e6, v] = np.nan  # Census sentinel for "not available"
    return df.rename(columns=variables)


def tiger_url(year: int, state: str) -> str:
    return f"https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/tl_{year}_{state}_tract.zip"


def fetch_tracts(year: int, state: str, counties: list[str], raw_dir, timeout: int = 180):
    """Tract polygons for the given counties (downloads and caches the state zip)."""
    import geopandas as gpd  # imported lazily so ACS-only code does not need geopandas

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / f"tl_{year}_{state}_tract.zip"
    if not zip_path.exists():
        r = requests.get(tiger_url(year, state), headers={"User-Agent": USER_AGENT}, timeout=timeout)
        r.raise_for_status()
        zip_path.write_bytes(r.content)
    gdf = gpd.read_file(zip_path)
    return gdf[gdf["COUNTYFP"].isin(counties)][["GEOID", "ALAND", "geometry"]].copy()
