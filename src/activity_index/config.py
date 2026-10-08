"""Project-wide settings. See docs/project_spec.md for the reasoning behind each choice."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW, INTERIM, PROCESSED = DATA / "raw", DATA / "interim", DATA / "processed"
RESULTS = ROOT / "results"

# Two non-overlapping ACS 5-year window pairs. A vintage's last year sets its tract geography:
# vintages ending 2019 or earlier use 2010 tracts, 2022 or later use 2020 tracts, so the
# T1 side must be crosswalked back to 2010 tracts. Select a pair with the ACTIVITY_WINDOW
# environment variable (PowerShell: $env:ACTIVITY_WINDOW = "robustness").
# "robustness" assumes the ACS 2020-24 release is out; confirm against the Census API.
WINDOWS = {
    "primary": {
        "t0": {"year": 2018, "label": "2014-18", "tract_vintage": 2010},
        "t1": {"year": 2023, "label": "2019-23", "tract_vintage": 2020},
        # Alt-data features use trips only up to the end of the T0 window (so they are predictive).
        "activity_years": range(2015, 2019),
    },
    "robustness": {
        "t0": {"year": 2019, "label": "2015-19", "tract_vintage": 2010},
        "t1": {"year": 2024, "label": "2020-24", "tract_vintage": 2020},
        "activity_years": range(2016, 2020),
    },
}
WINDOW = os.environ.get("ACTIVITY_WINDOW", "primary")
if WINDOW not in WINDOWS:
    raise ValueError(f"ACTIVITY_WINDOW must be one of {list(WINDOWS)}, got {WINDOW!r}")
ACS_T0, ACS_T1 = WINDOWS[WINDOW]["t0"], WINDOWS[WINDOW]["t1"]
ACTIVITY_YEARS = WINDOWS[WINDOW]["activity_years"]

# City -> {state FIPS: [county FIPS]}. Service-area approximations; verify before use.
# Tracts without nearby stations are dropped later, so over-inclusive counties are harmless.
CITIES = {
    "nyc": {"36": ["005", "047", "061", "081"]},
    "chicago": {"17": ["031"]},
    "boston": {"25": ["025", "017", "021"]},
    "dc": {"11": ["001"], "51": ["013", "510"]},
}

# Public S3 buckets holding trip files, and the key prefix that selects one year of files.
# Do not redistribute raw files (see licenses in docs/project_spec.md).
TRIP_BUCKETS = {
    "nyc": "https://s3.amazonaws.com/tripdata",
    "chicago": "https://divvy-tripdata.s3.amazonaws.com",
    "boston": "https://s3.amazonaws.com/hubway-data",
    "dc": "https://s3.amazonaws.com/capitalbikeshare-data",
}
TRIP_PREFIXES = {
    "nyc": "{year}",
    "chicago": "Divvy_Trips_{year}",
    "boston": "{year}",
    "dc": "{year}",
}

# GBFS feeds list current stations with coordinates, used when trip files lack them.
# Only stations still active today appear, so retired stations stay unmatched.
# The DC URL is a guess based on the pattern of the others; the check script reports failures.
GBFS = {
    "nyc": "https://gbfs.citibikenyc.com/gbfs/2.3/gbfs.json",
    "chicago": "https://gbfs.divvybikes.com/gbfs/2.3/gbfs.json",
    "boston": "https://gbfs.bluebikes.com/gbfs/gbfs.json",
    "dc": "https://gbfs.capitalbikeshare.com/gbfs/gbfs.json",
}

# ACS variables. Aggregate income and households are additive, so they crosswalk correctly
# between tract vintages (medians do not); mean household income = aggregate / households.
ACS_VARS = {
    "B19025_001E": "agg_household_income",
    "B25003_001E": "households",
    "B25003_003E": "renter_households",
    "B01003_001E": "population",
    "B15003_001E": "edu_total",
    "B15003_022E": "edu_bachelors",
    "B15003_023E": "edu_masters",
    "B15003_024E": "edu_professional",
    "B15003_025E": "edu_doctorate",
    "B25064_001E": "median_gross_rent",
    "B25077_001E": "median_home_value",
    "B01002_001E": "median_age",
    # margins of error (90% level) for the two quantities behind mean household income
    "B19025_001M": "agg_household_income_moe",
    "B25003_001M": "households_moe",
}

# Prior ACS window, used only for the "momentum" baseline (income-rank change BEFORE the study period).
# Same 2010 tract geography as T0, so no crosswalk is needed.
PRIOR_ACS = {"primary": {"year": 2013, "label": "2009-13"},
             "robustness": {"year": 2014, "label": "2010-14"}}[WINDOW]

# Approximate downtown point per city (lat, lon), for the distance-to-center control.
CITY_CENTERS = {"nyc": (40.7580, -73.9855), "chicago": (41.8819, -87.6278),
                "boston": (42.3554, -71.0605), "dc": (38.8977, -77.0365)}

# Trip-level filters: rides outside this range (minutes) are treated as tests, re-docks or lost bikes.
MIN_TRIP_MIN, MAX_TRIP_MIN = 1, 180
# A station counts as "mature" if it has trips in an active month of the first AND the last activity year.
