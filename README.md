# The Activity Index

Duke Alternative Data term project. Do bikeshare activity features from 2015-2018 (Citi Bike, Divvy, Bluebikes, Capital Bikeshare) predict which neighbourhoods gain income rank between ACS 2014-18 and 2019-23, beyond what free census data already says?

**Start with [docs/project_summary.md](docs/project_summary.md)** (data, target, cleaning, EDA, baselines and how to read the results). Reasoning behind the design: [docs/project_spec.md](docs/project_spec.md).

## Layout
```
src/activity_index/   package: config, trips, census, crosswalk, features, noise, modeling
scripts/              the pipeline, run in order (below)
tests/                offline unit tests
docs/                 project_summary (read first), project_spec, eda_findings, noise_ceiling
data/                 raw / interim / processed (gitignored; bikeshare licenses forbid redistribution)
results/              tables and figures (gitignored)
```

## Setup
```
python -m venv .venv ; .venv\Scripts\activate
pip install -r requirements.txt
# create a file named .env containing the line CENSUS_API_KEY=your_key (gitignored; never commit it)
```

## Run (from the repo root)
```
python scripts/01_download_trips.py       # trip files (about 5 GB)
python scripts/02_fetch_acs_tiger.py      # ACS tables + tract polygons (also: --prior-only)
python scripts/04_build_crosswalk.py      # 2020 tracts -> 2010 tracts
python scripts/05_aggregate_trips.py      # trips -> station-month tables (resumable)
python scripts/06_build_tract_table.py    # tract features and target
python scripts/07_eda.py                  # figures + results/eda_summary.md
python scripts/09_noise_ceiling.py        # ACS margins of error -> noise in the target
python scripts/10_baselines.py            # baselines; writes results/baseline_summary.csv
python -m pytest                          # unit tests
```
`results/baseline_summary.csv` is the one table to look at (Chicago, NYC, DC; Boston is excluded because of noise, see the summary doc; `baseline_summary_with_boston.csv` is the sensitivity run); `baselines_metrics.csv` and `baselines_lift.csv` hold the full detail (LightGBM, unweighted fits, no-Boston sample, per-city scores).
Robustness window (ACS 2015-19 to 2020-24): `ACTIVITY_WINDOW=robustness` (PowerShell: `$env:ACTIVITY_WINDOW = "robustness"`).
