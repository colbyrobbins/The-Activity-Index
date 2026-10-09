# The Activity Index

Duke Alternative Data term project. Do bikeshare activity features from 2015-2018 (Citi Bike, Divvy, Bluebikes, Capital Bikeshare) predict which neighbourhoods gain income rank between ACS 2014-18 and 2019-23, beyond what free census data already says?

**Start with [docs/project_summary.md](docs/project_summary.md)** (tract study). The yearly PUMA panel study is in [docs/puma_study.md](docs/puma_study.md) (data, target, cleaning, EDA, baselines and how to read the results). Reasoning behind the design: [docs/project_spec.md](docs/project_spec.md).

## Layout
```
src/activity_index/   package: config, trips, census, crosswalk, features, noise, modeling
scripts/              the pipeline, run in order (below); 11-14 are the PUMA time study
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

## PUMA time study (branch `time-study-data-pipeline`)
```
python scripts/01_download_trips.py --years 2019     # one extra trip year
python scripts/05_aggregate_trips.py --panel
python scripts/11_fetch_puma.py
python scripts/12_build_puma_panel.py
python scripts/13_puma_eda.py
python scripts/14_puma_baselines.py                  # writes results/puma/baseline_summary.csv
python scripts/15_puma_multiyear.py                  # multi-year outcome variant
```

## Nowcast (time-series, PUMA panel)
```
python scripts/22_area_nowcast.py                    # dollar nowcast on the area panel (scripts 20, 05 --areas, 21 first); see docs/nowcast.md
```
Estimates an area's year-T mean household income in dollars from its history and year-T bikeshare record, validated forward in time. (`17_puma_nowcast.py` and `16_levels_nowcast.py` are earlier versions kept for reference.)
