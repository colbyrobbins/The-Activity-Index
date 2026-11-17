# The Activity Index

Duke Alternative Data (AIPI) term project. Can public bikeshare trip data nowcast a neighbourhood's household income before the Census
Bureau posts it?

**Result in one paragraph.** For 20 areas in Boston, Chicago, DC and New York, adding one bikeshare feature (matched-station trip
growth) to an income-history baseline lowers the average nowcast miss from $8,495 to $7,995, about eight months before the ACS 1-year
figure is posted. The gain passes placebo tests (p = 0.001 to 0.006), survives a weather adjustment and shows up in every city and test
year, but its interval across areas still includes zero, and the signal is negative: faster trip growth goes with slower income growth.
Full write-up: **[docs/final_results.md](docs/final_results.md)**. Demo: open `demo/activity_index_demo.html` in a browser.

## How the project got here
| Stage | Question | Outcome | Write-up |
|---|---|---|---|
| 1. Tract study | Do 2015-18 bikeshare features predict a tract's income-rank change over five years? | No robust gain over free census data; 60% of the target is survey noise | [project_summary](docs/project_summary.md), [eda_findings](docs/eda_findings.md), [noise_ceiling](docs/noise_ceiling.md) |
| 2. PUMA yearly study | Does yearly bikeshare change track yearly income change? | No bikeshare group beat income history | [puma_study](docs/puma_study.md) |
| 3. Dollar nowcast (final) | Can year-T trips nowcast year-T income in dollars? | Trip growth helps a little; stress-tested | [nowcast](docs/nowcast.md), [final_results](docs/final_results.md) |

The project began as a Strava-based idea; without Strava Metro access it moved to open bikeshare trip data
(Citi Bike, Divvy, Capital Bikeshare, Bluebikes). Original design reasoning: [project_spec](docs/project_spec.md).

## Layout
```
src/activity_index/   package (data loading, features, panels, models, and the final checks: robustness, weather, trip_purpose, decision)
scripts/              the final pipeline, run in number order (below)
scripts/legacy/       stages 1 and 2, kept for the record (see scripts/legacy/README.md)
tests/                offline unit tests
docs/                 write-ups; final_results.md is the one to read
demo/                 the area screen (template.html is the source, activity_index_demo.html is built by script 29)
results/              final tables and figures (robustness/, decision/, area_*.csv); everything else is gitignored
data/                 raw / interim / processed (gitignored: bikeshare licenses forbid redistributing trip files)
```

## Setup
```
python -m venv .venv ; .venv\Scripts\activate
pip install -r requirements.txt
# create a file named .env containing the line CENSUS_API_KEY=your_key (gitignored; never commit it)
```

## Run the final pipeline (from the repo root)
```
python scripts/01_download_trips.py --years 2015 2016 2017 2018 2019 2022 2023 2024    # trip files (tens of GB; NYC 2024 alone is 8.7 GB)
python scripts/05_aggregate_trips.py --panel       # station-month tables for the PUMA panel
python scripts/05_aggregate_trips.py --areas       # same for the nowcast years
python scripts/11_fetch_puma.py                    # ACS 1-year by PUMA + 2010 PUMA polygons
python scripts/12_build_puma_panel.py              # PUMA panel (the area builder reads it)
python scripts/20_build_areas.py                   # areas that can be followed across the 2020 PUMA redraw
python scripts/21_build_area_panel.py              # area-year panel
python scripts/22_area_nowcast.py                  # the nowcast: B0, B1, B2g, ... in dollars
python scripts/23_area_gains.py                    # gain per area
python scripts/24_placebo.py                       # placebo tests, interval, by city / year
python scripts/25_fetch_weather.py                 # daily weather (Open-Meteo, no key)
python scripts/26_weather_control.py               # weather-adjusted trip growth, warm vs cold months
python scripts/27_trip_purpose.py                  # resident-like vs commuter-inflow trips
python scripts/28_decision_screen.py               # the screen scored as a decision; which way the signal points
python scripts/29_build_demo.py                    # demo/activity_index_demo.html
python -m pytest                                   # unit tests
```
`18_check_new_years.py` and `19_check_puma_boundaries.py` are the pre-flight checks that were run before extending the data to 2024.

## Data sources
Citi Bike, Divvy, Capital Bikeshare and Bluebikes public trip files; ACS 1-year and 5-year estimates and TIGER boundaries
(US Census Bureau); Open-Meteo historical weather (ERA5). Links are in [project_spec](docs/project_spec.md) and the script headers.
