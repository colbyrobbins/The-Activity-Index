## Summary

Final phase of the term project. Stress-tests the mid-semester result (trip growth lowers the income nowcast's average miss from $8,495 to $7,995), turns the nowcast into a decision screen with a demo, and reorganises the repo around the final pipeline.

Full write-up: `docs/final_results.md`.

## What the tests found

All results are forward-validated on 80 area-years (20 areas in Boston, Chicago, DC and New York; test years 2018, 2019, 2023, 2024).

| Check | Result |
|---|---|
| Headline (B2g vs B1) | −$501, 95% interval −$1,020 to +$121 (still includes zero) |
| Placebo: trip growth shuffled between areas | p = 0.001 |
| Placebo: trip growth shuffled between years | p = 0.001 |
| Placebo: best of 19 features, records swapped | p = 0.006 |
| By city and year | Gain in all four cities and all four test years; −$160 without NYC |
| Weather | Explains 47% of city-wide monthly trip growth, 5% of area growth relative to its city; gain with weather-adjusted growth is −$465 |
| Residents vs commuters | Commuter-inflow growth +$3, resident-like growth −$183; gain is −$908 in residential areas, −$168 in job centers |
| Direction of the signal | Negative in every forward fold and all four cities: faster trip growth goes with slower income growth |
| As a decision screen | 61% of ahead/behind calls held vs 57% from income history alone (intervals overlap); ranking by raw trip growth gets 36% |

Not done: additional cities.

## Changes

**New scripts**
- `24_placebo.py`: placebo tests, bootstrap interval, gain by city and year, leave-one-city-out
- `25_fetch_weather.py`: daily weather per city from Open-Meteo
- `26_weather_control.py`: weather-adjusted trip growth, warm vs cold months
- `27_trip_purpose.py`: resident-like vs commuter-inflow trip growth
- `28_decision_screen.py`: the nowcast scored as a ranking decision; direction of the signal; income-level control
- `29_build_demo.py`: builds `demo/activity_index_demo.html`

**New modules** in `src/activity_index/`: `robustness`, `weather`, `trip_purpose`, `decision`

**Demo**: `demo/template.html` (source) and `demo/activity_index_demo.html` (built, self-contained; open in a browser)

**Repo cleanup**
- Tract-study and PUMA-study scripts moved to `scripts/legacy/` with a README (they still run; import paths updated)
- `README.md` rewritten around the final result and pipeline
- `docs/final_results.md` added
- `.gitignore` now tracks the final result tables and figures (`results/robustness/`, `results/decision/`, `results/area_*.csv`); raw and processed data stay ignored
- `area_inputs.py` saves `data/interim/area_station_month.csv.gz` so scripts 26 and 27 run without geopandas

**Tests**: `tests/test_final_checks.py` (5 new tests)

## How to verify

```
python scripts/22_area_nowcast.py
python scripts/24_placebo.py
python scripts/26_weather_control.py
python scripts/27_trip_purpose.py
python scripts/28_decision_screen.py
python scripts/29_build_demo.py
python -m pytest
```

Last run: 85 tests passed. Scripts 26 and 27 need `data/interim/area_station_month.csv.gz` (written by script 21) and script 26 needs `data/raw/weather_daily.csv` (written by script 25).

## Limits

- 20 areas in four cities; the dollar interval includes zero.
- The cause of the negative relationship is untested.
- The warm-month, trip-purpose and area-type splits were run after the main result and are not corrected for multiple comparisons.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_0182kh9462B79fzDfyk6bz1m
