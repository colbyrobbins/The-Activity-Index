# Nowcast: time-series design (PUMA panel)

## What the question is
In year T, can a PUMA's trailing-12-month bikeshare record tell us its year-T income before the ACS publishes it, beyond what its own published history already says? Rows are PUMA-years; this is a time study, not a cross-section.

## Design (`src/activity_index/puma_nowcast.py`, `scripts/17_puma_nowcast.py`)
- Label: ACS 1-year log mean household income of year T, relative to the city's median PUMA that year.
- No static descriptors (density, distance to center, size): they do not change over time, so they only re-describe the PUMA.
- Own published history: income of T-1, last year's change. Free census: education, renters, rent, home value, age as of T-1.
- Bikeshare: year-T level of each rate, the year-over-year change, and the deviation from the PUMA's own average over earlier years only (the test year never enters its own baseline).
- Rungs: T0 last published, T1 + momentum, T2 + free census, T3 bikeshare only, T4 = T1 + bikeshare (the nowcast question), T5 = T2 + bikeshare, T4_<group>.
- Validation: forward in time (train on earlier years, test year Y; the last year is a true nowcast) is the headline; held-out PUMAs as a secondary check.
- With last year's income in the model, level and change are the same prediction. The model predicts the change vs last published directly (level = prev + change). Scoring the level minus prev instead gave a biased gain (+0.10 on pure noise); a test guards against it.

## Decisions (agreed)
- Target: the PUMA-area's mean household income in DOLLARS for year T; success = how close the predicted average is to the posted one (MAE in $, median % error). A predicted change is prediction minus last year's posting. Also reported: the error after removing the city-wide shift of that year (the part bikeshare could add), and each model's error next to the ACS sampling-noise floor.
- COVID: no 2020 or 2021 targets and no 2020/2021 trips. 2021 income is used only as an input (momentum for 2023). 2022 is a baseline year only (the "last published" figure for 2023).
- Target years: 2016-2019, 2023, 2024. 2025 is not used.
- Geography: ACS 1-year uses 2010 PUMAs through 2021 and 2020 PUMAs from 2022, and most panel PUMAs were renumbered, merged or split. `scripts/19_check_puma_boundaries.py` showed all 35 panel PUMAs sit inside groups that cover the same ground before and after (24 areas, 20 with every member in the panel). `scripts/20_build_areas.py` builds those areas and sums the additive ACS aggregates (income, households, counts) over members of the matching vintage.

## Build order
1. Areas and their income series (script 20; no trip downloads). Done when the continuity table looks sane across 2021 to 2022.
2. Trips for 2022-2024 only (NYC 2024 alone is 8.7 GB), with stations matched by location because NYC and Boston changed their station ids.
3. Area-year bikeshare panel, then the nowcast on dollars.

## Run (current pipeline: areas, dollars)
    python scripts/20_build_areas.py
    python scripts/01_download_trips.py --years 2022 2023 2024 --cities dc boston chicago nyc
    python scripts/05_aggregate_trips.py --areas
    python scripts/21_build_area_panel.py
    python scripts/22_area_nowcast.py         # B0 last posting x city trend; B1 + momentum; F_* + bikeshare; dollar error, forward validation

Older PUMA-level versions, kept for reference:
    python -m pytest
    python scripts/17_puma_nowcast.py       # needs script 12
Writes results/puma/nowcast_summary.csv and nowcast_gains.csv.

`scripts/16_levels_nowcast.py` (tract level, cross-sectional) is kept only as a side check; it is not the time study.
