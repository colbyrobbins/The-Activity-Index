# PUMA time study: yearly bikeshare change vs yearly income change

Branch `time-study-data-pipeline`. The tract study ([project_summary.md](project_summary.md)) had one before/after pair per tract
and level features. This study builds a panel: each PUMA appears once per year, with the **change in bikeshare use during the
year** predicting the **change in income over the same year**.

## Design decisions
| Question | Decision | Why |
|---|---|---|
| Outcome geography | PUMAs (about 100,000 people) | ACS 1-year estimates exist for them every year (except 2020). Tracts only get 5-year averages. |
| Outcome | `dlog_rel` = change in log mean household income from year-1 to year, minus the city's median change that year | Mean income = aggregate income / households. The city median removes recessions and city-wide trends, like the within-city rank in the tract study. Noise comes from the ACS margins of error, as before. |
| Feature timing | **Year-over-year on matched months**, not month-to-month | (1) The outcome is one annual average (ACS pools interviews over all twelve months), so a monthly feature has no monthly target. (2) Month-to-month movement is dominated by season and weather, not income; `13_puma_eda.py` measures this (the calendar month alone explains most within-year variation in monthly trips). (3) About 12x fewer features, which matters with a few hundred rows. (4) Comparing the same calendar months in both years cancels season and weather exactly, and counting only stations with trips in that month in both years stops a new or closed station from looking like a trend. |
| "Predict before the year" | Features use year t's bikeshare data to estimate the t-1 to t income change, which the ACS 1-year release publishes in September of t+1 | Bikeshare data is complete in January, so the nowcast is available about 8 months earlier. A January-June version of every feature is also built (`puma_panel_h1.csv`) to see whether a mid-year forecast keeps the signal. |
| Sample | PUMAs with at least 5 stations active in both 2015 and 2019 | An established system, so the change is not just an opening. |
| Years | Outcome years 2016-2019 (bike data 2015-2019) | 2020 has no standard ACS 1-year release and COVID breaks the pattern, so it is left out. |
| Boston | Included | At PUMA size its income noise is much smaller than at tract level. The EDA reports the noise ceiling by city, so the same check as in the tract study can drop it if needed. |

## Features (all nine are year-over-year changes, measured relative to the city's median PUMA that year)
- **usage:** `d_log_trips` (log growth of trips on matched station-months), `station_growth` (log change in active stations),
  `new_station_share` (share of this year's trips from stations with no trips last year). The last two control for system expansion.
- **rider_mix:** change in casual share, weekend share, round-trip share.
- **commute_role:** change in net morning and evening flow (positive = mostly a departure point).
- **trip_character:** change in log mean trip duration.

Controls (ACS 1-year, previous year, relative to the city): education, renter share, log rent, log home value, median age,
log density, km to center, log households.

## Baselines (same idea as the tract study)
| Baseline | Inputs |
|---|---|
| B0 | income level vs the city at the start of the year (+ square): mean reversion only |
| B1 | B0 + the eight free controls |
| B2 | B1 + last year's income change (momentum) |
| ALT_usage / rider_mix / commute_role / trip_character | B2 + that group |
| ALT_all | B2 + all nine bikeshare changes |

Ridge regression with inverse-variance weights. Three validations, each scored on rows the model never trained on:
`puma` (5 folds, a PUMA is never in train and test together), `loco` (leave one city out), `forward` (a year predicted from
earlier years only; 2018 and 2019 are scored). Gains come with intervals from resampling whole PUMAs, because the rows of one PUMA
are correlated across years. The one table to read is `results/puma/baseline_summary.csv`.

## Run order (from the repo root, on this branch)
```
python scripts/01_download_trips.py --years 2019     # the one extra trip year (2015-2018 are already downloaded)
python scripts/05_aggregate_trips.py --panel         # trips -> station-month tables for 2015-2019 (separate cache; NYC is slow)
python scripts/11_fetch_puma.py                      # ACS 1-year PUMA tables + PUMA polygons
python scripts/12_build_puma_panel.py                # the PUMA-year panel
python scripts/13_puma_eda.py                        # EDA, noise ceiling, monthly-vs-yearly evidence
python scripts/14_puma_baselines.py                  # baseline ladder
python scripts/15_puma_multiyear.py                  # multi-year outcome variant (h = 2, 3)
python -m pytest
```
The tract study (scripts 01-10) is untouched: the panel uses its own `*_panel` files.

## What the one-year panel showed (130 rows, 35 PUMAs)
- 66% of the yearly income change is ACS sampling noise (best possible R² 0.34); DC's is 100%.
- Last year's change correlates -0.42 with this year's, which is what pure measurement error produces (about -0.5). B2's large
  gain over B1 (+0.41 Spearman) is therefore mostly the model predicting the reversal of last year's survey error, not income dynamics.
- No bikeshare group beat B2 with an interval above 0; commute role was significantly worse under forward validation.
- Calendar month alone explains 88-94% of within-year variation in monthly trips, which supports year-over-year matched months.

## Multi-year variant (`15_puma_multiyear.py`)
Same panel, longer window: the outcome is the income change over h years (h = 2 and 3, `config.PUMA["horizons"]`) and the bikeshare
changes cover the same h years. Real change accumulates over a longer window while the survey error of the two end years does not, so the
noise share should fall. Differences from the one-year panel:
- End years are those whose window starts inside the bike data (h=3: 2018 and 2019; h=2: 2017-2019), so there are fewer rows
  (about 70 and 105 on 35 PUMAs) and windows overlap (the PUMA-cluster bootstrap handles that).
- Controls are measured at the window start.
- Momentum is the change from t-2h to t-h-1: it ends a year before the window begins, so it shares no survey year with the outcome and
  cannot predict the start year's noise. This removes the artifact described above.
- Validation: `puma` and `loco` only. A forward scheme would train on windows that overlap the test window.
- The script prints the noise share by horizon next to the one-year value, so the first thing to read is whether the noise share dropped.

## Checked on real data
The ACS 1-year PUMA geography and the TIGER 2010 PUMA polygons work (346 PUMAs in the five states); all 2019 trip files parse; 36 PUMAs
qualify (Boston 7, Chicago 11, DC 8, NYC 10), giving 144 PUMA-year rows, 130 usable (14 lose `edu_share_ba`, cause not yet checked).

## Limits to keep in mind
Systems add stations (handled by matching and by `station_growth`); PUMA income covers everyone in the PUMA while stations cover
part of it; four annual changes per PUMA are not independent; 2020 is excluded; the scope rule looks at 2019 activity, which is
a mild look-ahead for the 2016 rows.
