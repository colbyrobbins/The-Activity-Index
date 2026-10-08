# Project spec: bikeshare activity as a leading indicator of neighborhood income growth

Status as of 2026-10-08: planning. Data availability checked from official pages and archive listings; no trip files downloaded or inspected yet.

## Why not Strava
The original idea (Strava activity predicting income or retail site selection) hit two walls. The Global Heatmap is cross-sectional and refreshed too coarsely for a time study, and its high-resolution tiles need a logged-in account while Strava's terms prohibit automated collection. Strava Metro (the sanctioned route) is not available until November. Bikeshare trips are open, official, and have a decade of history. Strava Metro can be added later as an extension. The earlier Strava site-selection baseline is in `archive/strava_baseline/`; its leave-one-group-out evaluation harness (`04_train.py`) is the template for this project's evaluation.

## Question
Do bikeshare activity features measured 2015-2018 predict the change in a tract's within-city income rank between ACS 2014-18 and ACS 2019-23, beyond what 2018-era free data already tells you?

## Data
| Source | What | Status |
|---|---|---|
| Citi Bike (NYC) | monthly/annual trip zips, S3 | 2013 annual file confirmed; 2014+ assumed same pattern |
| Divvy (Chicago) | quarterly trip zips | 2015-2019 files confirmed |
| Bluebikes (Boston) | monthly trip zips (Hubway bucket) | 2015-2018 files confirmed |
| Capital Bikeshare (DC) | yearly 2010-2017, monthly 2018-2019 | confirmed |
| ACS 5-year, tract level | 2014-18 (2010 tracts), 2019-23 (2020 tracts) | via Census API; boundary vintages confirmed on census.gov |
| Tract crosswalk 2020 to 2010 | area/population weights | Drexel UHC crosswalk on Brown LTDB site; coverage beyond Philadelphia not confirmed |

**Licenses.** Citi Bike and Divvy agreements allow use in non-commercial analyses and studies; no redistribution as a stand-alone dataset (do not commit raw files); no linking to rider identities. Bluebikes and Capital Bikeshare agreements not yet read. Not legal advice.

**Access check, 2018 sample (run 2026-10-08, `scripts/00_check_access.py`).** One 2018 file per city downloaded and parsed.

| City | Parsed | Start-station coordinates | Notes |
|---|---|---|---|
| NYC | yes | 100%, in the trip files | The annual zip repeats each month (full CSV, split parts, and month-folder copies), so naive reads double-count: 36.4M rows vs about half that. `select_csvs` keeps one copy. |
| Boston | yes | 100%, in the trip files | files renamed hubway to bluebikes mid-2018; prefix match handles both |
| DC | yes | 94% via current GBFS feed only | trip files carry station number and name, no coordinates; unmatched are presumably retired stations |
| Chicago | yes | 83% via current GBFS feed only (below the 90% bar) | trip files carry no coordinates and no station file is bundled in the 2018 Q3 zip |

**Access check, 2015 sample (run 2026-10-08).**

| City | Sample rows | Start-station coordinates | Notes |
|---|---|---|---|
| NYC | 1.2M | 100% in file | no duplicate CSVs in the 2015 zip |
| Boston | 175k | 100% in file | |
| DC | 1.4M | 95% via current GBFS | unmatched include busy stations (Dupont Circle, Eastern Market, National Mall, M St & Penn Ave) |
| Chicago | 1.0M | 77% via current GBFS (WARN) | unmatched include the busiest lakefront stations (Streeter Dr & Illinois, Lake Shore Dr & Monroe, Lake Shore Dr & North Blvd) |

The unmatched stations are busy, not marginal, so they are probably renamed or re-ID'd rather than retired. Fix before feature building: a historical station list (Chicago Data Portal Divvy stations, DC open data) or name-based matching. Until then, Chicago is the weakest city and results should be reported with and without it.

**Known gaps.** The Census API now requires a free key: the unkeyed call returned a "Missing Key" HTML page (HTTP 200). Get one at https://api.census.gov/data/key_signup.html and set `CENSUS_API_KEY`. The client now says this explicitly. Divvy 2013-2014 file names unconfirmed (not needed). Bluebikes and Capital Bikeshare licenses unread. Full trip download for 2015-2018 is roughly 3.3 GB (NYC about 2.6 GB), plus Bluebikes.

**Tract geography.** Work on 2010 tracts. Crosswalk the 2019-23 data back to 2010 tracts using the additive quantities (aggregate household income, households), never medians. Mean household income = aggregate / households.

## Target
Primary: change in within-city percentile rank of mean household income, T1 (2019-23) minus T0 (2014-18). Rank within city removes inflation and citywide trends. Secondary: classification of top-quartile upward movers.

## Alt-data features (trips 2015-2018, aggregated to tracts)
Trips per station-month and 2015-to-2018 growth; member vs casual share; weekday/weekend ratio; morning-peak departures minus arrivals; round-trip share; trip distance and destination diversity. Restrict to stations open since 2015 and normalize by station-months to separate usage from system expansion.

## Baselines (layered, so lift is attributable)
1. B0: initial income rank only (mean reversion).
2. B1: B0 plus education, rent and home value, renter share, age mix, density, distance to downtown.
3. B2: B1 plus prior momentum (ACS 2009-13 to 2014-18 income change; vintage and tract geometry to be checked).
4. Alt: B2 plus bikeshare features. The difference vs B2 is the result.

## Evaluation
Regression: Spearman correlation (primary), out-of-sample R2, MAE of rank change, bootstrap CIs on lift. Classification: PR-AUC and precision at top 20 tracts per city. Calibration: actual mean rank change by predicted decile. Validation: leave-one-city-out plus spatial-block CV within cities (only 4 cities, so use both). Models: elastic net and gradient-boosted trees. Also report a noise ceiling from ACS margins of error.

## Risks
- Reverse causality: cities expand bikeshare into areas already changing. Use only pre-2019 trips, only long-running stations, and describe results as a leading indicator, not causal.
- ACS tract income is noisy; accuracy has a ceiling.
- The 2019-23 window includes COVID, which hit bikeshare and urban cores unevenly.
- Coverage skews to the urban core; only tracts near stations are in scope.

## Next steps
1. Download and inspect trip files; confirm fields and station coordinates per city.
2. Fetch ACS for both windows and build the 2010-tract crosswalk.
3. Build B0-B2 and report baseline numbers before adding bikeshare features.
4. Add bikeshare features and run the comparison.

## Sources
- Citi Bike: https://citibikenyc.com/system-data and https://citibikenyc.com/data-sharing-policy
- Divvy: https://divvybikes.com/system-data and https://divvybikes.com/data-license-agreement
- Bluebikes: https://bluebikes.com/system-data
- Capital Bikeshare: https://capitalbikeshare.com/system-data
- ACS tract vintage: https://www.census.gov/programs-surveys/acs/geography-acs/geography-boundaries-by-year.2018.html and ...year.2022.html
- Crosswalk: https://drexel.edu/uhc/news-events/news/2025/february/census-tract-crosswalk
