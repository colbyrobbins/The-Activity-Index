# EDA findings (before any modelling)

Run 2026-10-08 on the primary window: ACS 2014-18 to 2019-23, trips 2015-2018. Numbers come from
`results/eda_summary.md`; figures are in `results/eda/`. Read the caveats at the end before quoting any of this.

## What the data looks like
- **Volume and quality.** 90.9M valid trips: NYC 57.6M, Chicago 14.2M, DC 13.8M, Boston 5.4M. Between 99.5% and 99.8% of rows in every city
  survive the filters (start year in 2015-18, 1 to 180 minutes).
- **Coordinates.** NYC and Boston come from the trip files, Chicago from the bundled station files (97%) plus the current feed (3%), DC entirely
  from the current feed. 94.7% of DC trips have coordinates; 99 of DC's 498 located stations fall outside the DC, Arlington and Alexandria
  tracts (probably suburban Maryland and Fairfax County; not checked), so they are not in the study.
- **Seasonality is large.** Summer months run 2.6x (DC) to 9x (Boston) the January level, so every rate is per active station-month and no feature
  uses a raw annual total.
- **Systems grew.** Active stations 2015 to 2018: NYC 488 to 818, Boston 156 to 315, DC 357 to 528, Chicago 475 to 621. Trips grew 77% in NYC and
  57% in Boston, but only 13% in Chicago and 11% in DC, and both Chicago and DC fell about 6% in 2018 (cause not checked). Only 48% of NYC and Boston
  stations (in tracts) were active in both 2015 and 2018, versus 74-79% in DC and Chicago, which is why growth features use "mature" stations only.

## Who is in the sample
Only **724 of 4,216 tracts (17%)** contain a station active in both 2015 and 2018: Chicago 264, NYC 232, DC 140, Boston 88. Coverage is 52% of DC
tracts but only 12% of NYC's. These tracts differ sharply from the rest of each metro area: bachelor's share 62% vs 35% (1.1 SD), 5.5 km vs 16 km
from downtown, initial income rank 0.63 vs 0.47, more renters, younger. Anything we find is about urban-core bikeshare neighbourhoods, not cities
as a whole.

## The target
Rank change has a standard deviation of 0.10 to 0.13 (10 to 13 percentile points) per city, and 26% of in-scope tracts move more than 0.10 in either direction,
so there is something to predict. Mean reversion is strong: Spearman(initial rank, change) is -0.38 in NYC, -0.24 in DC, -0.16 in Chicago and -0.11
in Boston. There are a few extreme moves (Chicago +0.87, NYC -0.62) that need a look before modelling.

## Does bikeshare track income?
**In levels, yes.** Trips per station-month correlates 0.59 with initial income rank, and round-trip share -0.48, trip duration -0.29. Bikeshare
usage is a decent cross-sectional wealth signal, which is what the original Strava idea assumed. But initial rank is already known from the ACS, so
a level signal only helps if it carries information about change.

**For change, mostly no.** Raw correlation of trips per station-month with rank change is -0.07, but after removing initial rank and city it is
-0.02 (95% interval -0.11 to 0.06). The pattern in the quartile table (lowest-usage quartile gains +0.036, highest -0.010) is just mean reversion in
disguise. Usage growth is null (-0.04). The features that survive the adjustment are weak, about 0.1 in size:

| feature | adjusted rho | 95% interval |
|---|---|---|
| AM net flow (+ means mostly origins) | +0.107 | 0.029 to 0.173 |
| PM net flow | -0.107 | -0.176 to -0.041 |
| Round-trip share | -0.106 | -0.167 to -0.028 |
| Casual-rider share | -0.079 | -0.151 to -0.009 |

AM and PM net flow are probably close to mirror images (correlation not checked), so they may count as roughly one signal. A plausible reading is that residential-type stations
(people leave in the morning) sit in tracts that gain rank, while destination and leisure stations do not; ACS income is measured on residents, so
tracts with few residents are noisy. With 19 features tested, one or two intervals excluding zero would be expected by chance, so treat these as
leads, not findings.

## What else predicts change
After removing initial rank: median home value +0.18 (0.10 to 0.26), renter share -0.16, number of households -0.12, and **prior momentum -0.11**.
Momentum has the opposite sign from what the baseline assumed. That is the signature of noise in the 2014-18 income rank: if a tract's rank was
measured too high, it looks like it had positive momentum and then falls back. So the ACS noise, not persistent trends, drives much of what we see.

## Implications
1. The incremental signal is small, around 0.1 Spearman for individual features, and the sample is 724 tracts, so a lift of +0.01 to +0.03 would
   be a good result and its interval will be wide. The baseline will be dominated by mean reversion and a few controls.
2. The "momentum" baseline (B2) will probably add little or hurt, for the noise reason above.
3. Reasonable next steps, in order of value: fetch ACS margins of error (B19025_001M) to estimate the noise ceiling and drop or down-weight noisy
   tracts; widen scope to tracts within about 500 m of an established station; check the outlier tracts; add suburban DC counties.

## Caveats
Station coordinates in DC come from today's feed, so retired stations are missing (5% of trips). The window is one period only (the robustness
window 2015-19 to 2020-24 has not been run). The 2019-23 window includes COVID. Correlations are descriptive, not causal: cities placed stations
in areas that were already changing.
