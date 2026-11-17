# The Activity Index: final results

**Question.** In January of year T+1, can a neighbourhood's bikeshare record for year T tell us its year-T mean household income, about
eight months before the ACS 1-year estimate is posted, better than its own posted income history can?

**Answer.** A little, and not in the direction the project started from. One feature, matched-station trip growth, lowers the average
dollar miss from $8,495 to $7,995. The gain beats every placebo, survives a weather adjustment, and appears in all four cities and all
four test years. But it is small next to the ACS's own sampling noise ($4,752), its interval across areas still includes zero, and its
sign is negative: areas whose trips grew faster than the rest of their city saw income grow **slower**.

Everything below is scored forward in time (a test year is predicted from earlier years only) on 80 area-years: 20 areas in Boston,
Chicago, DC and New York, test years 2018, 2019, 2023 and 2024. Design details are in [nowcast.md](nowcast.md).

## 1. Headline (script 22)
| Rung | Inputs | Average miss | Typical miss |
|---|---|---|---|
| B0 | last posted income x the city's usual growth | $10,194 | 6.1% |
| B1 | B0 + the area's last income change | $8,495 | 5.1% |
| **B2g** | **B1 + trip growth** | **$7,995** | **4.8%** |
| F_all | B1 + all 19 bikeshare features | $9,754 | 6.1% |
| floor | what a perfect model would still miss (ACS sampling error) | $4,752 | |

Trip growth is the only one of 19 bikeshare features that helps on its own (-$501); the next best is -$106 and most make the nowcast
worse. Gain of B2g over B1: -$501, 95% interval -$1,020 to +$121 (resampling whole areas).

## 2. Is the gain real? Placebo tests (script 24)
Each placebo keeps the model and the sample and breaks one link. p = share of placebo draws that helped as much as the real feature.

| Placebo | What it breaks | Placebo gain (mean) | p |
|---|---|---|---|
| Trip growth shuffled between areas of a city-year | whose growth it is | +$152 (worse) | 0.001 |
| Trip growth shuffled between an area's own years | which year it is | +$244 (worse) | 0.001 |
| Whole bikeshare records swapped, best of 19 features kept | corrects for having picked the best feature | -$105 | 0.006 |

Where the gain comes from: every city (Boston -$521, Chicago -$116, DC -$323, NYC -$1,124) and every test year (2018 -$195,
2019 -$585, 2023 -$703, 2024 -$520). Leaving one city out, the gain stays between -$437 and -$490 except without NYC, where it falls
to -$160. NYC carries much of the effect.

How to read the two results together: the placebo tests say the link between an area's trip growth and its income growth is unlikely to
be chance; the interval says that with 20 areas we cannot yet pin down how many dollars it is worth.

## 3. Is it weather or seasonality? (scripts 25, 26)
- Weather moves a whole city: the change in monthly temperature, wet days and precipitation explains 47% of city-wide monthly trip
  growth (about +5% trips per degree C, -0.7% per extra wet day).
- The model feature is growth on matched calendar months, relative to the city's median area, so city-wide weather is already removed.
  Weather explains only 5% of an area's monthly growth relative to its city.
- Removing each area's fitted weather effect month by month and rebuilding the annual growth leaves the gain almost unchanged:
  -$465 (weather-adjusted) vs -$501 (raw). The weather part on its own makes the nowcast worse (+$234).
- Seasonality: growth measured in the warm months (Apr-Oct) carries the signal (-$603, interval -$1,054 to -$71); cold-month growth
  does not (+$215). This split was run after seeing the main result, so treat it as a lead.

## 4. Residents or commuters? (script 27)
Trips have no rider address, so the split uses time and direction: morning departures plus evening arrivals are people who live in the
area; morning arrivals plus evening departures are people who come to work there.

| Growth of | Gain vs B1 |
|---|---|
| all trips (B2g) | -$501 |
| resident-like trips | -$183 |
| commuter-inflow trips | +$3 |
| weekday / weekend | -$447 / -$142 |
| member / casual | -$190 / +$76 |

No subset beats all trips, and every interval includes zero. The pattern still points one way: growth in commuter-inflow trips carries
nothing, and the all-trip gain lands in the more residential areas (-$908) rather than the job centers (-$168). So the signal is about
the people who live in an area, not the people passing through it.

## 5. Which way does it point? (script 28)
The fitted slope of income growth on trip growth is negative in every forward fold (-0.24 to -0.47: an area whose trips grew 10% faster
than its city is expected to see income grow 2 to 5% slower), the correlation is negative in all four cities, and it is -0.36 after
removing income momentum. It is not a stand-in for how rich the area already is: with the income level in the model, trip growth still
adds -$506.

We did not test why. Plausible readings: bikeshare use grows fastest where new, younger, lower-income households are arriving (so mean
income is diluted); or higher earners ride less as their work and travel habits change. Either way "more activity = getting richer" is
not what these four cities show.

## 6. As a decision: the area screen (scripts 28, 29)
A retail or real-estate team asks which areas of a city are pulling ahead. Scored as a ranking inside each of 16 city-years:

| Screen | Calls that held (ahead / behind city) | Income gap, top third vs bottom third |
|---|---|---|
| Income history only (B1) | 57% | $7,740 |
| Income history + trip growth (B2g) | 61% | $11,185 |
| Follow the trips (rank by trip growth, no model) | 36% | -$5,139 |

Intervals overlap between the first two rows (direction: 46-68% vs 53-71%), so the improvement is suggestive. The third row is the
practical warning: ranking areas by raw bikeshare growth picks the wrong ones.
`demo/activity_index_demo.html` is the screen itself: map, ranking, nowcast and, on request, what the ACS later posted.

## 7. Limits
- 20 areas, four cities, four test years. Nothing here is proven at the size a product would need.
- Bikeshare covers dense urban cores and its riders are not a random sample of residents; areas without stations are invisible.
- Mean household income is one number per area of roughly 100,000 or more people. It says nothing about who gains or loses inside the area.
- The feature list was explored before the stress tests (19 features, several variants). The best-of-19 placebo corrects for the
  first; the later splits (warm months, trip purpose, area type) are uncorrected.
- 2020 and 2021 are excluded, and 2023's momentum term is the COVID recovery.
- Station coordinates for some stations come from the live station feed, so a re-run long after November 2026 could differ slightly.
- Not done: additional cities. The 2025 ACS 1-year estimates (posted September 2026) would give one fully out-of-sample year once 2025
  trips are added.
