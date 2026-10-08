# The Activity Index: what we did and what the numbers mean

Question: do bikeshare activity features from 2015-2018 predict how a neighbourhood's income rank changes between ACS 2014-18 and ACS 2019-23, beyond what free census data already says?

## 1. The data
| Piece | What it is | Source |
|---|---|---|
| Bikeshare trips 2015-2018 | 91M valid trips: NYC 57.6M, Chicago 14.2M, DC 13.8M, Boston 5.4M (Citi Bike, Divvy, Capital Bikeshare, Bluebikes). Start/end time, start/end station. | Public trip files (not redistributable; `data/` is gitignored) |
| Station locations | Latitude/longitude per station. | Trip files (NYC, Boston), bundled station files (Chicago), current GBFS feed (DC) |
| ACS 5-year tables | Income, education, rent, home value, age, households, with margins of error. Windows 2009-13 (momentum), 2014-18 (start), 2019-23 (end). | Census API |
| Tract polygons | 2010 tracts (2014-18) and 2020 tracts (2019-23). | TIGER |

## 2. What we predict
**rank_change** = a tract's within-city percentile rank of mean household income in 2019-23 minus the same rank in 2014-18. Positive means the tract moved up relative to the rest of its city. Tracts with fewer than 50 households are not ranked. The standard deviation is about 0.10 to 0.13 (10 to 13 percentile points).

**Sample.** A tract is in scope if it contains a "mature" station (trips in both 2015 and 2018) and has a rank change: 724 of 4,216 tracts (Chicago 264, NYC 232, DC 140, Boston 88). 12 more are dropped for very imprecise income (coefficient of variation above 0.3), leaving 712. **Boston is then excluded from the models (see "Why Boston is excluded"), so the headline sample is 624 tracts in Chicago, NYC and DC.** This is the urban core of the cities, not whole metro areas.

**Why Boston is excluded.** Boston went through the data work, EDA and the first baseline like the other cities. The noise ceiling (script 09, built after that first baseline) then showed that 92% of its tract rank change is ACS sampling noise (maximum achievable correlation 0.29, against 0.61 to 0.68 elsewhere), because it has only 88 usable tracts with imprecise incomes. The decision rests on that measurement, which uses no bikeshare features, not on how bikeshare performed. We still report the four-city run as a sensitivity check (`results/baseline_summary_with_boston.csv`). Boston remains in the EDA and in `model_table.csv` (`use_in_models` = False).

**Noise ceiling.** ACS incomes are survey estimates. Re-drawing them within their margins of error shows **about 60% of the variance in rank change is sampling noise** (92% in Boston). So the best any model could do is roughly **R² 0.39 / correlation 0.63**. Our models should be judged against that, not against 1.0.

## 3. Cleaning and fixes
- Trips kept only if they start in 2015-18 and last 1 to 180 minutes (99.5% to 99.8% survive). Column names are normalised across years (Divvy 2018 Q1 used long headers; an `end_time` alias was missing, which had silently dropped Chicago 2017-18 until fixed).
- Station coordinates: trip file, then pooled historical station files, then the current feed. Depot/test stations are removed by name, stations more than 60 km from the city center or outside every tract are dropped.
- Seasonality is large (summer is 2.6x to 9x January), so every rate is per active station-month, never an annual total.
- Stations opened at different times, so growth and rates use only "mature" stations.
- 2020-tract ACS data are moved onto 2010 tracts by area share (additive quantities only; mean income is rebuilt as aggregate income divided by households), so both windows describe the same tracts.
- **City-relative features:** every control and bikeshare feature is converted to its within-city percentile, because raw scales differ by city (median trips per station-month is about 1,140 in NYC and 250 in Chicago).

## 4. EDA in six lines
1. Bikeshare usage tracks income in levels (trips per station-month vs initial rank: Spearman 0.59), but initial rank is already known, so only information about change helps.
2. Mean reversion is strong (initial rank vs change: -0.38 NYC, -0.24 DC, -0.16 Chicago, -0.11 Boston); part of it is pure noise.
3. After removing initial rank and city, usage and usage growth have no relation to change (-0.02 and -0.04).
4. Four features have weak adjusted correlations of about 0.1: AM net flow +, PM net flow -, round-trip share -, casual-rider share -. With 19 features tested, treat them as leads.
5. Free controls matter more: home value +0.18, renter share -0.16.
6. Prior momentum is negative (-0.11), a sign of noise in the starting rank, not of persistent trends.

Detail: [eda_findings.md](eda_findings.md), [noise_ceiling.md](noise_ceiling.md), figures in `results/eda/`.

## 5. The baselines
Each is a ridge regression with the listed inputs. The target is rank change and the metrics are computed on predictions for tracts the model did not train on. Each row adds to the one it is compared with.

| Baseline | Inputs | Question it answers |
|---|---|---|
| B0 | initial rank, rank squared | How much do we get from mean reversion alone? |
| B1 | B0 + 8 free controls: bachelor's share, renter share, median rent, median home value, median age, log density, km to center, log households | How much does free census data add? |
| B2 | B1 + momentum (rank change 2009-13 to 2014-18) | Does trend add anything? **B2 is the bar the bikeshare data must clear.** |
| ALT_usage | B2 + trips per station-month, log growth | Does volume help? |
| ALT_rider_mix | B2 + casual share, weekend share, round-trip share | Does who rides help? |
| ALT_commute_role | B2 + net AM flow, net PM flow | Does residential vs destination role help? |
| ALT_trip_character | B2 + mean duration, mean distance, destination entropy | Does what trips look like help? |
| ALT_all | B2 + all 10 | Do they help together? |

All fits are weighted by how precise each tract's rank change is (1 / (signal variance + noise variance)).

## 6. How the numbers read
Two validation schemes, each predicting only tracts the model has not seen:
- **LOCO (leave one city out):** train on three cities, predict the fourth. Strict: does it generalize to a new city?
- **Spatial:** hold out 3 km blocks (5 folds), with city dummies and city-specific mean reversion allowed. Easier and more favourable to the model.

Two metrics, pooled over all 712 tracts:
- **Spearman:** rank correlation between predicted and actual rank change. 0 is no skill, 1 is perfect, about 0.63 is the noise ceiling.
- **R²:** share of variance explained. 0 equals predicting the average; the ceiling is about 0.39.

**Gain** is the change in Spearman against the baseline in `compared_to`, with a 95% bootstrap interval. If the interval includes 0, we cannot tell the difference from luck.

Results, Chicago + NYC + DC, 624 tracts (`results/baseline_summary.csv`, from `python scripts/10_baselines.py`):

| Baseline | Compared to | Spearman LOCO | R² LOCO | Spearman spatial | R² spatial | Gain LOCO | Gain spatial |
|---|---|---|---|---|---|---|---|
| B0 | | 0.109 | 0.018 | 0.187 | 0.028 | | |
| B1 | B0 | 0.248 | 0.082 | 0.284 | 0.109 | +0.139 (0.059 to 0.211) | +0.097 (0.021 to 0.172) |
| B2 | B1 | 0.236 | 0.075 | 0.299 | 0.107 | -0.012 (-0.027 to 0.003) | +0.015 (-0.004 to 0.033) |
| ALT_usage | B2 | 0.217 | 0.066 | 0.291 | 0.101 | -0.019 (-0.050 to 0.010) | -0.008 (-0.022 to 0.009) |
| ALT_rider_mix | B2 | 0.240 | 0.071 | 0.304 | 0.105 | +0.004 (-0.021 to 0.032) | +0.005 (-0.015 to 0.027) |
| ALT_commute_role | B2 | 0.267 | 0.098 | 0.310 | 0.111 | +0.032 (-0.002 to 0.065) | +0.012 (-0.011 to 0.035) |
| ALT_trip_character | B2 | 0.259 | 0.100 | 0.314 | 0.119 | +0.023 (-0.004 to 0.052) | +0.016 (-0.007 to 0.038) |
| ALT_all | B2 | 0.255 | 0.091 | 0.296 | 0.092 | +0.019 (-0.024 to 0.068) | -0.003 (-0.041 to 0.033) |
| noise ceiling | | corr about 0.65 | R² about 0.43 | | | | |

Sensitivity, with Boston (712 tracts): the same pattern, except that rider mix (+0.021), commute role (+0.019), trip character (+0.034) and all ten (+0.031) clear 0 under spatial validation. They do not once Boston is removed, which is one more reason not to lean on them.

**Reading it:**
1. Free census data is the real step up (B1 vs B0: about +0.10 to +0.14, clearly above 0).
2. Momentum adds nothing (B2 vs B1 includes 0 in both schemes).
3. Usage never helps. Rider mix, commute role and trip character are slightly positive, with commute role the most consistent (+0.012 to +0.032, positive in every cut), but every interval includes 0.
4. About 40 comparisons were run (including LightGBM and the sensitivity sample in `results/baselines_lift.csv`), the groups were chosen after the EDA saw the same tracts, and the spatial gains seen with Boston disappear without it. So **there is no robust evidence that bikeshare features add anything beyond free data; commute role is a lead worth retesting.** Absolute performance is modest (R² about 0.1 against a ceiling of about 0.43).

## 7. Open limitations
712 urban-core tracts; DC coordinates come from today's feed (5% of trips unlocated); area-weighted crosswalk; the 2019-23 window includes COVID; correlations are descriptive, not causal; the robustness window (ACS 2015-19 to 2020-24) has not been run.

## 8. Next (not started)
Re-test rider mix, commute role and trip character on the robustness window and on a wider tract scope (within about 500 m of a mature station) before treating them as real; then the final models and evaluation.
