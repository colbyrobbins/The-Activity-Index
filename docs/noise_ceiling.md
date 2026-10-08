# Noise ceiling for the target

Run 2026-10-08 (`scripts/09_noise_ceiling.py`). ACS margins of error turned into standard errors of each tract's mean household income, incomes
redrawn 300 times, re-ranked within city. Treats the two ACS windows as independent samples (they are non-overlapping) and approximates the
2020-to-2010 crosswalk error, so the noise is if anything understated. In-scope tracts only (724).

## How much of the target is noise
| scope | tracts | noise share of variance | ceiling R2 | ceiling correlation | moves beyond 2 SE |
|---|---|---|---|---|---|
| all in scope | 724 | 61% | 0.40 | 0.63 | 10.6% |
| Chicago | 264 | 54% | 0.46 | 0.68 | 11.7% |
| DC | 140 | 63% | 0.37 | 0.61 | 12.9% |
| NYC | 232 | 60% | 0.40 | 0.64 | 10.3% |
| Boston | 88 | 92% | 0.08 | 0.29 | 4.5% |

"Ceiling" is what a predictor that knew the true change perfectly could score against the observed, noisy change. About 40% of the variance is
real signal, so there is room: the first baselines (R2 about 0.03) are far below it. A pure-noise world would put 4.5% of tracts beyond 2 SE;
Boston is exactly there, so it shows no detectable real movers, and its ceiling correlation is only 0.29 on 88 tracts.

## Where the noise is
- Median coefficient of variation of tract income is 0.09 (2014-18) and 0.10 (2019-23).
- Observed SD of rank change by noise tercile: 0.033 (low noise), 0.097 (middle), 0.174 (high). Most of the apparent movement is in the noisiest tracts.
- Several of the largest moves are noise: Chicago 17031280900 (+0.87) has a 64% CV on its 2019-23 income. Others look real: NYC (Kings County) tracts
  36047027300 (+0.47), 36047026300 (+0.45) and 36047028501 (-0.62) are 4 to 4.6 SE moves.

## Mean reversion: how much is noise?
Spearman(initial rank, rank change), noise alone vs observed:

| city | noise only | observed |
|---|---|---|
| Boston | -0.20 | -0.11 |
| Chicago | -0.18 | -0.16 |
| DC | -0.15 | -0.24 |
| NYC | -0.17 | -0.38 |

In Boston and Chicago the observed reversion is no stronger than noise alone would create, so there is no evidence of real convergence there.
NYC shows more than twice what noise implies, which looks like real catch-up by lower-ranked tracts. That explains why the baseline does not transfer
across cities: the mean-reversion slope is city-specific.

## What follows for the models
1. Use the standard errors. Weight training by 1/noise variance, and report metrics on the lower-noise half as well as overall.
2. Allow city-specific mean reversion (city by initial-rank terms) in the validation that keeps cities together; keep leave-one-city-out as the strict test.
3. Treat Boston as a sensitivity case (with and without); it adds mostly noise.
4. Drop or flag tracts with a coefficient of variation above about 0.3 in either window; do not winsorize blindly, since several large moves are real.
5. Expect small lifts. Even a good feature moves Spearman by a few hundredths against a ceiling of 0.63.
