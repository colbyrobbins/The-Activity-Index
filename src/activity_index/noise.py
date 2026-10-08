"""How much of the observed income-rank change can be sampling noise in the ACS?

ACS tract incomes are survey estimates with published margins of error (MOE, 90% level). We turn the MOEs
into standard errors, redraw each tract's T0 and T1 mean income many times, re-rank within city, and see how
much the rank change moves. The spread of that movement is the noise in the target.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

Z90 = 1.645


def ratio_se(num, num_moe, den, den_moe):
    """Standard error of num/den from the two MOEs (Census 'approximating the MOE of a ratio' formula)."""
    num, num_moe, den, den_moe = (pd.Series(x, dtype=float) if not isinstance(x, pd.Series) else x.astype(float)
                                  for x in (num, num_moe, den, den_moe))
    r = num / den
    rad = num_moe ** 2 - r ** 2 * den_moe ** 2
    rad = rad.where(rad >= 0, num_moe ** 2 + r ** 2 * den_moe ** 2)   # the formula's fallback when negative
    return (np.sqrt(rad) / den) / Z90


def _rank(values: np.ndarray, city: np.ndarray, groups: dict) -> np.ndarray:
    out = np.full(len(values), np.nan)
    for idx in groups.values():
        v = pd.Series(values[idx])
        out[idx] = v.rank(pct=True).values
    return out


def simulate(df: pd.DataFrame, n_draws: int = 200, seed: int = 0, scope: np.ndarray | None = None) -> dict:
    """Monte Carlo noise in rank change.

    df needs: city, mean0, se0, mean1, se1 (NaN means the tract is not ranked). Returns
      noise_sd:   per-tract SD of rank change over draws (the standard error of the target)
      pure_noise_rho: per city, Spearman(rank_t0, rank_change) in a world with NO true change, i.e. what noise
                      alone does to mean reversion
    `scope` marks the tracts to report the pure-noise correlation on (default: all ranked tracts).
    """
    rng = np.random.default_rng(seed)
    ok = df[["mean0", "se0", "mean1", "se1"]].notna().all(axis=1).values
    sub = df[ok].reset_index(drop=True)
    city = sub["city"].values
    groups = {c: np.where(city == c)[0] for c in np.unique(city)}
    m0, s0, m1, s1 = (sub[c].values.astype(float) for c in ("mean0", "se0", "mean1", "se1"))
    sc = np.ones(len(sub), bool) if scope is None else scope[ok]

    changes = np.empty((n_draws, len(sub)))
    for d in range(n_draws):
        r0 = _rank(np.clip(m0 + s0 * rng.standard_normal(len(sub)), 0, None), city, groups)
        r1 = _rank(np.clip(m1 + s1 * rng.standard_normal(len(sub)), 0, None), city, groups)
        changes[d] = r1 - r0
    noise_sd = pd.Series(np.nan, index=df.index)
    noise_sd[ok] = changes.std(axis=0, ddof=1)

    rho = {c: [] for c in groups}
    for d in range(min(n_draws, 60)):
        r0 = _rank(np.clip(m0 + s0 * rng.standard_normal(len(sub)), 0, None), city, groups)
        r1 = _rank(np.clip(m0 + s1 * rng.standard_normal(len(sub)), 0, None), city, groups)   # same truth (T0 mean)
        for c, idx in groups.items():
            keep = idx[sc[idx]]
            if len(keep) > 20:
                rho[c].append(pd.Series(r0[keep]).rank().corr(pd.Series((r1 - r0)[keep]).rank()))
    return {"noise_sd": noise_sd, "pure_noise_rho": {c: float(np.mean(v)) for c, v in rho.items() if v}}


def ceiling(observed_change: pd.Series, noise_sd: pd.Series) -> dict:
    """Share of the observed variance in rank change that is sampling noise, and the resulting ceilings."""
    m = observed_change.notna() & noise_sd.notna()
    var_obs = float(observed_change[m].var())
    var_noise = float((noise_sd[m] ** 2).mean())
    reliability = max(0.0, 1 - var_noise / var_obs)
    return {"n": int(m.sum()), "var_observed": var_obs, "var_noise": var_noise,
            "noise_share": var_noise / var_obs, "reliability": reliability,
            "max_r2": reliability, "max_corr": float(np.sqrt(reliability))}
