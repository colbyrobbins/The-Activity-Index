import numpy as np
import pandas as pd

from activity_index import crosswalk as X
from activity_index import noise as N


def test_ratio_se_matches_hand_calculation():
    # ratio 100, num MOE 1000 on num 100000, den 1000 with MOE 100: radicand = 1e6 - 1e4*1e4 < 0 -> fallback (sum)
    se = N.ratio_se(pd.Series([100000.0]), pd.Series([1000.0]), pd.Series([1000.0]), pd.Series([100.0]))
    assert np.isclose(se.iloc[0], np.sqrt(1000.0 ** 2 + 100.0 ** 2 * 100.0 ** 2) / 1000.0 / 1.645)
    se2 = N.ratio_se(pd.Series([100000.0]), pd.Series([20000.0]), pd.Series([1000.0]), pd.Series([100.0]))
    assert np.isclose(se2.iloc[0], np.sqrt(20000.0 ** 2 - 100.0 ** 2 * 100.0 ** 2) / 1000.0 / 1.645)


def test_apply_crosswalk_moe_root_sum_of_squares():
    xw = pd.DataFrame({"GEOID": ["A", "B"], "GEOID_dst": ["P", "P"], "weight": [0.5, 1.0], "dst_cover": [1.0, 1.0]})
    v = pd.DataFrame({"GEOID": ["A", "B"], "m": [6.0, 4.0]})
    out = X.apply_crosswalk_moe(v, xw, ["m"]).set_index("GEOID")
    assert np.isclose(out.loc["P", "m"], np.sqrt((3.0) ** 2 + 4.0 ** 2))


def _frame(n=600, se=0.0, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(100, 20, n)
    return pd.DataFrame({"city": np.where(np.arange(n) % 2 == 0, "a", "b"), "mean0": base, "se0": se,
                         "mean1": base + rng.normal(0, 5, n), "se1": se})


def test_no_noise_means_no_noise_sd_and_full_reliability():
    df = _frame(se=1e-9)
    s = N.simulate(df, n_draws=30)
    assert (s["noise_sd"] < 1e-6).all()
    chg = pd.Series(np.random.default_rng(1).normal(0, 0.1, len(df)))
    assert N.ceiling(chg, s["noise_sd"])["reliability"] > 0.999


def test_heavy_noise_creates_mean_reversion_and_lowers_ceiling():
    df = _frame(se=20.0)       # noise as large as the spread of incomes
    s = N.simulate(df, n_draws=60)
    assert s["noise_sd"].mean() > 0.1
    assert all(r < -0.3 for r in s["pure_noise_rho"].values())      # high ranks fall back, low ranks rise, by noise alone
    c = N.ceiling(pd.Series(np.random.default_rng(1).normal(0, 0.15, len(df))), s["noise_sd"])
    assert c["noise_share"] > 0.3 and c["max_corr"] < 0.85
