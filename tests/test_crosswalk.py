import pandas as pd
import pytest

from activity_index import crosswalk as X


def _xw():
    # 2020 tract A is split 75/25 between 2010 tracts P and Q; 2020 tract B is exactly 2010 tract Q's rest
    return pd.DataFrame({
        "GEOID": ["A", "A", "B"], "GEOID_dst": ["P", "Q", "Q"],
        "weight": [0.75, 0.25, 1.0], "dst_cover": [1.0, 0.25, 0.75]})


def test_apply_crosswalk_conserves_totals_and_splits_by_weight():
    vals = pd.DataFrame({"GEOID": ["A", "B"], "households": [100.0, 40.0], "agg_income": [1e7, 2e6]})
    out = X.apply_crosswalk(vals, _xw(), ["households", "agg_income"]).set_index("GEOID")
    assert out.loc["P", "households"] == 75 and out.loc["Q", "households"] == 65
    assert out["households"].sum() == 140 and out["agg_income"].sum() == 1.2e7
    assert (out["share_with_data"] == 1.0).all()


def test_apply_crosswalk_reports_missing_source_data():
    vals = pd.DataFrame({"GEOID": ["A", "B"], "households": [100.0, None]})
    out = X.apply_crosswalk(vals, _xw(), ["households"]).set_index("GEOID")
    assert out.loc["P", "share_with_data"] == 1.0
    assert out.loc["Q", "share_with_data"] == pytest.approx(0.25)   # only A's piece (25% of Q) has data


def test_diagnostics_flags_unchanged_tracts():
    d = X.diagnostics(_xw()).set_index("GEOID_dst")
    assert d.loc["P", "n_sources"] == 1 and d.loc["P", "largest_piece"] == 1.0
    assert d.loc["Q", "n_sources"] == 2 and d.loc["Q", "largest_piece"] == 0.75


def test_build_crosswalk_from_polygons():
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    # 2020: one 4x1 tract A and one 2x1 tract B; 2010: P covers x 0-3, Q covers x 3-6 (4000 m units)
    m = 1000.0
    src = gpd.GeoDataFrame({"GEOID": ["A", "B"]}, geometry=[box(0, 0, 4 * m, m), box(4 * m, 0, 6 * m, m)], crs="EPSG:5070")
    dst = gpd.GeoDataFrame({"GEOID": ["P", "Q"]}, geometry=[box(0, 0, 3 * m, m), box(3 * m, 0, 6 * m, m)], crs="EPSG:5070")
    xw = X.build_crosswalk(src, dst).set_index(["GEOID", "GEOID_dst"])
    assert xw.loc[("A", "P"), "weight"] == pytest.approx(0.75)
    assert xw.loc[("A", "Q"), "weight"] == pytest.approx(0.25)
    assert xw.loc[("B", "Q"), "weight"] == pytest.approx(1.0)
    assert xw.loc[("A", "Q"), "dst_cover"] == pytest.approx(1 / 3)
