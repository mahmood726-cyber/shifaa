"""Tests for shifaa matrix builder."""
import pandas as pd
from shifaa.lakehouse.join import build_shifaa_matrix

def _make_dalys():
    return pd.DataFrame({"iso3c": ["PAK", "PAK", "GBR", "GBR"], "year": [2020]*4,
        "cause_id": [294, 426, 294, 426], "cause_name": ["IHD", "T2DM", "IHD", "T2DM"],
        "dalys": [4200000, 1500000, 1100000, 800000]})

def _make_trials():
    return pd.DataFrame({"iso3c": ["PAK", "GBR", "GBR"], "year": [2020]*3,
        "gbd_cause_id": [294, 294, 426], "trial_density": [0.5, 12.4, 8.1], "trial_count_binary": [2, 847, 520]})

def _make_wb():
    return pd.DataFrame({"iso3c": ["PAK", "GBR"], "year": [2020]*2,
        "gdp_pc": [1194.0, 41059.0], "governance": [-0.78, 1.69],
        "health_spend_pct": [2.9, 10.2], "physicians": [1.0, 2.8],
        "hci": [0.41, 0.78], "population": [220892340, 67886011]})

def test_build_matrix_has_required_columns():
    matrix = build_shifaa_matrix(_make_dalys(), _make_trials(), _make_wb())
    required = {"iso3c", "year", "gbd_cause_id", "dalys", "trial_density", "gdp_pc", "governance", "rei_score"}
    assert required.issubset(set(matrix.columns))

def test_build_matrix_rei_negative_for_deserts():
    matrix = build_shifaa_matrix(_make_dalys(), _make_trials(), _make_wb())
    pak_ihd = matrix[(matrix["iso3c"] == "PAK") & (matrix["gbd_cause_id"] == 294)]
    if len(pak_ihd) > 0:
        assert pak_ihd.iloc[0]["rei_score"] < 0

def test_build_matrix_fills_zero_trials():
    matrix = build_shifaa_matrix(_make_dalys(), _make_trials(), _make_wb())
    pak_dm = matrix[(matrix["iso3c"] == "PAK") & (matrix["gbd_cause_id"] == 426)]
    assert len(pak_dm) == 1
    assert pak_dm.iloc[0]["trial_density"] == 0

def test_build_matrix_row_count():
    matrix = build_shifaa_matrix(_make_dalys(), _make_trials(), _make_wb())
    assert len(matrix) == 4
