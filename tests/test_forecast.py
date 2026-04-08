"""Tests for REI forecast."""
import pandas as pd
from shifaa.analysis.forecast import project_rei

def test_project_rei_returns_future_years():
    current = pd.DataFrame({"iso3c": ["PAK", "GBR"], "gbd_cause_id": [294, 294],
        "rei_score": [-3.2, 1.8], "dalys": [4.2e6, 1.1e6], "trial_density": [0.5, 12.4], "year": [2020, 2020]})
    projected = project_rei(current, target_year=2036)
    assert len(projected) == 2 and all(projected["year"] == 2036) and "rei_projected" in projected.columns

def test_project_deserts_deepen():
    current = pd.DataFrame({"iso3c": ["NGA"], "gbd_cause_id": [294], "rei_score": [-5.0],
        "dalys": [3e6], "trial_density": [0.01], "dalys_growth_rate": [0.03], "trial_growth_rate": [0.0], "year": [2020]})
    projected = project_rei(current, target_year=2036)
    assert projected.iloc[0]["rei_projected"] < -5.0
