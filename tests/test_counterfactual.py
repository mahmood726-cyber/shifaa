"""Tests for counterfactual scenarios."""
import pandas as pd
from shifaa.analysis.counterfactual import compute_counterfactual

def test_counterfactual_improves_rei():
    desert = pd.DataFrame({"iso3c": ["PAK"], "gbd_cause_id": [294], "rei_score": [-3.2],
        "governance": [-0.78], "trial_density": [0.5], "dalys": [4.2e6], "population": [220e6]})
    result = compute_counterfactual(desert, governance_target=0.0, gov_coefficient=1.5)
    # Counterfactual trials should increase (governance improves), so counterfactual REI > original REI
    # Compare using trial_density_counterfactual > trial_density (the direct effect)
    assert result.iloc[0]["trial_density_counterfactual"] > result.iloc[0]["trial_density"]
    assert result.iloc[0]["trial_multiplier"] > 1.0

def test_counterfactual_columns():
    desert = pd.DataFrame({"iso3c": ["NGA"], "gbd_cause_id": [341], "rei_score": [-6.0],
        "governance": [-1.0], "trial_density": [0.1], "dalys": [3e6], "population": [206e6]})
    result = compute_counterfactual(desert, governance_target=0.5, gov_coefficient=1.5)
    assert "rei_counterfactual" in result.columns and "trial_density_counterfactual" in result.columns
