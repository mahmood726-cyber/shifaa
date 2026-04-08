"""Tests for multilevel Poisson regression."""
import numpy as np
import pandas as pd
from shifaa.analysis.regression import fit_poisson_model, extract_coefficients

def _make_data():
    np.random.seed(42)
    n = 200
    iso3c = np.random.choice(["PAK", "GBR", "USA", "NGA", "IND", "BRA", "DEU", "ZAF"], n)
    governance = np.where(np.isin(iso3c, ["GBR", "USA", "DEU"]), 1.5, -0.5) + np.random.normal(0, 0.1, n)
    gdp_pc = np.where(np.isin(iso3c, ["GBR", "USA", "DEU"]), 40000, 2000) + np.random.normal(0, 500, n)
    dalys = np.random.exponential(1e6, n)
    population = np.where(np.isin(iso3c, ["IND", "USA", "BRA"]), 200e6, 50e6)
    trial_count = np.random.poisson(np.exp(1 + 1.5 * governance + 0.0001 * np.log(dalys)), n)
    return pd.DataFrame({"iso3c": iso3c, "gbd_cause_id": np.random.choice([294, 426], n), "year": 2020,
        "trial_count_binary": trial_count, "dalys": dalys, "gdp_pc": gdp_pc, "governance": governance,
        "health_spend_pct": np.random.uniform(2, 12, n), "hci": np.random.uniform(0.3, 0.9, n),
        "physicians": np.random.uniform(0.5, 3.0, n), "population": population})

def test_fit_model_converges():
    result = fit_poisson_model(_make_data())
    assert result is not None and result["converged"] is True

def test_coefficients_include_governance():
    coefs = extract_coefficients(fit_poisson_model(_make_data()))
    assert "governance" in coefs["variable"].values

def test_governance_coefficient_positive():
    coefs = extract_coefficients(fit_poisson_model(_make_data()))
    gov = coefs[coefs["variable"] == "governance"]
    assert gov.iloc[0]["coefficient"] > 0

def test_coefficients_have_ci():
    coefs = extract_coefficients(fit_poisson_model(_make_data()))
    assert "ci_lower" in coefs.columns and "ci_upper" in coefs.columns
