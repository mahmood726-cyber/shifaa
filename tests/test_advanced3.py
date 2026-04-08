"""Tests for advanced stats tier 3."""
import numpy as np
import pandas as pd

from shifaa.analysis.advanced3 import (
    esteban_ray_polarization,
    fit_tobit_model,
    gam_threshold_analysis,
    lasso_variable_selection,
    rosenbaum_bounds,
    spatial_lag_effect,
)


# ── LASSO ────────────────────────────────────────────────────────────────────

def test_lasso_selects_variables():
    np.random.seed(42)
    n = 100
    gdp = np.random.lognormal(8, 1.5, n)
    df = pd.DataFrame({
        "trial_density": gdp / 1000 + np.random.normal(0, 1, n),
        "dalys": np.random.exponential(1e6, n),
        "gdp_pc": gdp,
        "governance": np.random.normal(0, 1, n),
        "health_spend_pct": np.random.uniform(2, 12, n),
        "physicians": np.random.uniform(0.5, 3, n),
        "noise1": np.random.normal(0, 1, n),
        "noise2": np.random.normal(0, 1, n),
    })
    result = lasso_variable_selection(df, candidate_indicators=[
        "dalys", "gdp_pc", "governance", "health_spend_pct", "physicians", "noise1", "noise2"])
    selected = result[result["selected"]]
    assert len(selected) >= 1
    assert "gdp_pc" in selected["variable"].values


# ── GAM Threshold ────────────────────────────────────────────────────────────

def test_gam_finds_threshold():
    np.random.seed(42)
    n = 100
    gov = np.random.uniform(-2, 2, n)
    trials = np.where(gov > 0, gov * 10 + np.random.normal(0, 2, n), np.random.normal(0, 1, n))
    df = pd.DataFrame({"governance": gov, "trial_density": trials.clip(min=0)})
    result = gam_threshold_analysis(df)
    assert "threshold_estimate" in result
    assert -0.5 < result["threshold_estimate"] < 0.5


def test_gam_returns_partial_dependence():
    np.random.seed(42)
    df = pd.DataFrame({
        "governance": np.random.uniform(-2, 2, 60),
        "trial_density": np.random.exponential(5, 60),
    })
    result = gam_threshold_analysis(df)
    assert len(result["partial_dependence_x"]) > 3
    assert len(result["partial_dependence_y"]) == len(result["partial_dependence_x"])


# ── Spatial Lag ──────────────────────────────────────────────────────────────

def test_spatial_lag_positive():
    # Clustered pattern: nearby countries have similar values
    values = {"NGA": 5, "GHA": 6, "BEN": 4, "TGO": 5, "CMR": 4,
              "FRA": 50, "DEU": 55, "BEL": 48, "NLD": 52, "GBR": 60,
              "USA": 70, "CAN": 65, "MEX": 15, "BRA": 20, "ARG": 18}
    result = spatial_lag_effect(values)
    assert result["rho"] > 0


# ── Tobit ────────────────────────────────────────────────────────────────────

def test_tobit_runs():
    np.random.seed(42)
    n = 100
    gdp = np.random.lognormal(8, 1.5, n)
    latent = 0.5 * np.log(gdp) + np.random.normal(0, 2, n) - 4
    trials = np.maximum(0, np.exp(latent))
    df = pd.DataFrame({
        "trial_density": trials,
        "dalys": np.random.exponential(1e6, n),
        "gdp_pc": gdp,
        "governance": np.random.normal(0, 1, n),
        "health_spend_pct": np.random.uniform(2, 12, n),
        "physicians": np.random.uniform(0.5, 3, n),
    })
    result = fit_tobit_model(df)
    if result is None:
        return
    assert "coefficients" in result
    assert "selection_bias" in result
    assert result["n_censored"] >= 0


# ── Polarization ─────────────────────────────────────────────────────────────

def test_polarization_high_for_bimodal():
    # Two distinct clusters
    values = np.concatenate([np.random.normal(5, 1, 50), np.random.normal(100, 5, 50)])
    result = esteban_ray_polarization(values, n_groups=2)
    assert result["polarization"] > 0


def test_polarization_low_for_uniform():
    values = np.random.uniform(10, 20, 100)
    result = esteban_ray_polarization(values, n_groups=3)
    assert result["polarization"] < esteban_ray_polarization(
        np.concatenate([np.zeros(50), np.full(50, 100)]), n_groups=2
    )["polarization"]


# ── Rosenbaum Bounds ─────────────────────────────────────────────────────────

def test_rosenbaum_significant_at_gamma_1():
    treated = np.array([50, 60, 40, 55, 70, 45, 65, 80, 50, 60])
    control = np.array([5, 3, 2, 8, 4, 6, 1, 7, 3, 5])
    result = rosenbaum_bounds(treated, control)
    gamma_1 = result[result["gamma"] == 1.0]
    assert gamma_1.iloc[0]["significant"]


def test_rosenbaum_loses_significance_at_high_gamma():
    treated = np.array([10, 12, 11, 13, 10])
    control = np.array([8, 9, 7, 10, 8])
    result = rosenbaum_bounds(treated, control)
    high_gamma = result[result["gamma"] >= 3.0]
    # At high gamma, even real effects become non-significant
    assert not high_gamma.iloc[-1]["significant"]
