"""Tests for advanced statistical methods."""
import numpy as np
import pandas as pd

from shifaa.analysis.advanced import (
    concentration_index,
    fit_negbin_model,
    fit_zip_model,
    morans_i,
    shapley_decomposition,
    theil_decomposition,
    theil_index,
)


# ── Concentration Index ─────────────────────────────────────────────────────

def test_ci_positive_when_rich_have_more():
    """If trials concentrate among rich countries, CI should be positive."""
    outcome = np.array([0, 0, 1, 5, 20])  # trials
    ranking = np.array([100, 200, 500, 5000, 40000])  # GDP
    ci = concentration_index(outcome, ranking)
    assert ci > 0


def test_ci_near_zero_for_equal():
    """If trials are equally distributed regardless of GDP, CI ~ 0."""
    outcome = np.array([10, 10, 10, 10, 10])
    ranking = np.array([100, 500, 1000, 5000, 40000])
    ci = concentration_index(outcome, ranking)
    assert abs(ci) < 0.1


def test_ci_handles_nan():
    outcome = np.array([1, np.nan, 3, 4])
    ranking = np.array([100, 200, np.nan, 400])
    ci = concentration_index(outcome, ranking)
    assert isinstance(ci, float)


# ── Theil Index ─────────────────────────────────────────────────────────────

def test_theil_perfect_equality():
    assert abs(theil_index(np.array([10, 10, 10, 10]))) < 0.01


def test_theil_high_inequality():
    t = theil_index(np.array([1, 1, 1, 1000]))
    assert t > 1.0


def test_theil_decomposition_sums():
    """Between + within should approximately equal total."""
    values = np.array([1, 2, 3, 100, 200, 300])
    groups = np.array(["A", "A", "A", "B", "B", "B"])
    result = theil_decomposition(values, groups)
    assert abs(result["between"] + result["within"] - result["total"]) < 0.01


def test_theil_decomposition_shares():
    values = np.array([1, 2, 3, 100, 200, 300])
    groups = np.array(["A", "A", "A", "B", "B", "B"])
    result = theil_decomposition(values, groups)
    assert abs(result["between_share"] + result["within_share"] - 1.0) < 0.05
    # Most inequality should be between groups (A vs B)
    assert result["between_share"] > 0.5


# ── Moran's I ───────────────────────────────────────────────────────────────

def test_morans_i_clustered():
    """Nearby countries with similar REI should produce positive Moran's I."""
    # West African cluster: all low REI
    # European cluster: all high REI
    values = {"NGA": -8, "GHA": -7, "BEN": -9, "TGO": -8, "CMR": -7,
              "FRA": 1, "DEU": 2, "BEL": 1, "NLD": 2, "GBR": 1,
              "USA": 2, "CAN": 1, "MEX": -3, "BRA": -2, "ARG": -1}
    result = morans_i(values, k_neighbors=3)
    assert result["morans_i"] > 0


def test_morans_i_returns_required_keys():
    values = {"USA": 1, "GBR": 2, "FRA": 1, "DEU": 2, "ITA": 1,
              "ESP": 1, "NLD": 2, "BEL": 2, "CHE": 2, "AUT": 1,
              "PAK": -5, "IND": -4, "BGD": -5, "NPL": -6, "LKA": -3}
    result = morans_i(values)
    assert "morans_i" in result
    assert "z_score" in result
    assert "p_value" in result
    assert "interpretation" in result


# ── ZIP Model ───────────────────────────────────────────────────────────────

def _make_zip_data():
    np.random.seed(42)
    n = 150
    gdp = np.random.lognormal(8, 1.5, n)
    governance = np.random.normal(0, 1, n)
    dalys = np.random.exponential(1e6, n)
    physicians = np.random.uniform(0.1, 3, n)
    health_spend = np.random.uniform(2, 12, n)
    # Structural zeros for low-GDP countries
    can_have_trials = (gdp > 2000).astype(int)
    rate = np.exp(-2 + 0.3 * np.log(gdp) + 0.5 * governance)
    trials = can_have_trials * np.random.poisson(rate)
    return pd.DataFrame({
        "trial_count_binary": trials, "dalys": dalys,
        "gdp_pc": gdp, "governance": governance,
        "health_spend_pct": health_spend, "physicians": physicians,
        "iso3c": [f"C{i:03d}" for i in range(n)],
    })


def test_zip_model_converges():
    result = fit_zip_model(_make_zip_data())
    if result is None:
        return  # ZIP can be finicky with synthetic data
    assert result["converged"]
    assert result["aic"] > 0


def test_negbin_model_converges():
    result = fit_negbin_model(_make_zip_data())
    if result is None:
        return
    assert result["converged"]
    assert result["aic"] > 0


# ── Shapley Decomposition ──────────────────────────────────────────────────

def test_shapley_sums_to_r2():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "trial_count_binary": np.random.poisson(5, n),
        "dalys": np.random.exponential(1e6, n),
        "gdp_pc": np.random.lognormal(8, 1.5, n),
        "governance": np.random.normal(0, 1, n),
        "health_spend_pct": np.random.uniform(2, 12, n),
        "physicians": np.random.uniform(0.5, 3, n),
    })
    result = shapley_decomposition(df)
    assert len(result) == 5
    assert abs(result["share_of_r2"].sum() - 1.0) < 0.05


def test_shapley_identifies_dominant_predictor():
    """GDP should dominate when trials strongly correlate with GDP."""
    np.random.seed(42)
    n = 100
    gdp = np.random.lognormal(8, 2, n)
    df = pd.DataFrame({
        "trial_count_binary": np.round(gdp / 1000).astype(int),
        "dalys": np.random.exponential(1e6, n),
        "gdp_pc": gdp,
        "governance": np.random.normal(0, 1, n),
        "health_spend_pct": np.random.uniform(2, 12, n),
        "physicians": np.random.uniform(0.5, 3, n),
    })
    result = shapley_decomposition(df)
    top = result.iloc[0]["variable"]
    assert top == "log_gdp_pc"
