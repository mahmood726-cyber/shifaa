"""Tests for equity trend analysis."""
import numpy as np
import pandas as pd
from shifaa.analysis.equity_trend import compute_gini, compute_lorenz, compute_annual_gini

def test_gini_perfect_equality():
    assert abs(compute_gini(np.array([10.0, 10.0, 10.0, 10.0])) - 0.0) < 0.01

def test_gini_perfect_inequality():
    assert compute_gini(np.array([0.0, 0.0, 0.0, 100.0])) > 0.7

def test_gini_moderate():
    g = compute_gini(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert 0.1 < g < 0.5

def test_lorenz_endpoints():
    pop, inc = compute_lorenz(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert abs(pop[0]) < 0.01 and abs(pop[-1] - 1.0) < 0.01 and abs(inc[-1] - 1.0) < 0.01

def test_annual_gini():
    matrix = pd.DataFrame({"iso3c": ["A", "B", "C"]*3, "year": [2019]*3 + [2020]*3 + [2021]*3,
        "trial_density": [1, 10, 100, 2, 20, 200, 3, 30, 300], "dalys": [100]*9})
    trend = compute_annual_gini(matrix)
    assert len(trend) == 3 and "gini" in trend.columns
