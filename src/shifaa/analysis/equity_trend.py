"""Gini coefficient and Lorenz curve for research equity trends (Paper 5)."""
from __future__ import annotations
import numpy as np
import pandas as pd

def compute_gini(values):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0 or values.sum() == 0:
        return 0.0
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_vals) / (n * np.sum(sorted_vals))) - (n + 1) / n

def compute_lorenz(values):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    sorted_vals = np.sort(values)
    cumulative = np.cumsum(sorted_vals)
    total = cumulative[-1] if cumulative[-1] > 0 else 1.0
    pop_frac = np.concatenate([[0], np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)])
    income_frac = np.concatenate([[0], cumulative / total])
    return pop_frac, income_frac

def compute_annual_gini(matrix, value_col="trial_density"):
    rows = []
    for year, group in matrix.groupby("year"):
        country_totals = group.groupby("iso3c")[value_col].sum().values
        rows.append({"year": year, "gini": compute_gini(country_totals), "n_countries": len(country_totals)})
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
