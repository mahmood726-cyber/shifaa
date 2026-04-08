"""Counterfactual scenarios for Paper 4 (The Path / Siraat)."""
from __future__ import annotations
import numpy as np
import pandas as pd

def compute_counterfactual(deserts, governance_target=0.0, gov_coefficient=1.5):
    df = deserts.copy()
    gov_change = governance_target - df["governance"]
    trial_multiplier = np.exp(gov_coefficient * gov_change)
    df["trial_density_counterfactual"] = df["trial_density"] * trial_multiplier
    eps = 1e-10
    population = df.get("population", pd.Series(1.0, index=df.index))
    dalys_rate = df["dalys"] / (population / 100_000)
    trial_rate_cf = df["trial_density_counterfactual"] / (population / 100_000)
    df["rei_counterfactual"] = np.log10((trial_rate_cf + eps) / (dalys_rate + eps)).clip(-10, 10)
    df["governance_target"] = governance_target
    df["trial_multiplier"] = trial_multiplier
    return df
