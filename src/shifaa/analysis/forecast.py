"""REI projection to future years (Paper 3)."""
from __future__ import annotations
import numpy as np
import pandas as pd

def project_rei(current, target_year=2036, default_dalys_growth=0.02, default_trial_growth=0.01):
    df = current.copy()
    base_year = df["year"].max() if "year" in df.columns else 2020
    years_ahead = target_year - base_year
    dalys_growth = df.get("dalys_growth_rate", pd.Series(default_dalys_growth, index=df.index))
    trial_growth = df.get("trial_growth_rate", pd.Series(default_trial_growth, index=df.index))
    projected_dalys = df["dalys"] * (1 + dalys_growth) ** years_ahead
    projected_trials = df["trial_density"] * (1 + trial_growth) ** years_ahead
    eps = 1e-10
    population = df.get("population", pd.Series(1.0, index=df.index))
    dalys_rate = projected_dalys / (population / 100_000)
    trial_rate = projected_trials / (population / 100_000)
    result = df[["iso3c", "gbd_cause_id"]].copy()
    result["year"] = target_year
    result["rei_current"] = df["rei_score"]
    result["rei_projected"] = np.log10((trial_rate + eps) / (dalys_rate + eps)).clip(-10, 10)
    result["dalys_projected"] = projected_dalys
    result["rei_change"] = result["rei_projected"] - result["rei_current"]
    return result
