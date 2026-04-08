"""Multilevel Poisson regression for Paper 2 (The Balance)."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def fit_poisson_model(matrix):
    required = ["trial_count_binary", "dalys", "gdp_pc", "governance",
                 "health_spend_pct", "hci", "physicians", "population"]
    missing_cols = [c for c in required if c not in matrix.columns]
    if missing_cols:
        print(f"WARNING: regression missing columns: {missing_cols}", file=sys.stderr)
        return None

    df = matrix.dropna(subset=required).copy()
    if len(df) < 20:
        print(f"WARNING: regression has only {len(df)} complete rows (need 20+)", file=sys.stderr)
        return None
    df["log_dalys"] = np.log(df["dalys"].clip(lower=1))
    df["log_gdp_pc"] = np.log(df["gdp_pc"].clip(lower=1))
    df["log_population"] = np.log(df["population"].clip(lower=1))
    try:
        model = smf.gee(
            "trial_count_binary ~ log_dalys + log_gdp_pc + governance + health_spend_pct + hci + physicians",
            groups="iso3c", data=df, family=sm.families.Poisson(),
            cov_struct=sm.cov_struct.Exchangeable(), offset=df["log_population"],
        )
        result = model.fit()
        return {"model": result, "converged": True, "n_obs": len(df), "n_countries": df["iso3c"].nunique()}
    except Exception as exc:
        print(f"WARNING: regression failed: {exc}", file=sys.stderr)
        return None


def extract_coefficients(result):
    model = result["model"]
    ci = model.conf_int()
    coefs = pd.DataFrame({
        "variable": model.params.index,
        "coefficient": model.params.values,
        "std_err": model.bse.values,
        "z_value": model.tvalues.values,
        "p_value": model.pvalues.values,
        "ci_lower": ci.iloc[:, 0].values,
        "ci_upper": ci.iloc[:, 1].values,
    }).reset_index(drop=True)
    return coefs[coefs["variable"] != "Intercept"].reset_index(drop=True)
