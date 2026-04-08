"""Multilevel Poisson regression for Paper 2 (The Balance)."""
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

def fit_poisson_model(matrix):
    df = matrix.dropna(subset=["trial_count_binary", "dalys", "gdp_pc", "governance", "health_spend_pct", "hci", "physicians", "population"]).copy()
    if len(df) < 20:
        return None
    df["log_dalys"] = np.log(df["dalys"].clip(lower=1))
    df["log_gdp_pc"] = np.log(df["gdp_pc"].clip(lower=1))
    df["log_population"] = np.log(df["population"].clip(lower=1))
    try:
        model = smf.gee("trial_count_binary ~ log_dalys + log_gdp_pc + governance + health_spend_pct + hci + physicians",
            groups="iso3c", data=df, family=sm.families.Poisson(), cov_struct=sm.cov_struct.Exchangeable(), offset=df["log_population"])
        result = model.fit()
        return {"model": result, "converged": True, "n_obs": len(df), "n_countries": df["iso3c"].nunique()}
    except Exception:
        return None

def extract_coefficients(result):
    model = result["model"]
    summary = model.summary2().tables[1]
    coefs = pd.DataFrame({"variable": summary.index, "coefficient": summary["Coef."].values,
        "std_err": summary["Std.Err."].values, "z_value": summary["z"].values,
        "p_value": summary["P>|z|"].values, "ci_lower": summary["[0.025"].values,
        "ci_upper": summary["0.975]"].values}).reset_index(drop=True)
    return coefs[coefs["variable"] != "Intercept"].reset_index(drop=True)
