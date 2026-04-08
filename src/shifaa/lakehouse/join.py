"""Build the unified shifaa_matrix from all data sources."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd


def build_shifaa_matrix(dalys_df, trials_df, wb_wide_df, who_df=None):
    matrix = dalys_df.rename(columns={"cause_id": "gbd_cause_id"}).copy()
    matrix = matrix.merge(trials_df, on=["iso3c", "year", "gbd_cause_id"], how="left")
    matrix["trial_density"] = matrix["trial_density"].fillna(0)
    matrix["trial_count_binary"] = matrix.get("trial_count_binary", pd.Series(0, index=matrix.index)).fillna(0).astype(int)
    matrix = matrix.merge(wb_wide_df, on=["iso3c", "year"], how="left")
    if who_df is not None and not who_df.empty:
        who_wide = who_df.pivot_table(index=["iso3c", "year"], columns="indicator_code", values="value", aggfunc="first").reset_index()
        who_wide.columns.name = None
        matrix = matrix.merge(who_wide, on=["iso3c", "year"], how="left")

    population = matrix.get("population", pd.Series(dtype="float64"))
    pop_missing = population.isna().sum()
    if pop_missing > 0:
        print(f"WARNING: {pop_missing}/{len(matrix)} rows missing population — REI rates will use raw counts", file=sys.stderr)
    if population.notna().any():
        dalys_per_100k = matrix["dalys"] / (population / 100_000)
        trial_density_per_100k = matrix["trial_density"] / (population / 100_000)
    else:
        dalys_per_100k = matrix["dalys"]
        trial_density_per_100k = matrix["trial_density"]

    matrix["dalys_per_100k"] = dalys_per_100k
    eps = 1e-10
    ratio = (trial_density_per_100k + eps) / (dalys_per_100k + eps)
    matrix["rei_score"] = np.log10(ratio).clip(-10, 10)
    return matrix
