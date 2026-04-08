"""REI computation, ranking, and summary for Paper 1 (The Mercy Map)."""
from __future__ import annotations
import pandas as pd

def rank_evidence_deserts(matrix, top_n=50, year=None):
    df = matrix.copy()
    if year is not None:
        df = df[df["year"] == year]
    elif "year" in df.columns:
        df = df[df["year"] == df["year"].max()]
    return df[["iso3c", "year", "gbd_cause_id", "cause_name", "dalys", "trial_density", "rei_score"]].sort_values("rei_score", ascending=True).head(top_n).reset_index(drop=True)

def summarize_rei_by_country(matrix, year=None):
    df = matrix.copy()
    if year is not None:
        df = df[df["year"] == year]
    elif "year" in df.columns:
        df = df[df["year"] == df["year"].max()]
    return df.groupby("iso3c").agg(mean_rei=("rei_score", "mean"), median_rei=("rei_score", "median"),
        n_diseases=("gbd_cause_id", "nunique"), total_dalys=("dalys", "sum"),
        total_trials=("trial_density", "sum")).reset_index().sort_values("mean_rei", ascending=True).reset_index(drop=True)

def summarize_rei_by_disease(matrix, year=None):
    df = matrix.copy()
    if year is not None:
        df = df[df["year"] == year]
    elif "year" in df.columns:
        df = df[df["year"] == df["year"].max()]
    return df.groupby(["gbd_cause_id", "cause_name"]).agg(mean_rei=("rei_score", "mean"),
        n_countries=("iso3c", "nunique"), total_dalys=("dalys", "sum")).reset_index().sort_values("mean_rei", ascending=True).reset_index(drop=True)
