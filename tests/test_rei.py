"""Tests for REI computation and ranking."""
import pandas as pd
from shifaa.analysis.rei import rank_evidence_deserts, summarize_rei_by_country

def _make_matrix():
    return pd.DataFrame({"iso3c": ["PAK", "PAK", "GBR", "GBR", "NGA", "NGA"], "year": [2020]*6,
        "gbd_cause_id": [294, 426, 294, 426, 294, 426], "cause_name": ["IHD", "T2DM"]*3,
        "dalys": [4.2e6, 1.5e6, 1.1e6, 0.8e6, 3.0e6, 1.0e6],
        "trial_density": [0.5, 0.1, 12.4, 8.1, 0.0, 0.0],
        "rei_score": [-3.2, -4.0, 1.8, 1.5, -10.0, -10.0],
        "population": [220e6]*2 + [68e6]*2 + [206e6]*2})

def test_rank_deserts_returns_sorted():
    deserts = rank_evidence_deserts(_make_matrix(), top_n=50)
    scores = deserts["rei_score"].tolist()
    assert scores == sorted(scores)

def test_rank_deserts_nga_worst():
    deserts = rank_evidence_deserts(_make_matrix(), top_n=10)
    assert deserts.iloc[0]["iso3c"] == "NGA"

def test_summarize_by_country():
    summary = summarize_rei_by_country(_make_matrix())
    assert "mean_rei" in summary.columns
    nga = summary[summary["iso3c"] == "NGA"]
    assert nga.iloc[0]["mean_rei"] < -5

def test_summarize_includes_disease_count():
    summary = summarize_rei_by_country(_make_matrix())
    pak = summary[summary["iso3c"] == "PAK"]
    assert pak.iloc[0]["n_diseases"] == 2
