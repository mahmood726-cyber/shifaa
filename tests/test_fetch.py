"""Tests for CT.gov batch fetcher."""
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from shifaa.ctgov.fetch import build_trial_matrix, fetch_trials_for_cause

FIXTURES = Path(__file__).parent / "fixtures"


def _mock_search(*args, **kwargs):
    data = json.loads((FIXTURES / "ctgov_study_sample.json").read_text("utf-8"))
    from shifaa.ctgov.api import parse_study
    return [parse_study(s) for s in data["studies"]]


def test_fetch_trials_for_cause(tmp_path):
    cause = {
        "gbd_cause_id": 426,
        "gbd_cause_name": "Diabetes mellitus type 2",
        "mesh_terms": "Type 2 Diabetes",
    }
    with patch("shifaa.ctgov.fetch.search_studies_by_condition", side_effect=_mock_search):
        result = fetch_trials_for_cause(cause, cache_dir=tmp_path)
    assert result["gbd_cause_id"] == 426
    assert result["total_studies"] >= 1


def test_build_trial_matrix():
    trials = [
        {
            "nct_id": "NCT001",
            "start_year": 2020,
            "gbd_cause_id": 426,
            "allocation": {"PAK": 0.25, "GBR": 0.5, "USA": 0.25},
        },
        {
            "nct_id": "NCT002",
            "start_year": 2020,
            "gbd_cause_id": 426,
            "allocation": {"PAK": 1.0},
        },
    ]
    matrix = build_trial_matrix(trials)
    pak_row = matrix[
        (matrix["iso3c"] == "PAK")
        & (matrix["year"] == 2020)
        & (matrix["gbd_cause_id"] == 426)
    ]
    assert len(pak_row) == 1
    assert abs(pak_row.iloc[0]["trial_density"] - 1.25) < 0.01


def test_build_trial_matrix_binary_column():
    trials = [
        {
            "nct_id": "NCT001",
            "start_year": 2020,
            "gbd_cause_id": 294,
            "allocation": {"PAK": 0.5, "GBR": 0.5},
        },
    ]
    matrix = build_trial_matrix(trials)
    pak = matrix[matrix["iso3c"] == "PAK"]
    assert pak.iloc[0]["trial_count_binary"] == 1


def test_build_trial_matrix_empty():
    matrix = build_trial_matrix([])
    assert len(matrix) == 0
    assert "trial_density" in matrix.columns
