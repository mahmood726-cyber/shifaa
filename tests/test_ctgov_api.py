"""Tests for CT.gov API client (mocked responses)."""
import json
from pathlib import Path
from unittest.mock import MagicMock

from shifaa.ctgov.api import parse_study, search_studies_by_condition

FIXTURES = Path(__file__).parent / "fixtures"


def _load_fixture():
    return json.loads((FIXTURES / "ctgov_study_sample.json").read_text("utf-8"))


def test_parse_study_extracts_nct_id():
    parsed = parse_study(_load_fixture()["studies"][0])
    assert parsed["nct_id"] == "NCT00000001"


def test_parse_study_extracts_conditions():
    parsed = parse_study(_load_fixture()["studies"][0])
    assert "Diabetes Mellitus, Type 2" in parsed["conditions"]
    assert len(parsed["conditions"]) == 2


def test_parse_study_extracts_locations():
    parsed = parse_study(_load_fixture()["studies"][0])
    assert len(parsed["locations"]) == 4
    countries = [loc["country"] for loc in parsed["locations"]]
    assert "Pakistan" in countries


def test_parse_study_extracts_start_year():
    parsed = parse_study(_load_fixture()["studies"][0])
    assert parsed["start_year"] == 2020


def test_parse_study_extracts_enrollment():
    parsed = parse_study(_load_fixture()["studies"][0])
    assert parsed["enrollment"] == 500


def test_search_studies_mocked():
    data = _load_fixture()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = data
    mock_resp.raise_for_status.return_value = None
    session = MagicMock()
    session.get.return_value = mock_resp
    studies = search_studies_by_condition("diabetes", session=session, max_pages=1)
    assert len(studies) == 1
    assert studies[0]["nct_id"] == "NCT00000001"
