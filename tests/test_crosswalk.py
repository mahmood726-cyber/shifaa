"""Tests for GBD-MeSH crosswalk loader."""
from pathlib import Path
from shifaa.crosswalk.loader import load_crosswalk, match_condition

FIXTURES = Path(__file__).parent / "fixtures"

def test_load_crosswalk():
    cw = load_crosswalk(FIXTURES / "crosswalk_sample.csv")
    assert len(cw) == 3
    assert cw.iloc[0]["gbd_cause_id"] == 294

def test_match_exact():
    cw = load_crosswalk(FIXTURES / "crosswalk_sample.csv")
    result = match_condition("Type 2 Diabetes", cw)
    assert result is not None
    assert result["gbd_cause_id"] == 426

def test_match_case_insensitive():
    cw = load_crosswalk(FIXTURES / "crosswalk_sample.csv")
    result = match_condition("hiv", cw)
    assert result is not None
    assert result["gbd_cause_id"] == 341

def test_match_substring():
    cw = load_crosswalk(FIXTURES / "crosswalk_sample.csv")
    result = match_condition("Acute Myocardial Infarction", cw)
    assert result is not None
    assert result["gbd_cause_id"] == 294

def test_no_match_returns_none():
    cw = load_crosswalk(FIXTURES / "crosswalk_sample.csv")
    result = match_condition("Alien Abduction Syndrome", cw)
    assert result is None

def test_match_returns_required_keys():
    cw = load_crosswalk(FIXTURES / "crosswalk_sample.csv")
    result = match_condition("Angina", cw)
    assert "gbd_cause_id" in result
    assert "gbd_cause_name" in result
