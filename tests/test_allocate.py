"""Tests for trial site-weighted country allocation."""
from shifaa.ctgov.allocate import allocate_trial, build_country_iso3_map


def test_allocate_single_country():
    trial = {"nct_id": "NCT001", "locations": [
        {"country": "Pakistan", "facility": "A", "city": "Lahore"},
        {"country": "Pakistan", "facility": "B", "city": "Karachi"},
    ]}
    alloc = allocate_trial(trial)
    assert len(alloc) == 1
    assert alloc["PAK"] == 1.0


def test_allocate_multi_country_weighted():
    trial = {"nct_id": "NCT002", "locations": [
        {"country": "United Kingdom", "facility": "A", "city": "London"},
        {"country": "United Kingdom", "facility": "B", "city": "London"},
        {"country": "Pakistan", "facility": "C", "city": "Lahore"},
        {"country": "United States", "facility": "D", "city": "NYC"},
    ]}
    alloc = allocate_trial(trial)
    assert abs(alloc["GBR"] - 0.5) < 0.01
    assert abs(alloc["PAK"] - 0.25) < 0.01
    assert abs(alloc["USA"] - 0.25) < 0.01


def test_allocate_empty_locations():
    alloc = allocate_trial({"nct_id": "NCT003", "locations": []})
    assert alloc == {}


def test_allocate_unknown_country_skipped():
    alloc = allocate_trial({"nct_id": "NCT004", "locations": [
        {"country": "Atlantis", "facility": "X", "city": "Y"},
    ]})
    assert alloc == {}


def test_build_country_map_has_common_entries():
    cmap = build_country_iso3_map()
    assert cmap["United States"] == "USA"
    assert cmap["Pakistan"] == "PAK"
    assert len(cmap) > 100
