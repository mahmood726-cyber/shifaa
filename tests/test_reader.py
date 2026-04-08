"""Tests for lakehouse reader."""
import pandas as pd
import pytest

from shifaa.lakehouse.reader import (
    pivot_wb_wide,
    read_ihme_dalys,
    read_wb_indicators,
)


def test_read_wb_indicators_returns_dataframe():
    df = read_wb_indicators()
    if df is None:
        pytest.skip("WB lakehouse not available")
    assert "iso3c" in df.columns
    assert "year" in df.columns
    assert len(df) > 0


def test_read_ihme_dalys_returns_dataframe():
    df = read_ihme_dalys()
    if df is None or len(df) == 0:
        pytest.skip("IHME lakehouse not available or crosswalk missing")
    assert "cause_id" in df.columns
    assert "iso3c" in df.columns
    assert len(df) > 0


def test_pivot_wb_wide():
    long = pd.DataFrame(
        {
            "iso3c": ["PAK", "PAK", "GBR", "GBR"],
            "year": [2020, 2020, 2020, 2020],
            "indicator_code": [
                "wb_NY.GDP.PCAP.CD",
                "wb_GE.EST",
                "wb_NY.GDP.PCAP.CD",
                "wb_GE.EST",
            ],
            "value": [1194.0, -0.78, 41059.0, 1.69],
        }
    )
    mapping = {"gdp_pc": "wb_NY.GDP.PCAP.CD", "governance": "wb_GE.EST"}
    wide = pivot_wb_wide(long, mapping)
    assert "gdp_pc" in wide.columns
    assert "governance" in wide.columns
    assert len(wide) == 2
    pak = wide[wide["iso3c"] == "PAK"]
    assert abs(pak.iloc[0]["gdp_pc"] - 1194.0) < 0.01


def test_pivot_wb_wide_handles_missing_indicators():
    long = pd.DataFrame(
        {
            "iso3c": ["PAK"],
            "year": [2020],
            "indicator_code": ["wb_NY.GDP.PCAP.CD"],
            "value": [1194.0],
        }
    )
    mapping = {"gdp_pc": "wb_NY.GDP.PCAP.CD", "governance": "wb_GE.EST"}
    wide = pivot_wb_wide(long, mapping)
    assert "gdp_pc" in wide.columns
    assert len(wide) == 1


def test_read_wb_has_required_columns():
    df = read_wb_indicators()
    if df is None:
        pytest.skip("WB lakehouse not available")
    # All rows must have non-null iso3c, year, and value (enforced by dropna)
    assert df["iso3c"].notna().all()
    assert df["year"].notna().all()
    assert df["value"].notna().all()
