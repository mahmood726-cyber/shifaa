"""Read harmonized/native Parquet from IHME, WB, and WHO lakehouses."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from shifaa.config import (
    IHME_REFERENCE,
    IHME_SILVER,
    WB_INDICATORS,
    WB_SILVER,
    WHO_SILVER,
)


def read_wb_indicators(silver_dir: Path | None = None) -> pd.DataFrame | None:
    """Read all WB harmonized parquet files into long format."""
    if silver_dir is None:
        silver_dir = WB_SILVER
    frames: list[pd.DataFrame] = []
    for domain_dir in silver_dir.iterdir():
        harm_dir = domain_dir / "harmonized"
        if not harm_dir.is_dir():
            continue
        for pq in harm_dir.glob("*.parquet"):
            df = pd.read_parquet(
                pq, columns=["iso3c", "year", "indicator_code", "value"]
            )
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["iso3c", "year", "indicator_code", "value"])
    combined = pd.concat(frames, ignore_index=True)
    return combined.dropna(subset=["iso3c", "year", "value"])


def pivot_wb_wide(
    long_df: pd.DataFrame,
    indicator_mapping: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Pivot WB long-format data to wide, one column per indicator."""
    if indicator_mapping is None:
        indicator_mapping = WB_INDICATORS
    codes_needed = set(indicator_mapping.values())
    filtered = long_df[long_df["indicator_code"].isin(codes_needed)].copy()
    code_to_name = {v: k for k, v in indicator_mapping.items()}
    filtered["indicator_name_short"] = filtered["indicator_code"].map(code_to_name)
    wide = (
        filtered.pivot_table(
            index=["iso3c", "year"],
            columns="indicator_name_short",
            values="value",
            aggfunc="first",
        )
        .reset_index()
    )
    wide.columns.name = None
    return wide


def read_ihme_dalys(
    silver_dir: Path | None = None,
    reference_dir: Path | None = None,
) -> pd.DataFrame | None:
    """Read IHME native DALYs parquet, filtered to both-sexes/all-ages/number."""
    if silver_dir is None:
        silver_dir = IHME_SILVER
    if reference_dir is None:
        reference_dir = IHME_REFERENCE
    dalys_path = silver_dir / "gbd_results" / "native" / "dalys.parquet"
    if not dalys_path.exists():
        return pd.DataFrame(columns=["iso3c", "year", "cause_id", "cause_name", "dalys"])
    df = pd.read_parquet(dalys_path)
    # Filter: both sexes (3), all ages (22), number metric (1)
    df = df[
        (df["sex_id"] == 3) & (df["age_id"] == 22) & (df["metric_id"] == 1)
    ].copy()
    crosswalk_path = reference_dir / "location_to_iso3.csv"
    if crosswalk_path.exists():
        loc_map = pd.read_csv(crosswalk_path)
        df = df.merge(loc_map[["location_id", "iso3c"]], on="location_id", how="left")
    else:
        df["iso3c"] = None
    result = df[["iso3c", "year", "cause_id", "cause_name", "val"]].copy()
    result = result.rename(columns={"val": "dalys"})
    return result.dropna(subset=["iso3c", "dalys"])


def read_who_health_system(
    silver_dir: Path | None = None,
) -> pd.DataFrame | None:
    """Read WHO World Health Statistics and HIDR parquet files."""
    if silver_dir is None:
        silver_dir = WHO_SILVER
    frames: list[pd.DataFrame] = []

    whs_dir = silver_dir / "world_health_statistics"
    if whs_dir.is_dir():
        for pq in whs_dir.glob("*.parquet"):
            df = pd.read_parquet(pq)
            if "dim_geo_code" in df.columns and "value_numeric" in df.columns:
                mapped = pd.DataFrame(
                    {
                        "iso3c": df["dim_geo_code"],
                        "year": pd.to_numeric(
                            df.get("dim_time_year", pd.Series(dtype="Int64")),
                            errors="coerce",
                        ).astype("Int64"),
                        "indicator_code": df.get("ind_code", ""),
                        "value": pd.to_numeric(
                            df["value_numeric"], errors="coerce"
                        ),
                    }
                )
                frames.append(mapped.dropna(subset=["iso3c", "value"]))

    hidr_dir = silver_dir / "hidr"
    if hidr_dir.is_dir():
        for pq in hidr_dir.glob("*.parquet"):
            df = pd.read_parquet(pq)
            if "iso3" in df.columns and "estimate" in df.columns:
                mapped = pd.DataFrame(
                    {
                        "iso3c": df["iso3"],
                        "year": pd.to_numeric(
                            df.get("date", pd.Series(dtype="Int64")),
                            errors="coerce",
                        ).astype("Int64"),
                        "indicator_code": df.get("indicator_abbr", ""),
                        "value": pd.to_numeric(df["estimate"], errors="coerce"),
                    }
                )
                frames.append(mapped.dropna(subset=["iso3c", "value"]))

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)
