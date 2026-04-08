"""Batch-fetch CT.gov trials for all crosswalk causes and build trial matrix."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from shifaa.config import CTGOV_CACHE_DIR
from shifaa.ctgov.allocate import allocate_trial
from shifaa.ctgov.api import build_session, search_studies_by_condition
from shifaa.crosswalk.loader import load_crosswalk


def fetch_trials_for_cause(
    cause_row: dict,
    cache_dir: Path | None = None,
    session=None,
    max_pages: int = 50,
) -> dict:
    """Fetch CT.gov trials for a single GBD cause, with JSON caching.

    Parameters
    ----------
    cause_row : dict
        Row from crosswalk with at least ``gbd_cause_id``, ``gbd_cause_name``,
        and ``mesh_terms`` (semicolon-separated).
    cache_dir : Path | None
        Directory for per-cause JSON caches.  Defaults to ``CTGOV_CACHE_DIR``.
    session : requests.Session | None
        Reusable session.  Built automatically if *None*.
    max_pages : int
        Maximum API pages to fetch.

    Returns
    -------
    dict
        Summary with keys: gbd_cause_id, gbd_cause_name, search_term,
        total_studies, allocated_studies, trials.
    """
    if cache_dir is None:
        cache_dir = CTGOV_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    gbd_id = cause_row["gbd_cause_id"]
    mesh_terms = str(cause_row.get("mesh_terms", "")).split(";")
    primary_term = mesh_terms[0].strip() if mesh_terms else cause_row["gbd_cause_name"]

    cache_file = cache_dir / f"cause_{gbd_id}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text("utf-8"))

    if session is None:
        session = build_session()

    studies = search_studies_by_condition(
        primary_term, session=session, max_pages=max_pages,
    )

    allocated: list[dict] = []
    for study in studies:
        alloc = allocate_trial(study)
        if alloc:
            allocated.append({
                "nct_id": study["nct_id"],
                "start_year": study["start_year"],
                "gbd_cause_id": gbd_id,
                "allocation": alloc,
            })

    result = {
        "gbd_cause_id": gbd_id,
        "gbd_cause_name": cause_row.get("gbd_cause_name", ""),
        "search_term": primary_term,
        "total_studies": len(studies),
        "allocated_studies": len(allocated),
        "trials": allocated,
    }
    cache_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def fetch_all_causes(
    crosswalk_path: Path | None = None,
    cache_dir: Path | None = None,
) -> list[dict]:
    """Fetch trials for every cause in the crosswalk.

    Parameters
    ----------
    crosswalk_path : Path | None
        CSV path.  Defaults to ``CROSSWALK_PATH``.
    cache_dir : Path | None
        Cache directory.  Defaults to ``CTGOV_CACHE_DIR``.

    Returns
    -------
    list[dict]
        One summary dict per crosswalk cause.
    """
    cw = load_crosswalk(crosswalk_path)
    session = build_session()
    results: list[dict] = []
    try:
        for _, row in cw.iterrows():
            results.append(
                fetch_trials_for_cause(
                    row.to_dict(), cache_dir=cache_dir, session=session,
                ),
            )
    finally:
        session.close()
    return results


def build_trial_matrix(allocated_trials: list[dict]) -> pd.DataFrame:
    """Aggregate allocated trials into a country-year-cause density matrix.

    Parameters
    ----------
    allocated_trials : list[dict]
        Flat list of trial dicts, each with ``start_year``, ``gbd_cause_id``,
        and ``allocation`` (iso3c -> weight mapping).

    Returns
    -------
    pd.DataFrame
        Columns: iso3c, year, gbd_cause_id, trial_density, trial_count_binary.
    """
    rows: list[dict] = []
    for trial in allocated_trials:
        year = trial.get("start_year")
        gbd_id = trial.get("gbd_cause_id")
        alloc = trial.get("allocation", {})
        if not year or not gbd_id or not alloc:
            continue
        for iso3c, weight in alloc.items():
            rows.append({
                "iso3c": iso3c,
                "year": year,
                "gbd_cause_id": gbd_id,
                "trial_weight": weight,
            })

    if not rows:
        return pd.DataFrame(
            columns=["iso3c", "year", "gbd_cause_id", "trial_density", "trial_count_binary"],
        )

    df = pd.DataFrame(rows)
    grouped = (
        df.groupby(["iso3c", "year", "gbd_cause_id"])
        .agg(
            trial_density=("trial_weight", "sum"),
            trial_count_binary=("trial_weight", "count"),
        )
        .reset_index()
    )
    return grouped
