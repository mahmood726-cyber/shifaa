"""Load and query the GBD-MeSH-ICD crosswalk."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from shifaa.config import CROSSWALK_PATH

def load_crosswalk(path: Path | None = None) -> pd.DataFrame:
    if path is None:
        path = CROSSWALK_PATH
    return pd.read_csv(path, dtype={"gbd_cause_id": int})

def match_condition(condition_text: str, crosswalk: pd.DataFrame) -> dict | None:
    cond_lower = condition_text.lower()
    for _, row in crosswalk.iterrows():
        mesh_terms = str(row["mesh_terms"]).split(";")
        for term in mesh_terms:
            term_lower = term.strip().lower()
            if not term_lower:
                continue
            if term_lower in cond_lower or cond_lower in term_lower:
                return {
                    "gbd_cause_id": int(row["gbd_cause_id"]),
                    "gbd_cause_name": row["gbd_cause_name"],
                    "matched_term": term.strip(),
                }
    return None

def match_conditions_batch(conditions: list[str], crosswalk: pd.DataFrame) -> list[dict | None]:
    return [match_condition(c, crosswalk) for c in conditions]
