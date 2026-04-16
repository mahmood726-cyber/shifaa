"""Paths to lakehouses, CT.gov API config, and constants."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _lake_root(env_var: str, default_drive: str, repo: str) -> Path:
    override = os.environ.get(env_var)
    if override:
        return Path(override)
    for drive in (default_drive, "D", "C"):
        candidate = Path(f"{drive}:/Projects/{repo}")
        if candidate.exists():
            return candidate
    return Path(f"{default_drive}:/Projects/{repo}")


_IHME_ROOT = _lake_root("SHIFAA_IHME_ROOT", "D", "ihme-data-lakehouse")
_WB_ROOT = _lake_root("SHIFAA_WB_ROOT", "D", "wb-data-lakehouse")
_WHO_ROOT = _lake_root("SHIFAA_WHO_ROOT", "D", "who-data-lakehouse")

IHME_SILVER = _IHME_ROOT / "data" / "silver"
WB_SILVER = _WB_ROOT / "data" / "silver"
WHO_SILVER = _WHO_ROOT / "data" / "silver"
IHME_REFERENCE = _IHME_ROOT / "data" / "reference"

CTGOV_BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
CTGOV_CACHE_DIR = ROOT / "data" / "cache" / "ctgov"
CTGOV_PAGE_SIZE = 100
CTGOV_TIMEOUT = 60

ANALYSIS_YEARS = range(2005, 2026)
REI_MIN_TRIALS = 0
REI_MIN_DALYS = 100

CROSSWALK_PATH = ROOT / "src" / "shifaa" / "crosswalk" / "gbd_mesh_icd.csv"
OUTPUT_DIR = ROOT / "output"

WB_INDICATORS = {
    "gdp_pc": "wb_NY.GDP.PCAP.CD",
    "health_spend_pct": "wb_SH.XPD.CHEX.GD.ZS",
    "governance": "wb_GE.EST",
    "hci": "wb_HD.HCI.OVRL",
    "physicians": "wb_SH.MED.PHYS.ZS",
    "uhc_index": "wb_SH.UHC.SRVS.CV.XD",
    "population": "wb_SP.POP.TOTL",
    "life_expectancy": "wb_SP.DYN.LE00.IN",
}

WHO_INDICATORS = {
    "hospital_beds": "SH.MED.BEDS.ZS",
}
