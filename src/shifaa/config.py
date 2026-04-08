"""Paths to lakehouses, CT.gov API config, and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

IHME_SILVER = Path("C:/Projects/ihme-data-lakehouse/data/silver")
WB_SILVER = Path("C:/Projects/wb-data-lakehouse/data/silver")
WHO_SILVER = Path("C:/Projects/who-data-lakehouse/data/silver")
IHME_REFERENCE = Path("C:/Projects/ihme-data-lakehouse/data/reference")

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
