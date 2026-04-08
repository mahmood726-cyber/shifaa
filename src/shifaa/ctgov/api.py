"""CT.gov API v2 client: search studies by condition, parse responses."""
from __future__ import annotations

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from shifaa.config import CTGOV_BASE_URL, CTGOV_PAGE_SIZE, CTGOV_TIMEOUT

FIELDS = ",".join([
    "protocolSection.identificationModule.nctId",
    "protocolSection.identificationModule.briefTitle",
    "protocolSection.statusModule.overallStatus",
    "protocolSection.statusModule.startDateStruct",
    "protocolSection.conditionsModule.conditions",
    "protocolSection.designModule.enrollmentInfo",
    "protocolSection.contactsLocationsModule.locations",
])


def build_session() -> requests.Session:
    """Create a requests session with retry logic for CT.gov API."""
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.headers.update({"User-Agent": "shifaa/0.1"})
    session.mount("https://", adapter)
    return session


def search_studies_by_condition(
    condition: str,
    session: requests.Session | None = None,
    max_pages: int = 50,
) -> list[dict]:
    """Search CT.gov for studies matching a condition, paginating through results."""
    if session is None:
        session = build_session()
    params: dict = {
        "query.cond": condition,
        "pageSize": CTGOV_PAGE_SIZE,
        "fields": FIELDS,
    }
    all_studies: list[dict] = []
    page_token = None
    for _ in range(max_pages):
        if page_token:
            params["pageToken"] = page_token
        resp = session.get(CTGOV_BASE_URL, params=params, timeout=CTGOV_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        for study in data.get("studies", []):
            parsed = parse_study(study)
            if parsed:
                all_studies.append(parsed)
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return all_studies


def parse_study(study: dict) -> dict | None:
    """Extract key fields from a CT.gov study record."""
    proto = study.get("protocolSection", {})
    ident = proto.get("identificationModule", {})
    status = proto.get("statusModule", {})
    conds = proto.get("conditionsModule", {})
    design = proto.get("designModule", {})
    contacts = proto.get("contactsLocationsModule", {})

    nct_id = ident.get("nctId")
    if not nct_id:
        return None

    start_date = status.get("startDateStruct", {}).get("date", "")
    start_year = None
    if start_date and len(start_date) >= 4:
        try:
            start_year = int(start_date[:4])
        except ValueError:
            pass

    enrollment = design.get("enrollmentInfo", {}).get("count")

    locations = []
    for loc in contacts.get("locations", []):
        if loc.get("country"):
            locations.append({
                "facility": loc.get("facility", ""),
                "city": loc.get("city", ""),
                "country": loc["country"],
            })

    return {
        "nct_id": nct_id,
        "title": ident.get("briefTitle", ""),
        "status": status.get("overallStatus", ""),
        "start_year": start_year,
        "enrollment": enrollment,
        "conditions": conds.get("conditions", []),
        "locations": locations,
    }
