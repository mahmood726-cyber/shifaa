"""Site-weighted country allocation for multi-country trials."""
from __future__ import annotations
from collections import Counter

_COUNTRY_ISO3: dict[str, str] = {
    "Afghanistan": "AFG", "Albania": "ALB", "Algeria": "DZA", "Angola": "AGO",
    "Argentina": "ARG", "Armenia": "ARM", "Australia": "AUS", "Austria": "AUT",
    "Azerbaijan": "AZE", "Bahrain": "BHR", "Bangladesh": "BGD", "Belarus": "BLR",
    "Belgium": "BEL", "Benin": "BEN", "Bolivia": "BOL", "Bosnia and Herzegovina": "BIH",
    "Botswana": "BWA", "Brazil": "BRA", "Brunei": "BRN", "Bulgaria": "BGR",
    "Burkina Faso": "BFA", "Burundi": "BDI", "Cambodia": "KHM", "Cameroon": "CMR",
    "Canada": "CAN", "Chad": "TCD", "Chile": "CHL", "China": "CHN",
    "Colombia": "COL", "Congo": "COG",
    "Congo, The Democratic Republic of the": "COD",
    "Costa Rica": "CRI", "Croatia": "HRV", "Cuba": "CUB", "Cyprus": "CYP",
    "Czech Republic": "CZE", "Czechia": "CZE",
    "Denmark": "DNK", "Dominican Republic": "DOM",
    "Ecuador": "ECU", "Egypt": "EGY", "El Salvador": "SLV", "Estonia": "EST",
    "Ethiopia": "ETH", "Finland": "FIN", "France": "FRA", "Gabon": "GAB",
    "Gambia": "GMB", "Georgia": "GEO", "Germany": "DEU", "Ghana": "GHA",
    "Greece": "GRC", "Guatemala": "GTM", "Guinea": "GIN", "Haiti": "HTI",
    "Honduras": "HND", "Hong Kong": "HKG", "Hungary": "HUN", "Iceland": "ISL",
    "India": "IND", "Indonesia": "IDN", "Iran": "IRN",
    "Iran, Islamic Republic of": "IRN",
    "Iraq": "IRQ", "Ireland": "IRL", "Israel": "ISR", "Italy": "ITA",
    "Jamaica": "JAM", "Japan": "JPN", "Jordan": "JOR", "Kazakhstan": "KAZ",
    "Kenya": "KEN", "Korea, Republic of": "KOR", "South Korea": "KOR",
    "Kuwait": "KWT", "Kyrgyzstan": "KGZ",
    "Latvia": "LVA", "Lebanon": "LBN", "Libya": "LBY", "Lithuania": "LTU",
    "Luxembourg": "LUX", "Madagascar": "MDG", "Malawi": "MWI", "Malaysia": "MYS",
    "Mali": "MLI", "Malta": "MLT", "Mauritius": "MUS", "Mexico": "MEX",
    "Moldova": "MDA", "Mongolia": "MNG", "Montenegro": "MNE", "Morocco": "MAR",
    "Mozambique": "MOZ", "Myanmar": "MMR", "Namibia": "NAM", "Nepal": "NPL",
    "Netherlands": "NLD", "New Zealand": "NZL", "Nicaragua": "NIC", "Niger": "NER",
    "Nigeria": "NGA", "North Macedonia": "MKD", "Norway": "NOR", "Oman": "OMN",
    "Pakistan": "PAK", "Panama": "PAN", "Paraguay": "PRY", "Peru": "PER",
    "Philippines": "PHL", "Poland": "POL", "Portugal": "PRT", "Puerto Rico": "PRI",
    "Qatar": "QAT", "Romania": "ROU", "Russian Federation": "RUS", "Russia": "RUS",
    "Rwanda": "RWA", "Saudi Arabia": "SAU", "Senegal": "SEN", "Serbia": "SRB",
    "Sierra Leone": "SLE", "Singapore": "SGP", "Slovakia": "SVK", "Slovenia": "SVN",
    "Somalia": "SOM", "South Africa": "ZAF", "Spain": "ESP", "Sri Lanka": "LKA",
    "Sudan": "SDN", "Sweden": "SWE", "Switzerland": "CHE", "Syria": "SYR",
    "Taiwan": "TWN", "Tajikistan": "TJK", "Tanzania": "TZA",
    "Tanzania, United Republic of": "TZA",
    "Thailand": "THA", "Togo": "TGO", "Trinidad and Tobago": "TTO",
    "Tunisia": "TUN", "Turkey": "TUR", "Turkiye": "TUR",
    "Uganda": "UGA", "Ukraine": "UKR",
    "United Arab Emirates": "ARE", "United Kingdom": "GBR", "United States": "USA",
    "Uruguay": "URY", "Uzbekistan": "UZB", "Venezuela": "VEN",
    "Viet Nam": "VNM", "Vietnam": "VNM",
    "Yemen": "YEM", "Zambia": "ZMB", "Zimbabwe": "ZWE",
}


def build_country_iso3_map() -> dict[str, str]:
    """Return a copy of the country-name to ISO 3166-1 alpha-3 mapping."""
    return dict(_COUNTRY_ISO3)


def allocate_trial(trial: dict) -> dict[str, float]:
    """Compute site-weighted country allocation for a trial.

    Each facility location counts as one site.  The allocation for each
    country is the fraction of total sites located in that country.

    Parameters
    ----------
    trial : dict
        Trial record with an optional ``locations`` list.  Each location
        must have a ``country`` key matching a CT.gov English country name.

    Returns
    -------
    dict[str, float]
        Mapping of ISO 3166-1 alpha-3 codes to fractional allocations
        summing to 1.0.  Empty dict if no recognised locations.
    """
    locations = trial.get("locations", [])
    if not locations:
        return {}
    country_counts: Counter[str] = Counter()
    for loc in locations:
        country_name = loc.get("country", "")
        iso3c = _COUNTRY_ISO3.get(country_name)
        if iso3c:
            country_counts[iso3c] += 1
    if not country_counts:
        return {}
    total = sum(country_counts.values())
    return {iso3c: count / total for iso3c, count in country_counts.items()}
