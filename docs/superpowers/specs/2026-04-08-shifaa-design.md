# Shifaa (شفاء) — Global Healing Equity Intelligence

> Date: 2026-04-08
> Status: APPROVED
> Location: `C:\Projects\shifaa\`
> Inspiration: Al-Fatiha and Quran 17:82 — "We send down of the Quran that which is a healing (shifaa) and a mercy (rahma)"

## Purpose

Healing knowledge — the evidence from clinical trials that saves lives — is a mercy that belongs to all of humanity. But it is distributed as if some lives matter more than others. Shifaa quantifies this, explains it, predicts it, and shows the path to correcting it.

The system links four global data sources — IHME GBD (disease burden), ClinicalTrials.gov (research activity), World Bank (socioeconomic determinants), and WHO (health system capacity) — to compute a Research Equity Index (REI) for every country-disease pair on Earth. Five E156 papers and a live HTML dashboard expose where the mercy gaps are, what drives them, where they're heading, and what would close them.

## Data Sources (read-only inputs)

| Source | Location | What we use | Join key |
|--------|----------|-------------|----------|
| IHME GBD | `C:\Projects\ihme-data-lakehouse\data\silver\*\harmonized\` | DALYs by cause × country × year | iso3c + year + cause_id |
| World Bank | `C:\Projects\wb-data-lakehouse\data\silver\*\harmonized\` | GDP, governance, health spend, HCI, education, UHC, ESG (~537 indicators) | iso3c + year |
| WHO GHO | `C:\Projects\who-data-lakehouse\data\silver\` | Physicians, beds, UHC coverage, health workforce | iso3c + year |
| CT.gov API | `https://clinicaltrials.gov/api/v2/studies` | Trial records: conditions, facility locations, dates, status | NCT ID → condition → GBD cause via crosswalk |

Shifaa never copies lakehouse data. It reads harmonized Parquet in place. CT.gov data is fetched live and cached locally.

## The Crosswalk: GBD ↔ MeSH ↔ ICD

The critical linkage. Hybrid approach (C):

1. **Automated backbone**: MeSH→ICD→GBD mapping using NLM's published MeSH-to-ICD crosswalk
2. **Manual curation for top 50 GBD causes**: Cover ~85% of global DALYs. Each cause gets curated MeSH synonyms and ICD-10 code ranges.

File: `crosswalk/gbd_mesh_icd.csv`
```
gbd_cause_id,gbd_cause_name,gbd_level,mesh_terms,icd10_codes
294,Ischemic heart disease,3,"Coronary Artery Disease;Myocardial Infarction;Angina;Acute Coronary Syndrome",I20-I25
295,Stroke,3,"Stroke;Cerebrovascular Accident;Brain Ischemia;Cerebral Hemorrhage",I60-I69
587,Diabetes mellitus type 2,3,"Diabetes Mellitus, Type 2;Type 2 Diabetes;T2DM",E11
...50 rows total
```

Matching algorithm: CT.gov condition text is matched against MeSH terms (case-insensitive substring match). Unmatched conditions are logged and excluded from analysis (not silently dropped).

## Trial-to-Country Allocation

CT.gov trials list facility locations. A single trial may span multiple countries.

**Primary method (D): Site-count weighted**
```
country_weight = sites_in_country / total_sites_in_trial
trial_density[iso3c][cause][year] += country_weight
```

**Sensitivity analysis (A): Binary presence**
```
If trial has any site in country, country gets 1.
```

Both are computed; primary analysis uses (D), sensitivity analysis uses (A) to verify robustness.

## The Shifaa Matrix

The single analytical table that all 5 papers query:

```
iso3c | year | gbd_cause_id | gbd_cause_name | dalys | dalys_per_100k | trial_density | trial_count_binary | gdp_pc | governance | health_spend_pct | physicians_per_1k | hci | uhc_index | rei_score
PAK   | 2020 | 294          | IHD            | 4.2M  | 1,920          | 0.03          | 2                  | 1,194  | -0.78      | 2.9              | 1.0               | 0.41| 45        | -3.2
GBR   | 2020 | 294          | IHD            | 1.1M  | 1,640          | 12.4          | 847                | 41,059 | 1.69       | 10.2             | 2.8               | 0.78| 87        | +1.8
```

**REI (Research Equity Index):**
```
REI = log10(trial_density_per_100k_population / dalys_per_100k)
```
- Negative = evidence desert (high burden, low research)
- Positive = well-researched relative to burden
- Zero = research proportional to burden

## Causal Model (Paper 2)

Multilevel Poisson regression:

```
Trial_count ~ offset(log(population)) + log(DALYs) + log(GDP_pc) + Governance + Health_spend + HCI + Physicians + (1|country) + (1|disease)
```

- Outcome: trial count (site-weighted) per country-disease-year
- Offset: log(population) to model rate
- Fixed effects: burden + socioeconomic determinants
- Random intercepts: country, disease

Key question answered: **Does burden drive trials, or do wealth/governance override burden?** If governance coefficient dominates DALYs coefficient, that's the Lancet finding.

## The 5 E156 Papers

### Paper 1 — The Mercy Map (Rahma)
- **Estimand**: REI per country-disease pair, cross-sectional 2020
- **Method**: Descriptive. Compute REI for 50 causes × ~190 countries. Rank evidence deserts.
- **Key output**: Global heatmap. The 50 worst country-disease evidence deserts named.
- **Dashboard panel**: Interactive choropleth, drill by disease

### Paper 2 — The Balance (Mizan)
- **Estimand**: Governance coefficient on trial density, adjusted for burden and GDP
- **Method**: Multilevel Poisson regression on shifaa_matrix (2015-2023 pooled)
- **Key finding**: Variance decomposition — what % of trial distribution is explained by burden vs. wealth vs. governance?
- **Dashboard panel**: Coefficient forest plot, partial dependence curves

### Paper 3 — The Forecast (Qadr)
- **Estimand**: Projected REI change 2025→2036
- **Method**: IHME GBD Foresight burden projections + WB development trajectory extrapolation. Apply Paper 2 model to projected covariates.
- **Key finding**: 20 countries where mercy gap widens most by 2036
- **Dashboard panel**: Animated timeline, future evidence desert map

### Paper 4 — The Path (Siraat)
- **Estimand**: DALYs avertable if top 30 evidence deserts reached median REI
- **Method**: Counterfactual. Set governance/spend to achievable targets (e.g., median LMIC) → predict trial density from Paper 2 model → estimate health impact using published evidence-to-outcome lags
- **Key finding**: Specific prescriptions per country. "If Pakistan's governance reached India's level, trial activity would increase X-fold."
- **Dashboard panel**: What-if sliders per country

### Paper 5 — The Reckoning (Hisab)
- **Estimand**: Global Gini coefficient of REI, 2005-2025 time-series
- **Method**: Annual REI distribution → Gini + Lorenz curve. Breakpoint analysis at COVID (2020).
- **Key finding**: Is the world moving toward research equity or away?
- **Dashboard panel**: Animated Lorenz curve, pre/post-COVID comparison

## Project Structure

```
C:\Projects\shifaa\
├── src/shifaa/
│   ├── __init__.py
│   ├── config.py                 # Lakehouse paths, CT.gov API config, constants
│   ├── crosswalk/
│   │   ├── __init__.py
│   │   ├── gbd_mesh_icd.csv      # Top 50 GBD↔MeSH↔ICD crosswalk
│   │   ├── loader.py             # Load crosswalk, match conditions→causes
│   │   └── aggregate_codes.py    # WB aggregate region filter
│   ├── ctgov/
│   │   ├── __init__.py
│   │   ├── api.py                # CT.gov API v2 client
│   │   ├── fetch.py              # Fetch trials by condition batch
│   │   └── allocate.py           # Site-weighted country allocation
│   ├── lakehouse/
│   │   ├── __init__.py
│   │   ├── reader.py             # Read harmonized Parquet from 3 lakehouses
│   │   └── join.py               # Build shifaa_matrix
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── rei.py                # REI computation
│   │   ├── regression.py         # Multilevel Poisson (Paper 2)
│   │   ├── forecast.py           # Projection model (Paper 3)
│   │   ├── counterfactual.py     # What-if scenarios (Paper 4)
│   │   └── equity_trend.py       # Gini/Lorenz time-series (Paper 5)
│   └── dashboard/
│       └── shifaa_atlas.html     # Single-file HTML dashboard
├── crosswalk/
│   └── gbd_mesh_icd.csv
├── data/
│   └── cache/ctgov/              # Cached CT.gov API responses
├── tests/
│   ├── conftest.py
│   ├── fixtures/
│   ├── test_crosswalk.py
│   ├── test_ctgov.py
│   ├── test_reader.py
│   ├── test_rei.py
│   ├── test_regression.py
│   └── test_equity_trend.py
├── pyproject.toml
├── E156-PROTOCOL.md
└── docs/superpowers/specs/
```

## Dependencies

```toml
dependencies = [
    "pandas>=2.2",
    "pyarrow>=15.0",
    "requests>=2.31",
    "statsmodels>=0.14",
    "scipy>=1.12",
]
```

No R. All Python. No API keys needed (CT.gov API v2 is open).

## Testing (~45 tests)

All offline using fixture data. No network calls in tests.

| Module | Tests | Coverage |
|--------|-------|----------|
| test_crosswalk.py | 6 | Load CSV, match conditions, handle unmatched, case insensitivity |
| test_ctgov.py | 6 | API parsing, site extraction, allocation weights, binary fallback |
| test_reader.py | 5 | Read from each lakehouse, harmonized schema validation |
| test_rei.py | 6 | REI computation, edge cases (zero trials, zero burden), aggregation |
| test_regression.py | 6 | Model fitting, coefficient extraction, random effects, prediction |
| test_join.py | 5 | Matrix construction, join completeness, missing data handling |
| test_counterfactual.py | 5 | Scenario computation, bounds checking |
| test_equity_trend.py | 6 | Gini computation, Lorenz curve, time-series, COVID breakpoint |

## Dashboard: Shifaa Atlas

Single-file HTML. 5 panels matching the 5 papers:

1. **Mercy Map** — World choropleth coloured by REI. Click country → disease breakdown table. Disease dropdown filter.
2. **Balance** — Forest plot of regression coefficients. Toggle between fixed effects. Partial dependence curves.
3. **Forecast** — Dual map: 2025 vs 2036 projected REI. Animated slider.
4. **Path** — Country selector + what-if sliders (governance, health spend). Real-time REI prediction.
5. **Reckoning** — Animated Lorenz curve (2005-2025). Gini trend line. COVID breakpoint annotation.

Reads pre-computed JSON results. No live computation. Fully offline after build.

## Non-Goals

- Not building a new lakehouse (reads existing ones)
- Not real-time monitoring (batch analysis, refreshable)
- Not covering all ~370 GBD causes (top 50, expandable)
- Not modelling individual trial outcomes (aggregate country-level analysis)
- Not prescribing specific interventions (identifying where investment is needed, not what drugs to trial)

## Success Criteria

1. Shifaa matrix covers 50 causes × 190 countries × 20 years
2. REI computed for every cell with data
3. Regression model converges with interpretable coefficients
4. At least one finding that is genuinely novel (e.g., governance explains more trial variance than burden)
5. Dashboard loads in <3 seconds, works fully offline
6. All 5 E156 papers written
7. 45+ tests passing
