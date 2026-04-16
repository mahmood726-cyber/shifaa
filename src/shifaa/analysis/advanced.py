"""Advanced statistical methods for Shifaa papers.

Zero-Inflated Poisson (Lambert 1992), Negative Binomial, Concentration Index
(Wagstaff 2005), Theil decomposition (Shorrocks 1980), Moran's I (1950),
Shapley-Shorrocks decomposition (2013).
"""
from __future__ import annotations

import math
import sys
from itertools import combinations

import numpy as np
import pandas as pd


# ── Paper 2: Zero-Inflated Poisson ──────────────────────────────────────────

def fit_zip_model(matrix: pd.DataFrame) -> dict | None:
    """Zero-Inflated Poisson regression for trial counts.

    Models two processes:
    1. Inflate (logistic): probability a country is a "structural zero"
       (can't have trials regardless of covariates)
    2. Count (Poisson): trial count conditional on being in the count regime

    Lambert (1992), Technometrics 34(1):1-14.
    """
    import statsmodels.api as sm
    from statsmodels.discrete.count_model import ZeroInflatedPoisson

    required = ["trial_count_binary", "log_dalys", "log_gdp_pc", "governance",
                 "health_spend_pct", "physicians"]
    df = matrix.copy()

    # Prepare log transforms if not present
    if "log_dalys" not in df.columns:
        df["log_dalys"] = np.log(df["dalys"].clip(lower=1))
    if "log_gdp_pc" not in df.columns:
        df["log_gdp_pc"] = np.log(df["gdp_pc"].clip(lower=1))

    df = df.dropna(subset=required)
    if len(df) < 30:
        print(f"WARNING: ZIP needs 30+ obs, got {len(df)}", file=sys.stderr)
        return None

    endog = df["trial_count_binary"].astype(int)
    exog = sm.add_constant(df[["log_dalys", "log_gdp_pc", "governance",
                                "health_spend_pct", "physicians"]])
    # Inflate model: use GDP as predictor of structural zeros
    exog_inflate = sm.add_constant(df[["log_gdp_pc"]])

    try:
        model = ZeroInflatedPoisson(
            endog, exog, exog_infl=exog_inflate, inflation="logit",
        )
        result = model.fit(disp=0, maxiter=200)
        return {
            "model": result,
            "converged": result.mle_retvals["converged"],
            "n_obs": len(df),
            "aic": result.aic,
            "bic": result.bic,
            "llf": result.llf,
        }
    except Exception as exc:
        print(f"WARNING: ZIP failed: {exc}", file=sys.stderr)
        return None


def extract_zip_coefficients(result: dict) -> pd.DataFrame:
    """Extract coefficients from ZIP model, separating count and inflate parts."""
    model = result["model"]
    params = model.params
    pvalues = model.pvalues
    ci = model.conf_int()

    rows = []
    for i, name in enumerate(params.index):
        rows.append({
            "variable": name,
            "coefficient": params.iloc[i],
            "p_value": pvalues.iloc[i],
            "ci_lower": ci.iloc[i, 0],
            "ci_upper": ci.iloc[i, 1],
            "part": "inflate" if name.startswith("inflate_") else "count",
        })
    return pd.DataFrame(rows)


# ── Paper 2: Negative Binomial ──────────────────────────────────────────────

def fit_negbin_model(matrix: pd.DataFrame) -> dict | None:
    """Negative Binomial regression as robustness check for overdispersion.

    Cameron & Trivedi (1986), J Econometrics.
    """
    import statsmodels.api as sm
    from statsmodels.discrete.discrete_model import NegativeBinomial

    df = matrix.copy()
    if "log_dalys" not in df.columns:
        df["log_dalys"] = np.log(df["dalys"].clip(lower=1))
    if "log_gdp_pc" not in df.columns:
        df["log_gdp_pc"] = np.log(df["gdp_pc"].clip(lower=1))

    required = ["trial_count_binary", "log_dalys", "log_gdp_pc", "governance",
                 "health_spend_pct", "physicians"]
    df = df.dropna(subset=required)
    if len(df) < 20:
        return None

    endog = df["trial_count_binary"].astype(int)
    exog = sm.add_constant(df[["log_dalys", "log_gdp_pc", "governance",
                                "health_spend_pct", "physicians"]])

    try:
        model = NegativeBinomial(endog, exog)
        result = model.fit(disp=0, maxiter=200)
        return {
            "model": result,
            "converged": result.mle_retvals["converged"],
            "n_obs": len(df),
            # sentinel:skip-line P1-empty-dataframe-access  (guarded by the "alpha" in ... string check)
            "alpha": float(result.params.iloc[-1]) if "alpha" in str(result.params.index[-1]).lower() else None,
            "aic": result.aic,
            "bic": result.bic,
        }
    except Exception as exc:
        print(f"WARNING: NegBin failed: {exc}", file=sys.stderr)
        return None


# ── Paper 5: Concentration Index ────────────────────────────────────────────

def concentration_index(outcome: np.ndarray, ranking: np.ndarray) -> float:
    """Compute the Concentration Index for health inequality.

    Orders observations by ranking variable (e.g., GDP) and measures
    how concentrated the outcome (e.g., trial density) is among
    higher-ranked units.

    CI = 2/mean(y) * cov(y, R) where R is fractional rank.

    Wagstaff (2005), J Health Economics 24(4):627-641.

    Returns:
        CI in [-1, 1]. Positive = outcome concentrated among higher-ranked.
        Negative = concentrated among lower-ranked.
    """
    outcome = np.asarray(outcome, dtype=float)
    ranking = np.asarray(ranking, dtype=float)

    # Remove NaN pairs
    valid = ~(np.isnan(outcome) | np.isnan(ranking))
    outcome = outcome[valid]
    ranking = ranking[valid]

    if len(outcome) < 3 or outcome.mean() == 0:
        return 0.0

    # Sort by ranking variable
    order = np.argsort(ranking)
    y_sorted = outcome[order]
    n = len(y_sorted)

    # Fractional rank
    frac_rank = (np.arange(1, n + 1) - 0.5) / n

    # CI = (2 / mean(y)) * cov(y, R)
    mean_y = y_sorted.mean()
    ci = (2.0 / mean_y) * np.cov(y_sorted, frac_rank)[0, 1]
    return ci


# ── Paper 5: Theil Index Decomposition ──────────────────────────────────────

def theil_index(values: np.ndarray) -> float:
    """Compute Theil T index (GE(1)) for inequality measurement.

    T = (1/n) * sum(y_i/mean(y) * ln(y_i/mean(y)))

    Theil (1967); generalized entropy index with alpha=1.
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    values = values[values > 0]  # Theil undefined for zero values

    if len(values) < 2:
        return 0.0

    mean_y = values.mean()
    if mean_y == 0:
        return 0.0

    ratios = values / mean_y
    return float(np.mean(ratios * np.log(ratios)))


def theil_decomposition(
    values: np.ndarray,
    groups: np.ndarray,
) -> dict:
    """Decompose Theil index into between-group and within-group components.

    Shorrocks (1980), Economica 47(188):613-625.

    Returns dict with: total, between, within, between_share, within_share
    """
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)

    valid = ~np.isnan(values) & (values > 0)
    values = values[valid]
    groups = groups[valid]

    if len(values) < 2:
        return {"total": 0, "between": 0, "within": 0,
                "between_share": 0, "within_share": 0}

    total = theil_index(values)
    grand_mean = values.mean()
    n = len(values)

    unique_groups = np.unique(groups)
    between = 0.0
    within = 0.0

    for g in unique_groups:
        mask = groups == g
        y_g = values[mask]
        n_g = len(y_g)
        if n_g == 0:
            continue
        mean_g = y_g.mean()
        share_g = (n_g / n) * (mean_g / grand_mean)

        # Between component
        if mean_g > 0 and grand_mean > 0:
            between += share_g * np.log(mean_g / grand_mean)

        # Within component
        within += share_g * theil_index(y_g)

    total_check = between + within

    return {
        "total": total,
        "between": between,
        "within": within,
        "between_share": between / total if total > 0 else 0,
        "within_share": within / total if total > 0 else 0,
    }


# ── Paper 1: Moran's I ─────────────────────────────────────────────────────

# Centroids for ~190 countries (lat, lon). Subset of most important.
_COUNTRY_CENTROIDS: dict[str, tuple[float, float]] = {
    "AFG": (33.9, 67.7), "AGO": (-12.4, 18.5), "ALB": (41.2, 20.2),
    "ARE": (23.4, 53.8), "ARG": (-38.4, -63.6), "ARM": (40.1, 45.0),
    "AUS": (-25.3, 133.8), "AUT": (47.5, 14.6), "AZE": (40.1, 47.6),
    "BDI": (-3.4, 29.9), "BEL": (50.5, 4.5), "BEN": (9.3, 2.3),
    "BFA": (12.2, -1.6), "BGD": (23.7, 90.4), "BGR": (42.7, 25.5),
    "BHR": (26.0, 50.6), "BOL": (-16.3, -63.6), "BRA": (-14.2, -51.9),
    "CAF": (6.6, 20.9), "CAN": (56.1, -106.3), "CHE": (46.8, 8.2),
    "CHL": (-35.7, -71.5), "CHN": (35.9, 104.2), "CIV": (7.5, -5.5),
    "CMR": (7.4, 12.4), "COD": (-4.0, 21.8), "COG": (-0.2, 15.8),
    "COL": (4.6, -74.3), "CUB": (21.5, -78.0), "CZE": (49.8, 15.5),
    "DEU": (51.2, 10.5), "DNK": (56.3, 9.5), "DOM": (18.7, -70.2),
    "DZA": (28.0, 1.7), "ECU": (-1.8, -78.2), "EGY": (26.8, 30.8),
    "ESP": (40.5, -3.7), "ETH": (9.1, 40.5), "FIN": (61.9, 25.7),
    "FRA": (46.2, 2.2), "GBR": (55.4, -3.4), "GHA": (7.9, -1.0),
    "GRC": (39.1, 21.8), "GTM": (15.8, -90.2), "GIN": (9.9, -11.4),
    "HND": (15.2, -86.2), "HRV": (45.1, 15.2), "HTI": (19.0, -72.3),
    "HUN": (47.2, 19.5), "IDN": (-0.8, 113.9), "IND": (20.6, 79.0),
    "IRL": (53.1, -8.2), "IRN": (32.4, 53.7), "IRQ": (33.2, 43.7),
    "ISL": (64.6, -19.0), "ISR": (31.0, 34.9), "ITA": (41.9, 12.6),
    "JAM": (18.1, -77.3), "JOR": (30.6, 36.2), "JPN": (36.2, 138.3),
    "KAZ": (48.0, 68.0), "KEN": (-0.0, 37.9), "KGZ": (41.2, 74.8),
    "KHM": (12.6, 105.0), "KOR": (35.9, 128.0), "KWT": (29.3, 47.5),
    "LAO": (19.9, 102.5), "LBN": (33.9, 35.9), "LBR": (6.4, -9.4),
    "LBY": (26.3, 17.2), "LKA": (7.9, 80.8), "MAR": (31.8, -7.1),
    "MDG": (-18.8, 46.9), "MEX": (23.6, -102.6), "MLI": (17.6, -4.0),
    "MMR": (21.9, 96.0), "MNG": (46.9, 103.8), "MOZ": (-18.7, 35.5),
    "MRT": (21.0, -10.9), "MWI": (-13.3, 34.3), "MYS": (4.2, 101.9),
    "NAM": (-22.0, 17.1), "NER": (17.6, 8.1), "NGA": (9.1, 8.7),
    "NIC": (12.9, -85.2), "NLD": (52.1, 5.3), "NOR": (60.5, 8.5),
    "NPL": (28.4, 84.1), "NZL": (-40.9, 174.9), "OMN": (21.5, 55.9),
    "PAK": (30.4, 69.3), "PAN": (8.5, -80.8), "PER": (-9.2, -75.0),
    "PHL": (12.9, 121.8), "POL": (51.9, 19.1), "PRT": (39.4, -8.2),
    "PRY": (-23.4, -58.4), "QAT": (25.4, 51.2), "ROU": (45.9, 25.0),
    "RUS": (61.5, 105.3), "RWA": (-1.9, 29.9), "SAU": (23.9, 45.1),
    "SDN": (12.9, 30.2), "SEN": (14.5, -14.5), "SGP": (1.4, 103.8),
    "SLE": (8.5, -11.8), "SLV": (13.8, -88.9), "SOM": (5.2, 46.2),
    "SRB": (44.0, 21.0), "SSD": (6.9, 31.3), "SWE": (60.1, 18.6),
    "SYR": (34.8, 39.0), "TCD": (15.5, 18.7), "TGO": (8.6, 1.2),
    "THA": (15.9, 100.9), "TJK": (38.9, 71.3), "TLS": (-8.9, 126.0),
    "TUN": (33.9, 9.5), "TUR": (39.0, 35.2), "TZA": (-6.4, 34.9),
    "UGA": (1.4, 32.3), "UKR": (48.4, 31.2), "URY": (-32.5, -55.8),
    "USA": (37.1, -95.7), "UZB": (41.4, 64.6), "VEN": (6.4, -66.6),
    "VNM": (14.1, 108.3), "YEM": (15.6, 48.5), "ZAF": (-30.6, 22.9),
    "ZMB": (-13.1, 27.8), "ZWE": (-19.0, 29.2),
}


def morans_i(
    values: dict[str, float],
    centroids: dict[str, tuple[float, float]] | None = None,
    k_neighbors: int = 5,
) -> dict:
    """Compute Moran's I statistic for spatial autocorrelation.

    Tests whether evidence deserts are spatially clustered or random.

    Moran (1950), Biometrika 37(1):17-23.

    Args:
        values: {iso3c: REI_score}
        centroids: {iso3c: (lat, lon)}. Uses built-in centroids if None.
        k_neighbors: Number of nearest neighbors for spatial weights.

    Returns:
        dict with: morans_i, expected_i, z_score, p_value, interpretation
    """
    if centroids is None:
        centroids = _COUNTRY_CENTROIDS

    # Filter to countries with both values and centroids
    common = sorted(set(values.keys()) & set(centroids.keys()))
    if len(common) < 10:
        return {"morans_i": 0, "z_score": 0, "p_value": 1.0,
                "interpretation": "insufficient data"}

    n = len(common)
    y = np.array([values[c] for c in common])
    coords = np.array([centroids[c] for c in common])

    # Build k-nearest-neighbor spatial weight matrix
    from scipy.spatial.distance import cdist
    dist_matrix = cdist(coords, coords, metric="euclidean")

    W = np.zeros((n, n))
    for i in range(n):
        neighbors = np.argsort(dist_matrix[i])[1:k_neighbors + 1]
        for j in neighbors:
            W[i, j] = 1.0

    # Row-standardize
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W = W / row_sums

    # Moran's I = (n / S0) * (y'Wy / y'y)
    y_dev = y - y.mean()
    S0 = W.sum()
    numerator = float(y_dev @ W @ y_dev)
    denominator = float(y_dev @ y_dev)

    if denominator == 0 or S0 == 0:
        return {"morans_i": 0, "z_score": 0, "p_value": 1.0,
                "interpretation": "no variance"}

    I = (n / S0) * (numerator / denominator)
    E_I = -1.0 / (n - 1)

    # Variance under normality assumption (simplified)
    S1 = 0.5 * np.sum((W + W.T) ** 2)
    S2 = np.sum((W.sum(axis=0) + W.sum(axis=1)) ** 2)
    k = (np.sum(y_dev ** 4) / n) / ((np.sum(y_dev ** 2) / n) ** 2)

    A = n * ((n ** 2 - 3 * n + 3) * S1 - n * S2 + 3 * S0 ** 2)
    B = k * ((n ** 2 - n) * S1 - 2 * n * S2 + 6 * S0 ** 2)
    C = (n - 1) * (n - 2) * (n - 3) * S0 ** 2

    var_I = (A - B) / C - E_I ** 2 if C > 0 else 0.01
    var_I = max(var_I, 1e-10)

    z = (I - E_I) / np.sqrt(var_I)

    # Two-sided p-value from normal approximation
    from scipy.stats import norm
    p_value = 2 * (1 - norm.cdf(abs(z)))

    if I > 0 and p_value < 0.05:
        interp = "significant positive spatial autocorrelation (deserts are clustered)"
    elif I < 0 and p_value < 0.05:
        interp = "significant negative spatial autocorrelation (deserts are dispersed)"
    else:
        interp = "no significant spatial pattern"

    return {
        "morans_i": float(I),
        "expected_i": float(E_I),
        "z_score": float(z),
        "p_value": float(p_value),
        "n_countries": n,
        "interpretation": interp,
    }


# ── Paper 2: Shapley-Shorrocks Decomposition ───────────────────────────────

def shapley_decomposition(
    matrix: pd.DataFrame,
    outcome_col: str = "trial_count_binary",
    predictors: list[str] | None = None,
) -> pd.DataFrame:
    """Shapley-Shorrocks value decomposition of R-squared.

    Attributes the explained variance (R-squared) of trial counts to each
    predictor, accounting for all possible orderings.

    Shorrocks (2013), J Economic Inequality 11(1):99-126.

    Returns DataFrame: variable, shapley_value, share_of_r2
    """
    import statsmodels.api as sm

    if predictors is None:
        predictors = ["log_dalys", "log_gdp_pc", "governance",
                       "health_spend_pct", "physicians"]

    df = matrix.copy()
    if "log_dalys" not in df.columns:
        df["log_dalys"] = np.log(df["dalys"].clip(lower=1))
    if "log_gdp_pc" not in df.columns:
        df["log_gdp_pc"] = np.log(df["gdp_pc"].clip(lower=1))

    df = df.dropna(subset=[outcome_col] + predictors)
    if len(df) < 20:
        return pd.DataFrame(columns=["variable", "shapley_value", "share_of_r2"])

    y = df[outcome_col].values

    def r2_for_subset(cols):
        if not cols:
            return 0.0
        X = sm.add_constant(df[list(cols)].values)
        try:
            model = sm.OLS(y, X).fit()
            return model.rsquared
        except Exception:
            return 0.0

    n = len(predictors)
    shapley = {p: 0.0 for p in predictors}

    # For each predictor, compute marginal contribution across all subsets
    for p in predictors:
        others = [x for x in predictors if x != p]
        for size in range(n):
            for subset in combinations(others, size):
                subset_with = set(subset) | {p}
                subset_without = set(subset)
                marginal = r2_for_subset(subset_with) - r2_for_subset(subset_without)
                # Weight: |S|!(n-|S|-1)! / n!
                weight = (math.factorial(size) *
                          math.factorial(n - size - 1) /
                          math.factorial(n))
                shapley[p] += weight * marginal

    total = sum(shapley.values())
    rows = [{"variable": p, "shapley_value": v,
             "share_of_r2": v / total if total > 0 else 0}
            for p, v in shapley.items()]

    return pd.DataFrame(rows).sort_values("shapley_value", ascending=False).reset_index(drop=True)
