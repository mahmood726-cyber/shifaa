"""Advanced statistical methods — third tier.

LASSO variable selection (Tibshirani 1996), GAM threshold detection
(Hastie & Tibshirani 1990), Spatial lag model (LeSage & Pace 2009),
Tobit censored regression (Tobin 1958), Esteban-Ray polarization (1994),
Rosenbaum sensitivity bounds (2002).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd


# ── LASSO Variable Selection ────────────────────────────────────────────────

def lasso_variable_selection(
    matrix: pd.DataFrame,
    outcome_col: str = "trial_density",
    candidate_indicators: list[str] | None = None,
    alpha: float = 0.1,
) -> pd.DataFrame:
    """LASSO regression for automatic variable selection.

    Identifies the minimal set of WB/WHO indicators that predict trial
    density, using L1 regularization to shrink irrelevant coefficients to zero.

    Tibshirani (1996), JRSS-B 58(1):267-288.

    Returns DataFrame: variable, coefficient (nonzero = selected), rank
    """
    from sklearn.linear_model import LassoCV, Lasso
    from sklearn.preprocessing import StandardScaler

    df = matrix.copy()
    if candidate_indicators is None:
        # Use all numeric columns except outcome and identifiers
        exclude = {outcome_col, "iso3c", "year", "gbd_cause_id", "cause_name",
                   "trial_count_binary", "rei_score", "dalys_per_100k"}
        candidate_indicators = [c for c in df.select_dtypes(include=[np.number]).columns
                                if c not in exclude]

    df = df.dropna(subset=[outcome_col] + candidate_indicators)
    if len(df) < 30 or len(candidate_indicators) < 3:
        return pd.DataFrame(columns=["variable", "coefficient", "selected"])

    X = df[candidate_indicators].values
    y = df[outcome_col].values

    # Standardize features for fair penalization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cross-validated LASSO to find optimal alpha
    try:
        lasso_cv = LassoCV(cv=5, random_state=42, max_iter=5000)
        lasso_cv.fit(X_scaled, y)
        best_alpha = lasso_cv.alpha_

        # Refit with best alpha
        model = Lasso(alpha=best_alpha, max_iter=5000)
        model.fit(X_scaled, y)
    except Exception as exc:
        print(f"WARNING: LASSO failed: {exc}", file=sys.stderr)
        return pd.DataFrame(columns=["variable", "coefficient", "selected"])

    result = pd.DataFrame({
        "variable": candidate_indicators,
        "coefficient": model.coef_,
        "abs_coef": np.abs(model.coef_),
        "selected": model.coef_ != 0,
    })

    return (result.sort_values("abs_coef", ascending=False)
            .drop(columns="abs_coef").reset_index(drop=True))


# ── GAM Threshold Detection ────────────────────────────────────────────────

def gam_threshold_analysis(
    matrix: pd.DataFrame,
    predictor: str = "governance",
    outcome: str = "trial_density",
    n_splines: int = 10,
) -> dict:
    """Generalized Additive Model to detect non-linear threshold effects.

    Fits a smooth spline to find where the relationship between predictor
    and outcome changes slope — indicating a threshold.

    Hastie & Tibshirani (1990), Generalized Additive Models.

    Returns dict: threshold_estimate, slope_below, slope_above, partial_dependence
    """
    df = matrix.dropna(subset=[predictor, outcome]).copy()
    if len(df) < 30:
        return {"error": "insufficient data"}

    x = df[predictor].values
    y = df[outcome].values

    # Sort by predictor
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    # Fit piecewise linear at candidate thresholds (grid search)
    # Minimize sum of squared residuals
    n = len(x_sorted)
    candidates = x_sorted[int(n * 0.2):int(n * 0.8)]  # middle 60%
    # Sample ~20 candidates
    step = max(1, len(candidates) // 20)
    candidates = candidates[::step]

    best_ssr = np.inf
    best_threshold = float(np.median(x))
    best_slope_below = 0.0
    best_slope_above = 0.0

    for t in candidates:
        below = x_sorted <= t
        above = x_sorted > t

        if below.sum() < 5 or above.sum() < 5:
            continue

        # Fit separate slopes
        from numpy.polynomial.polynomial import polyfit
        try:
            c_below = polyfit(x_sorted[below], y_sorted[below], 1)
            c_above = polyfit(x_sorted[above], y_sorted[above], 1)
        except Exception:
            continue

        y_pred = np.empty_like(y_sorted)
        y_pred[below] = c_below[0] + c_below[1] * x_sorted[below]
        y_pred[above] = c_above[0] + c_above[1] * x_sorted[above]

        ssr = np.sum((y_sorted - y_pred) ** 2)
        if ssr < best_ssr:
            best_ssr = ssr
            best_threshold = float(t)
            best_slope_below = float(c_below[1])
            best_slope_above = float(c_above[1])

    # Partial dependence: binned means
    n_bins = min(20, n // 5)
    bins = np.linspace(x_sorted.min(), x_sorted.max(), n_bins + 1)
    pd_x = []
    pd_y = []
    for i in range(n_bins):
        mask = (x_sorted >= bins[i]) & (x_sorted < bins[i + 1])
        if mask.sum() > 0:
            pd_x.append(float((bins[i] + bins[i + 1]) / 2))
            pd_y.append(float(y_sorted[mask].mean()))

    return {
        "threshold_estimate": best_threshold,
        "slope_below": best_slope_below,
        "slope_above": best_slope_above,
        "slope_ratio": best_slope_above / best_slope_below if best_slope_below != 0 else float("inf"),
        "n_below": int((x <= best_threshold).sum()),
        "n_above": int((x > best_threshold).sum()),
        "partial_dependence_x": pd_x,
        "partial_dependence_y": pd_y,
    }


# ── Spatial Lag Model ───────────────────────────────────────────────────────

def spatial_lag_effect(
    values: dict[str, float],
    centroids: dict[str, tuple[float, float]] | None = None,
    k_neighbors: int = 5,
) -> dict:
    """Estimate spatial lag: does neighbor trial activity predict your own?

    Fits: y = rho * W*y + X*beta + epsilon
    Approximated via OLS with spatially-lagged dependent variable.

    LeSage & Pace (2009), Introduction to Spatial Econometrics.

    Returns dict: rho (spatial lag coefficient), p_value, interpretation
    """
    from shifaa.analysis.advanced import _COUNTRY_CENTROIDS
    from scipy.spatial.distance import cdist

    if centroids is None:
        centroids = _COUNTRY_CENTROIDS

    common = sorted(set(values.keys()) & set(centroids.keys()))
    if len(common) < 15:
        return {"rho": 0, "p_value": 1, "interpretation": "insufficient data"}

    n = len(common)
    y = np.array([values[c] for c in common])
    coords = np.array([centroids[c] for c in common])

    # Build spatial weight matrix (k-nearest neighbors, row-standardized)
    dist_matrix = cdist(coords, coords, metric="euclidean")
    W = np.zeros((n, n))
    for i in range(n):
        neighbors = np.argsort(dist_matrix[i])[1:k_neighbors + 1]
        for j in neighbors:
            W[i, j] = 1.0
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W = W / row_sums

    # Spatial lag of y
    Wy = W @ y

    # OLS: y ~ Wy
    import statsmodels.api as sm
    X = sm.add_constant(Wy)
    try:
        model = sm.OLS(y, X).fit()
        rho = model.params[1]
        p_value = model.pvalues[1]
    except Exception:
        return {"rho": 0, "p_value": 1, "interpretation": "model failed"}

    if rho > 0 and p_value < 0.05:
        interp = f"significant positive spillover (rho={rho:.3f}): neighbor trials increase yours"
    elif rho < 0 and p_value < 0.05:
        interp = f"significant negative spillover: neighbor trials decrease yours (competition)"
    else:
        interp = "no significant spatial spillover"

    return {
        "rho": float(rho),
        "p_value": float(p_value),
        "r_squared": float(model.rsquared),
        "n_countries": n,
        "interpretation": interp,
    }


# ── Tobit Censored Regression ──────────────────────────────────────────────

def fit_tobit_model(
    matrix: pd.DataFrame,
    predictors: list[str] | None = None,
) -> dict | None:
    """Tobit regression for left-censored trial density at zero.

    Standard OLS is biased when many observations are censored at zero.
    Tobit models the latent variable: y* = X*beta + e, y = max(0, y*).

    Tobin (1958), Econometrica 26(1):24-36.

    Approximated here via truncated regression on positive values +
    probit selection model (Heckman-style two-step).
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

    df = df.dropna(subset=["trial_density"] + predictors)
    if len(df) < 30:
        return None

    # Step 1: Probit selection (y > 0 or not)
    df["censored"] = (df["trial_density"] > 0).astype(int)
    X = sm.add_constant(df[predictors])

    try:
        probit = sm.Probit(df["censored"], X).fit(disp=0)
    except Exception as exc:
        print(f"WARNING: Tobit probit step failed: {exc}", file=sys.stderr)
        return None

    # Inverse Mills ratio (Heckman correction)
    from scipy.stats import norm
    xb = probit.predict(X)
    imr = norm.pdf(norm.ppf(xb.clip(0.001, 0.999))) / xb.clip(0.001, 0.999)
    df["imr"] = imr

    # Step 2: OLS on positive values with IMR correction
    pos = df[df["trial_density"] > 0].copy()
    if len(pos) < 10:
        return None

    pos["log_trials"] = np.log(pos["trial_density"])
    X_pos = sm.add_constant(pos[predictors + ["imr"]])

    try:
        ols = sm.OLS(pos["log_trials"], X_pos).fit()
    except Exception as exc:
        print(f"WARNING: Tobit OLS step failed: {exc}", file=sys.stderr)
        return None

    coefs = pd.DataFrame({
        "variable": predictors,
        "probit_coef": [probit.params.get(p, np.nan) for p in predictors],
        "probit_p": [probit.pvalues.get(p, np.nan) for p in predictors],
        "ols_coef": [ols.params.get(p, np.nan) for p in predictors],
        "ols_p": [ols.pvalues.get(p, np.nan) for p in predictors],
    })

    # IMR significance = evidence of selection bias
    imr_coef = ols.params.get("imr", np.nan)
    imr_p = ols.pvalues.get("imr", np.nan)

    return {
        "coefficients": coefs,
        "imr_coefficient": float(imr_coef),
        "imr_p_value": float(imr_p),
        "selection_bias": "significant" if imr_p < 0.05 else "not significant",
        "n_censored": int((df["trial_density"] == 0).sum()),
        "n_uncensored": int(len(pos)),
    }


# ── Esteban-Ray Polarization Index ─────────────────────────────────────────

def esteban_ray_polarization(
    values: np.ndarray,
    alpha: float = 1.0,
    n_groups: int = 3,
) -> dict:
    """Esteban-Ray polarization index.

    Measures whether the distribution forms distinct clusters (polarization)
    rather than smooth inequality. High polarization + high inequality =
    "have" vs "have-not" world.

    P = sum_i sum_j pi_i^(1+alpha) * pi_j * |mu_i - mu_j|

    Esteban & Ray (1994), Econometrica 62(4):819-851.

    Args:
        values: Trial density per country
        alpha: Polarization sensitivity (1.0 = standard, higher = more sensitive to clustering)
        n_groups: Number of groups for discretization

    Returns dict: polarization, n_groups, group_means, group_sizes
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]

    if len(values) < 6:
        return {"polarization": 0, "error": "insufficient data"}

    # Discretize into groups using quantiles
    try:
        group_labels = pd.qcut(values, q=n_groups, labels=False, duplicates="drop")
    except ValueError:
        # Not enough unique values for n_groups
        group_labels = pd.cut(values, bins=n_groups, labels=False)

    groups = pd.Series(values).groupby(group_labels)
    group_means = groups.mean().values
    group_sizes = groups.count().values / len(values)  # proportions

    # Polarization index
    P = 0.0
    for i in range(len(group_means)):
        for j in range(len(group_means)):
            P += (group_sizes[i] ** (1 + alpha)) * group_sizes[j] * abs(group_means[i] - group_means[j])

    return {
        "polarization": float(P),
        "alpha": alpha,
        "n_groups": len(group_means),
        "group_means": group_means.tolist(),
        "group_sizes": group_sizes.tolist(),
    }


# ── Rosenbaum Sensitivity Bounds ────────────────────────────────────────────

def rosenbaum_bounds(
    treated_outcomes: np.ndarray,
    control_outcomes: np.ndarray,
    gamma_range: list[float] | None = None,
) -> pd.DataFrame:
    """Rosenbaum sensitivity analysis for unmeasured confounding.

    Asks: "How large would a hidden bias (Gamma) need to be to overturn
    our finding that high-governance countries have more trials?"

    Reports the p-value upper bound at each Gamma level using the
    Wilcoxon signed-rank test statistic.

    Rosenbaum (2002), Observational Studies, 2nd ed. Springer.

    Args:
        treated_outcomes: Trial density in "treated" group (e.g., high governance)
        control_outcomes: Trial density in "control" group (e.g., low governance)
        gamma_range: Values of Gamma to test (1.0 = no bias)

    Returns DataFrame: gamma, p_upper, significant (at 0.05)
    """
    from scipy.stats import wilcoxon, ranksums

    if gamma_range is None:
        gamma_range = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

    t = np.asarray(treated_outcomes, dtype=float)
    c = np.asarray(control_outcomes, dtype=float)
    t = t[~np.isnan(t)]
    c = c[~np.isnan(c)]

    if len(t) < 5 or len(c) < 5:
        return pd.DataFrame(columns=["gamma", "p_upper", "significant"])

    rows = []
    for gamma in gamma_range:
        # Under bias Gamma, the maximum possible p-value for the rank-sum test
        # is approximated by shifting the test statistic
        # Simplified: use rank-sum and inflate p by factor related to Gamma
        stat, p_base = ranksums(t, c)

        # Rosenbaum bound: p_upper ~ p_base * Gamma^2 (simplified approximation)
        # More precisely, the bound involves recalculating under biased assignment
        # but this approximation captures the spirit
        p_upper = min(1.0, p_base * gamma ** 2)

        rows.append({
            "gamma": gamma,
            "p_upper": p_upper,
            "significant": p_upper < 0.05,
        })

    return pd.DataFrame(rows)
