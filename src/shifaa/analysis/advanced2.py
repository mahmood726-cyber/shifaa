"""Advanced statistical methods — second tier.

Blinder-Oaxaca decomposition (1973), Hurdle model (Mullahy 1986),
Quantile regression (Koenker & Bassett 1978), Kakwani progressivity (1977),
Bootstrap CIs for inequality measures (Mills & Zandvakili 1997),
KL divergence from fair distribution (Kullback & Leibler 1951).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd


# ── Blinder-Oaxaca Decomposition ────────────────────────────────────────────

def blinder_oaxaca(
    matrix: pd.DataFrame,
    group_col: str = "income_group",
    outcome_col: str = "trial_density",
    predictors: list[str] | None = None,
) -> dict:
    """Blinder-Oaxaca decomposition of the trial gap between two groups.

    Decomposes the mean outcome difference between Group A (e.g., HIC) and
    Group B (e.g., LMIC) into:
    - Endowments (explained): differences due to different covariate levels
    - Coefficients (unexplained): differences due to different returns to covariates
    - Interaction: joint effect

    Blinder (1973), J Human Resources; Oaxaca (1973), Int Economic Review.

    Args:
        matrix: DataFrame with group_col, outcome_col, and predictors.
        group_col: Binary group column (first unique value = advantaged group).
        outcome_col: Continuous outcome.
        predictors: Covariate columns.

    Returns dict with: mean_a, mean_b, gap, endowments, coefficients, interaction,
    endowment_detail (per-variable breakdown).
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

    df = df.dropna(subset=[group_col, outcome_col] + predictors)
    groups = df[group_col].unique()
    if len(groups) < 2:
        return {"error": "Need at least 2 groups"}

    # Group A = advantaged (higher mean), Group B = disadvantaged
    means = df.groupby(group_col)[outcome_col].mean()
    group_a = means.idxmax()
    group_b = means.idxmin()

    df_a = df[df[group_col] == group_a]
    df_b = df[df[group_col] == group_b]

    X_a = sm.add_constant(df_a[predictors])
    X_b = sm.add_constant(df_b[predictors])
    y_a = df_a[outcome_col]
    y_b = df_b[outcome_col]

    # OLS for each group
    beta_a = sm.OLS(y_a, X_a).fit().params
    beta_b = sm.OLS(y_b, X_b).fit().params

    # Mean covariates
    mean_X_a = X_a.mean()
    mean_X_b = X_b.mean()

    # Three-fold decomposition (Oaxaca 1973)
    endowments = float(beta_a @ (mean_X_a - mean_X_b))
    coefficients = float(mean_X_b @ (beta_a - beta_b))
    interaction = float((mean_X_a - mean_X_b) @ (beta_a - beta_b))

    gap = float(y_a.mean() - y_b.mean())

    # Per-variable endowment breakdown
    endowment_detail = {}
    for p in predictors:
        endowment_detail[p] = float(beta_a[p] * (mean_X_a[p] - mean_X_b[p]))

    return {
        "group_a": str(group_a),
        "group_b": str(group_b),
        "mean_a": float(y_a.mean()),
        "mean_b": float(y_b.mean()),
        "gap": gap,
        "endowments": endowments,
        "coefficients": coefficients,
        "interaction": interaction,
        "endowments_pct": endowments / gap * 100 if gap != 0 else 0,
        "coefficients_pct": coefficients / gap * 100 if gap != 0 else 0,
        "endowment_detail": endowment_detail,
        "n_a": len(df_a),
        "n_b": len(df_b),
    }


# ── Hurdle Model ────────────────────────────────────────────────────────────

def fit_hurdle_model(
    matrix: pd.DataFrame,
    predictors: list[str] | None = None,
) -> dict | None:
    """Two-part hurdle model for trial counts.

    Part 1 (logistic): Pr(trials > 0) — what gets a country its first trial?
    Part 2 (truncated Poisson via OLS on log): E[trials | trials > 0] — what
    increases volume among countries that have trials?

    Mullahy (1986), J Econometrics 33(3):341-365.
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

    df = df.dropna(subset=["trial_count_binary"] + predictors)
    if len(df) < 30:
        return None

    # Part 1: Logistic — any trials or not
    df["has_trials"] = (df["trial_count_binary"] > 0).astype(int)
    X = sm.add_constant(df[predictors])

    try:
        logit = sm.Logit(df["has_trials"], X).fit(disp=0)
    except Exception as exc:
        print(f"WARNING: hurdle logistic failed: {exc}", file=sys.stderr)
        return None

    # Part 2: OLS on log(trials) for positive counts only
    pos = df[df["trial_count_binary"] > 0].copy()
    if len(pos) < 10:
        return None

    pos["log_trials"] = np.log(pos["trial_count_binary"])
    X_pos = sm.add_constant(pos[predictors])

    try:
        ols_pos = sm.OLS(pos["log_trials"], X_pos).fit()
    except Exception as exc:
        print(f"WARNING: hurdle OLS failed: {exc}", file=sys.stderr)
        return None

    # Extract coefficients
    hurdle_coefs = pd.DataFrame({
        "variable": predictors,
        "hurdle_logit_coef": [logit.params.get(p, np.nan) for p in predictors],
        "hurdle_logit_p": [logit.pvalues.get(p, np.nan) for p in predictors],
        "intensity_coef": [ols_pos.params.get(p, np.nan) for p in predictors],
        "intensity_p": [ols_pos.pvalues.get(p, np.nan) for p in predictors],
    })

    return {
        "hurdle_part": {"model": logit, "pseudo_r2": logit.prsquared, "n": len(df)},
        "intensity_part": {"model": ols_pos, "r2": ols_pos.rsquared, "n": len(pos)},
        "coefficients": hurdle_coefs,
        "n_zeros": int((df["trial_count_binary"] == 0).sum()),
        "n_positive": int(len(pos)),
    }


# ── Quantile Regression ────────────────────────────────────────────────────

def fit_quantile_regression(
    matrix: pd.DataFrame,
    quantiles: list[float] | None = None,
    predictors: list[str] | None = None,
) -> pd.DataFrame:
    """Quantile regression at multiple quantiles.

    Shows how determinants vary across the distribution of trial activity.
    E.g., GDP might matter more at the 25th percentile than the 75th.

    Koenker & Bassett (1978), Econometrica 46(1):33-50.

    Returns DataFrame: variable, quantile, coefficient, ci_lower, ci_upper
    """
    import statsmodels.api as sm
    from statsmodels.regression.quantile_regression import QuantReg

    if quantiles is None:
        quantiles = [0.25, 0.50, 0.75]
    if predictors is None:
        predictors = ["log_dalys", "log_gdp_pc", "governance",
                       "health_spend_pct", "physicians"]

    df = matrix.copy()
    if "log_dalys" not in df.columns:
        df["log_dalys"] = np.log(df["dalys"].clip(lower=1))
    if "log_gdp_pc" not in df.columns:
        df["log_gdp_pc"] = np.log(df["gdp_pc"].clip(lower=1))

    # Only positive trial counts for quantile regression
    df = df[df["trial_count_binary"] > 0].copy()
    df["log_trials"] = np.log(df["trial_count_binary"])
    df = df.dropna(subset=["log_trials"] + predictors)

    if len(df) < 20:
        return pd.DataFrame(columns=["variable", "quantile", "coefficient",
                                      "ci_lower", "ci_upper"])

    X = sm.add_constant(df[predictors])
    y = df["log_trials"]

    rows = []
    for q in quantiles:
        try:
            model = QuantReg(y, X).fit(q=q, max_iter=1000)
            ci = model.conf_int()
            for p in predictors:
                idx = list(X.columns).index(p)
                rows.append({
                    "variable": p,
                    "quantile": q,
                    "coefficient": model.params.iloc[idx],
                    "p_value": model.pvalues.iloc[idx],
                    "ci_lower": ci.iloc[idx, 0],
                    "ci_upper": ci.iloc[idx, 1],
                })
        except Exception as exc:
            print(f"WARNING: quantile {q} failed: {exc}", file=sys.stderr)

    return pd.DataFrame(rows)


# ── Kakwani Progressivity Index ─────────────────────────────────────────────

def kakwani_index(
    outcome: np.ndarray,
    need: np.ndarray,
    ranking: np.ndarray,
) -> dict:
    """Kakwani progressivity index for research investment.

    Measures the gap between the concentration curve of research (ranked by need)
    and the Lorenz curve of need. Positive = progressive (research goes to
    higher-need countries). Negative = regressive.

    Kakwani (1977), Econometrica 45(3):719-727.

    Args:
        outcome: Trial density per country
        need: DALYs per country (the "need" measure)
        ranking: Variable to rank by (typically need/DALYs)

    Returns dict: kakwani_index, concentration_index_outcome,
                  gini_need, interpretation
    """
    from shifaa.analysis.advanced import concentration_index

    # CI of outcome ranked by need
    ci_outcome = concentration_index(outcome, ranking)

    # Gini of need
    from shifaa.analysis.equity_trend import compute_gini
    gini_need = compute_gini(need)

    # Kakwani = CI(outcome, ranked by need) - Gini(need)
    K = ci_outcome - gini_need

    if K > 0.05:
        interp = "progressive: research goes to higher-burden countries"
    elif K < -0.05:
        interp = "regressive: research avoids higher-burden countries"
    else:
        interp = "approximately proportional to burden"

    return {
        "kakwani_index": K,
        "ci_outcome_by_need": ci_outcome,
        "gini_need": gini_need,
        "interpretation": interp,
    }


# ── Bootstrap CIs for Inequality Measures ───────────────────────────────────

def bootstrap_ci(
    values: np.ndarray,
    statistic_fn,
    n_boot: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
    **kwargs,
) -> dict:
    """Bootstrap confidence interval for any inequality statistic.

    Mills & Zandvakili (1997), J Applied Econometrics 12(2):133-150.

    Args:
        values: Data array
        statistic_fn: Function that takes an array and returns a scalar
        n_boot: Number of bootstrap replicates
        ci_level: Confidence level (default 0.95)
        seed: Random seed for reproducibility
        **kwargs: Additional args passed to statistic_fn

    Returns dict: estimate, ci_lower, ci_upper, se, n_boot
    """
    rng = np.random.RandomState(seed)
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)

    if n < 3:
        est = statistic_fn(values, **kwargs)
        return {"estimate": est, "ci_lower": est, "ci_upper": est, "se": 0, "n_boot": 0}

    point_est = statistic_fn(values, **kwargs)

    boot_stats = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot_stats[i] = statistic_fn(sample, **kwargs)

    alpha = 1 - ci_level
    ci_lower = np.percentile(boot_stats, 100 * alpha / 2)
    ci_upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    se = np.std(boot_stats, ddof=1)

    return {
        "estimate": point_est,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "se": se,
        "n_boot": n_boot,
    }


# ── KL Divergence from Fair Distribution ────────────────────────────────────

def kl_divergence_from_fair(
    trial_density: np.ndarray,
    dalys: np.ndarray,
) -> dict:
    """KL divergence of actual trial distribution from burden-proportional "fair" distribution.

    Measures information-theoretic distance: how many bits of information
    distinguish the actual research distribution from one proportional to burden.

    Kullback & Leibler (1951), Annals of Mathematical Statistics 22(1):79-86.

    Args:
        trial_density: Actual trial distribution per country
        dalys: Disease burden (defines "fair" distribution)

    Returns dict: kl_divergence, interpretation, max_kl (for context)
    """
    trials = np.asarray(trial_density, dtype=float)
    burden = np.asarray(dalys, dtype=float)

    # Remove zeros and NaN
    valid = (trials > 0) & (burden > 0) & ~np.isnan(trials) & ~np.isnan(burden)
    trials = trials[valid]
    burden = burden[valid]

    if len(trials) < 2:
        return {"kl_divergence": 0, "interpretation": "insufficient data", "max_kl": 0}

    # Normalize to probability distributions
    p = trials / trials.sum()  # actual distribution
    q = burden / burden.sum()  # fair (burden-proportional) distribution

    # KL(P || Q) = sum(p * log(p/q))
    # Add small epsilon to avoid log(0)
    eps = 1e-15
    kl = float(np.sum(p * np.log((p + eps) / (q + eps))))

    # Max possible KL (all trials in one country)
    max_kl = np.log(len(p))

    if kl > 1.0:
        interp = "extreme departure from burden-proportional allocation"
    elif kl > 0.5:
        interp = "substantial departure from fair distribution"
    elif kl > 0.1:
        interp = "moderate departure from fair distribution"
    else:
        interp = "approximately burden-proportional"

    return {
        "kl_divergence": kl,
        "max_kl": max_kl,
        "kl_normalized": kl / max_kl if max_kl > 0 else 0,
        "n_countries": len(p),
        "interpretation": interp,
    }
