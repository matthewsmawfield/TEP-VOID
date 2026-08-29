#!/usr/bin/env python3
"""
Step 51: Bayesian hierarchical analysis of the band-dependence test

The step_49 OLS regression found a slope of +8.1e5 ± 1.5e5 (5.35sigma)
for the Madore & Freedman (2023) band-dependence test — correct TEP
sign but ~4.5x steeper than predicted, with chi^2/dof = 5.19 indicating
substantial intrinsic scatter.

This script performs a Bayesian hierarchical analysis that:
1. Models intrinsic scatter (sigma_int) as a free parameter
2. Uses MCMC to sample the posterior of slope, intercept, sigma_int
3. Compares the fitted slope to the TEP prediction
4. Performs leave-one-out influence analysis
5. Tests for outlier contamination

The hierarchical model is:
  delta_mu_i ~ Normal(slope * X_i + intercept, sqrt(err_i^2 + sigma_int^2))

with priors:
  slope ~ Uniform(-1e7, 1e7)
  intercept ~ Normal(0, 0.1)
  sigma_int ~ HalfCauchy(0, 0.1)
"""

import json
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats as sps
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.utils.logger import TEPLogger, set_step_logger, print_status

# TEP constants
C_KMS = 299792.458
U_REF = (87.165) ** 2
B_V = -2.76
B_H = -3.26
KAPPA_P_DEFAULT = 0.365e6
KAPPA_P_JOINT = 0.400e6
PREDICTED_SLOPE_DEFAULT = (abs(B_H) - abs(B_V)) * KAPPA_P_DEFAULT  # 0.50 * 0.365e6
PREDICTED_SLOPE_JOINT = (abs(B_H) - abs(B_V)) * KAPPA_P_JOINT


def load_data():
    """Load the band-dependence matched catalog."""
    path = PROJECT_ROOT / 'data' / 'processed' / 'band_dependence_matched.csv'
    if not path.exists():
        print_status(f"Band-dependence catalog not found at {path}", "ERROR")
        return None
    df = pd.read_csv(path)
    print_status(f"Loaded {len(df)} galaxies from band-dependence catalog", "INFO")
    return df


def mcmc_hierarchical(x, y, yerr, n_burn=5000, n_sample=10000, seed=42):
    """MCMC sampling of the hierarchical linear model with intrinsic scatter.

    Model: y_i ~ Normal(a*x_i + b, sqrt(yerr_i^2 + sigma_int^2))
    Priors: a ~ Uniform(-1e7, 1e7), b ~ Normal(0, 0.1), sigma_int ~ HalfCauchy(0, 0.1)
    """
    rng = np.random.default_rng(seed)
    n = len(x)

    # Initialize
    a = 5e5  # start near OLS estimate
    b = 0.0
    sigma_int = 0.05

    # Log posterior
    def log_post(a, b, sigma_int):
        if sigma_int <= 0 or abs(a) > 1e7:
            return -np.inf
        total_var = yerr**2 + sigma_int**2
        log_lik = -0.5 * np.sum(np.log(2 * np.pi * total_var) + (y - a*x - b)**2 / total_var)
        # Priors
        log_prior_a = 0.0  # uniform
        log_prior_b = -0.5 * (b / 0.1)**2  # normal(0, 0.1)
        log_prior_sigma = -np.log(sigma_int) - (sigma_int / 0.1)  # half-cauchy
        return log_lik + log_prior_a + log_prior_b + log_prior_sigma

    # Metropolis-Hastings
    samples = np.zeros((n_sample, 3))
    current_lp = log_post(a, b, sigma_int)
    n_accept = 0

    # Proposal scales (tuned during burn-in)
    prop_a = 1e5
    prop_b = 0.01
    prop_s = 0.01

    for i in range(n_burn + n_sample):
        # Propose
        a_new = a + rng.normal(0, prop_a)
        b_new = b + rng.normal(0, prop_b)
        s_new = sigma_int + rng.normal(0, prop_s)

        new_lp = log_post(a_new, b_new, s_new)

        if np.log(rng.uniform()) < new_lp - current_lp:
            a, b, sigma_int = a_new, b_new, s_new
            current_lp = new_lp
            n_accept += 1

        # Adapt proposal scales during burn-in
        if i < n_burn and (i + 1) % 500 == 0:
            recent_rate = n_accept / (i + 1)
            if recent_rate < 0.2:
                prop_a *= 0.8
                prop_b *= 0.8
                prop_s *= 0.8
            elif recent_rate > 0.5:
                prop_a *= 1.2
                prop_b *= 1.2
                prop_s *= 1.2

        if i >= n_burn:
            samples[i - n_burn] = [a, b, sigma_int]

    accept_rate = n_accept / (n_burn + n_sample)
    return samples, accept_rate


def bootstrap_regression(x, y, yerr, n_boot=10000, seed=42):
    """Bootstrap regression with resampling of residuals."""
    rng = np.random.default_rng(seed)

    # OLS fit
    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(lambda x, a, b: a*x + b, x, y,
                          sigma=yerr, absolute_sigma=True, p0=[5e5, 0])
    a_ols, b_ols = popt
    residuals = y - (a_ols * x + b_ols)

    boot_slopes = np.zeros(n_boot)
    boot_intercepts = np.zeros(n_boot)

    for i in range(n_boot):
        # Resample residuals
        y_boot = a_ols * x + b_ols + rng.choice(residuals, size=len(y), replace=True)
        try:
            popt_b, _ = curve_fit(lambda x, a, b: a*x + b, x, y_boot,
                                 p0=[a_ols, b_ols], maxfev=1000)
            boot_slopes[i] = popt_b[0]
            boot_intercepts[i] = popt_b[1]
        except:
            boot_slopes[i] = np.nan
            boot_intercepts[i] = np.nan

    return boot_slopes, boot_intercepts


def run_analysis():
    """Run the full Bayesian hierarchical analysis."""
    print_status("=" * 60, "INFO")
    print_status("Step 51: Bayesian Hierarchical Band-Dependence Analysis", "INFO")
    print_status("=" * 60, "INFO")

    df = load_data()
    if df is None:
        return

    # Use primary sample (MF2023)
    primary = df[df['sample'].str.startswith('primary')] if 'sample' in df.columns else df
    print_status(f"\nPrimary sample (MF2023): {len(primary)} galaxies", "INFO")

    # Get valid data
    if 'delta_mu_band' not in primary.columns:
        print_status("No delta_mu_band column", "ERROR")
        return

    valid = primary.dropna(subset=['delta_mu_band', 'X_i'])
    if 'delta_mu_band_err' in valid.columns:
        yerr = valid['delta_mu_band_err'].values
        yerr = np.where(np.isfinite(yerr) & (yerr > 0), yerr, 0.05)
    else:
        yerr = np.full(len(valid), 0.05)

    x = valid['X_i'].values
    y = valid['delta_mu_band'].values

    print_status(f"Valid galaxies: {len(valid)}", "INFO")
    print_status(f"X_i range: {x.min():.2e} to {x.max():.2e}", "INFO")
    print_status(f"delta_mu_band range: {y.min():.4f} to {y.max():.4f}", "INFO")

    # 1. OLS regression (baseline)
    print_status("\n--- OLS Regression (baseline) ---", "PROCESS")
    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(lambda x, a, b: a*x + b, x, y,
                          sigma=yerr, absolute_sigma=True, p0=[5e5, 0])
    slope_ols = popt[0]
    slope_err_ols = np.sqrt(pcov[0, 0])
    intercept_ols = popt[1]
    print_status(f"OLS: slope = {slope_ols:.3e} ± {slope_err_ols:.3e} "
                 f"({slope_ols/slope_err_ols:.2f}sigma)", "TEST")
    print_status(f"OLS: intercept = {intercept_ols:.4f}", "TEST")

    # Chi^2
    residuals = y - (slope_ols * x + intercept_ols)
    chi2 = np.sum(residuals**2 / yerr**2)
    dof = len(y) - 2
    print_status(f"OLS: chi^2/dof = {chi2/dof:.2f}", "TEST")

    # 2. MCMC hierarchical
    print_status("\n--- MCMC Hierarchical (with intrinsic scatter) ---", "PROCESS")
    print_status("Running MCMC (5000 burn-in + 10000 samples)...", "INFO")
    samples, accept_rate = mcmc_hierarchical(x, y, yerr, n_burn=5000, n_sample=10000)
    print_status(f"Acceptance rate: {accept_rate:.2f}", "INFO")

    slope_mcmc = np.median(samples[:, 0])
    slope_err_mcmc = np.std(samples[:, 0])
    slope_lo = np.percentile(samples[:, 0], 16)
    slope_hi = np.percentile(samples[:, 0], 84)
    intercept_mcmc = np.median(samples[:, 1])
    intercept_err_mcmc = np.std(samples[:, 1])
    sigma_int = np.median(samples[:, 2])
    sigma_int_err = np.std(samples[:, 2])
    sigma_int_lo = np.percentile(samples[:, 2], 16)
    sigma_int_hi = np.percentile(samples[:, 2], 84)

    print_status(f"MCMC: slope = {slope_mcmc:.3e} (+{slope_hi-slope_mcmc:.3e} / "
                 f"{slope_mcmc-slope_lo:.3e})", "TEST")
    print_status(f"MCMC: slope sigma = {slope_mcmc/slope_err_mcmc:.2f}sigma", "TEST")
    print_status(f"MCMC: intercept = {intercept_mcmc:.4f} ± {intercept_err_mcmc:.4f}", "TEST")
    print_status(f"MCMC: sigma_int = {sigma_int:.4f} (+{sigma_int_hi-sigma_int:.4f} / "
                 f"{sigma_int-sigma_int_lo:.4f})", "TEST")

    # Compare to TEP prediction
    print_status(f"\n--- TEP Prediction Comparison ---", "PROCESS")
    print_status(f"TEP predicted slope (default kappa_P): {PREDICTED_SLOPE_DEFAULT:.3e}", "INFO")
    print_status(f"TEP predicted slope (joint kappa_P): {PREDICTED_SLOPE_JOINT:.3e}", "INFO")
    ratio = slope_mcmc / PREDICTED_SLOPE_DEFAULT
    print_status(f"Observed/predicted ratio: {ratio:.2f}", "TEST")

    # Probability that slope > 0 (TEP direction)
    p_positive = np.mean(samples[:, 0] > 0)
    print_status(f"P(slope > 0) = {p_positive:.4f}", "TEST")

    # Probability that slope > predicted
    p_above_pred = np.mean(samples[:, 0] > PREDICTED_SLOPE_DEFAULT)
    print_status(f"P(slope > predicted) = {p_above_pred:.4f}", "TEST")

    # 3. Bootstrap regression
    print_status("\n--- Bootstrap Regression (10,000 resamples) ---", "PROCESS")
    boot_slopes, boot_intercepts = bootstrap_regression(x, y, yerr, n_boot=10000)
    boot_slopes = boot_slopes[np.isfinite(boot_slopes)]
    slope_boot = np.median(boot_slopes)
    slope_boot_err = np.std(boot_slopes)
    slope_boot_lo = np.percentile(boot_slopes, 2.5)
    slope_boot_hi = np.percentile(boot_slopes, 97.5)
    print_status(f"Bootstrap: slope = {slope_boot:.3e} ± {slope_boot_err:.3e}", "TEST")
    print_status(f"Bootstrap: 95% CI = [{slope_boot_lo:.3e}, {slope_boot_hi:.3e}]", "TEST")

    # 4. Leave-one-out influence analysis
    print_status("\n--- Leave-One-Out Influence Analysis ---", "PROCESS")
    loo_slopes = []
    loo_galaxies = []
    for i in range(len(valid)):
        mask = np.ones(len(valid), dtype=bool)
        mask[i] = False
        try:
            popt_loo, _ = curve_fit(lambda x, a, b: a*x + b,
                                   x[mask], y[mask],
                                   sigma=yerr[mask], absolute_sigma=True,
                                   p0=[5e5, 0])
            loo_slopes.append(popt_loo[0])
            gal_name = valid.iloc[i].get('galaxy', f'galaxy_{i}')
            loo_galaxies.append(gal_name)
        except:
            loo_slopes.append(np.nan)
            loo_galaxies.append(valid.iloc[i].get('galaxy', f'galaxy_{i}'))

    loo_slopes = np.array(loo_slopes)
    most_influential_idx = np.nanargmin(np.abs(loo_slopes - slope_ols))
    print_status(f"Most influential: {loo_galaxies[most_influential_idx]} "
                 f"(slope -> {loo_slopes[most_influential_idx]:.3e})", "TEST")
    print_status(f"LOO slope range: {np.nanmin(loo_slopes):.3e} to "
                 f"{np.nanmax(loo_slopes):.3e}", "TEST")

    # 5. Outlier detection (sigma-clip)
    print_status("\n--- Outlier Analysis ---", "PROCESS")
    # Compute standardized residuals with MCMC parameters
    total_var = yerr**2 + sigma_int**2
    std_resid = (y - (slope_mcmc * x + intercept_mcmc)) / np.sqrt(total_var)
    outliers = np.abs(std_resid) > 3
    n_outliers = outliers.sum()
    print_status(f"Outliers (|sigma| > 3): {n_outliers}", "TEST")
    if n_outliers > 0:
        for i in np.where(outliers)[0]:
            gal = valid.iloc[i].get('galaxy', f'galaxy_{i}')
            print_status(f"  {gal}: std_resid = {std_resid[i]:.2f}, "
                        f"delta_mu = {y[i]:.4f}, X_i = {x[i]:.2e}", "INFO")

    # 6. Sigma-clipped regression
    if n_outliers > 0:
        clip_mask = ~outliers
        popt_clip, pcov_clip = curve_fit(lambda x, a, b: a*x + b,
                                         x[clip_mask], y[clip_mask],
                                         sigma=yerr[clip_mask], absolute_sigma=True,
                                         p0=[5e5, 0])
        slope_clip = popt_clip[0]
        slope_err_clip = np.sqrt(pcov_clip[0, 0])
        print_status(f"Sigma-clipped (N={clip_mask.sum()}): slope = "
                     f"{slope_clip:.3e} ± {slope_err_clip:.3e} "
                     f"({slope_clip/slope_err_clip:.2f}sigma)", "TEST")
    else:
        slope_clip = slope_ols
        slope_err_clip = slope_err_ols

    # Save results
    results = {
        'step': 51,
        'description': 'Bayesian hierarchical analysis of band-dependence test',
        'sample': 'primary (MF2023)',
        'n_galaxies': len(valid),
        'ols': {
            'slope': float(slope_ols),
            'slope_err': float(slope_err_ols),
            'slope_sigma': float(slope_ols / slope_err_ols),
            'intercept': float(intercept_ols),
            'chi2_dof': float(chi2 / dof),
        },
        'mcmc_hierarchical': {
            'slope_median': float(slope_mcmc),
            'slope_std': float(slope_err_mcmc),
            'slope_16th': float(slope_lo),
            'slope_84th': float(slope_hi),
            'intercept_median': float(intercept_mcmc),
            'intercept_std': float(intercept_err_mcmc),
            'sigma_int_median': float(sigma_int),
            'sigma_int_std': float(sigma_int_err),
            'sigma_int_16th': float(sigma_int_lo),
            'sigma_int_84th': float(sigma_int_hi),
            'acceptance_rate': float(accept_rate),
            'p_slope_positive': float(p_positive),
            'p_slope_above_predicted': float(p_above_pred),
        },
        'bootstrap': {
            'slope_median': float(slope_boot),
            'slope_std': float(slope_boot_err),
            'slope_2_5': float(slope_boot_lo),
            'slope_97_5': float(slope_boot_hi),
        },
        'tep_prediction': {
            'predicted_slope_default': float(PREDICTED_SLOPE_DEFAULT),
            'predicted_slope_joint': float(PREDICTED_SLOPE_JOINT),
            'observed_over_predicted': float(ratio),
        },
        'loo_analysis': {
            'most_influential_galaxy': loo_galaxies[most_influential_idx],
            'loo_slope_without_most_influential': float(loo_slopes[most_influential_idx]),
            'loo_slope_min': float(np.nanmin(loo_slopes)),
            'loo_slope_max': float(np.nanmax(loo_slopes)),
        },
        'outlier_analysis': {
            'n_outliers': int(n_outliers),
            'sigma_clipped_slope': float(slope_clip),
            'sigma_clipped_slope_err': float(slope_err_clip),
        },
    }

    os.makedirs(PROJECT_ROOT / 'results' / 'outputs', exist_ok=True)
    output_path = PROJECT_ROOT / 'results' / 'outputs' / 'step_51_band_bayesian.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print_status(f"\nSaved results to {output_path}", "INFO")

    print_status("\n" + "=" * 60, "INFO")
    print_status("Step 51 complete", "INFO")
    print_status("=" * 60, "INFO")

    return results


if __name__ == '__main__':
    run_analysis()
