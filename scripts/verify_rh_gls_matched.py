#!/usr/bin/env python3
"""
Verification: Matched-GLS R_H check for KBC model predictions.

The existing step_32 computes the KBC predicted R_H as the ratio of
arithmetic means of the KBC H0(z) curve in each bin, while the observed
R_H uses a covariance-aware GLS estimator.  This script runs the KBC
model vectors through the SAME two-bin GLS operator and reports whether
the 8.4/9.7sigma exclusion survives under the matched estimator.

Method:
  - For each SN, the KBC model predicts H0_KBC(z_i), giving a model
    distance-modulus shift s_i = 5*log10(H_ref / H_KBC(z_i)).
  - The GLS zero-point estimator is applied to s_i in each bin using
    the SAME Pantheon+ STAT+SYS covariance submatrix:
        a_hat = (1^T C^-1 s) / (1^T C^-1 1)
        H0_GLS = H_ref * 10^(-a_hat/5)
  - R_H_KBC_GLS = H0_GLS(high) / H0_GLS(low)
  - Significance: z = (R_H_obs - R_H_KBC_GLS) / sigma_R_H_obs
    (the uncertainty is on the observed R_H; the model prediction
     is deterministic given the digitized curve).

Output:
  results/outputs/step_32_rh_gls_matched_check.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import curve_fit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

H0_REF = 73.04
C_KMS = 299792.458
OMEGA_M = 0.302


def comoving_distance_integral(z, omega_m=OMEGA_M):
    """D_C(z) = integral_0^z dz'/E(z') in units of c/H0, for given Omega_m."""
    result, _ = quad(
        lambda zp: 1.0 / np.sqrt(omega_m * (1 + zp) ** 3 + (1 - omega_m)),
        0, z,
    )
    return result


def compute_mu_ref(z_array, omega_m=OMEGA_M):
    """
    Reference distance modulus mu_ref(z) at H0_ref for given Omega_m.

    mu_ref(z) = 5*log10((1+z) * D_C(z) * c / H0_ref) + 25
    where D_C(z) = integral_0^z dz'/E(z') and
          E(z) = sqrt(Omega_m*(1+z)^3 + (1-Omega_m))
    """
    mu_ref = np.zeros(len(z_array))
    for i in range(len(z_array)):
        d_c = comoving_distance_integral(z_array[i], omega_m)
        mu_ref[i] = 5 * np.log10((1 + z_array[i]) * d_c * C_KMS / H0_REF) + 25
    return mu_ref


def load_digitized_curve(profile):
    """Load digitized KBC curve from Mazurenko et al. 2025."""
    curve_path = PROJECT_ROOT / "data" / "raw" / "external" / "mazurenko_curves" / f"{profile}_method3.json"
    if not curve_path.exists():
        raise FileNotFoundError(f"Digitized curve not found: {curve_path}")
    with open(curve_path) as f:
        curve_data = json.load(f)
    z_curve = np.array([p["z"] for p in curve_data])
    h0_curve = np.array([p["H0"] for p in curve_data])

    sort_idx = np.argsort(z_curve)
    z_curve = z_curve[sort_idx]
    h0_curve = h0_curve[sort_idx]

    if profile == "gaussian":
        diffs = np.diff(h0_curve)
        sign_changes = int(np.sum(np.abs(np.diff(np.sign(diffs))) > 0))
        if sign_changes > 10:
            def _gaussian_decline(z, h_inf, amplitude, sigma_z):
                return h_inf + amplitude * np.exp(-z ** 2 / (2 * sigma_z ** 2))
            popt, _ = curve_fit(_gaussian_decline, z_curve, h0_curve,
                               p0=[67.4, 6.0, 0.15], maxfev=10000)
            h0_curve = _gaussian_decline(z_curve, *popt)

    return z_curve, h0_curve


def evaluate_curve(z_array, z_curve, h0_curve):
    """Interpolate KBC H0(z) at observed SN redshifts."""
    log_z = np.log10(np.clip(z_array, z_curve.min(), z_curve.max()))
    log_z_curve = np.log10(z_curve)
    h0_void = np.interp(log_z, log_z_curve, h0_curve)
    h0_void = np.where(z_array < z_curve.min(), h0_curve[0], h0_void)
    h0_void = np.where(z_array > z_curve.max(), h0_curve[-1], h0_void)
    return h0_void


def gls_h0(s_vector, cov_submatrix):
    """GLS zero-point estimator for H0 from model shifts.

    a_hat = (1^T C^-1 s) / (1^T C^-1 1)
    H0_GLS = H_ref * 10^(-a_hat/5)
    """
    n = len(s_vector)
    ones = np.ones(n)
    try:
        cov_inv = np.linalg.inv(cov_submatrix)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov_submatrix)
    denom = float(ones @ cov_inv @ ones)
    a_hat = float(s_vector @ cov_inv @ ones) / denom
    h0_gls = H0_REF * 10 ** (-a_hat / 5)
    return h0_gls, a_hat, cov_inv, denom


def main():
    # Load Pantheon+ data
    data_path = PROJECT_ROOT / "data" / "raw" / "Pantheon+SH0ES.dat"
    cov_path = PROJECT_ROOT / "data" / "raw" / "Pantheon+SH0ES_STAT+SYS.cov"

    df = pd.read_csv(data_path, sep=" ", comment="#")
    z = pd.to_numeric(df["zCMB"], errors="coerce").values
    mu_obs = pd.to_numeric(df["MU_SH0ES"], errors="coerce").values
    mask = np.isfinite(z) & np.isfinite(mu_obs) & (z > 0)
    z = z[mask]
    mu_obs = mu_obs[mask]

    # Load covariance (flat format: first line = N, then N*N values one per line)
    with open(cov_path) as f:
        n_sne = int(f.readline().strip())
        cov_flat = np.array([float(f.readline().strip()) for _ in range(n_sne * n_sne)])
    cov = cov_flat.reshape(n_sne, n_sne)
    assert cov.shape == (n_sne, n_sne), f"Covariance shape {cov.shape} != ({n_sne}, {n_sne})"

    # Define bins (matching step_32)
    z_low_min, z_low_max = 0.05, 0.15
    z_high_min = 0.25

    low_mask = (z >= z_low_min) & (z < z_low_max)
    high_mask = z >= z_high_min

    # Compute observed R_H using GLS
    mu_ref = compute_mu_ref(z, OMEGA_M)
    delta_mu_obs = mu_obs - mu_ref

    # GLS zero-point for observed in each bin
    cov_low = cov[np.ix_(low_mask, low_mask)]
    cov_high = np.ix_(high_mask, high_mask)
    cov_high = cov[np.ix_(high_mask, high_mask)]

    s_low_obs = delta_mu_obs[low_mask]
    s_high_obs = delta_mu_obs[high_mask]

    h0_low_obs, a_low, cov_inv_low, denom_low = gls_h0(s_low_obs, cov_low)
    h0_high_obs, a_high, cov_inv_high, denom_high = gls_h0(s_high_obs, cov_high)
    R_H_obs = h0_high_obs / h0_low_obs

    # Uncertainty on R_H from joint covariance (matching step_32 approach)
    n_low = low_mask.sum()
    n_high = high_mask.sum()
    ones_low = np.ones(n_low)
    ones_high = np.ones(n_high)

    cross_cov = cov[np.ix_(low_mask, high_mask)]

    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        # Cross-covariance of the two zero-point estimators
        # Cov(a_low, a_high) = (1_l^T C_ll^{-1} C_lh C_hh^{-1} 1_h) / (denom_l * denom_h)
        cross_term = float(ones_low @ cov_inv_low @ cross_cov @ cov_inv_high @ ones_high) / (denom_low * denom_high)
        cov_a_lh = cross_term

        sigma_a_l = 1.0 / np.sqrt(denom_low)
        sigma_a_h = 1.0 / np.sqrt(denom_high)

        var_diff = sigma_a_h ** 2 + sigma_a_l ** 2 - 2.0 * cov_a_lh
        sigma_diff = np.sqrt(max(var_diff, 0.0))
        sigma_R_H = R_H_obs * sigma_diff * np.log(10.0) / 5.0

    # Load KBC curves and compute matched-GLS R_H
    results = {
        "description": "Matched-GLS R_H check: KBC model vectors through the same two-bin GLS operator",
        "H0_REF": H0_REF,
        "omega_m": OMEGA_M,
        "bins": {"low": [z_low_min, z_low_max], "high": [z_high_min, None]},
        "n_low": int(n_low),
        "n_high": int(n_high),
        "observed": {
            "H0_low": h0_low_obs,
            "H0_high": h0_high_obs,
            "R_H": R_H_obs,
            "sigma_R_H": sigma_R_H,
            "a_low": a_low,
            "a_high": a_high,
        },
    }

    for profile in ["gaussian", "exponential"]:
        z_curve, h0_curve = load_digitized_curve(profile)
        h0_kbc = evaluate_curve(z, z_curve, h0_curve)

        # Model distance-modulus shift
        s_kbc = 5 * np.log10(H0_REF / h0_kbc)

        s_low_kbc = s_kbc[low_mask]
        s_high_kbc = s_kbc[high_mask]

        h0_low_kbc, a_low_kbc, _, _ = gls_h0(s_low_kbc, cov_low)
        h0_high_kbc, a_high_kbc, _, _ = gls_h0(s_high_kbc, cov_high)
        R_H_kbc_gls = h0_high_kbc / h0_low_kbc

        # Significance
        z_sigma = (R_H_obs - R_H_kbc_gls) / sigma_R_H

        results[f"kbc_{profile}"] = {
            "H0_low_gls": h0_low_kbc,
            "H0_high_gls": h0_high_kbc,
            "R_H_kbc_gls": R_H_kbc_gls,
            "sigma_exclusion": z_sigma,
            "a_low": a_low_kbc,
            "a_high": a_high_kbc,
        }

        print(f"KBC {profile}: R_H_KBC_GLS = {R_H_kbc_gls:.6f}, "
              f"exclusion = {z_sigma:.2f}sigma")

    # Also compute the arithmetic-mean version for comparison
    for profile in ["gaussian", "exponential"]:
        z_curve, h0_curve = load_digitized_curve(profile)
        h0_kbc = evaluate_curve(z, z_curve, h0_curve)
        h0_low_arith = np.mean(h0_kbc[low_mask])
        h0_high_arith = np.mean(h0_kbc[high_mask])
        R_H_arith = h0_high_arith / h0_low_arith
        z_arith = (R_H_obs - R_H_arith) / sigma_R_H
        results[f"kbc_{profile}_arithmetic_mean"] = {
            "R_H_kbc_arith": R_H_arith,
            "sigma_exclusion_arith": z_arith,
        }
        print(f"KBC {profile} (arithmetic): R_H = {R_H_arith:.6f}, "
              f"exclusion = {z_arith:.2f}sigma")

    print(f"\nObserved: R_H = {R_H_obs:.6f} +/- {sigma_R_H:.6f}")
    print(f"Matched-GLS exclusion: "
          f"{results['kbc_gaussian']['sigma_exclusion']:.2f}sigma (Gaussian), "
          f"{results['kbc_exponential']['sigma_exclusion']:.2f}sigma (Exponential)")

    # Save
    out_path = PROJECT_ROOT / "results" / "outputs" / "step_32_rh_gls_matched_check.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
