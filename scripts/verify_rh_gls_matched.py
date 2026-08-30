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
OMEGA_M = 0.302  # matches the value used in step_32_omega_m_sensitivity.py


def comoving_distance_integral(z, omega_m):
    result, _ = quad(
        lambda zp: 1.0 / np.sqrt(omega_m * (1 + zp) ** 3 + (1 - omega_m)),
        0, z,
    )
    return result


def compute_mu_ref(z_array, omega_m):
    mu_ref = np.zeros(len(z_array))
    for i in range(len(z_array)):
        d_c = comoving_distance_integral(z_array[i], omega_m)
        mu_ref[i] = 5 * np.log10((1 + z_array[i]) * d_c * C_KMS / H0_REF) + 25
    return mu_ref


def load_digitized_curve(curve_path, profile):
    with open(curve_path) as f:
        curve_data = json.load(f)
    z_curve = np.array([p["z"] for p in curve_data])
    h0_curve = np.array([p["H0"] for p in curve_data])
    sort_idx = np.argsort(z_curve)
    z_s = z_curve[sort_idx]
    h0_s = h0_curve[sort_idx]
    diffs = np.diff(h0_s)
    sign_changes = int(np.sum(np.abs(np.diff(np.sign(diffs))) > 0))
    if profile == "gaussian" and sign_changes > 10:
        def _gaussian_decline(z, h_inf, amplitude, sigma_z):
            return h_inf + amplitude * np.exp(-z ** 2 / (2 * sigma_z ** 2))
        popt, _ = curve_fit(
            _gaussian_decline, z_curve, h0_curve,
            p0=[67.4, 6.0, 0.15], maxfev=10000,
        )
        h0_curve = _gaussian_decline(z_s, *popt)
        z_curve = z_s
    else:
        z_curve = z_s
        h0_curve = h0_s
    return z_curve, h0_curve


def evaluate_curve(z_array, z_curve, h0_curve):
    log_z = np.log10(np.clip(z_array, z_curve.min(), z_curve.max()))
    log_z_curve = np.log10(z_curve)
    h0_void = np.interp(log_z, log_z_curve, h0_curve)
    h0_void = np.where(z_array < z_curve.min(), h0_curve[0], h0_void)
    h0_void = np.where(z_array > z_curve.max(), h0_curve[-1], h0_void)
    return h0_void


def gls_h0(d_shift, mask, indices, cov_full):
    """Apply the GLS zero-point estimator to a model shift vector."""
    idx_sub = indices[mask]
    cov_sub = cov_full[np.ix_(idx_sub, idx_sub)]
    n_sub = int(mask.sum())
    diag_pos = np.diag(cov_sub)[np.diag(cov_sub) > 0]
    diag_med = np.median(diag_pos) if len(diag_pos) > 0 else 1.0
    cov_sub_reg = cov_sub + 1e-8 * diag_med * np.eye(n_sub)

    ones_sub = np.ones(n_sub)
    s_sub = d_shift[mask]
    try:
        cov_inv_ones = np.linalg.solve(cov_sub_reg, ones_sub)
        denom = float(ones_sub @ cov_inv_ones)
        cov_inv_s = np.linalg.solve(cov_sub_reg, s_sub)
        a_hat = float(ones_sub @ cov_inv_s) / denom
    except np.linalg.LinAlgError:
        cov_inv_sub = np.linalg.pinv(cov_sub_reg)
        denom = float(ones_sub @ cov_inv_sub @ ones_sub)
        a_hat = float(ones_sub @ cov_inv_sub @ s_sub) / denom

    sigma_a = 1.0 / np.sqrt(denom)
    h0 = H0_REF * (10.0 ** (-a_hat / 5.0))
    sigma_h0 = h0 * (np.log(10.0) / 5.0) * sigma_a
    return h0, sigma_h0, n_sub


def main():
    data_raw = PROJECT_ROOT / "data" / "raw"
    results_dir = PROJECT_ROOT / "results" / "outputs"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load Pantheon+ data
    dat_path = data_raw / "Pantheon+SH0ES.dat"
    df = pd.read_csv(dat_path, sep=r"\s+")
    z = pd.to_numeric(df["zCMB"], errors="coerce")
    mu = pd.to_numeric(df["MU_SH0ES"], errors="coerce")
    mask = z.notna() & mu.notna() & (z > 0)
    z = z[mask].values
    mu = mu[mask].values
    indices = mask[mask].index.values

    # Load covariance
    cov_path = data_raw / "Pantheon+SH0ES_STAT+SYS.cov"
    with open(cov_path) as f:
        n = int(f.readline().strip())
        data = np.fromstring(f.read(), sep="\n")
    cov_full = data[: n * n].reshape(n, n)
    mu_ref = compute_mu_ref(z, OMEGA_M)
    d_obs = mu - mu_ref

    low_mask = (z >= 0.05) & (z < 0.15)
    high_mask = z > 0.25

    # Observed R_H (GLS)
    h0_low_obs, sigma_h0_low, n_low = gls_h0(d_obs, low_mask, indices, cov_full)
    h0_high_obs, sigma_h0_high, n_high = gls_h0(d_obs, high_mask, indices, cov_full)
    r_h_obs = h0_high_obs / h0_low_obs

    # Cross-covariance between the two bins (from the full Pantheon+ covariance)
    idx_low = indices[low_mask]
    idx_high = indices[high_mask]
    n_l, n_h = int(low_mask.sum()), int(high_mask.sum())
    ones_l, ones_h = np.ones(n_l), np.ones(n_h)
    cov_ll = cov_full[np.ix_(idx_low, idx_low)]
    cov_hh = cov_full[np.ix_(idx_high, idx_high)]
    cov_lh = cov_full[np.ix_(idx_low, idx_high)]
    diag_l = np.diag(cov_ll)[np.diag(cov_ll) > 0]
    diag_h = np.diag(cov_hh)[np.diag(cov_hh) > 0]
    cov_ll_reg = cov_ll + 1e-6 * np.median(diag_l) * np.eye(n_l)
    cov_hh_reg = cov_hh + 1e-6 * np.median(diag_h) * np.eye(n_h)

    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        try:
            inv_ones_l = np.linalg.solve(cov_ll_reg, ones_l)
            denom_l = float(ones_l @ inv_ones_l)
            inv_ones_h = np.linalg.solve(cov_hh_reg, ones_h)
            denom_h = float(ones_h @ inv_ones_h)
            cross_product = float(inv_ones_l @ cov_lh @ inv_ones_h)
        except np.linalg.LinAlgError:
            cov_inv_ll = np.linalg.pinv(cov_ll_reg)
            cov_inv_hh = np.linalg.pinv(cov_hh_reg)
            denom_l = float(ones_l @ cov_inv_ll @ ones_l)
            denom_h = float(ones_h @ cov_inv_hh @ ones_h)
            cross_product = float(ones_l @ cov_inv_ll @ cov_lh @ cov_inv_hh @ ones_h)

        sigma_a_l = 1.0 / np.sqrt(denom_l)
        sigma_a_h = 1.0 / np.sqrt(denom_h)
        cov_a_lh = cross_product / (denom_l * denom_h)
    var_diff = sigma_a_h ** 2 + sigma_a_l ** 2 - 2.0 * cov_a_lh
    sigma_diff = np.sqrt(max(var_diff, 0.0))
    sigma_r_h = r_h_obs * sigma_diff * np.log(10.0) / 5.0

    # Load KBC curves
    curves_dir = data_raw / "external" / "mazurenko_curves"
    z_g, h0_g_curve = load_digitized_curve(
        curves_dir / "gaussian_method3.json", "gaussian")
    z_e, h0_e_curve = load_digitized_curve(
        curves_dir / "exponential_method3.json", "exponential")

    h0_g_eval = evaluate_curve(z, z_g, h0_g_curve)
    h0_e_eval = evaluate_curve(z, z_e, h0_e_curve)

    # KBC model shift: s_i = 5*log10(H_ref / H_KBC(z_i))
    s_g = 5.0 * np.log10(H0_REF / h0_g_eval)
    s_e = 5.0 * np.log10(H0_REF / h0_e_eval)

    # Matched-GLS R_H for KBC models
    h0_low_g, _, _ = gls_h0(s_g, low_mask, indices, cov_full)
    h0_high_g, _, _ = gls_h0(s_g, high_mask, indices, cov_full)
    rh_kbc_g_gls = h0_high_g / h0_low_g

    h0_low_e, _, _ = gls_h0(s_e, low_mask, indices, cov_full)
    h0_high_e, _, _ = gls_h0(s_e, high_mask, indices, cov_full)
    rh_kbc_e_gls = h0_high_e / h0_low_e

    # Arithmetic-mean R_H (existing method)
    rh_kbc_g_arith = float(np.mean(h0_g_eval[high_mask]) / np.mean(h0_g_eval[low_mask]))
    rh_kbc_e_arith = float(np.mean(h0_e_eval[high_mask]) / np.mean(h0_e_eval[low_mask]))

    # Significance under matched GLS
    z_kbc_g_gls = (r_h_obs - rh_kbc_g_gls) / sigma_r_h
    z_kbc_e_gls = (r_h_obs - rh_kbc_e_gls) / sigma_r_h

    # Significance under arithmetic mean (existing)
    z_kbc_g_arith = (r_h_obs - rh_kbc_g_arith) / sigma_r_h
    z_kbc_e_arith = (r_h_obs - rh_kbc_e_arith) / sigma_r_h

    output = {
        "step": "32_rh_gls_matched_check",
        "description": (
            "Verification: KBC model R_H predictions run through the "
            "same two-bin GLS covariance-aware estimator used for the "
            "observed R_H, instead of arithmetic means of the KBC "
            "curve in each bin."
        ),
        "methodology": (
            "The KBC model shift s_i = 5*log10(H_ref/H_KBC(z_i)) is "
            "fed through the identical GLS zero-point estimator "
            "(1^T C^-1 s)/(1^T C^-1 1) in each bin using the same "
            "Pantheon+ STAT+SYS covariance submatrix. The resulting "
            "R_H_KBC_GLS is compared to the observed R_H using the "
            "observed sigma_R_H."
        ),
        "omega_m": OMEGA_M,
        "h0_ref": H0_REF,
        "observed": {
            "R_H": float(r_h_obs),
            "sigma_R_H": float(sigma_r_h),
            "H0_low": float(h0_low_obs),
            "H0_high": float(h0_high_obs),
            "n_low": int(n_low),
            "n_high": int(n_high),
        },
        "kbc_gaussian": {
            "R_H_gls_matched": float(rh_kbc_g_gls),
            "R_H_arithmetic_mean": float(rh_kbc_g_arith),
            "Z_gls_matched": float(z_kbc_g_gls),
            "Z_arithmetic_mean": float(z_kbc_g_arith),
            "H0_low_gls": float(h0_low_g),
            "H0_high_gls": float(h0_high_g),
        },
        "kbc_exponential": {
            "R_H_gls_matched": float(rh_kbc_e_gls),
            "R_H_arithmetic_mean": float(rh_kbc_e_arith),
            "Z_gls_matched": float(z_kbc_e_gls),
            "Z_arithmetic_mean": float(z_kbc_e_arith),
            "H0_low_gls": float(h0_low_e),
            "H0_high_gls": float(h0_high_e),
        },
        "verdict": (
            "SURVIVES" if abs(z_kbc_g_gls) > 5 and abs(z_kbc_e_gls) > 5
            else "WEAKENED"
        ),
    }

    out_path = results_dir / "step_32_rh_gls_matched_check.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print("=" * 72)
    print("MATCHED-GLS R_H CHECK FOR KBC MODEL PREDICTIONS")
    print("=" * 72)
    print(f"Observed R_H = {r_h_obs:.6f} +/- {sigma_r_h:.6f}")
    print(f"  H0_low  = {h0_low_obs:.4f} (N={n_low})")
    print(f"  H0_high = {h0_high_obs:.4f} (N={n_high})")
    print()
    print("KBC Gaussian:")
    print(f"  R_H (GLS matched)     = {rh_kbc_g_gls:.6f}  Z = {z_kbc_g_gls:.2f}sigma")
    print(f"  R_H (arithmetic mean) = {rh_kbc_g_arith:.6f}  Z = {z_kbc_g_arith:.2f}sigma")
    print()
    print("KBC Exponential:")
    print(f"  R_H (GLS matched)     = {rh_kbc_e_gls:.6f}  Z = {z_kbc_e_gls:.2f}sigma")
    print(f"  R_H (arithmetic mean) = {rh_kbc_e_arith:.6f}  Z = {z_kbc_e_arith:.2f}sigma")
    print()
    print(f"Verdict: {output['verdict']}")
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
