#!/usr/bin/env python3
"""
Step 32 (supplement): Digitization Sensitivity for H0(z) Rejection of KBC Void Curves
=======================================================================================
Recomputes the native mu-space DeltaAIC (void model vs flat model) for the
digitized KBC H0(z) curves under additive H0 shifts of +/-0.5 and +/-1.0
km/s/Mpc applied to BOTH the Gaussian and Exponential Method-3 curves.

This tests whether digitization noise in the curve-tracing could materially
affect the rejection. The primary result (zero shift) must reproduce the
published values:
    Full sample (no cut):  DeltaAIC = 179.4 (Gaussian) / 310.3 (Exponential)
    z >= 0.05:             DeltaAIC =  91.5 (Gaussian) / 106.9 (Exponential)

Method (identical to step_32 _compute_mu_space_comparison):
    - ALL 1701 Pantheon+ rows (no binning, no deduplication)
    - Native mu-space (no Jacobian, no H0 inversion)
    - Full 1701x1701 STAT+SYS covariance matrix
    - KBC H0(z) -> s_i = 5*log10(H_ref / H_KBC(z_i))  with H_ref = 73.04
    - Common zero-point M analytically marginalized
    - Both models k=1 => DeltaAIC = DeltaChi2

Output:
    results/outputs/step_32_digitization_sensitivity.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import curve_fit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status


# ---------------------------------------------------------------------------
# Cosmological constants (match step_32)
# ---------------------------------------------------------------------------
H0_REF = 73.04          # km/s/Mpc  (SH0ES reference; marginalization makes this irrelevant)
H0_CMB = 67.4           # km/s/Mpc  (Planck 2018)
OMEGA_M = 0.302
C_KMS = 299792.458


def load_pantheon_data(data_raw):
    """Load ALL 1701 Pantheon+ rows from Pantheon+SH0ES.dat."""
    dat_path = data_raw / "Pantheon+SH0ES.dat"
    if not dat_path.exists():
        raise FileNotFoundError(f"Pantheon+SH0ES.dat not found at {dat_path}")
    df = pd.read_csv(dat_path, sep=r"\s+")
    print_status(f"Loaded {len(df)} rows from {dat_path.name}", "PROCESS")
    return df


def load_pantheon_covariance(data_raw):
    """Load the 1701x1701 STAT+SYS covariance matrix."""
    cov_path = data_raw / "Pantheon+SH0ES_STAT+SYS.cov"
    if not cov_path.exists():
        raise FileNotFoundError(f"Covariance matrix not found at {cov_path}")
    print_status(f"Loading covariance matrix from {cov_path.name}...", "PROCESS")
    with open(cov_path) as f:
        n = int(f.readline().strip())
        data = np.fromstring(f.read(), sep="\n")
    cov = data[: n * n].reshape(n, n)
    print_status(f"Loaded {n}x{n} covariance matrix", "SUCCESS")
    return cov, n


def comoving_distance_integral(z):
    """D_C(z) = integral_0^z dz'/E(z') in units of c/H0."""
    result, _ = quad(
        lambda zp: 1.0 / np.sqrt(OMEGA_M * (1 + zp) ** 3 + (1 - OMEGA_M)),
        0, z,
    )
    return result


def compute_mu_ref(z_array):
    """Reference distance modulus mu_ref(z) at H0_ref, Omega_m."""
    mu_ref = np.zeros(len(z_array))
    for i in range(len(z_array)):
        d_c = comoving_distance_integral(z_array[i])
        mu_ref[i] = 5 * np.log10((1 + z_array[i]) * d_c * C_KMS / H0_REF) + 25
    return mu_ref


def load_digitized_curve(curve_path, profile):
    """
    Load a digitized KBC H0(z) curve and apply the same denoising as step_32.

    For the Gaussian Method-3 curve, digitization artifacts (interleaved
    upper/lower envelope tracing) produce non-monotonic noise. A parametric
    Gaussian-decline fit H(z) = H_inf + A*exp(-z^2/(2*sigma^2)) is applied
    to denoise, exactly as in step_32.void_model_prediction.

    The Exponential curve is already clean and requires no denoising.
    """
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
        # Parametric denoising fit — identical to step_32
        def _gaussian_decline(z, h_inf, amplitude, sigma_z):
            return h_inf + amplitude * np.exp(-z ** 2 / (2 * sigma_z ** 2))

        popt, _ = curve_fit(
            _gaussian_decline, z_curve, h0_curve,
            p0=[67.4, 6.0, 0.15], maxfev=10000,
        )
        h0_curve = _gaussian_decline(z_s, *popt)
        z_curve = z_s
        print_status(
            f"  Gaussian curve denoised: H_inf={popt[0]:.2f}, "
            f"A={popt[1]:.2f}, sigma={popt[2]:.3f} "
            f"(sign_changes={sign_changes} -> 0)",
            "DEBUG",
        )
    else:
        # Exponential or already-clean: use sorted raw points
        z_curve = z_s
        h0_curve = h0_s
        print_status(f"  {profile} curve: clean ({sign_changes} sign changes)", "DEBUG")

    return z_curve, h0_curve


def evaluate_curve(z_array, z_curve, h0_curve, h0_shift=0.0):
    """
    Interpolate the (optionally denoised) KBC H0(z) curve at the SN
    redshifts, applying an additive H0 shift to test digitization
    sensitivity.  Interpolation is in log(z) space, matching step_32.
    """
    log_z = np.log10(np.clip(z_array, z_curve.min(), z_curve.max()))
    log_z_curve = np.log10(z_curve)
    h0_void = np.interp(log_z, log_z_curve, h0_curve)
    # Clamp outside range
    h0_void = np.where(z_array < z_curve.min(), h0_curve[0], h0_void)
    h0_void = np.where(z_array > z_curve.max(), h0_curve[-1], h0_void)
    # Apply digitization shift
    h0_void = h0_void + h0_shift
    return h0_void


def compute_delta_aic(d, s_model, cov_inv, ones, denom):
    """
    Compute DeltaAIC = AIC(void) - AIC(flat) with marginalized zero-point.

    chi2_marg(r) = r^T C^{-1} r - (r^T C^{-1} 1)^2 / (1^T C^{-1} 1)

    Flat model:  r = d            (s = 0)
    Void model:  r = d - s_model

    Both models have k=1 (zero-point only), so DeltaAIC = DeltaChi2.
    """
    # Flat
    r_flat = d
    chi2_flat = float(r_flat @ cov_inv @ r_flat)
    chi2_flat -= float((r_flat @ cov_inv @ ones) ** 2 / denom)

    # Void
    r_void = d - s_model
    chi2_void = float(r_void @ cov_inv @ r_void)
    chi2_void -= float((r_void @ cov_inv @ ones) ** 2 / denom)

    delta_aic = chi2_void - chi2_flat  # k equal => DeltaAIC = DeltaChi2
    return chi2_flat, chi2_void, delta_aic


def main():
    logger = TEPLogger(
        "step_32_digitization_sensitivity",
        log_file_path=PROJECT_ROOT / "logs" / "step_32_digitization_sensitivity.log",
    )
    set_step_logger(logger)

    data_raw = PROJECT_ROOT / "data" / "raw"
    data_external = data_raw / "external"
    curves_dir = data_external / "mazurenko_curves"
    results_dir = PROJECT_ROOT / "results" / "outputs"
    results_dir.mkdir(parents=True, exist_ok=True)

    print_status("=" * 72, "TEST")
    print_status("Digitization Sensitivity: H0(z) Rejection of KBC Void Curves", "TEST")
    print_status("Native mu-space, full 1701x1701 STAT+SYS covariance", "TEST")
    print_status("=" * 72, "TEST")

    print_status(
        "Scientific context: The primary step_32 result rejects the KBC void "
        "H0(z) curves at large DeltaAIC in native mu-space. This sensitivity "
        "test addresses whether digitization noise introduced during curve-"
        "tracing of the published KBC figures could materially alter that "
        "rejection. Additive H0 shifts of +/-0.5 and +/-1.0 km/s/Mpc are "
        "applied to both the Gaussian and Exponential Method-3 curves, "
        "bracketing plausible digitization uncertainty. If the rejection "
        "persists across all shifts, the falsification is robust to "
        "curve-tracing artifacts.",
        "PROCESS",
    )

    # --- Load Pantheon+ data and covariance ---
    df = load_pantheon_data(data_raw)
    cov_full, n_cov = load_pantheon_covariance(data_raw)

    z = pd.to_numeric(df["zCMB"], errors="coerce")
    mu = pd.to_numeric(df["MU_SH0ES"], errors="coerce")
    mask = z.notna() & mu.notna() & (z > 0)
    z = z[mask].values
    mu = mu[mask].values
    indices = mask[mask].index.values
    n_sn = len(z)
    print_status(f"Valid SNe with zCMB and MU_SH0ES: {n_sn}", "PROCESS")

    # --- Reference cosmology moduli and data residual ---
    print_status("Computing reference cosmology moduli...", "PROCESS")
    mu_ref = compute_mu_ref(z)
    d = mu - mu_ref
    print_status(
        f"Data residual: mean={d.mean():.4f}, std={d.std():.4f}, "
        f"min={d.min():.4f}, max={d.max():.4f}",
        "PROCESS",
    )

    # --- Load digitized KBC curves ---
    print_status("Loading digitized KBC curves...", "PROCESS")
    print_status(
        "Methodology: Both digitized Method-3 curves (Gaussian and "
        "Exponential) are loaded and denoised identically to step_32. "
        "The Gaussian curve undergoes parametric Gaussian-decline "
        "denoising; the Exponential curve is used as-is. For each "
        "curve, additive H0 shifts in {-1.0, -0.5, 0.0, +0.5, +1.0} "
        "km/s/Mpc are applied, and DeltaAIC is recomputed in native "
        "mu-space with the full 1701x1701 STAT+SYS covariance and "
        "analytically marginalized zero-point. Two sample cuts are "
        "evaluated: full sample and z >= 0.05.",
        "PROCESS",
    )
    z_g, h0_g = load_digitized_curve(curves_dir / "gaussian_method3.json", "gaussian")
    z_e, h0_e = load_digitized_curve(curves_dir / "exponential_method3.json", "exponential")

    # --- Invert full covariance ---
    print_status("Inverting 1701x1701 covariance matrix...", "PROCESS")
    cov_sub = cov_full[np.ix_(indices, indices)]
    try:
        cov_inv = np.linalg.inv(cov_sub)
    except np.linalg.LinAlgError:
        print_status("Singular matrix — using pseudo-inverse", "WARNING")
        cov_inv = np.linalg.pinv(cov_sub)

    ones = np.ones(n_sn)
    denom = float(ones @ cov_inv @ ones)

    # --- Define sample cuts ---
    sample_cuts = {
        "full": {"z_min": 0.0, "label": "Full sample (no cut)"},
        "z_ge_0.05": {"z_min": 0.05, "label": "z >= 0.05"},
    }

    # --- Define shifts ---
    shifts = [-1.0, -0.5, 0.0, 0.5, 1.0]

    # --- Run the grid ---
    all_results = {}
    for cut_name, cut_info in sample_cuts.items():
        z_min = cut_info["z_min"]
        z_mask = z >= z_min
        n_cut = int(z_mask.sum())
        print_status(
            f"\n--- {cut_info['label']} (N={n_cut}) ---", "TEST"
        )

        d_sub = d[z_mask]
        idx_sub = indices[z_mask]
        cov_sub_z = cov_full[np.ix_(idx_sub, idx_sub)]
        try:
            cov_inv_z = np.linalg.inv(cov_sub_z)
        except np.linalg.LinAlgError:
            cov_inv_z = np.linalg.pinv(cov_sub_z)

        ones_z = np.ones(n_cut)
        denom_z = float(ones_z @ cov_inv_z @ ones_z)

        cut_results = {}
        for shift in shifts:
            z_sn = z[z_mask]

            # Gaussian
            h0_g_eval = evaluate_curve(z_sn, z_g, h0_g, h0_shift=shift)
            s_g = np.array([
                5 * np.log10(H0_REF / h) if h > 0 else 0.0 for h in h0_g_eval
            ])
            chi2_f_g, chi2_g, daic_g = compute_delta_aic(
                d_sub, s_g, cov_inv_z, ones_z, denom_z
            )

            # Exponential
            h0_e_eval = evaluate_curve(z_sn, z_e, h0_e, h0_shift=shift)
            s_e = np.array([
                5 * np.log10(H0_REF / h) if h > 0 else 0.0 for h in h0_e_eval
            ])
            chi2_f_e, chi2_e, daic_e = compute_delta_aic(
                d_sub, s_e, cov_inv_z, ones_z, denom_z
            )

            # chi2_flat is the same for both (s=0), report once
            shift_key = f"shift_{shift:+.1f}"
            cut_results[shift_key] = {
                "h0_shift_km_s_mpc": shift,
                "n_sne": n_cut,
                "chi2_flat": float(chi2_f_g),
                "gaussian": {
                    "chi2_void": float(chi2_g),
                    "delta_aic": float(daic_g),
                },
                "exponential": {
                    "chi2_void": float(chi2_e),
                    "delta_aic": float(daic_e),
                },
            }
            print_status(
                f"  shift={shift:+.1f}: "
                f"DeltaAIC_G={daic_g:+.1f}, "
                f"DeltaAIC_E={daic_e:+.1f}  "
                f"[chi2_flat={chi2_f_g:.1f}]",
                "TEST",
            )

        all_results[cut_name] = {
            "label": cut_info["label"],
            "z_min": z_min,
            "n_sne": n_cut,
            "shift_results": cut_results,
        }

    # --- Assemble output ---
    output = {
        "step": "32_digitization_sensitivity",
        "description": (
            "Digitization sensitivity of the H0(z) rejection of KBC void "
            "curves. Native mu-space DeltaAIC (void vs flat) recomputed "
            "under additive H0 shifts of +/-0.5 and +/-1.0 km/s/Mpc "
            "applied to both digitized Method-3 curves."
        ),
        "methodology": (
            "Native mu-space DeltaAIC computed for both digitized KBC "
            "Method-3 curves (Gaussian and Exponential) under additive "
            "H0 shifts of -1.0, -0.5, 0.0, +0.5, +1.0 km/s/Mpc. All "
            "1701 Pantheon+ rows are used without binning, with the full "
            "STAT+SYS covariance matrix and analytically marginalized "
            "zero-point. Two sample cuts are evaluated: full sample and "
            "z >= 0.05. Both flat and void models have k=1, so "
            "DeltaAIC = DeltaChi2."
        ),
        "provenance": {
            "data_sources": [
                "data/raw/Pantheon+SH0ES.dat",
                "data/raw/Pantheon+SH0ES_STAT+SYS.cov",
                "data/raw/external/mazurenko_curves/gaussian_method3.json",
                "data/raw/external/mazurenko_curves/exponential_method3.json",
            ],
            "pipeline_block": "Ic (sensitivity and replication)",
        },
        "scientific_context": (
            "The primary step_32 result rejects the KBC void H0(z) curves "
            "at large DeltaAIC in native mu-space. This sensitivity test "
            "evaluates whether digitization noise from curve-tracing of "
            "the published KBC figures could alter that rejection. "
            "Additive H0 shifts bracketing plausible digitization "
            "uncertainty are applied; persistence of the rejection "
            "across all shifts confirms robustness to curve-tracing "
            "artifacts."
        ),
        "tep_prediction": (
            "Under TEP, no redshift-dependent H0 evolution is expected; "
            "the flat reference cosmology should be preferred, yielding "
            "positive DeltaAIC (void rejected) at all digitization shifts."
        ),
        "void_prediction": (
            "Under the KBC void model, H0(z) declines with redshift; if "
            "the digitized curves were a faithful representation, "
            "DeltaAIC could be reduced toward zero under shifts that "
            "compensate tracing noise. The persistence of large positive "
            "DeltaAIC falsifies this expectation."
        ),
        "downstream_consumers": [
            "step_32_redshift_decay_profile",
            "manuscript_section_sensitivity",
        ],
        "method": "native_mu_space_full_covariance",
        "covariance_matrix": "Pantheon+SH0ES_STAT+SYS.cov (1701x1701)",
        "binned": False,
        "diagonal_errors": False,
        "zero_point_marginalized": True,
        "h0_ref": H0_REF,
        "omega_m": OMEGA_M,
        "k_per_model": 1,
        "delta_aic_definition": "AIC(void) - AIC(flat), both k=1 => DeltaAIC = DeltaChi2",
        "conversion": "s_i = 5*log10(H_ref / H_KBC(z_i))",
        "shift_definition": "H_KBC_shifted(z) = H_KBC(z) + shift (applied to both curves)",
        "gaussian_denoising": (
            "Gaussian Method-3 curve denoised via parametric fit "
            "H(z)=H_inf+A*exp(-z^2/(2*sigma^2)), identical to step_32. "
            "Exponential curve used as-is (already clean)."
        ),
        "samples": all_results,
        "note": (
            "All values computed in native mu-space using ALL Pantheon+ "
            "rows (no binning) with the full 1701x1701 STAT+SYS "
            "covariance matrix. A common zero-point is analytically "
            "marginalized. Both flat and void models have k=1. The table "
            "label is 'native mu-space, full covariance' — NOT 'binned, "
            "diagonal errors'."
        ),
    }

    out_path = results_dir / "step_32_digitization_sensitivity.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print_status(f"\nResults saved to {out_path}", "SUCCESS")

    # --- Summary table ---
    print_status("\n" + "=" * 72, "TEST")
    print_status("SUMMARY: Digitization Sensitivity Table", "TEST")
    print_status("Label: native mu-space, full covariance", "TEST")
    print_status("=" * 72, "TEST")
    header = (
        f"{'Sample':<22} {'Shift':>6} {'N':>6} "
        f"{'dAIC_Gauss':>12} {'dAIC_Exp':>12}"
    )
    print_status(header, "TEST")
    print_status("-" * len(header), "TEST")
    for cut_name, cut_data in all_results.items():
        for shift_key, sr in cut_data["shift_results"].items():
            shift = sr["h0_shift_km_s_mpc"]
            n = sr["n_sne"]
            dg = sr["gaussian"]["delta_aic"]
            de = sr["exponential"]["delta_aic"]
            print_status(
                f"{cut_data['label']:<22} {shift:>+6.1f} {n:>6} "
                f"{dg:>+12.1f} {de:>+12.1f}",
                "TEST",
            )

    # --- Verification of zero-shift against published values ---
    # Reference values are the canonical Omega_m = 0.302 (KBC-preferred) zero-shift
    # DeltaAIC from step_32_redshift_decay_profile, matching the cosmology used by
    # this step. The previously hardcoded values (179.4/310.3, 91.5/106.9) were the
    # Omega_m = 0.315 entries from step_32_omega_m_sensitivity and never matched.
    print_status("\n" + "=" * 72, "TEST")
    print_status("VERIFICATION (zero shift vs published step_32 values, Omega_m = 0.302)", "TEST")
    print_status("=" * 72, "TEST")
    expected = {
        "full": (194.3, 328.7),
        "z_ge_0.05": (101.5, 117.1),
    }
    all_ok = True
    for cut_name, (exp_g, exp_e) in expected.items():
        sr = all_results[cut_name]["shift_results"]["shift_+0.0"]
        got_g = sr["gaussian"]["delta_aic"]
        got_e = sr["exponential"]["delta_aic"]
        ok_g = abs(got_g - exp_g) < 1.0
        ok_e = abs(got_e - exp_e) < 1.0
        all_ok = all_ok and ok_g and ok_e
        print_status(
            f"  {cut_name}: Gaussian {got_g:.1f} (expected {exp_g}, "
            f"{'OK' if ok_g else 'MISMATCH'}), "
            f"Exponential {got_e:.1f} (expected {exp_e}, "
            f"{'OK' if ok_e else 'MISMATCH'})",
            "TEST",
        )
    if all_ok:
        print_status("All zero-shift values match published results.", "SUCCESS")
    else:
        print_status("WARNING: some zero-shift values do not match.", "WARNING")

    # --- Interpretation ---
    all_daic_positive = True
    for cut_data in all_results.values():
        for sr in cut_data["shift_results"].values():
            if sr["gaussian"]["delta_aic"] <= 0 or sr["exponential"]["delta_aic"] <= 0:
                all_daic_positive = False
    if all_daic_positive:
        print_status(
            "Interpretation: DeltaAIC remains strongly positive for all "
            "shifts and both curves across both sample cuts. The KBC void "
            "H0(z) prediction is rejected at every digitization shift "
            "tested, confirming that the falsification is robust to "
            "plausible curve-tracing noise.",
            "SUCCESS",
        )
    else:
        print_status(
            "Interpretation: one or more DeltaAIC values are non-positive "
            "under digitization shifts; the rejection may not be fully "
            "robust to curve-tracing noise.",
            "WARNING",
        )

    return output


class Step32DigitizationSensitivity:
    """Pipeline-compatible wrapper for the digitization sensitivity analysis."""

    def run(self):
        return main()


if __name__ == "__main__":
    main()
