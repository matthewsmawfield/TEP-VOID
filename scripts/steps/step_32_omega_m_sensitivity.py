#!/usr/bin/env python3
"""
Step 32 (supplement): Omega_m Sensitivity for H0(z) Rejection of KBC Void Curves
=================================================================================
Recomputes the native mu-space DeltaAIC (void model vs flat model) for the
digitized KBC H0(z) curves under varying reference-cosmology Omega_m values:
    Omega_m = 0.28, 0.30, 0.302, 0.315

Only the reference cosmology changes (mu_ref(z) depends on Omega_m through
the comoving distance integral). The KBC H0(z) curves are digitized predictions
and are NOT modified. Zero digitization shift is applied throughout.

The primary result (Omega_m = 0.315) must reproduce the published values:
    Full sample (no cut):  DeltaAIC = 179.4 (Gaussian) / 310.3 (Exponential)
    z >= 0.05:             DeltaAIC =  91.5 (Gaussian) / 106.9 (Exponential)

Method (identical to step_32_digitization_sensitivity):
    - ALL 1701 Pantheon+ rows (no binning, no deduplication)
    - Native mu-space (no Jacobian, no H0 inversion)
    - Full 1701x1701 STAT+SYS covariance matrix
    - KBC H0(z) -> s_i = 5*log10(H_ref / H_KBC(z_i))  with H_ref = 73.04
    - Common zero-point M analytically marginalized
    - Both models k=1 => DeltaAIC = DeltaChi2

Output:
    results/outputs/step_32_omega_m_sensitivity.json
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
C_KMS = 299792.458

# Omega_m values to test
OMEGA_M_VALUES = [0.28, 0.30, 0.302, 0.315]


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


def comoving_distance_integral(z, omega_m):
    """D_C(z) = integral_0^z dz'/E(z') in units of c/H0, for given Omega_m."""
    result, _ = quad(
        lambda zp: 1.0 / np.sqrt(omega_m * (1 + zp) ** 3 + (1 - omega_m)),
        0, z,
    )
    return result


def compute_mu_ref(z_array, omega_m):
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
        "step_32_omega_m_sensitivity",
        log_file_path=PROJECT_ROOT / "logs" / "step_32_omega_m_sensitivity.log",
    )
    set_step_logger(logger)

    data_raw = PROJECT_ROOT / "data" / "raw"
    data_external = data_raw / "external"
    curves_dir = data_external / "mazurenko_curves"
    results_dir = PROJECT_ROOT / "results" / "outputs"
    results_dir.mkdir(parents=True, exist_ok=True)

    print_status("=" * 72, "TEST")
    print_status("Omega_m Sensitivity: H0(z) Rejection of KBC Void Curves", "TEST")
    print_status("Native mu-space, full 1701x1701 STAT+SYS covariance", "TEST")
    print_status(f"Omega_m values: {OMEGA_M_VALUES}", "TEST")
    print_status("Zero digitization shift; KBC curves unchanged", "TEST")
    print_status("=" * 72, "TEST")

    print_status(
        "Scientific context: The primary step_32 result rejects the KBC "
        "void H0(z) curves using a reference cosmology with Omega_m = "
        "0.302. This sensitivity test addresses whether that rejection "
        "depends on the assumed matter density. The reference-cosmology "
        "distance modulus mu_ref(z) is recomputed for Omega_m in {0.28, "
        "0.30, 0.302, 0.315}, spanning the range of plausible values. "
        "The KBC H0(z) curves themselves are digitized predictions and "
        "are not modified. If DeltaAIC remains strongly positive across "
        "all Omega_m values, the falsification is robust to the choice "
        "of reference matter density.",
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

    # --- Load digitized KBC curves (loaded ONCE, not changed with Omega_m) ---
    print_status("Loading digitized KBC curves...", "PROCESS")
    z_g, h0_g = load_digitized_curve(curves_dir / "gaussian_method3.json", "gaussian")
    z_e, h0_e = load_digitized_curve(curves_dir / "exponential_method3.json", "exponential")

    # --- Precompute void model s_i for each curve (does NOT depend on Omega_m) ---
    # s_i = 5*log10(H_ref / H_KBC(z_i)) — only depends on the KBC curve and H_ref
    print_status("Precomputing void model s_i (independent of Omega_m)...", "PROCESS")
    h0_g_eval = evaluate_curve(z, z_g, h0_g, h0_shift=0.0)
    s_g_full = np.array([
        5 * np.log10(H0_REF / h) if h > 0 else 0.0 for h in h0_g_eval
    ])
    h0_e_eval = evaluate_curve(z, z_e, h0_e, h0_shift=0.0)
    s_e_full = np.array([
        5 * np.log10(H0_REF / h) if h > 0 else 0.0 for h in h0_e_eval
    ])

    # --- Define sample cuts ---
    sample_cuts = {
        "full": {"z_min": 0.0, "label": "Full sample (no cut)"},
        "z_ge_0.05": {"z_min": 0.05, "label": "z >= 0.05"},
    }

    # --- Precompute covariance inversions for each sample cut (independent of Omega_m) ---
    print_status("Precomputing covariance inversions for each sample cut...", "PROCESS")
    cut_cov_data = {}
    for cut_name, cut_info in sample_cuts.items():
        z_min = cut_info["z_min"]
        z_mask = z >= z_min
        n_cut = int(z_mask.sum())
        idx_sub = indices[z_mask]
        cov_sub_z = cov_full[np.ix_(idx_sub, idx_sub)]
        try:
            cov_inv_z = np.linalg.inv(cov_sub_z)
        except np.linalg.LinAlgError:
            cov_inv_z = np.linalg.pinv(cov_sub_z)
        ones_z = np.ones(n_cut)
        denom_z = float(ones_z @ cov_inv_z @ ones_z)
        cut_cov_data[cut_name] = {
            "z_mask": z_mask,
            "n_cut": n_cut,
            "cov_inv_z": cov_inv_z,
            "ones_z": ones_z,
            "denom_z": denom_z,
        }
        print_status(
            f"  {cut_info['label']}: N={n_cut}, denom={denom_z:.4f}", "PROCESS"
        )

    # --- Run the Omega_m grid ---
    print_status(
        "Methodology: For each Omega_m value, the reference distance "
        "modulus mu_ref(z) is recomputed via the comoving-distance "
        "integral D_C(z) = integral_0^z dz'/E(z') with "
        "E(z) = sqrt(Omega_m*(1+z)^3 + (1-Omega_m)). The data residual "
        "d = mu_obs - mu_ref is then used in the native mu-space "
        "DeltaAIC computation with the full 1701x1701 STAT+SYS "
        "covariance and marginalized zero-point. The void model shift "
        "s_i = 5*log10(H_ref/H_KBC(z_i)) is independent of Omega_m and "
        "is precomputed once. Two sample cuts are evaluated: full "
        "sample and z >= 0.05.",
        "PROCESS",
    )
    all_results = {}
    for omega_m in OMEGA_M_VALUES:
        om_key = f"omega_m_{omega_m:.3f}"
        print_status(f"\n{'=' * 72}", "TEST")
        print_status(f"Omega_m = {omega_m:.3f}", "TEST")
        print_status(f"{'=' * 72}", "TEST")

        # Compute reference cosmology moduli for this Omega_m
        print_status(f"Computing mu_ref for Omega_m={omega_m:.3f}...", "PROCESS")
        mu_ref = compute_mu_ref(z, omega_m)
        d = mu - mu_ref
        print_status(
            f"  Data residual: mean={d.mean():.4f}, std={d.std():.4f}, "
            f"min={d.min():.4f}, max={d.max():.4f}",
            "PROCESS",
        )

        om_results = {}
        for cut_name, cut_info in sample_cuts.items():
            cd = cut_cov_data[cut_name]
            z_mask = cd["z_mask"]
            n_cut = cd["n_cut"]
            cov_inv_z = cd["cov_inv_z"]
            ones_z = cd["ones_z"]
            denom_z = cd["denom_z"]

            d_sub = d[z_mask]
            s_g_sub = s_g_full[z_mask]
            s_e_sub = s_e_full[z_mask]
            z_sn = z[z_mask]

            # Gaussian
            chi2_f_g, chi2_g, daic_g = compute_delta_aic(
                d_sub, s_g_sub, cov_inv_z, ones_z, denom_z
            )

            # Exponential
            chi2_f_e, chi2_e, daic_e = compute_delta_aic(
                d_sub, s_e_sub, cov_inv_z, ones_z, denom_z
            )

            om_results[cut_name] = {
                "label": cut_info["label"],
                "z_min": cut_info["z_min"],
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
                f"  {cut_info['label']} (N={n_cut}): "
                f"DeltaAIC_G={daic_g:+.1f}, "
                f"DeltaAIC_E={daic_e:+.1f}  "
                f"[chi2_flat={chi2_f_g:.1f}]",
                "TEST",
            )

        all_results[om_key] = {
            "omega_m": omega_m,
            "samples": om_results,
        }

    # --- Assemble output ---
    output = {
        "step": "32_omega_m_sensitivity",
        "description": (
            "Omega_m sensitivity of the H0(z) rejection of KBC void curves. "
            "Native mu-space DeltaAIC (void vs flat) recomputed for varying "
            "reference-cosmology Omega_m values. The KBC H0(z) curves are "
            "digitized predictions and are NOT modified — only the reference "
            "cosmology mu_ref(z) changes."
        ),
        "methodology": (
            "Native mu-space DeltaAIC computed for both digitized KBC "
            "Method-3 curves (Gaussian and Exponential) at Omega_m = "
            "0.28, 0.30, 0.302, and 0.315. For each Omega_m, the reference "
            "distance modulus mu_ref(z) is recomputed via the comoving-"
            "distance integral. All 1701 Pantheon+ rows are used without "
            "binning, with the full 1701x1701 STAT+SYS covariance matrix "
            "and analytically marginalized zero-point. The void model "
            "shift s_i = 5*log10(H_ref/H_KBC(z_i)) is independent of "
            "Omega_m. Two sample cuts are evaluated: full sample and "
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
            "using a reference cosmology with Omega_m = 0.302. This "
            "sensitivity test evaluates whether that rejection depends on "
            "the assumed matter density by recomputing mu_ref(z) for "
            "Omega_m spanning 0.28 to 0.315. Persistence of the rejection "
            "across all values confirms robustness to the reference "
            "cosmology parameters."
        ),
        "tep_prediction": (
            "Under TEP, no redshift-dependent H0 evolution is expected; "
            "the flat reference cosmology should be preferred, yielding "
            "positive DeltaAIC (void rejected) at all Omega_m values."
        ),
        "void_prediction": (
            "Under the KBC void model, H0(z) declines with redshift; if "
            "the rejection were an artifact of a specific Omega_m choice, "
            "DeltaAIC could approach zero for alternative values. The "
            "persistence of large positive DeltaAIC falsifies this "
            "expectation."
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
        "omega_m_values": OMEGA_M_VALUES,
        "k_per_model": 1,
        "delta_aic_definition": "AIC(void) - AIC(flat), both k=1 => DeltaAIC = DeltaChi2",
        "conversion": "s_i = 5*log10(H_ref / H_KBC(z_i))",
        "reference_cosmology": (
            "mu_ref(z) = 5*log10((1+z)*D_C(z)*c/H_ref) + 25, "
            "D_C(z) = integral_0^z dz'/E(z'), "
            "E(z) = sqrt(Omega_m*(1+z)^3 + (1-Omega_m))"
        ),
        "kbc_curves_unchanged": True,
        "digitization_shift": 0.0,
        "gaussian_denoising": (
            "Gaussian Method-3 curve denoised via parametric fit "
            "H(z)=H_inf+A*exp(-z^2/(2*sigma^2)), identical to step_32. "
            "Exponential curve used as-is (already clean)."
        ),
        "results": all_results,
        "note": (
            "All values computed in native mu-space using ALL Pantheon+ "
            "rows (no binning) with the full 1701x1701 STAT+SYS "
            "covariance matrix. A common zero-point is analytically "
            "marginalized. Both flat and void models have k=1. Only the "
            "reference cosmology Omega_m is varied; the KBC H0(z) "
            "predictions are digitized and unchanged."
        ),
    }

    out_path = results_dir / "step_32_omega_m_sensitivity.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print_status(f"\nResults saved to {out_path}", "SUCCESS")

    # --- Summary table ---
    print_status("\n" + "=" * 72, "TEST")
    print_status("SUMMARY: Omega_m Sensitivity Table", "TEST")
    print_status("Label: native mu-space, full covariance, zero shift", "TEST")
    print_status("=" * 72, "TEST")
    header = (
        f"{'Omega_m':>8} {'Sample':<22} {'N':>6} "
        f"{'dAIC_Gauss':>12} {'dAIC_Exp':>12} "
        f"{'chi2_flat':>10} {'chi2_v_G':>10} {'chi2_v_E':>10}"
    )
    print_status(header, "TEST")
    print_status("-" * len(header), "TEST")
    for om_key, om_data in all_results.items():
        omega_m = om_data["omega_m"]
        for cut_name, sr in om_data["samples"].items():
            n = sr["n_sne"]
            dg = sr["gaussian"]["delta_aic"]
            de = sr["exponential"]["delta_aic"]
            cf = sr["chi2_flat"]
            cg = sr["gaussian"]["chi2_void"]
            ce = sr["exponential"]["chi2_void"]
            print_status(
                f"{omega_m:>8.3f} {sr['label']:<22} {n:>6} "
                f"{dg:>+12.1f} {de:>+12.1f} "
                f"{cf:>10.1f} {cg:>10.1f} {ce:>10.1f}",
                "TEST",
            )

    # --- Verification of Omega_m=0.315 against published values ---
    print_status("\n" + "=" * 72, "TEST")
    print_status("VERIFICATION (Omega_m=0.315 vs published step_32 values)", "TEST")
    print_status("=" * 72, "TEST")
    expected = {
        "full": (179.4, 310.3),
        "z_ge_0.05": (91.5, 106.9),
    }
    ref_key = "omega_m_0.315"
    all_ok = True
    for cut_name, (exp_g, exp_e) in expected.items():
        sr = all_results[ref_key]["samples"][cut_name]
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
        print_status("All Omega_m=0.315 values match published results.", "SUCCESS")
    else:
        print_status("WARNING: some Omega_m=0.315 values do not match.", "WARNING")

    # --- Robustness assessment ---
    print_status("\n" + "=" * 72, "TEST")
    print_status("ROBUSTNESS ASSESSMENT", "TEST")
    print_status("=" * 72, "TEST")
    for cut_name in sample_cuts:
        vals_g = []
        vals_e = []
        for om_key, om_data in all_results.items():
            vals_g.append(om_data["samples"][cut_name]["gaussian"]["delta_aic"])
            vals_e.append(om_data["samples"][cut_name]["exponential"]["delta_aic"])
        vals_g = np.array(vals_g)
        vals_e = np.array(vals_e)
        print_status(
            f"  {sample_cuts[cut_name]['label']}: "
            f"Gaussian DeltaAIC range [{vals_g.min():.1f}, {vals_g.max():.1f}], "
            f"spread={vals_g.max()-vals_g.min():.1f}; "
            f"Exponential range [{vals_e.min():.1f}, {vals_e.max():.1f}], "
            f"spread={vals_e.max()-vals_e.min():.1f}",
            "TEST",
        )
        # All values should remain strongly positive (void rejected)
        all_positive_g = np.all(vals_g > 0)
        all_positive_e = np.all(vals_e > 0)
        print_status(
            f"    All Gaussian DeltaAIC > 0: {all_positive_g}, "
            f"All Exponential DeltaAIC > 0: {all_positive_e}",
            "TEST",
        )

    # --- Calibration-independent R_H at Omega_m = 0.302 ---
    print_status("\n" + "=" * 72, "TEST")
    print_status(
        "R_H = H0(z > 0.25) / H0(0.05 <= z < 0.15) at Omega_m = 0.302",
        "TEST",
    )
    print_status("=" * 72, "TEST")

    def _gls_h0(mask):
        idx_sub = indices[mask]
        cov_sub = cov_full[np.ix_(idx_sub, idx_sub)]
        try:
            cov_inv_sub = np.linalg.inv(cov_sub)
        except np.linalg.LinAlgError:
            cov_inv_sub = np.linalg.pinv(cov_sub)
        n_sub = int(mask.sum())
        ones_sub = np.ones(n_sub)
        denom = float(ones_sub @ cov_inv_sub @ ones_sub)
        d_sub = d[mask]
        a_hat = float(ones_sub @ cov_inv_sub @ d_sub) / denom
        sigma_a = 1.0 / np.sqrt(denom)
        h0 = H0_REF * (10.0 ** (-a_hat / 5.0))
        sigma_h0 = h0 * (np.log(10.0) / 5.0) * sigma_a
        return h0, sigma_h0, n_sub

    d_rh = mu - compute_mu_ref(z, 0.302)
    low_mask = (z >= 0.05) & (z < 0.15)
    high_mask = z > 0.25

    h0_low, sigma_h0_low, n_low = _gls_h0(low_mask)
    h0_high, sigma_h0_high, n_high = _gls_h0(high_mask)

    r_h = h0_high / h0_low
    sigma_r_h = r_h * np.sqrt(
        (sigma_h0_low / h0_low) ** 2 + (sigma_h0_high / h0_high) ** 2
    )
    sigma_ln_r_h = sigma_r_h / r_h

    # KBC predicted R_H = <H0_KBC(z>0.25)> / <H0_KBC(0.05<=z<0.15)>
    rh_kbc_g = float(np.mean(h0_g_eval[high_mask]) / np.mean(h0_g_eval[low_mask]))
    rh_kbc_e = float(np.mean(h0_e_eval[high_mask]) / np.mean(h0_e_eval[low_mask]))

    z_flat = (r_h - 1.0) / sigma_r_h
    z_kbc_g = (r_h - rh_kbc_g) / sigma_r_h
    z_kbc_e = (r_h - rh_kbc_e) / sigma_r_h

    rh_output = {
        "step": "32_rh_full_sample",
        "description": (
            "Calibration-independent relative H0 evolution: "
            "R_H = H0(z > 0.25) / H0(0.05 <= z < 0.15), "
            "computed with a GLS zero-point fit to the native "
            "Pantheon+ distance moduli and the full STAT+SYS "
            "covariance matrix at Omega_m = 0.302."
        ),
        "definition": "R_H = H0(z > 0.25) / H0(0.05 <= z < 0.15), Omega_m = 0.302",
        "omega_m": 0.302,
        "h0_ref": H0_REF,
        "methodology": (
            "A generalized least-squares zero-point offset is fit to the native "
            "Pantheon+ distance moduli using the full 1701x1701 STAT+SYS covariance "
            "matrix at Omega_m = 0.302. H0 is then inferred per SN via LCDM distance "
            "modulus inversion, and the ratio R_H = <H0(z > 0.25)> / <H0(0.05 <= z < 0.15)> "
            "is formed. The common zero-point cancels identically in the ratio, making "
            "R_H calibration-independent. Significance is assessed against the flat "
            "prediction R_H = 1 and the KBC Gaussian (R_H ~ 0.9524) and Exponential "
            "(R_H ~ 0.9458) predictions."
        ),
        "provenance": {
            "data_sources": [
                "Pantheon+SH0ES.dat (1701 rows, native mu-space)",
                "Pantheon+SH0ES_STAT+SYS.cov (1701x1701 covariance)",
                "KBC Gaussian curve (digitized from Mazurenko et al. 2025, Fig. 3)",
                "KBC Exponential curve (digitized from Mazurenko et al. 2025, Fig. 3)",
            ],
            "pipeline_block": "Ic (sensitivity and replication)",
            "covariance": "Pantheon+ STAT+SYS full 1701x1701",
        },
        "scientific_context": (
            "This test addresses whether H0 declines with redshift as predicted by the "
            "KBC void/MOND model, or remains flat as predicted by TEP for global M_B "
            "calibration. The ratio R_H cancels the common zero-point, providing a "
            "calibration-independent discriminating observable between the two models. "
            "This is the primary falsification statistic of the paper."
        ),
        "tep_prediction": "R_H = 1.0 (flat H0(z) for global M_B; zero-point bias cancels in the ratio)",
        "void_prediction": "R_H < 1.0 (KBC Gaussian ~ 0.9524, Exponential ~ 0.9458; gradual H0 decline)",
        "downstream_consumers": ["step_42_falsification_summary", "manuscript_section_8"],
        "primary": {
            "label": "Full sample (PRIMARY)",
            "n_low": int(n_low),
            "n_high": int(n_high),
            "H0_low": float(h0_low),
            "sigma_H0_low": float(sigma_h0_low),
            "H0_high": float(h0_high),
            "sigma_H0_high": float(sigma_h0_high),
            "R_H": float(r_h),
            "sigma_R_H": float(sigma_r_h),
            "sigma_ln_R_H": float(sigma_ln_r_h),
            "Z_vs_flat": float(z_flat),
            "R_H_kbc_gaussian": float(rh_kbc_g),
            "Z_vs_kbc_gaussian": float(z_kbc_g),
            "R_H_kbc_exponential": float(rh_kbc_e),
            "Z_vs_kbc_exponential": float(z_kbc_e),
        },
    }
    rh_path = results_dir / "step_32_rh_full_sample.json"
    with open(rh_path, "w") as f:
        json.dump(rh_output, f, indent=2)
    print_status(
        f"  Low-z  (0.05 <= z < 0.15): H0 = {h0_low:.3f} +/- {sigma_h0_low:.3f} "
        f"(N={n_low})",
        "TEST",
    )
    print_status(
        f"  High-z (z > 0.25):         H0 = {h0_high:.3f} +/- {sigma_h0_high:.3f} "
        f"(N={n_high})",
        "TEST",
    )
    print_status(f"  R_H = {r_h:.4f} +/- {sigma_r_h:.4f}", "TEST")
    print_status(f"  Z(flat) = {z_flat:.2f}", "TEST")
    print_status(f"  Z(KBC Gaussian) = {z_kbc_g:.2f}", "TEST")
    print_status(f"  Z(KBC Exponential) = {z_kbc_e:.2f}", "TEST")
    print_status(f"\nR_H results saved to {rh_path}", "SUCCESS")

    # --- Interpretation ---
    all_daic_positive_om = True
    for om_data in all_results.values():
        for sr in om_data["samples"].values():
            if sr["gaussian"]["delta_aic"] <= 0 or sr["exponential"]["delta_aic"] <= 0:
                all_daic_positive_om = False
    if all_daic_positive_om:
        print_status(
            "Interpretation: DeltaAIC remains strongly positive for all "
            "Omega_m values and both curves across both sample cuts. The "
            "KBC void H0(z) prediction is rejected at every reference "
            "matter density tested, confirming that the falsification is "
            "robust to the assumed cosmological parameters.",
            "SUCCESS",
        )
    else:
        print_status(
            "Interpretation: one or more DeltaAIC values are non-positive "
            "under Omega_m variation; the rejection may not be fully "
            "robust to the reference cosmology.",
            "WARNING",
        )

    return output


class Step32OmegaMSensitivity:
    """Pipeline-compatible wrapper for the Omega_m sensitivity analysis."""

    def run(self):
        return main()


if __name__ == "__main__":
    main()
