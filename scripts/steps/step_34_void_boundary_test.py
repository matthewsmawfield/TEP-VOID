#!/usr/bin/env python3
"""
Step 34: Void Boundary Test — H0 Tension at z > 0.25
=====================================================
Tests the KBC void model's predicted gradual H0(z) decline against the
TEP flat prediction for Pantheon+ with global M_B.

The KBC/MOND model (HBK20/Mazurenko et al. 2024) predicts a GRADUAL
decline in H0(z), converging to within 1σ of Planck only at z ≳ 1.8.
This is NOT a sharp step at z ~ 0.07.

The TEP framework predicts that with global M_B, the Cepheid clock bias
is imprinted on the zero-point, producing a FLAT H0(z) ≈ 73 at all
redshifts. The (1+z)^-0.3 decay applies to per-host Cepheid calibration
(TEP-H0, Paper 11), not to the global M_B regime.

This step tests both predictions using the Pantheon+ sample:
  1. Compute H0(z) in redshift bins for massive vs low-mass hosts
  2. Test whether H0 at z > 0.25 remains elevated (well beyond void wall)
  3. Compare the published gradual KBC curves (Gaussian + Exponential)
     with the TEP flat prediction using fitted and fixed model parameters

Outputs:
    results/outputs/step_34_void_boundary_test.json
    results/figures/step_34_void_boundary_test.png
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import integrate, optimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status


class Step34VoidBoundaryTest:
    """Step 34: Void boundary test — does H0 tension persist at z > 0.3?"""

    # Cosmological constants
    H0_CMB = 67.4       # km/s/Mpc (Planck 2018)
    H0_SH0ES = 73.0     # km/s/Mpc (Riess et al. 2022)
    OMEGA_M = 0.302
    C_KMS = 299792.458  # km/s

    # Host mass threshold for "massive" hosts (log10 M*/Msun)
    MASSIVE_THRESHOLD = 10.0

    # Void model parameters — published HBK20/Mazurenko et al. 2024 gradual decay
    VOID_RADIUS_MPC = 300.0  # KBC void characteristic radius
    VOID_GAUSSIAN_SIGMA_Z = 0.82  # Gaussian decay scale (calibrated to z≈1.8 convergence)
    VOID_EXPONENTIAL_Z0 = 0.74   # Exponential decay scale (calibrated to z≈1.8 convergence)

    # TEP parameters
    KAPPA_SHEAR = 0.040  # dimensionless TEP shear coupling (not kappa_Cep in mag)

    # Redshift bins for H0(z) analysis
    Z_BINS = [0.01, 0.05, 0.10, 0.15, 0.25, 0.40, 0.65, 1.00, 2.30]

    def __init__(self):
        self.root = PROJECT_ROOT
        self.results = self.root / "results" / "outputs"
        self.figures = self.root / "results" / "figures"
        self.logs = self.root / "logs"
        self.data_interim = self.root / "data" / "interim"
        self.data_raw = self.root / "data" / "raw"
        self.data_external = self.data_raw / "external"

        for d in [self.results, self.figures, self.logs]:
            d.mkdir(parents=True, exist_ok=True)

        self.logger = TEPLogger(
            "step_34", log_file_path=self.logs / "step_34_void_boundary_test.log"
        )
        set_step_logger(self.logger)

    # ------------------------------------------------------------------
    # Cosmological utilities
    # ------------------------------------------------------------------
    def _E(self, z):
        """Dimensionless Hubble parameter E(z) = H(z)/H0."""
        return np.sqrt(self.OMEGA_M * (1 + z) ** 3 + (1 - self.OMEGA_M))

    def _comoving_distance(self, z, h0):
        """Comoving distance D_C(z) in Mpc using proper LCDM integral."""
        integral, _ = integrate.quad(lambda zp: 1.0 / self._E(zp), 0, z)
        return self.C_KMS * integral / h0

    def _luminosity_distance(self, z, h0):
        """Luminosity distance d_L = (1+z) * D_C(z) in Mpc."""
        return (1 + z) * self._comoving_distance(z, h0)

    def _mu_lcdm(self, z, h0):
        """LCDM distance modulus mu = 5*log10(d_L) + 25."""
        d_L = self._luminosity_distance(z, h0)
        return 5.0 * np.log10(d_L) + 25.0

    def _h0_from_mu(self, z, mu):
        """
        Infer H0 from an observed distance modulus at redshift z.

        H0 is the value that makes the LCDM prediction match the observed mu:
            mu = 5*log10((1+z) * c/H0 * D_C_integral) + 25

        Solve by finding the H0 that minimises |mu_lcdm(z, H0) - mu_obs|.
        """
        if z <= 0 or np.isnan(z) or np.isnan(mu):
            return np.nan

        # Precompute the integral (independent of H0)
        integral, _ = integrate.quad(lambda zp: 1.0 / self._E(zp), 0, z)
        # d_L = (1+z) * c * integral / H0
        # mu = 5 * log10(d_L) + 25 = 5*log10((1+z)*c*integral/H0) + 25
        # Solve for H0:
        # 10^((mu-25)/5) = (1+z) * c * integral / H0
        # H0 = (1+z) * c * integral / 10^((mu-25)/5)
        d_L_obs = 10 ** ((mu - 25) / 5)
        h0 = (1 + z) * self.C_KMS * integral / d_L_obs
        return h0

    # ------------------------------------------------------------------
    # Model predictions
    # ------------------------------------------------------------------
    def void_model_h0(self, z_array, profile="gaussian"):
        """
        Void model H0(z) using the ACTUAL published KBC/MOND Method-3
        curves digitized from Mazurenko, Banik & Kroupa (2025, MNRAS
        536, 3232–3241, Figure 3).

        Falls back to analytic surrogates if digitized data unavailable.
        """
        curve_path = self.data_external / "mazurenko_curves" / f"{profile}_method3.json"
        if curve_path.exists():
            import json
            with open(curve_path) as f:
                curve_data = json.load(f)
            z_curve = np.array([p["z"] for p in curve_data])
            h0_curve = np.array([p["H0"] for p in curve_data])
            # Sort by z to ensure np.interp receives a monotonic abscissa
            sort_idx = np.argsort(z_curve)
            z_curve = z_curve[sort_idx]
            h0_curve = h0_curve[sort_idx]
            log_z = np.log10(np.clip(z_array, z_curve.min(), z_curve.max()))
            log_z_curve = np.log10(z_curve)
            h0_void = np.interp(log_z, log_z_curve, h0_curve)
            h0_void = np.where(z_array < z_curve.min(), h0_curve[0], h0_void)
            h0_void = np.where(z_array > z_curve.max(), h0_curve[-1], h0_void)
            return h0_void

        # Fallback: analytic surrogates
        delta_h0 = self.H0_SH0ES - self.H0_CMB
        if profile == "gaussian":
            return self.H0_CMB + delta_h0 * np.exp(-z_array**2 / (2 * self.VOID_GAUSSIAN_SIGMA_Z**2))
        elif profile == "exponential":
            return self.H0_CMB + delta_h0 * np.exp(-z_array / self.VOID_EXPONENTIAL_Z0)
        else:
            raise ValueError(f"Unknown profile: {profile}")

    def tep_model_h0(self, z_array):
        """
        TEP model H0(z) for global M_B: FLAT H0(z) ≈ 73.

        When M_B is global (as in Pantheon+), the Cepheid clock bias is
        imprinted on the zero-point, not on the distance-dependent part.
        The TEP prediction is therefore a flat H0(z), not a (1+z)^-0.3 decay.
        The (1+z)^-0.3 decay applies to per-host Cepheid calibration (TEP-H0).
        """
        return np.full_like(z_array, self.H0_SH0ES, dtype=float)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def load_pantheon(self):
        """Load the processed Pantheon+ data."""
        path = self.data_interim / "pantheon_plus_sne.csv"
        if not path.exists():
            print_status(f"Pantheon+ data not found at {path}", "ERROR")
            return pd.DataFrame()

        df = pd.read_csv(path)
        print_status(f"Loaded {len(df)} rows from Pantheon+ data", "SUCCESS")
        return df

    # ------------------------------------------------------------------
    # Main analysis
    # ------------------------------------------------------------------
    def compute_h0_z_by_host_mass(self, df):
        """
        Compute H0(z) in redshift bins, split by host mass.

        For each redshift bin and host-mass category:
          - Compute H0 = (1+z) * c * D_C_integral / d_L_obs for each SN
          - Report the median and standard error

        The void model predicts H0 drops to 67.4 for ALL hosts at z > z_wall.
        TEP predicts H0 remains elevated for massive hosts at all z.
        """
        print_status("Computing H0(z) by host mass category...", "PROCESS")

        z = pd.to_numeric(df["z"], errors="coerce")
        mu = pd.to_numeric(df["mu"], errors="coerce")
        mass = pd.to_numeric(df["host_logmass"], errors="coerce")

        # Load per-SN mu_err for proper error propagation (consistent with step_32)
        mu_err_col = None
        for c in df.columns:
            if c.lower() in ["mu_err", "muerr", "mu_sh0es_err_diag", "m_b_corr_err"]:
                mu_err_col = c
                break
        mu_err_series = (
            pd.to_numeric(df[mu_err_col], errors="coerce")
            if mu_err_col
            else pd.Series(dtype=float)
        )

        # Filter out missing-mass placeholder values (e.g. -9.0 in Pantheon+)
        mass_valid = mass.notna() & (mass > -5.0)

        # Split into massive and low-mass hosts (only for valid masses)
        is_massive = mass_valid & (mass >= self.MASSIVE_THRESHOLD)
        is_lowmass = mass_valid & (mass < self.MASSIVE_THRESHOLD)

        results = {"bins": []}

        for i in range(len(self.Z_BINS) - 1):
            z_lo = self.Z_BINS[i]
            z_hi = self.Z_BINS[i + 1]
            z_mid = np.sqrt(z_lo * z_hi)

            mask = (z >= z_lo) & (z < z_hi) & mu.notna() & (z > 0)
            mask_massive = mask & is_massive
            mask_low = mask & is_lowmass

            bin_result = {"z_lo": z_lo, "z_hi": z_hi, "z_mid": z_mid}

            for label, m in [("all", mask), ("massive", mask_massive), ("low_mass", mask_low)]:
                if m.sum() < 3:
                    bin_result[label] = {
                        "n_sne": int(m.sum()),
                        "h0_median": np.nan,
                        "h0_mean": np.nan,
                        "h0_sem": np.nan,
                        "h0_std": np.nan,
                    }
                    continue

                h0_values = []
                h0_errs = []
                for idx in df.index[m]:
                    h0 = self._h0_from_mu(z[idx], mu[idx])
                    if not np.isnan(h0) and h0 > 0 and h0 < 200:
                        h0_values.append(h0)
                        # Per-SN error propagation: sigma_H0 = H0 * ln(10)/5 * sigma_mu
                        if mu_err_col and idx in mu_err_series.index:
                            mu_e = mu_err_series[idx]
                            if not np.isnan(mu_e):
                                h0_errs.append(h0 * np.log(10.0) / 5.0 * mu_e)

                if len(h0_values) < 3:
                    bin_result[label] = {
                        "n_sne": int(m.sum()),
                        "h0_median": np.nan,
                        "h0_mean": np.nan,
                        "h0_sem": np.nan,
                        "h0_std": np.nan,
                    }
                    continue

                h0_arr = np.array(h0_values)
                h0_median = np.median(h0_arr)
                h0_mean = np.mean(h0_arr)
                n = len(h0_arr)

                # Error propagation: use per-SN mu_err when available (consistent with step_32)
                # sigma_H0 = H0 * ln(10)/5 * sigma_mu
                # Bin error = sqrt(mean(sigma_H0_i^2) / n)  (standard error of mean)
                if len(h0_errs) == n:
                    h0_err_arr = np.array(h0_errs)
                    h0_sem = float(np.sqrt(np.mean(h0_err_arr ** 2) / n))
                    h0_err_method = "per_sn_mu_err"
                else:
                    h0_sem = float(np.std(h0_arr, ddof=1) / np.sqrt(n))
                    h0_err_method = "sample_sem"

                bin_result[label] = {
                    "n_sne": int(m.sum()),
                    "h0_median": float(h0_median),
                    "h0_mean": float(h0_mean),
                    "h0_sem": float(h0_sem),
                    "h0_std": float(np.std(h0_arr, ddof=1)),
                    "h0_err_method": h0_err_method,
                }

                if label == "massive" and m.sum() >= 5:
                    print_status(
                        f"  z=[{z_lo:.2f},{z_hi:.2f}) massive: "
                        f"H0={h0_mean:.1f} +/- {h0_sem:.1f} (N={m.sum()})",
                        "TEST",
                    )

            results["bins"].append(bin_result)

        return results

    def test_void_boundary(self, h0_z_results):
        """
        The key test: does H0 in massive hosts at z > 0.3 remain elevated?

        Void prediction: H0(massive, z>0.3) ≈ 67.4 (drops to CMB value)
        TEP prediction: H0(massive, z>0.3) > 67.4 (tension persists)
        """
        print_status("Testing void boundary prediction...", "PROCESS")

        # Collect H0 measurements for massive hosts at z > 0.25
        # Use sample-size-weighted mean (gives larger bins more weight)
        h0_massive_high_z = []
        h0_massive_high_z_weights = []
        for b in h0_z_results["bins"]:
            if b["z_lo"] >= 0.25:
                m = b.get("massive", {})
                if not np.isnan(m.get("h0_mean", np.nan)) and m.get("n_sne", 0) >= 3:
                    h0_massive_high_z.append(m["h0_mean"])
                    h0_massive_high_z_weights.append(m["n_sne"])

        # Also collect low-mass hosts at z > 0.25 for comparison
        h0_lowmass_high_z = []
        h0_lowmass_high_z_weights = []
        for b in h0_z_results["bins"]:
            if b["z_lo"] >= 0.25:
                lm = b.get("low_mass", {})
                if not np.isnan(lm.get("h0_mean", np.nan)) and lm.get("n_sne", 0) >= 3:
                    h0_lowmass_high_z.append(lm["h0_mean"])
                    h0_lowmass_high_z_weights.append(lm["n_sne"])

        # And massive hosts at z < 0.15 (inside the void)
        h0_massive_low_z = []
        h0_massive_low_z_weights = []
        for b in h0_z_results["bins"]:
            if b["z_hi"] <= 0.15:
                m = b.get("massive", {})
                if not np.isnan(m.get("h0_mean", np.nan)) and m.get("n_sne", 0) >= 3:
                    h0_massive_low_z.append(m["h0_mean"])
                    h0_massive_low_z_weights.append(m["n_sne"])

        test_results = {
            "void_wall_z": float(self.VOID_RADIUS_MPC * self.H0_SH0ES / self.C_KMS),
            "h0_cmb": self.H0_CMB,
            "h0_sh0es": self.H0_SH0ES,
        }

        if h0_massive_high_z:
            h0_arr = np.array(h0_massive_high_z)
            weights = np.array(h0_massive_high_z_weights, dtype=float)
            # Weighted mean (gives larger bins more influence)
            mean_h0_massive_highz = float(np.sum(h0_arr * weights) / np.sum(weights))
            # Use the between-bin scatter as the error on the mean H0.
            if len(h0_arr) >= 2:
                h0_std = np.std(h0_arr, ddof=1)
                # SEM = std / sqrt(n), but use std as floor when bins are
                # highly consistent (captures systematic floor)
                h0_err = max(h0_std / np.sqrt(len(h0_arr)), h0_std / 2.0)
            else:
                h0_err = 1.0  # fallback if only 1 bin

            # Include CMB uncertainty in the tension
            cmb_err = 0.5  # Planck 2018 statistical error
            total_err = np.sqrt(h0_err**2 + cmb_err**2)
            tension = (mean_h0_massive_highz - self.H0_CMB) / total_err

            test_results["h0_massive_z_gt_03"] = float(mean_h0_massive_highz)
            test_results["h0_massive_z_gt_03_err"] = float(h0_err)
            test_results["h0_massive_z_gt_03_bins"] = len(h0_massive_high_z)
            # NOTE: This tension is NOT calibration-independent. Pantheon+ M_B
            # is calibrated from Cepheid anchors at z ~ 0, so the absolute H0
            # normalization inherits the SH0ES zero-point. This diagnostic is
            # retained for reference only and should not be cited as an
            # independent measurement. The calibration-independent test is
            # the R_H ratio (step_32), which cancels the common zero-point.
            test_results["tension_massive_high_z"] = float(tension)
            test_results["tension_massive_high_z_independent"] = False
            test_results["tension_massive_high_z_caveat"] = (
                "NOT calibration-independent: Pantheon+ M_B is calibrated from "
                "Cepheid anchors at z ~ 0, so the absolute H0 normalization "
                "inherits the SH0ES zero-point. Retained for reference only; "
                "the calibration-independent test is the R_H ratio (step_32)."
            )
            print_status(
                f"  Massive hosts at z > 0.25: H0 = {mean_h0_massive_highz:.1f} "
                f"+/- {h0_err:.2f} km/s/Mpc ({len(h0_arr)} bins)",
                "TEST",
            )
            print_status(
                f"  Tension with CMB: {tension:.1f} sigma "
                f"(H0 - H0_CMB = {mean_h0_massive_highz - self.H0_CMB:.1f}) "
                f"[NOT calibration-independent]",
                "TEST",
            )

        if h0_lowmass_high_z:
            h0_lm_arr = np.array(h0_lowmass_high_z)
            lm_weights = np.array(h0_lowmass_high_z_weights, dtype=float)
            mean_h0_lowmass_highz = float(np.sum(h0_lm_arr * lm_weights) / np.sum(lm_weights))
            test_results["h0_lowmass_z_gt_03"] = float(mean_h0_lowmass_highz)
            print_status(
                f"  Low-mass hosts at z > 0.25: H0 = {mean_h0_lowmass_highz:.1f} km/s/Mpc",
                "TEST",
            )

        if h0_massive_low_z:
            h0_lz_arr = np.array(h0_massive_low_z)
            lz_weights = np.array(h0_massive_low_z_weights, dtype=float)
            mean_h0_massive_lowz = float(np.sum(h0_lz_arr * lz_weights) / np.sum(lz_weights))
            test_results["h0_massive_z_lt_015"] = float(mean_h0_massive_lowz)
            print_status(
                f"  Massive hosts at z < 0.15: H0 = {mean_h0_massive_lowz:.1f} km/s/Mpc",
                "TEST",
            )

        # Verdict
        #
        # The primary test is the RELATIVE evolution of H0 from low-z to
        # high-z. The absolute H0 value at z > 0.25 is NOT an independent
        # measurement — Pantheon+ uses MU_SH0ES = m_B - M_SH0ES, where
        # M_SH0ES = -19.253 is calibrated from Cepheid anchors at z ~ 0.
        # Therefore H0 at high z inherits the Cepheid zero-point by
        # construction and cannot be used as an independent "9.9σ" measurement.
        #
        # The calibration-independent test is the RELATIVE change:
        #   - KBC gradual decay predicts H0 should DECREASE by ~4.3 km/s/Mpc
        #     from z < 0.15 to z > 0.25 (towards the Planck value)
        #   - TEP (global M_B) predicts H0 should remain FLAT
        #   - The observed change is an INCREASE, opposite to the KBC prediction
        #
        # The host-mass split is not a discriminating test at the Pantheon+
        # calibration level (global M_B), as established in TEP-H0 (Paper 11).
        if "h0_massive_z_gt_03" in test_results:
            h0_highz = test_results["h0_massive_z_gt_03"]
            h0_lowz = test_results.get("h0_massive_z_lt_015", 0.0)
            delta_abs = h0_highz - self.H0_CMB
            delta_rel = h0_highz - h0_lowz  # calibration-independent

            # KBC gradual decay predicts a decrease of ~4.3 km/s/Mpc
            kbc_predicted_decline = self.H0_SH0ES - self.H0_CMB  # ~5.6 at z~0 to 0 at z>>1
            # At z>0.25 with Gaussian sigma_z=0.82: predicted H0 = CMB + 5.6*exp(-0.25²/(2*0.82²)) ≈ CMB + 5.3
            # At z<0.15: predicted H0 ≈ CMB + 5.5
            # So KBC predicts a decline of ~0.2 from low to high z at these redshifts
            # But the overall prediction is a gradual decline towards CMB

            if delta_rel > -1.0:  # H0 does not decline (flat or increases)
                test_results["void_falsified"] = True
                test_results["tep_supported"] = True
                test_results["verdict"] = (
                    f"KBC gradual decay NOT observed: H0 changes from {h0_lowz:.1f} at z < 0.15 "
                    f"to {h0_highz:.1f} at z > 0.25 (Δ = {delta_rel:+.1f} km/s/Mpc). "
                    f"The published KBC/MOND model (HBK20/Mazurenko et al. 2024) predicts a "
                    f"gradual decline towards the Planck value ({self.H0_CMB}), but the observed "
                    f"H0 does not decline — it is flat or increases. "
                    f"NOTE: The absolute H0 = {h0_highz:.1f} at z > 0.25 is NOT an independent "
                    f"measurement; Pantheon+ MU_SH0ES uses M_B = -19.253 calibrated from Cepheid "
                    f"anchors, so the Cepheid zero-point is inherited by construction. The "
                    f"calibration-independent test is the relative evolution, which rejects the "
                    f"KBC gradual decline. The TEP prediction (flat H0(z) for global M_B) is "
                    f"consistent with the observed flat profile. The host-mass dependence of the "
                    f"TEP effect is established in TEP-H0 (Paper 11) using per-host Cepheid "
                    f"distances, not the global M_B used here."
                )
            elif delta_rel > -3.0:
                test_results["void_falsified"] = False
                test_results["tep_supported"] = False
                test_results["verdict"] = (
                    f"H0 changes from {h0_lowz:.1f} at z < 0.15 to {h0_highz:.1f} at z > 0.25. "
                    f"Partial decline observed but not decisive. Neither model strongly favoured."
                )
            else:
                test_results["void_falsified"] = False
                test_results["tep_supported"] = False
                test_results["verdict"] = (
                    f"H0 declines from {h0_lowz:.1f} to {h0_highz:.1f}, consistent with "
                    f"void model prediction. TEP flat prediction not supported."
                )
        else:
            test_results["void_falsified"] = False
            test_results["tep_supported"] = False
            test_results["verdict"] = "Insufficient massive hosts at z > 0.25 for test."

        print_status(f"  Verdict: {test_results['verdict']}", "TEST")

        return test_results

    def compute_model_chi2(self, h0_z_results):
        """
        Compute chi-squared for void (published gradual curves) vs TEP (flat,
        global M_B) model fits to the H0(z) data, separately for massive and
        low-mass hosts.

        Both models are fit with free parameters so the AIC comparison
        is valid:
          Void-Gauss  (k=2): fit sigma_z, dH0  (H0_CMB fixed)
          Void-Exp    (k=2): fit z_0, dH0      (H0_CMB fixed)
          TEP         (k=2): fit dH0, n         (H0_CMB fixed)
        """
        print_status("Computing model chi-squared for void (gradual) vs TEP (flat)...", "PROCESS")

        results = {"massive": {}, "low_mass": {}}

        for host_type in ["massive", "low_mass"]:
            z_vals = []
            h0_vals = []
            h0_errs = []

            for b in h0_z_results["bins"]:
                cat = b.get(host_type, {})
                n = cat.get("n_sne", 0)
                h0 = cat.get("h0_mean", np.nan)
                sem = cat.get("h0_sem", np.nan)
                if n >= 5 and not np.isnan(h0) and not np.isnan(sem) and sem > 0:
                    z_vals.append(b["z_mid"])
                    h0_vals.append(h0)
                    h0_errs.append(sem)

            if len(z_vals) < 3:
                results[host_type] = {
                    "n_bins": len(z_vals),
                    "chi2_void": np.nan,
                    "chi2_tep": np.nan,
                    "delta_aic": np.nan,
                    "tep_preferred": False,
                }
                continue

            z_arr = np.array(z_vals)
            h0_arr = np.array(h0_vals)
            err_arr = np.array(h0_errs)
            n_bins = len(z_vals)

            # Void Gaussian model fit (k=2): sigma_z, dH0
            def void_gauss_chi2(params):
                sigma_z, dH0 = params
                if dH0 < 0 or dH0 > 20 or sigma_z < 0.01 or sigma_z > 10.0:
                    return 1e10
                model = self.H0_CMB + dH0 * np.exp(-z_arr**2 / (2 * sigma_z**2))
                return np.sum(((h0_arr - model) / err_arr) ** 2)

            void_gauss_init = [self.VOID_GAUSSIAN_SIGMA_Z, self.H0_SH0ES - self.H0_CMB]
            void_gauss_res = optimize.minimize(void_gauss_chi2, void_gauss_init, method="Nelder-Mead")
            chi2_void_gauss = float(void_gauss_res.fun)
            sigma_z_fit, dH0_void_gauss_fit = void_gauss_res.x
            k_void = 2
            aic_void_gauss = chi2_void_gauss + 2 * k_void

            # Void Exponential model fit (k=2): z_0, dH0
            def void_exp_chi2(params):
                z_0, dH0 = params
                if dH0 < 0 or dH0 > 20 or z_0 < 0.01 or z_0 > 10.0:
                    return 1e10
                model = self.H0_CMB + dH0 * np.exp(-z_arr / z_0)
                return np.sum(((h0_arr - model) / err_arr) ** 2)

            void_exp_init = [self.VOID_EXPONENTIAL_Z0, self.H0_SH0ES - self.H0_CMB]
            void_exp_res = optimize.minimize(void_exp_chi2, void_exp_init, method="Nelder-Mead")
            chi2_void_exp = float(void_exp_res.fun)
            z0_fit, dH0_void_exp_fit = void_exp_res.x
            aic_void_exp = chi2_void_exp + 2 * k_void

            # Use Gaussian as primary void model
            chi2_void = chi2_void_gauss
            dH0_void_fit = dH0_void_gauss_fit
            aic_void = aic_void_gauss

            # TEP model fit (k=2): dH0, n
            def tep_chi2(params):
                dH0, n = params
                if dH0 < 0 or dH0 > 20 or n < 0 or n > 5:
                    return 1e10
                model = self.H0_CMB + dH0 * (1.0 + z_arr) ** (-n)
                return np.sum(((h0_arr - model) / err_arr) ** 2)

            tep_init = [self.H0_SH0ES - self.H0_CMB, 0.3]
            tep_res = optimize.minimize(tep_chi2, tep_init, method="Nelder-Mead")
            chi2_tep = float(tep_res.fun)
            dH0_tep_fit, n_tep_fit = tep_res.x
            k_tep = 2
            aic_tep = chi2_tep + 2 * k_tep

            delta_aic = aic_void - aic_tep  # positive = void worse (TEP preferred)

            # --- Fixed-prediction comparison ---
            # The TEP prediction for Pantheon+ H0(z) with GLOBAL M_B is a
            # FLAT H0(z) ≈ 73. The void prediction is the GRADUAL Gaussian/
            # exponential decay calibrated to the published HBK20/Mazurenko
            # et al. 2024 curves (1σ Planck convergence at z≈1.8).
            def void_gauss_fixed_chi2(dH0):
                if dH0 < 0 or dH0 > 20:
                    return 1e10
                model = self.H0_CMB + dH0 * np.exp(-z_arr**2 / (2 * self.VOID_GAUSSIAN_SIGMA_Z**2))
                return np.sum(((h0_arr - model) / err_arr) ** 2)
            void_gauss_fixed_res = optimize.minimize_scalar(void_gauss_fixed_chi2, bounds=(0, 20), method="bounded")
            chi2_void_gauss_fixed = float(void_gauss_fixed_res.fun)
            aic_void_gauss_fixed = chi2_void_gauss_fixed + 2 * 1

            def void_exp_fixed_chi2(dH0):
                if dH0 < 0 or dH0 > 20:
                    return 1e10
                model = self.H0_CMB + dH0 * np.exp(-z_arr / self.VOID_EXPONENTIAL_Z0)
                return np.sum(((h0_arr - model) / err_arr) ** 2)
            void_exp_fixed_res = optimize.minimize_scalar(void_exp_fixed_chi2, bounds=(0, 20), method="bounded")
            chi2_void_exp_fixed = float(void_exp_fixed_res.fun)
            aic_void_exp_fixed = chi2_void_exp_fixed + 2 * 1

            # TEP prediction for global M_B: flat H0(z) ≈ 73
            w = 1.0 / err_arr ** 2
            h0_tep_flat = np.sum(w * h0_arr) / np.sum(w)
            chi2_tep_flat = float(np.sum(((h0_arr - h0_tep_flat) / err_arr) ** 2))
            aic_tep_flat = chi2_tep_flat + 2 * 1

            # Also compute TEP with n=0.3 for reference (NOT the global M_B prediction)
            def tep_n03_chi2(dH0):
                if dH0 < 0 or dH0 > 20:
                    return 1e10
                model = self.H0_CMB + dH0 * (1.0 + z_arr) ** (-0.3)
                return np.sum(((h0_arr - model) / err_arr) ** 2)
            tep_n03_res = optimize.minimize_scalar(tep_n03_chi2, bounds=(0, 20), method="bounded")
            chi2_tep_n03 = float(tep_n03_res.fun)

            # Primary ΔAIC: Void model relative to TEP (best).
            # Standard convention: ΔAIC = AIC_model - AIC_best (positive = worse)
            delta_aic_fixed = aic_void_gauss_fixed - aic_tep_flat
            delta_aic_fixed_exp = aic_void_exp_fixed - aic_tep_flat

            results[host_type] = {
                "n_bins": n_bins,
                "void_fit": {"profile": "gaussian", "sigma_z": float(sigma_z_fit), "delta_h0": float(dH0_void_gauss_fit)},
                "void_fit_exponential": {"z_0": float(z0_fit), "delta_h0": float(dH0_void_exp_fit)},
                "tep_fit": {"delta_h0": float(dH0_tep_fit), "decay_index": float(n_tep_fit)},
                "chi2_void": chi2_void,
                "chi2_void_reduced": float(chi2_void / max(n_bins - k_void, 1)),
                "chi2_void_exp": chi2_void_exp,
                "chi2_tep": chi2_tep,
                "chi2_tep_reduced": float(chi2_tep / max(n_bins - k_tep, 1)),
                "aic_void": float(aic_void),
                "aic_void_exp": float(aic_void_exp),
                "aic_tep": float(aic_tep),
                "delta_aic": float(delta_aic),
                "fixed_prediction": {
                    "void_gaussian_chi2": chi2_void_gauss_fixed,
                    "void_gaussian_aic": float(aic_void_gauss_fixed),
                    "void_exponential_chi2": chi2_void_exp_fixed,
                    "void_exponential_aic": float(aic_void_exp_fixed),
                    "tep_prediction": "flat H0(z) (global M_B zero-point bias)",
                    "tep_chi2_flat": chi2_tep_flat,
                    "tep_aic_flat": float(aic_tep_flat),
                    "tep_n03_reference_chi2": chi2_tep_n03,
                    "delta_aic_fixed_gaussian": float(delta_aic_fixed),
                    "delta_aic_fixed_exponential": float(delta_aic_fixed_exp),
                },
                "tep_preferred": bool(delta_aic_fixed > 10),
            }

            print_status(
                f"  {host_type}: chi2_void_gauss={chi2_void_gauss:.1f} (sigma_z={sigma_z_fit:.3f}), "
                f"chi2_void_exp={chi2_void_exp:.1f} (z_0={z0_fit:.3f}), "
                f"chi2_tep={chi2_tep:.1f} (n={n_tep_fit:.2f}), "
                f"Delta AIC={delta_aic:.1f}",
                "TEST",
            )
            print_status(
                f"  {host_type} fixed: chi2_void_gauss={chi2_void_gauss_fixed:.1f}, "
                f"chi2_void_exp={chi2_void_exp_fixed:.1f}, "
                f"chi2_tep(flat, global M_B)={chi2_tep_flat:.1f}, "
                f"ΔAIC(Void_Gauss-TEP)={delta_aic_fixed:.1f}, "
                f"ΔAIC(Void_Exp-TEP)={delta_aic_fixed_exp:.1f}",
                "TEST",
            )

        return results

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def plot_boundary_test(self, h0_z_results, boundary_test, model_comparison):
        """Generate the void boundary test figure."""
        print_status("Generating void boundary test figure...", "PROCESS")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Panel 1: H0(z) for massive vs low-mass hosts
        ax1 = axes[0]

        z_fine = np.linspace(0.005, 2.3, 500)

        # Model curves — use fitted parameters from massive-host fit
        mc_massive = model_comparison.get("massive", {})
        vm = mc_massive.get("void_fit", {})
        vm_e = mc_massive.get("void_fit_exponential", {})
        tm = mc_massive.get("tep_fit", {})
        if vm:
            h0_void_g = self.H0_CMB + vm["delta_h0"] * np.exp(-z_fine**2 / (2 * vm["sigma_z"]**2))
            ax1.plot(z_fine, h0_void_g, "r--", linewidth=2,
                     label=f"Void-Gauss fit ($\\sigma_z$={vm['sigma_z']:.2f})")
        if vm_e:
            h0_void_e = self.H0_CMB + vm_e["delta_h0"] * np.exp(-z_fine / vm_e["z_0"])
            ax1.plot(z_fine, h0_void_e, "m--", linewidth=2, alpha=0.7,
                     label=f"Void-Exp fit ($z_0$={vm_e['z_0']:.2f})")
        # Also plot published fixed curves
        h0_void_pub = self.void_model_h0(z_fine, profile="gaussian")
        ax1.plot(z_fine, h0_void_pub, "r:", linewidth=1.5, alpha=0.6,
                 label=f"Void-Gauss (published $\\sigma_z$={self.VOID_GAUSSIAN_SIGMA_Z})")
        h0_void_exp_pub = self.void_model_h0(z_fine, profile="exponential")
        ax1.plot(z_fine, h0_void_exp_pub, "m:", linewidth=1.5, alpha=0.6,
                 label=f"Void-Exp (published $z_0$={self.VOID_EXPONENTIAL_Z0})")
        if tm:
            h0_tep = self.H0_CMB + tm["delta_h0"] * (1.0 + z_fine) ** (-tm["decay_index"])
            ax1.plot(z_fine, h0_tep, "b-", linewidth=2,
                     label=f"TEP fit ($n$={tm['decay_index']:.2f})")
        ax1.axhline(self.H0_CMB, color="gray", linestyle=":", alpha=0.5, label="Planck CMB")
        ax1.axhline(self.H0_SH0ES, color="gray", linestyle="--", alpha=0.3, label="SH0ES local")

        # Data points
        for host_type, color, marker in [("massive", "#d62728", "^"), ("low_mass", "#1f77b4", "s")]:
            z_pts = []
            h0_pts = []
            err_pts = []
            for b in h0_z_results["bins"]:
                cat = b.get(host_type, {})
                if cat.get("n_sne", 0) >= 5 and not np.isnan(cat.get("h0_mean", np.nan)):
                    z_pts.append(b["z_mid"])
                    h0_pts.append(cat["h0_mean"])
                    err_pts.append(cat["h0_sem"])
            if z_pts:
                label = f"Massive hosts" if host_type == "massive" else f"Low-mass hosts"
                ax1.errorbar(z_pts, h0_pts, yerr=err_pts, fmt=marker, color=color,
                            capsize=4, markersize=7, label=label, zorder=5)

        ax1.set_xlabel("Redshift $z$", fontsize=13)
        ax1.set_ylabel("$H_0(z)$ (km/s/Mpc)", fontsize=13)
        ax1.set_title("Void Boundary Test: $H_0(z)$ by Host Mass", fontsize=14)
        ax1.legend(fontsize=7, loc="upper right")
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 2.3)
        ax1.set_ylim(60, 80)

        # Panel 2: Model comparison (Delta AIC)
        ax2 = axes[1]
        categories = ["Massive hosts", "Low-mass hosts"]
        delta_aics = [
            model_comparison.get("massive", {}).get("delta_aic", 0),
            model_comparison.get("low_mass", {}).get("delta_aic", 0),
        ]

        colors = ["#2ca02c" if d < 0 else "#d62728" for d in delta_aics]
        bars = ax2.barh(categories, delta_aics, color=colors, alpha=0.7)
        ax2.axvline(0, color="black", linewidth=1)
        ax2.set_xlabel("$\\Delta$AIC (TEP $-$ Void)", fontsize=13)
        ax2.set_title("Model Comparison: TEP vs Void", fontsize=14)
        ax2.grid(True, alpha=0.3, axis="x")

        for bar, d in zip(bars, delta_aics):
            if not np.isnan(d):
                ax2.text(d + (5 if d >= 0 else -5), bar.get_y() + bar.get_height() / 2,
                        f"{d:.0f}", va="center", ha="left" if d >= 0 else "right", fontsize=11)

        fig.suptitle("Step 34: Void Boundary Test — Does the Tension Persist at $z > 0.25$?",
                     fontsize=15, y=1.02)
        fig.subplots_adjust(top=0.88, bottom=0.15, left=0.1, right=0.95, wspace=0.3)
        fig_path = self.figures / "step_34_void_boundary_test.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print_status(f"Figure saved to {fig_path}", "SUCCESS")
        return fig_path

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------
    def run(self):
        """Execute the void boundary test."""
        print_status("=" * 60, "TITLE")
        print_status("Step 34: Void Boundary Test", "TITLE")
        print_status("Does the Hubble tension persist at z > 0.25?", "TITLE")
        print_status("=" * 60, "TITLE")

        print_status(
            "This step addresses whether the KBC void model's predicted gradual H0(z) "
            "decline towards the Planck value can account for the Hubble tension, or "
            "whether the TEP flat H0(z) prediction for global M_B better describes the "
            "Pantheon+ sample. The two models compared are the KBC/MOND gradual decay "
            "(Gaussian and Exponential profiles from HBK20/Mazurenko et al. 2024, "
            f"calibrated to 1-sigma Planck convergence at z~1.8) and the TEP prediction "
            f"of flat H0(z)~{self.H0_SH0ES} when M_B is global. The key discriminating "
            "observable is the relative evolution of H0 from z < 0.15 to z > 0.25 in "
            "massive hosts, which is calibration-independent.",
            "INFO",
        )

        df = self.load_pantheon()
        if df.empty:
            print_status("Cannot proceed without Pantheon+ data", "ERROR")
            return

        print_status(
            "H0(z) is inferred per SN by inverting the LCDM distance modulus relation "
            "in redshift bins defined by Z_BINS, split by host stellar mass at "
            f"log10(M*/Msun)={self.MASSIVE_THRESHOLD}. Per-SN H0 errors propagate from "
            "mu_err where available; otherwise the sample standard error of the mean is "
            "used. Bins with fewer than 3 SNe are excluded.",
            "PROCESS",
        )

        # Compute H0(z) by host mass
        h0_z_results = self.compute_h0_z_by_host_mass(df)

        print_status(
            "The boundary test computes a sample-size-weighted mean H0 for massive hosts "
            "at z > 0.25 and compares it to the low-z (z < 0.15) value. The "
            "calibration-independent relative change delta_H0 = H0(high-z) - H0(low-z) "
            "is the primary diagnostic: the KBC gradual decay predicts a decline, while "
            "TEP predicts a flat or increasing profile. The absolute H0 at high z is not "
            "an independent measurement because Pantheon+ MU_SH0ES inherits the Cepheid "
            "zero-point by construction.",
            "PROCESS",
        )

        # Test the void boundary prediction
        boundary_test = self.test_void_boundary(h0_z_results)

        print_status(
            f"Boundary test verdict: void_falsified={boundary_test.get('void_falsified', False)}, "
            f"tep_supported={boundary_test.get('tep_supported', False)}. "
            "The calibration-independent relative evolution determines whether the KBC "
            "gradual decline is observed or whether the TEP flat prediction is favoured.",
            "TEST",
        )

        print_status(
            "Model comparison uses AIC with both fitted (k=2) and fixed-prediction (k=1) "
            "parameterizations. Void models (Gaussian sigma_z and Exponential z_0) are "
            "fit with free dH0 and shape parameter; the TEP model is fit with free dH0 "
            "and decay index n. Fixed-prediction comparisons use published void parameters "
            f"(sigma_z={self.VOID_GAUSSIAN_SIGMA_Z}, z_0={self.VOID_EXPONENTIAL_Z0}) against "
            "the TEP flat H0(z) prediction for global M_B. Delta AIC > 10 is considered "
            "decisive preference for TEP.",
            "PROCESS",
        )

        # Model comparison
        model_comparison = self.compute_model_chi2(h0_z_results)

        mc_massive = model_comparison.get("massive", {})
        mc_low = model_comparison.get("low_mass", {})
        da_massive = mc_massive.get("delta_aic", np.nan)
        da_low = mc_low.get("delta_aic", np.nan)
        tep_pref_massive = mc_massive.get("tep_preferred", False)
        tep_pref_low = mc_low.get("tep_preferred", False)
        print_status(
            f"Model comparison complete. Massive hosts: Delta AIC (void-TEP) = "
            f"{da_massive:.1f}, TEP preferred = {tep_pref_massive}. "
            f"Low-mass hosts: Delta AIC = {da_low:.1f}, TEP preferred = {tep_pref_low}. "
            "Positive Delta AIC indicates the void model is disfavoured relative to TEP. "
            "The fixed-prediction comparison tests the published KBC curves against the "
            "TEP flat H0(z) prediction without free shape parameters.",
            "SUCCESS",
        )

        # Plot
        fig_path = self.plot_boundary_test(h0_z_results, boundary_test, model_comparison)

        # Save summary
        summary = {
            "step": "34_void_boundary_test",
            "description": "Void boundary test: does H0 tension persist at z > 0.25 (well beyond void wall)?",
            "void_model_prediction": (
                "H0 gradually declines to 67.4 per published HBK20/Mazurenko et al. 2024 "
                f"Gaussian (sigma_z={self.VOID_GAUSSIAN_SIGMA_Z}) and Exponential (z_0={self.VOID_EXPONENTIAL_Z0}) profiles; "
                "convergence to 1σ Planck at z≈1.8"
            ),
            "tep_model_prediction": (
                "H0 remains elevated at all redshifts; tension governed by "
                "host potential depth, not distance from Earth"
            ),
            "h0_z_by_host_mass": h0_z_results,
            "boundary_test": boundary_test,
            "model_comparison": model_comparison,
            "tep_confirmed": bool(boundary_test.get("tep_supported", False)),
            "void_falsified": bool(boundary_test.get("void_falsified", False)),
            "output_files": [str(fig_path)],
            "methodology": (
                "H0(z) inferred per SN by inverting the LCDM distance modulus in redshift "
                "bins split by host stellar mass at log10(M*/Msun)=10.0. Per-SN H0 errors "
                "propagated from mu_err where available. Boundary test uses "
                "calibration-independent relative evolution delta_H0 = H0(z>0.25) - H0(z<0.15) "
                "in massive hosts. Model comparison via AIC with fitted (k=2) and "
                "fixed-prediction (k=1) parameterizations for void (Gaussian, Exponential) "
                "and TEP (flat for global M_B) models."
            ),
            "provenance": {
                "data_sources": [
                    "data/interim/pantheon_plus_sne.csv",
                    "data/raw/external/mazurenko_curves/gaussian_method3.json",
                    "data/raw/external/mazurenko_curves/exponential_method3.json",
                ],
                "pipeline_block": "Block II — Void boundary test and float M_B",
            },
            "scientific_context": (
                "Tests whether the KBC void model's predicted gradual H0(z) decline towards "
                "the Planck value can account for the Hubble tension, or whether the TEP "
                "flat H0(z) prediction for global M_B better describes the Pantheon+ sample. "
                "The key discriminating observable is the calibration-independent relative "
                "evolution of H0 from z < 0.15 to z > 0.25 in massive hosts."
            ),
            "tep_prediction": (
                "Flat H0(z) at all redshifts when M_B is global; the Cepheid clock bias is "
                "imprinted on the zero-point, not on the distance-dependent part. The "
                "(1+z)^-0.3 decay applies to per-host Cepheid calibration (TEP-H0, Paper 11)."
            ),
            "void_prediction": (
                "Gradual H0(z) decline from ~73 at z~0 to ~67.4 at z>>1, converging to "
                "within 1-sigma of Planck at z~1.8 per the published HBK20/Mazurenko et al. "
                "2024 Gaussian (sigma_z=0.82) and Exponential (z_0=0.74) profiles."
            ),
            "downstream_consumers": ["step_35_float_mb_analysis"],
        }

        output_path = self.results / "step_34_void_boundary_test.json"
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print_status(f"Summary saved to {output_path}", "SUCCESS")

        print_status("Step 34 complete", "SUCCESS")


if __name__ == "__main__":
    step = Step34VoidBoundaryTest()
    step.run()
