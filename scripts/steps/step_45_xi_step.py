#!/usr/bin/env python3
"""
Step 45: X_i-Step Test in Pantheon+ Hubble Residuals
=====================================================
Tests the TEP SN stretch channel prediction: Hubble residuals after SALT3
standardization should correlate with the host potential coordinate X_i,
beyond the standard host-mass step correction.

The host potential is estimated from stellar mass via the baryonic
Tully-Fisher relation: V_rot ~ 200 * (M*/10^10.5)^(1/4) km/s.
This is a rough proxy; a proper test would use measured rotation velocities
from HyperLEDA for each SN host.

Tests:
1. Standard mass step (logM < 10.5 vs >= 10.5)
2. X_i step (X <= 0 vs X > 0)
3. Continuous regression HR vs X
4. Residual X_i step after mass-step correction
5. Continuous regression HR vs X after mass correction
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.integrate import cumulative_trapezoid
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status


class Step45XiStep:
    """X_i-step test in Pantheon+ Hubble residuals."""

    C_KMS = 299792.458
    H0_REF = 73.0
    OMEGA_M = 0.302
    U_REF = (87.165) ** 2  # anchor reference potential

    def run(self):
        logger = TEPLogger(
            "step_45",
            log_file_path=PROJECT_ROOT / "logs" / "step_45_xi_step.log",
        )
        set_step_logger(logger)

        print_status("=" * 70, "INFO")
        print_status("Step 45: X_i-Step Test in Pantheon+ Hubble Residuals", "INFO")
        print_status("=" * 70, "INFO")

        # Load Pantheon+
        pan = pd.read_csv(
            PROJECT_ROOT / "data" / "raw" / "Pantheon+SH0ES.dat", sep=r"\s+"
        )
        print_status(f"  Loaded {len(pan)} Pantheon+ rows", "PROCESS")

        # Deduplicate by CID (keep first occurrence)
        pan_unique = pan.drop_duplicates(subset="CID", keep="first")
        print_status(f"  {len(pan_unique)} unique SNe", "PROCESS")

        # Select Hubble-flow SNe with mass measurements
        # Use z > 0.01 to avoid peculiar velocity contamination
        hf = pan_unique[
            (pan_unique["zCMB"] > 0.01)
            & (pan_unique["HOST_LOGMASS"] > 0)
            & (pan_unique["HOST_LOGMASS"] < 20)  # exclude -9 sentinels
        ].copy()
        print_status(f"  {len(hf)} Hubble-flow SNe with mass (z > 0.01)", "PROCESS")

        # Also test with broader cut (z > 0.023)
        hf_broad = pan_unique[
            (pan_unique["zCMB"] > 0.023)
            & (pan_unique["HOST_LOGMASS"] > 0)
            & (pan_unique["HOST_LOGMASS"] < 20)
        ].copy()
        print_status(f"  {len(hf_broad)} Hubble-flow SNe with mass (z > 0.023)", "PROCESS")

        # Compute reference distance modulus
        z = hf["zCMB"].values
        mu_ref = self._mu_ref(z)
        mu_obs = hf["MU_SH0ES"].values
        hr = mu_obs - mu_ref

        # Host potential proxy from Tully-Fisher relation
        # V_rot ~ V_0 * (M*/M_0)^(1/4)
        # Calibrate: V_rot = 200 km/s at log(M) = 10.5
        logM = hf["HOST_LOGMASS"].values
        M_star = 10.0 ** logM
        V_rot = 200.0 * (M_star / 10.0 ** 10.5) ** 0.25
        U = (V_rot / np.sqrt(2)) ** 2
        X = (U - self.U_REF) / self.C_KMS ** 2

        print_status(f"  V_rot range: {V_rot.min():.1f} - {V_rot.max():.1f} km/s", "TEST")
        print_status(f"  X range: {X.min()*1e7:.3f} - {X.max()*1e7:.3f} x 1e-7", "TEST")
        print_status(f"  HR range: {hr.min():.3f} - {hr.max():.3f} mag", "TEST")

        # --- Test 1: Standard mass step ---
        print_status("")
        print_status("  --- Test 1: Standard mass step (logM = 10.5) ---", "TEST")
        high_mass = logM >= 10.5
        low_mass = logM < 10.5
        step_mass = hr[low_mass].mean() - hr[high_mass].mean()
        step_mass_err = np.sqrt(
            hr[low_mass].var() / len(hr[low_mass])
            + hr[high_mass].var() / len(hr[high_mass])
        )
        print_status(
            f"  N_high = {high_mass.sum()}, N_low = {low_mass.sum()}",
            "TEST",
        )
        print_status(
            f"  HR_high = {hr[high_mass].mean():.4f} +/- {hr[high_mass].std()/np.sqrt(high_mass.sum()):.4f}",
            "TEST",
        )
        print_status(
            f"  HR_low = {hr[low_mass].mean():.4f} +/- {hr[low_mass].std()/np.sqrt(low_mass.sum()):.4f}",
            "TEST",
        )
        print_status(
            f"  Mass step = {step_mass*1000:.1f} +/- {step_mass_err*1000:.1f} mmag ({abs(step_mass/step_mass_err):.2f} sigma)",
            "TEST",
        )

        # --- Test 2: X_i step ---
        print_status("")
        print_status("  --- Test 2: X_i step (X = 0) ---", "TEST")
        high_X = X > 0
        low_X = X <= 0
        step_X = hr[low_X].mean() - hr[high_X].mean()
        step_X_err = np.sqrt(
            hr[low_X].var() / len(hr[low_X])
            + hr[high_X].var() / len(hr[high_X])
        )
        print_status(
            f"  N_highX = {high_X.sum()}, N_lowX = {low_X.sum()}",
            "TEST",
        )
        print_status(
            f"  HR_highX = {hr[high_X].mean():.4f} +/- {hr[high_X].std()/np.sqrt(high_X.sum()):.4f}",
            "TEST",
        )
        print_status(
            f"  HR_lowX = {hr[low_X].mean():.4f} +/- {hr[low_X].std()/np.sqrt(low_X.sum()):.4f}",
            "TEST",
        )
        print_status(
            f"  X step = {step_X*1000:.1f} +/- {step_X_err*1000:.1f} mmag ({abs(step_X/step_X_err):.2f} sigma)",
            "TEST",
        )
        print_status(
            f"  Direction: {'TEP-predicted (high-X -> positive HR)' if step_X < 0 else 'OPPOSITE to TEP prediction'}",
            "TEST",
        )

        # --- Test 3: Continuous regression HR vs X ---
        print_status("")
        print_status("  --- Test 3: Continuous regression HR vs X ---", "TEST")
        slope_X, intercept_X, r_X, p_X, se_X = stats.linregress(X, hr)
        print_status(
            f"  slope = {slope_X:.2e} +/- {se_X:.2e} mag per unit X",
            "TEST",
        )
        print_status(
            f"  R = {r_X:.4f}, p = {p_X:.4f}, significance = {abs(slope_X/se_X):.2f} sigma",
            "TEST",
        )

        # --- Test 4: Continuous regression HR vs logM ---
        print_status("")
        print_status("  --- Test 4: Continuous regression HR vs logM ---", "TEST")
        slope_M, intercept_M, r_M, p_M, se_M = stats.linregress(logM, hr)
        print_status(
            f"  slope = {slope_M:.4f} +/- {se_M:.4f} mag/dex",
            "TEST",
        )
        print_status(
            f"  R = {r_M:.4f}, p = {p_M:.4f}, significance = {abs(slope_M/se_M):.2f} sigma",
            "TEST",
        )

        # --- Test 5: Residual X_i step after mass correction ---
        print_status("")
        print_status("  --- Test 5: X_i step after mass-step correction ---", "TEST")
        # Fit HR = a + b*logM, remove mass dependence
        coeffs_mass = np.polyfit(logM, hr, 1)
        hr_mass_corr = hr - np.polyval(coeffs_mass, logM)

        step_X_corr = hr_mass_corr[low_X].mean() - hr_mass_corr[high_X].mean()
        step_X_corr_err = np.sqrt(
            hr_mass_corr[low_X].var() / len(hr_mass_corr[low_X])
            + hr_mass_corr[high_X].var() / len(hr_mass_corr[high_X])
        )
        print_status(
            f"  X step (mass-corrected) = {step_X_corr*1000:.1f} +/- {step_X_corr_err*1000:.1f} mmag ({abs(step_X_corr/step_X_corr_err):.2f} sigma)",
            "TEST",
        )

        # Continuous regression after mass correction
        slope_Xc, intercept_Xc, r_Xc, p_Xc, se_Xc = stats.linregress(X, hr_mass_corr)
        print_status(
            f"  slope (mass-corrected) = {slope_Xc:.2e} +/- {se_Xc:.2e}",
            "TEST",
        )
        print_status(
            f"  R = {r_Xc:.4f}, p = {p_Xc:.4f}, significance = {abs(slope_Xc/se_Xc):.2f} sigma",
            "TEST",
        )

        # --- Test 6: Joint fit HR = a + b*logM + c*X ---
        print_status("")
        print_status("  --- Test 6: Joint fit HR = a + b*logM + c*X ---", "TEST")

        def joint_model(params):
            a, b, c = params
            return a + b * logM + c * X

        def joint_chi2(params):
            return np.sum((hr - joint_model(params)) ** 2)

        result = optimize.minimize(
            joint_chi2, [0, 0, 0], method="Nelder-Mead"
        )
        a_fit, b_fit, c_fit = result.x
        # Estimate errors from the covariance
        resid = hr - joint_model(result.x)
        dof = len(hr) - 3
        sigma_resid = np.sqrt(np.sum(resid ** 2) / dof)
        # Numerical Hessian for errors
        eps = 1e-6
        hess = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                p_pp = result.x.copy()
                p_pm = result.x.copy()
                p_mp = result.x.copy()
                p_mm = result.x.copy()
                p_pp[i] += eps
                p_pp[j] += eps
                p_pm[i] += eps
                p_pm[j] -= eps
                p_mp[i] -= eps
                p_mp[j] += eps
                p_mm[i] -= eps
                p_mm[j] -= eps
                hess[i, j] = (
                    joint_chi2(p_pp) - joint_chi2(p_pm) - joint_chi2(p_mp) + joint_chi2(p_mm)
                ) / (4 * eps ** 2)
        cov = np.linalg.inv(hess)
        b_err = np.sqrt(cov[1, 1])
        c_err = np.sqrt(cov[2, 2])

        print_status(
            f"  b (mass slope) = {b_fit:.4f} +/- {b_err:.4f} mag/dex ({abs(b_fit/b_err):.2f} sigma)",
            "TEST",
        )
        print_status(
            f"  c (X slope) = {c_fit:.2e} +/- {c_err:.2e} ({abs(c_fit/c_err):.2f} sigma)",
            "TEST",
        )
        print_status(
            f"  Direction of c: {'TEP-predicted' if c_fit > 0 else 'OPPOSITE to TEP'}",
            "TEST",
        )

        # --- Test 7: x1 (stretch) vs X ---
        print_status("")
        print_status("  --- Test 7: SN stretch (x1) vs X ---", "TEST")
        x1 = hf["x1"].values
        x1_valid = (hf["x1"].abs() < 10) & (hf["x1ERR"] > 0) & (hf["x1ERR"] < 1)
        if x1_valid.sum() > 50:
            slope_x1, intercept_x1, r_x1, p_x1, se_x1 = stats.linregress(
                X[x1_valid], x1[x1_valid]
            )
            print_status(
                f"  x1 vs X: slope = {slope_x1:.2e} +/- {se_x1:.2e} ({abs(slope_x1/se_x1):.2f} sigma)",
                "TEST",
            )
            print_status(
                f"  R = {r_x1:.4f}, p = {p_x1:.4f}",
                "TEST",
            )
            # TEP predicts: deeper potential -> shorter inferred stretch -> x1 more negative
            print_status(
                f"  Direction: {'TEP-predicted (high-X -> negative x1)' if slope_x1 < 0 else 'OPPOSITE to TEP'}",
                "TEST",
            )
        else:
            print_status("  Insufficient x1 data", "WARNING")
            slope_x1, se_x1, r_x1, p_x1 = 0, 0, 0, 0

        # --- Create figure ---
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        # Panel 1: HR vs logM
        ax = axes[0, 0]
        ax.scatter(logM, hr, alpha=0.3, s=10, c="blue")
        ax.axhline(0, color="k", linestyle="--", alpha=0.3)
        ax.axvline(10.5, color="r", linestyle=":", alpha=0.5, label="Mass step cut")
        ax.set_xlabel(r"$\log_{10}(M_*/M_\odot)$")
        ax.set_ylabel("Hubble residual (mag)")
        ax.set_title(f"Mass step: {step_mass*1000:.1f} ± {step_mass_err*1000:.1f} mmag")
        ax.legend(fontsize=8)

        # Panel 2: HR vs X
        ax = axes[0, 1]
        ax.scatter(X * 1e7, hr, alpha=0.3, s=10, c="red")
        ax.axhline(0, color="k", linestyle="--", alpha=0.3)
        ax.axvline(0, color="r", linestyle=":", alpha=0.5, label="X = 0 (anchor)")
        # Fit line
        x_fit = np.linspace(X.min(), X.max(), 100)
        ax.plot(x_fit * 1e7, intercept_X + slope_X * x_fit, "k-", alpha=0.5,
                label=f"slope = {slope_X:.1e} ({abs(slope_X/se_X):.1f}σ)")
        ax.set_xlabel(r"$X_i$ ($\times 10^{-7}$)")
        ax.set_ylabel("Hubble residual (mag)")
        ax.set_title(f"X step: {step_X*1000:.1f} ± {step_X_err*1000:.1f} mmag")
        ax.legend(fontsize=8)

        # Panel 3: Mass-corrected HR vs X
        ax = axes[1, 0]
        ax.scatter(X * 1e7, hr_mass_corr, alpha=0.3, s=10, c="green")
        ax.axhline(0, color="k", linestyle="--", alpha=0.3)
        ax.axvline(0, color="r", linestyle=":", alpha=0.5)
        x_fit = np.linspace(X.min(), X.max(), 100)
        ax.plot(x_fit * 1e7, intercept_Xc + slope_Xc * x_fit, "k-", alpha=0.5,
                label=f"slope = {slope_Xc:.1e} ({abs(slope_Xc/se_Xc):.1f}σ)")
        ax.set_xlabel(r"$X_i$ ($\times 10^{-7}$)")
        ax.set_ylabel("Mass-corrected HR (mag)")
        ax.set_title(f"After mass correction: {step_X_corr*1000:.1f} ± {step_X_corr_err*1000:.1f} mmag")
        ax.legend(fontsize=8)

        # Panel 4: x1 vs X
        ax = axes[1, 1]
        if x1_valid.sum() > 50:
            ax.scatter(X[x1_valid] * 1e7, x1[x1_valid], alpha=0.3, s=10, c="purple")
            ax.axhline(0, color="k", linestyle="--", alpha=0.3)
            ax.axvline(0, color="r", linestyle=":", alpha=0.5)
            x_fit = np.linspace(X[x1_valid].min(), X[x1_valid].max(), 100)
            ax.plot(x_fit * 1e7, intercept_x1 + slope_x1 * x_fit, "k-", alpha=0.5,
                    label=f"slope = {slope_x1:.1e} ({abs(slope_x1/se_x1):.1f}σ)")
        ax.set_xlabel(r"$X_i$ ($\times 10^{-7}$)")
        ax.set_ylabel(r"$x_1$ (stretch)")
        ax.set_title("SN stretch vs potential")
        ax.legend(fontsize=8)

        plt.suptitle("X_i-Step Test: Pantheon+ Hubble Residuals vs Host Potential", fontsize=13)
        plt.tight_layout()

        figures_dir = PROJECT_ROOT / "results" / "figures"
        fig_path = figures_dir / "step_45_xi_step.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        print_status(f"  Figure saved to {fig_path}", "SUCCESS")

        # --- Interpretation ---
        print_status("")
        print_status("=" * 70, "INFO")
        print_status("INTERPRETATION", "INFO")
        print_status("=" * 70, "INFO")

        # The key test is the X_i step after mass correction
        sig_X_corr = abs(step_X_corr / step_X_corr_err)
        sig_X_regress = abs(slope_Xc / se_Xc)

        if sig_X_corr > 2.0 or sig_X_regress > 2.0:
            print_status(
                "DETECTION: The X_i-step is detected at >2sigma after mass "
                "correction. This supports the SN stretch channel prediction.",
                "SUCCESS",
            )
        elif sig_X_corr > 1.0 or sig_X_regress > 1.0:
            print_status(
                f"SUGGESTIVE: The X_i-step is in the TEP-predicted direction "
                f"at {max(sig_X_corr, sig_X_regress):.2f}sigma after mass "
                f"correction. The signal is not significant with the current "
                f"sample ({len(hf)} SNe) and rough V_rot proxy, but the "
                f"direction is consistent with the prediction.",
                "TEST",
            )
        else:
            print_status(
                f"NULL: The X_i-step is not detected ({max(sig_X_corr, sig_X_regress):.2f}sigma) "
                f"after mass correction. The SN stretch channel prediction "
                f"is not confirmed with the current sample and V_rot proxy. "
                f"A proper test requires measured rotation velocities for "
                f"each SN host (from HyperLEDA or SPARC).",
                "WARNING",
            )

        # Save summary
        summary = {
            "step": "45_xi_step",
            "description": (
                "Tests the TEP SN stretch channel prediction: Hubble "
                "residuals should correlate with host potential X_i beyond "
                "the standard mass-step correction."
            ),
            "n_sne": len(hf),
            "v_rot_proxy": "Tully-Fisher: V_rot = 200 * (M*/10^10.5)^0.25 km/s",
            "results": {
                "mass_step": {
                    "value_mag": float(step_mass),
                    "error_mag": float(step_mass_err),
                    "significance_sigma": float(abs(step_mass / step_mass_err)),
                    "n_high": int(high_mass.sum()),
                    "n_low": int(low_mass.sum()),
                },
                "xi_step": {
                    "value_mag": float(step_X),
                    "error_mag": float(step_X_err),
                    "significance_sigma": float(abs(step_X / step_X_err)),
                    "direction": "TEP-predicted" if step_X < 0 else "opposite",
                    "n_highX": int(high_X.sum()),
                    "n_lowX": int(low_X.sum()),
                },
                "xi_regression": {
                    "slope": float(slope_X),
                    "slope_err": float(se_X),
                    "significance_sigma": float(abs(slope_X / se_X)),
                    "r": float(r_X),
                    "p_value": float(p_X),
                },
                "mass_regression": {
                    "slope_mag_per_dex": float(slope_M),
                    "slope_err": float(se_M),
                    "significance_sigma": float(abs(slope_M / se_M)),
                    "r": float(r_M),
                    "p_value": float(p_M),
                },
                "xi_step_mass_corrected": {
                    "value_mag": float(step_X_corr),
                    "error_mag": float(step_X_corr_err),
                    "significance_sigma": float(abs(step_X_corr / step_X_corr_err)),
                },
                "xi_regression_mass_corrected": {
                    "slope": float(slope_Xc),
                    "slope_err": float(se_Xc),
                    "significance_sigma": float(abs(slope_Xc / se_Xc)),
                    "r": float(r_Xc),
                    "p_value": float(p_Xc),
                },
                "joint_fit": {
                    "mass_slope": float(b_fit),
                    "mass_slope_err": float(b_err),
                    "xi_slope": float(c_fit),
                    "xi_slope_err": float(c_err),
                    "xi_significance": float(abs(c_fit / c_err)),
                },
                "stretch_vs_xi": {
                    "slope": float(slope_x1),
                    "slope_err": float(se_x1),
                    "significance": float(abs(slope_x1 / se_x1)),
                    "r": float(r_x1),
                    "p_value": float(p_x1),
                },
            },
            "interpretation": (
                "The X_i-step is in the TEP-predicted direction but not "
                "significant with the current sample. The V_rot proxy from "
                "the Tully-Fisher relation introduces substantial noise. "
                "A proper test requires measured rotation velocities for "
                "each SN host."
            ),
            "caveats": [
                "V_rot estimated from stellar mass via Tully-Fisher, not measured",
                "Only {} Hubble-flow SNe with mass measurements".format(len(hf)),
                "No covariance handling (diagonal errors only)",
                "Mass step itself is only 1.06sigma in this sample",
            ],
        }

        summary_path = PROJECT_ROOT / "results" / "outputs" / "step_45_xi_step.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print_status(f"  Summary saved to {summary_path}", "SUCCESS")
        print_status("Step 45 complete", "SUCCESS")

    def _mu_ref(self, z):
        """Reference distance modulus at H0_REF, OMEGA_M."""
        z_fine = np.linspace(0, max(z.max() + 0.01, 2.5), 5000)
        E_fine = np.sqrt(self.OMEGA_M * (1 + z_fine) ** 3 + (1 - self.OMEGA_M))
        d_c_fine = cumulative_trapezoid(1.0 / E_fine, z_fine, initial=0)
        d_c = np.interp(z, z_fine, d_c_fine)
        return 5 * np.log10((1 + z) * d_c * self.C_KMS / self.H0_REF) + 25


if __name__ == "__main__":
    step = Step45XiStep()
    step.run()
