#!/usr/bin/env python3
"""
Step 33: Host-Mass z > 0.3 Survey Design
=========================================
Forward-looking survey design for SNe Ia in massive elliptical host
galaxies at z > 0.3, designed to test the TEP prediction that the
H0 inflation signal persists for massive hosts at high redshift.

Key Tasks:
1. Compute expected Delta_mu (distance modulus shift) for massive hosts
   at z > 0.3 under the TEP framework
2. Calculate the required sample size N for a 3-sigma detection
3. Estimate survey parameters: exposure times, field coverage, cadence
4. Compare with existing surveys (DES, LSST, ZTF) for feasibility

TEP Prediction:
    If TEP is correct, SNe Ia in massive elliptical hosts at z > 0.3
    should show a distance modulus offset Delta_mu ~ kappa * <X_i>(z)
    relative to the LambdaCDM prediction.  This signal should be
    detectable with a targeted survey of ~50-100 SNe in massive hosts.

Outputs:
    results/outputs/step_33_host_mass_z03_survey.json
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import integrate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status


class Step33HostMassZ03Survey:
    """Step 33: Survey design for SNe Ia in massive ellipticals at z > 0.3."""

    # TEP parameters
    KAPPA_CEP = 0.040  # TEP shear coupling constant
    H0_CMB = 67.4  # km/s/Mpc
    OMEGA_M = 0.302
    C_KMS = 299792.458

    # Survey design parameters
    Z_MIN = 0.30
    Z_MAX = 0.60
    MASSIVE_HOST_THRESHOLD = 10.0  # log10(M*/Msun) for "massive" — standard across all TEP-VOID steps
    SN_IA_INTRINSIC_SCATTER = 0.12  # mag, intrinsic scatter of SN Ia
    SN_IA_PHOTOMETRIC_ERR = 0.05  # mag, typical photometric error at z~0.4
    HOST_MASS_ERR = 0.20  # dex, typical stellar mass uncertainty

    # SN Ia rate in massive ellipticals (per 10^4 Mpc^3 per year)
    SN_RATE_MASSIVE = 0.04

    # ------------------------------------------------------------------
    # Cosmological utilities
    # ------------------------------------------------------------------
    def _E(self, z):
        """Dimensionless Hubble parameter E(z) = H(z)/H0."""
        return np.sqrt(self.OMEGA_M * (1 + z) ** 3 + (1 - self.OMEGA_M))

    def _comoving_distance(self, z, h0=None):
        """
        Comoving distance D_C(z) in Mpc using proper LCDM integral.

        D_C(z) = c/H0 * integral_0^z dz'/E(z')

        Valid at all redshifts (unlike the linear cz/H0 approximation).
        """
        if h0 is None:
            h0 = self.H0_CMB
        integral, _ = integrate.quad(lambda zp: 1.0 / self._E(zp), 0, z)
        return self.C_KMS * integral / h0

    def __init__(self):
        self.root = PROJECT_ROOT
        self.data_interim = self.root / "data" / "interim"
        self.data_processed = self.root / "data" / "processed"
        self.results = self.root / "results" / "outputs"
        self.logs = self.root / "logs"

        for d in [self.data_interim, self.data_processed, self.results, self.logs]:
            d.mkdir(parents=True, exist_ok=True)

        self.logger = TEPLogger(
            "step_33", log_file_path=self.logs / "step_33_host_mass_z03_survey.log"
        )
        set_step_logger(self.logger)

    def compute_expected_delta_mu(self):
        """
        Compute the expected distance modulus shift Delta_mu for massive
        hosts at z > 0.3 under the TEP framework.

        Delta_mu = 5/ln(10) * kappa * <X_i>(z) * massive_factor

        where <X_i>(z) = (1+z)^-0.3 is the temporal shear expectation value
        and massive_factor = 1.0 for massive elliptical hosts (the full
        TEP coupling applies to galaxies with deep gravitational potentials).
        """
        print_status("Computing expected Delta_mu for massive hosts at z > 0.3...", "PROCESS")

        z_values = np.linspace(self.Z_MIN, self.Z_MAX, 50)

        # TEP temporal shear expectation: <X_i>(z) = 1/(1+z)^0.3
        x_i = 1.0 / (1.0 + z_values) ** 0.3

        # For massive elliptical hosts, the full TEP coupling applies
        massive_factor = 1.0

        # Distance modulus shift: Delta_mu = 5/ln(10) * kappa * <X_i>(z) * massive_factor
        delta_mu = (5.0 / np.log(10.0)) * self.KAPPA_CEP * x_i * massive_factor

        # At z = 0.3
        z_03 = 0.30
        x_i_03 = 1.0 / (1.0 + z_03) ** 0.3
        delta_mu_03 = (5.0 / np.log(10.0)) * self.KAPPA_CEP * x_i_03 * massive_factor

        # At z = 0.4 (typical survey center)
        z_04 = 0.40
        x_i_04 = 1.0 / (1.0 + z_04) ** 0.3
        delta_mu_04 = (5.0 / np.log(10.0)) * self.KAPPA_CEP * x_i_04 * massive_factor

        print_status(f"  Delta_mu at z=0.30: {delta_mu_03:.4f} mag", "TEST")
        print_status(f"  Delta_mu at z=0.40: {delta_mu_04:.4f} mag", "TEST")
        print_status(f"  Delta_mu at z=0.60: {delta_mu[-1]:.4f} mag", "TEST")

        return {
            "z_values": z_values.tolist(),
            "delta_mu_values": delta_mu.tolist(),
            "delta_mu_z03": float(delta_mu_03),
            "delta_mu_z04": float(delta_mu_04),
            "delta_mu_z06": float(delta_mu[-1]),
            "mean_delta_mu": float(np.mean(delta_mu)),
        }

    def compute_required_sample_size(self, delta_mu_info):
        """Calculate required sample size N for 3-sigma detection."""
        print_status("Computing required sample size for 3-sigma detection...", "PROCESS")

        delta_mu = delta_mu_info["mean_delta_mu"]

        # Total error per SN: sqrt(intrinsic^2 + photometric^2)
        sigma_per_sn = np.sqrt(self.SN_IA_INTRINSIC_SCATTER**2 + self.SN_IA_PHOTOMETRIC_ERR**2)
        print_status(f"  Per-SN error: sigma = {sigma_per_sn:.3f} mag", "PROCESS")

        # For 3-sigma detection: N = (3 * sigma / delta_mu)^2
        n_3sigma = int(np.ceil((3.0 * sigma_per_sn / delta_mu) ** 2))
        print_status(f"  Required N for 3-sigma: {n_3sigma} SNe", "TEST")

        # For 5-sigma detection
        n_5sigma = int(np.ceil((5.0 * sigma_per_sn / delta_mu) ** 2))
        print_status(f"  Required N for 5-sigma: {n_5sigma} SNe", "TEST")

        # If we can reduce intrinsic scatter with a more homogeneous sample
        sigma_reduced = 0.10  # with better host matching
        n_3sigma_reduced = int(np.ceil((3.0 * sigma_reduced / delta_mu) ** 2))
        print_status(f"  Required N (reduced scatter) for 3-sigma: {n_3sigma_reduced} SNe", "TEST")

        return {
            "delta_mu_signal": float(delta_mu),
            "sigma_per_sn": float(sigma_per_sn),
            "sigma_reduced": float(sigma_reduced),
            "n_3sigma": n_3sigma,
            "n_5sigma": n_5sigma,
            "n_3sigma_reduced_scatter": n_3sigma_reduced,
            "snr_per_sn": float(delta_mu / sigma_per_sn),
        }

    def estimate_survey_parameters(self, n_required):
        """Estimate survey parameters for feasibility assessment."""
        print_status("Estimating survey parameters...", "PROCESS")

        # Survey volume between z=0.3 and z=0.6
        # V = 4/3 * pi * (d_C(z_max)^3 - d_C(z_min)^3)
        # Use proper LCDM comoving distance (not linear cz/H0, which is
        # only valid at z << 0.1 and gives ~15% error at z=0.5)
        d_min = self._comoving_distance(self.Z_MIN)  # Mpc
        d_max = self._comoving_distance(self.Z_MAX)  # Mpc
        volume = (4.0 / 3.0) * np.pi * (d_max**3 - d_min**3)  # Mpc^3
        print_status(f"  Survey volume (z=0.3-0.6): {volume:.2e} Mpc^3", "PROCESS")

        # Fraction of volume in massive ellipticals (~15%)
        massive_frac = 0.15
        volume_massive = volume * massive_frac

        # Expected SNe per year in massive ellipticals in this volume
        sn_per_year = self.SN_RATE_MASSIVE * volume_massive / 1e4
        print_status(f"  Expected SNe/yr in massive hosts: {sn_per_year:.1f}", "PROCESS")

        # Survey duration needed
        survey_duration = int(np.ceil(n_required["n_3sigma"] / sn_per_year))
        print_status(f"  Estimated survey duration: {survey_duration} years", "TEST")

        # With a deeper survey covering more area
        # LSST will discover ~10x more SNe in this range
        lsst_multiplier = 10.0
        sn_per_year_lsst = sn_per_year * lsst_multiplier
        survey_duration_lsst = max(1, int(np.ceil(n_required["n_3sigma"] / sn_per_year_lsst)))
        print_status(f"  With LSST: ~{sn_per_year_lsst:.0f} SNe/yr, ~{survey_duration_lsst} years", "TEST")

        return {
            "z_min": self.Z_MIN,
            "z_max": self.Z_MAX,
            "survey_volume_mpc3": float(volume),
            "massive_fraction": float(massive_frac),
            "sn_per_year_allsky": float(sn_per_year),
            "survey_duration_years": survey_duration,
            "lsst_sn_per_year": float(sn_per_year_lsst),
            "lsst_survey_duration_years": survey_duration_lsst,
            "massive_host_threshold_logmsun": self.MASSIVE_HOST_THRESHOLD,
        }

    def assess_feasibility(self, survey_params, n_required):
        """Assess feasibility of the survey design."""
        print_status("Assessing survey feasibility...", "PROCESS")

        # Check if LSST can deliver the required sample
        lsst_feasible = survey_params["lsst_sn_per_year"] * 10 >= n_required["n_3sigma"]  # 10-year LSST
        print_status(f"  LSST 10-year survey feasible: {lsst_feasible}", "TEST")

        # Required photometric precision
        required_precision = n_required["delta_mu_signal"] / 3.0  # for 3-sigma per SN
        print_status(f"  Required photometric precision: {required_precision:.3f} mag", "PROCESS")

        # Can current instruments achieve this?
        precision_achievable = self.SN_IA_PHOTOMETRIC_ERR < required_precision
        print_status(f"  Precision achievable with current instruments: {precision_achievable}", "TEST")

        return {
            "lsst_feasible": bool(lsst_feasible),
            "required_photometric_precision_mag": float(required_precision),
            "precision_achievable": bool(precision_achievable),
            "feasibility_assessment": "feasible" if (lsst_feasible and precision_achievable) else "challenging",
            "key_challenge": "Host mass classification at z > 0.3 requires deep imaging or spectroscopy",
        }

    def run(self):
        """Execute the full step."""
        print_status("Step 33: Host-Mass z > 0.3 Survey Design", "TITLE")

        print_status(
            "This step addresses whether the TEP-predicted H0 inflation "
            "signal persists for SNe Ia in massive elliptical hosts at "
            "z > 0.3, where the temporal shear expectation <X_i>(z) = "
            "(1+z)^-0.3 is reduced relative to the local universe. The "
            "discriminating observable is the distance modulus offset "
            "Delta_mu = (5/ln10) * kappa * <X_i>(z), which TEP predicts "
            "to be non-zero and decreasing with redshift, while the "
            "LambdaCDM null hypothesis predicts zero offset at all "
            "redshifts. The survey design quantifies the sample size, "
            "comoving volume, and observing time required for a "
            "statistically significant detection of this signal.",
            "INFO",
        )

        # Compute expected Delta_mu
        print_status(
            "Methodology: Delta_mu is computed as (5/ln10) * kappa_CEP * "
            f"<X_i>(z) * massive_factor with kappa_CEP = {self.KAPPA_CEP} "
            f"and <X_i>(z) = (1+z)^-0.3, evaluated over z = {self.Z_MIN} "
            f"to {self.Z_MAX}. A massive_factor of 1.0 is applied for "
            "elliptical hosts with deep gravitational potentials, "
            "representing the full TEP shear coupling.",
            "PROCESS",
        )
        delta_mu_info = self.compute_expected_delta_mu()

        # Compute required sample size
        print_status(
            "Methodology: Required sample size N is computed from "
            "N = (n_sigma * sigma / Delta_mu)^2, where sigma combines "
            f"intrinsic SN Ia scatter ({self.SN_IA_INTRINSIC_SCATTER} mag) "
            f"and photometric error ({self.SN_IA_PHOTOMETRIC_ERR} mag) in "
            "quadrature. Both 3-sigma and 5-sigma thresholds are "
            "evaluated, with a reduced-scatter scenario for improved "
            "host matching.",
            "PROCESS",
        )
        n_required = self.compute_required_sample_size(delta_mu_info)

        # Estimate survey parameters
        print_status(
            "Methodology: Survey volume is computed from the proper LCDM "
            f"comoving distance integral between z = {self.Z_MIN} and "
            f"z = {self.Z_MAX}, with a massive elliptical fraction of "
            f"15%. The SN Ia rate in massive hosts "
            f"({self.SN_RATE_MASSIVE} per 10^4 Mpc^3 per yr) yields the "
            "expected event count, scaled by an LSST detection multiplier "
            "of 10x.",
            "PROCESS",
        )
        survey_params = self.estimate_survey_parameters(n_required)

        # Assess feasibility
        feasibility = self.assess_feasibility(survey_params, n_required)

        print_status(
            f"Interpretation: The expected mean Delta_mu of "
            f"{delta_mu_info['mean_delta_mu']:.4f} mag requires "
            f"approximately {n_required['n_3sigma']} SNe in massive hosts "
            f"for a 3-sigma detection. Under TEP, this signal arises from "
            f"the temporal shear coupling and should be detectable with "
            f"LSST in approximately "
            f"{survey_params['lsst_survey_duration_years']} years. Under "
            f"the LambdaCDM null hypothesis, no redshift-dependent offset "
            f"is predicted, and the measured Delta_mu would be consistent "
            f"with zero within uncertainty.",
            "SUCCESS",
        )

        # Summary
        summary = {
            "step": "33_host_mass_z03_survey",
            "description": "Forward-looking survey design for SNe Ia in massive ellipticals at z > 0.3",
            "tep_parameters": {
                "kappa_ceph": self.KAPPA_CEP,
                "h0_cmb": self.H0_CMB,
            },
            "expected_delta_mu": delta_mu_info,
            "required_sample_size": n_required,
            "survey_parameters": survey_params,
            "feasibility": feasibility,
            "tep_prediction": "Massive hosts at z>0.3 should show Delta_mu ~ 0.08 mag offset",
            "key_finding": f"Need ~{n_required['n_3sigma']} SNe in massive hosts for 3-sigma detection",
            "output_files": [
                str(self.results / "step_33_host_mass_z03_survey.json"),
            ],
            "methodology": (
                "Delta_mu computed as (5/ln10) * kappa_CEP * <X_i>(z) * "
                "massive_factor with kappa_CEP = 0.040, <X_i>(z) = (1+z)^-0.3, "
                "and massive_factor = 1.0 for massive elliptical hosts. "
                "Required sample size N = (n_sigma * sigma / Delta_mu)^2 with "
                "sigma = sqrt(intrinsic^2 + photometric^2). Survey volume "
                "from proper LCDM comoving distance integral, with 15% "
                "massive elliptical fraction and SN Ia rate of 0.04 per "
                "10^4 Mpc^3 per yr."
            ),
            "provenance": {
                "data_sources": [
                    "TEP framework parameters (kappa_CEP, H0_CMB, Omega_m)",
                    "SN Ia rate in massive ellipticals (0.04 per 10^4 Mpc^3 per yr)",
                    "LSST survey capabilities (10x detection multiplier)",
                ],
                "pipeline_block": "Host mass and Xi regression",
            },
            "scientific_context": (
                "This step addresses whether the TEP-predicted H0 inflation "
                "signal persists for SNe Ia in massive elliptical hosts at "
                "z > 0.3. The temporal shear expectation <X_i>(z) = (1+z)^-0.3 "
                "decreases with redshift, producing a correspondingly smaller "
                "Delta_mu. The discriminating observable is the distance "
                "modulus offset, which TEP predicts to be non-zero and "
                "redshift-dependent, while LambdaCDM predicts zero offset "
                "at all redshifts."
            ),
            "void_prediction": (
                "No redshift-dependent distance modulus offset is predicted "
                "for massive hosts under the null hypothesis; any observed "
                "Delta_mu should be consistent with zero within measurement "
                "uncertainty."
            ),
            "downstream_consumers": [],
        }

        summary_path = self.results / "step_33_host_mass_z03_survey.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print_status(f"Summary saved to {summary_path}", "SUCCESS")

        print_status("Step 33 complete", "SUCCESS")


if __name__ == "__main__":
    step = Step33HostMassZ03Survey()
    step.run()
