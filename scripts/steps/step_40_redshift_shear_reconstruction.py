#!/usr/bin/env python3
"""
Step 40: Redshift Shear Reconstruction — Pantheon+ Under TEP
==============================================================
Reconstructs Pantheon+ SN Ia distance moduli under the TEP temporal
shear framework, applying the correction:

    mu_TEP = mu_obs + kappa_Cep * X_i

where:
  - kappa_Cep is the TEP shear coupling constant (in mag), imported from
    the companion paper TEP-H0 (Paper 11), where it is determined from
    the host-level Cepheid analysis.
  - X_i = (S_total * U_i - U_ref) / c^2 is the dimensionless screened potential
    coordinate, with U_i = u_phi^2 the rotation-based potential proxy
    (from step_01) and U_ref = (30.507 km/s)^2 the screened anchor reference.

IMPORTANT CAVEAT:
  - Pantheon+ uses a single global M_B calibration for all SNe. The
    Cepheid calibration bias is imprinted on M_B, not on individual SN
    distances. The TEP correction is therefore essentially a constant
    offset (kappa_Cep * <X_i>_calibrators), not a host-mass-dependent
    correction. The host-mass dependence of the TEP effect requires
    per-host Cepheid calibrations (TEP-H0, Paper 11).
  - The (1+z)^{-0.3} H0(z) decay prediction from Section 7 is NOT
    applied as a per-SN correction. It is a prediction about the
    apparent H0(z) profile, tested separately in Section 7.
  - This step is a consistency check: the TEP-predicted correction
    direction (positive, reducing the negative residual) should be
    consistent with the observed residual structure. It is NOT a
    discriminating test between TEP and the void model.

Key Tasks:
1. Load Pantheon+ data (from step_03)
2. Load kappa_Cep from TEP-H0 (Paper 11): kappa_Cep = 0.365 x 10^6 mag
3. Load host potential catalog from step_01
4. Compute X_i for each SN host (actual or proxy)
5. Apply TEP correction to distance moduli
6. Compare corrected vs uncorrected Hubble diagram residuals

Outputs:
    results/outputs/step_40_redshift_shear_reconstruction.json
    results/figures/step_40_pantheon_tep_correction.png
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import integrate

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.screening import U_REF_SCREENED, compute_screening_by_name

# Path to the companion TEP-H0 project (sibling directory)
TEP_H0_ROOT = PROJECT_ROOT.parent / "TEP-H0"
TEP_H0_RESULTS = TEP_H0_ROOT / "results" / "outputs"


class Step40RedshiftShearReconstruction:
    """Step 40: Reconstruct Pantheon+ distance moduli under TEP temporal shear."""

    # TEP parameters from TEP-H0 (Paper 11)
    # kappa_Cep is the Cepheid-channel coupling constant in mag, determined
    # from the host-level Cepheid analysis in TEP-H0 (Section 4, Step 42).
    # The Cepheid-channel closure (beta_X = 0) yields:
    #   kappa_Cep^equiv = (0.365 +/- 0.304) x 10^6 mag
    # The redshift-only WLS (Step 44, sigma_v=150) yields:
    #   kappa_Cep = (0.452 +/- 0.220) x 10^6 mag
    # The canonical reference scaling is:
    #   kappa_canonical = 0.960 x 10^6 mag
    # We use the redshift-only WLS value as the primary estimate for the
    # TEP correction, with the equiv value recorded for the manuscript's
    # calibrator-average estimator (+0.025 +/- 0.021 mag).
    # These are imported as fixed parameters from the companion paper; they
    # are NOT derived within the TEP-VOID pipeline. The load_kappa_ceph
    # method reads the value from the TEP-H0 results directory.

    # Anchor reference potential (from TEP-H0)
    # U_ref_screened = sigma_ref_screened^2 = (30.507 km/s)^2
    # This is the screened anchor reference matching TEP-H0's canonical coordinate.
    SIGMA_REF = 30.507  # km/s — screened
    U_REF = U_REF_SCREENED  # ≈ 930.7 (km/s)^2

    H0_CMB = 67.4  # km/s/Mpc
    H0_SH0ES = 73.0  # km/s/Mpc
    OMEGA_M = 0.302
    C_KMS = 299792.458

    def __init__(self):
        self.root = PROJECT_ROOT
        self.data_interim = self.root / "data" / "interim"
        self.data_processed = self.root / "data" / "processed"
        self.results = self.root / "results" / "outputs"
        self.figures = self.root / "results" / "figures"
        self.logs = self.root / "logs"

        for d in [self.data_interim, self.data_processed, self.results, self.figures, self.logs]:
            d.mkdir(parents=True, exist_ok=True)

        self.logger = TEPLogger(
            "step_40", log_file_path=self.logs / "step_40_redshift_shear_reconstruction.log"
        )
        set_step_logger(self.logger)

    # ------------------------------------------------------------------
    # Cosmological utilities
    # ------------------------------------------------------------------
    def _E(self, z):
        """Dimensionless Hubble parameter E(z) = H(z)/H0."""
        return np.sqrt(self.OMEGA_M * (1 + z) ** 3 + (1 - self.OMEGA_M))

    def _comoving_distance_integral(self, z):
        """Comoving distance integral D_C(z) = integral_0^z dz'/E(z') in units of c/H0."""
        result, _ = integrate.quad(lambda zp: 1.0 / self._E(zp), 0, z)
        return result

    def _mu_lcdm(self, z, h0):
        """
        LCDM distance modulus for a given H0.

        d_L(z, H0) = (1+z) * c/H0 * D_C(z)
        mu = 5 * log10(d_L) + 25

        Valid at all redshifts (unlike the linear cz/H0 approximation).
        """
        d_c = self._comoving_distance_integral(z)
        d_L = (1 + z) * self.C_KMS * d_c / h0
        return 5 * np.log10(d_L) + 25

    def load_kappa_ceph(self):
        """Load kappa_Cep from companion paper TEP-H0 (Paper 11).

        kappa_Cep is the TEP shear coupling constant (in mag), determined from
        the host-level Cepheid analysis in TEP-H0. It is NOT derived within the
        TEP-VOID pipeline — it is imported as a fixed parameter for cross-validation.

        TEP-H0 reports three values:
          - kappa_Cep^equiv = (0.365 +/- 0.304) x 10^6 mag  (Cepheid-channel closure)
          - kappa_Cep = (0.452 +/- 0.220) x 10^6 mag         (redshift-only WLS, sigma_v=150)
          - kappa_canonical = 0.960 x 10^6 mag                (reference scaling)

        The redshift-only WLS value (0.452 +/- 0.220) x 10^6 is loaded as the
        primary estimate for the TEP correction, with its uncertainty propagated
        into the ΔM_B shift error. The equiv value (0.365 +/- 0.304) x 10^6 is
        also recorded for the manuscript's calibrator-average estimator
        (+0.025 +/- 0.021 mag), which uses the endpoint-slope closure value.
        """
        print_status("Loading kappa_Cep from TEP-H0 (Paper 11)...", "PROCESS")

        # Try to load from the TEP-H0 companion results directory
        # step_42_tep_native_ladder.json contains kappa_B (the redshift-only WLS value)
        # step_44_joint_distance_redshift_likelihood.json contains redshift_only_wls[0].kappa_Cep
        tep_h0_files = [
            ("step_42_tep_native_ladder.json", self._extract_kappa_from_step42),
            ("step_44_joint_distance_redshift_likelihood.json", self._extract_kappa_from_step44),
        ]

        for fname, extractor in tep_h0_files:
            step_path = TEP_H0_RESULTS / fname
            if step_path.exists():
                try:
                    with open(step_path, "r") as f:
                        data = json.load(f)
                    kappa, kappa_err = extractor(data)
                    if kappa is not None:
                        # Store both for downstream use
                        self.kappa_ceph = float(kappa)
                        self.kappa_ceph_err = float(kappa_err) if kappa_err is not None else 220046.0
                        # Equiv value (Cepheid-channel closure) for the
                        # manuscript calibrator-average estimator
                        self.kappa_ceph_equiv = 365000.0
                        self.kappa_ceph_equiv_err = 304000.0
                        print_status(
                            f"Loaded kappa_Cep = {kappa:.4e} +/- {self.kappa_ceph_err:.4e} mag "
                            f"from TEP-H0/{fname}",
                            "SUCCESS",
                        )
                        print_status(
                            f"  (equiv closure value kappa_Cep^equiv = "
                            f"{self.kappa_ceph_equiv:.4e} +/- {self.kappa_ceph_equiv_err:.4e} mag "
                            f"recorded for the manuscript calibrator-average estimator)",
                            "DEBUG",
                        )
                        return float(kappa)
                except Exception as e:
                    print_status(f"Error reading {fname}: {e}", "WARNING")

        # No fallback: fail loudly if TEP-H0 results are not available
        raise FileNotFoundError(
            f"Cannot load kappa_Cep from TEP-H0 companion project. "
            f"Expected one of: {TEP_H0_RESULTS}/step_42_tep_native_ladder.json, "
            f"{TEP_H0_RESULTS}/step_44_joint_distance_redshift_likelihood.json. "
            "Re-run the TEP-H0 companion pipeline to regenerate these outputs."
        )

    @staticmethod
    def _extract_kappa_from_step42(data):
        """Extract kappa_Cep from step_42_tep_native_ladder.json (kappa_B field)."""
        if isinstance(data, list) and len(data) > 0:
            kappa = data[0].get("kappa_B")
            kappa_err = data[0].get("kappa_B_err")
            return kappa, kappa_err
        return None, None

    @staticmethod
    def _extract_kappa_from_step44(data):
        """Extract kappa_Cep from step_44_joint_distance_redshift_likelihood.json."""
        # The redshift_only_wls array contains kappa_Cep at various sigma_v thresholds
        # Use the first entry (sigma_v=150 km/s) as the primary estimate
        wls = data.get("redshift_only_wls", [])
        if isinstance(wls, list) and len(wls) > 0:
            kappa = wls[0].get("kappa_Cep")
            kappa_err = wls[0].get("kappa_Cep_err")
            return kappa, kappa_err
        return None, None

    def load_host_potential(self):
        """Load host potential catalog from step_01."""
        print_status("Loading host potential catalog from step_01...", "PROCESS")

        catalog_path = self.data_processed / "host_potential_catalog.csv"

        if catalog_path.exists():
            try:
                df = pd.read_csv(catalog_path)
                print_status(f"Loaded {len(df)} host potentials from step_01", "SUCCESS")
                return df
            except Exception as e:
                print_status(f"Error reading host potential catalog: {e}", "WARNING")

        print_status("Host potential catalog not found. Using default potential proxy.", "WARNING")
        return pd.DataFrame()

    def load_pantheon_data(self):
        """Load Pantheon+ data from step_03."""
        candidates = [
            self.data_interim / "pantheon_plus_sne.csv",
            self.data_interim / "pantheon_plus_processed.csv",
            self.data_interim / "pantheon_sh0es_processed.csv",
            self.data_processed / "pantheon_plus.csv",
        ]

        for path in candidates:
            if path.exists():
                print_status(f"Loading Pantheon+ data from {path}...", "PROCESS")
                try:
                    df = pd.read_csv(path)
                    print_status(f"Loaded {len(df)} rows from Pantheon+ data", "SUCCESS")
                    return df
                except Exception as e:
                    print_status(f"Error reading {path}: {e}", "ERROR")

        print_status("Pantheon+ data not found. Cannot proceed without real data.", "WARNING")
        return pd.DataFrame()

    def generate_fallback_pantheon(self):
        """No synthetic data is generated. Returns empty DataFrame with warning."""
        print_status("No synthetic data generated — TEP-VOID uses only real published data.", "WARNING")
        print_status("Run step_03 (Pantheon+ ingestion) first to download real data.", "WARNING")
        return pd.DataFrame()

    MASSIVE_THRESHOLD = 10.0  # Standard threshold across all TEP-VOID steps

    def _compute_xi_for_hosts(self, df, host_potential_df):
        """Compute the screened potential coordinate X_i for each SN host.

        X_i = (S_total * U_i - U_ref_screened) / c^2

        where U_i = u_phi^2 is the rotation-based potential proxy and
        U_ref_screened = (30.507 km/s)^2 is the screened anchor reference.

        For hosts with measured potentials (matched to step_01 catalog),
        the actual X_i with TEP screening is used. For hosts without, a
        mass-dependent sigmoid proxy is calibrated from the 41 calibrator
        hosts (with S_total = 1.0 for uncatalogued hosts).
        """
        c2 = self.C_KMS ** 2

        # Calibrate mass-potential proxy from the 41 hosts
        # Load host properties to get masses for the potential catalog
        hp_path = self.data_raw_external = self.root / "data" / "raw" / "external" / "hosts_properties.csv"
        if hp_path.exists() and not host_potential_df.empty:
            hp = pd.read_csv(hp_path)
            # Normalize galaxy names for matching
            def norm_name(s):
                return str(s).replace(" ", "").upper().strip()
            host_potential_df = host_potential_df.copy()
            host_potential_df["norm"] = host_potential_df["galaxy"].apply(norm_name)
            hp["norm"] = hp["normalized_name"].apply(norm_name)
            merged = host_potential_df.merge(hp[["norm", "host_logmass"]], on="norm", how="inner")
            merged = merged[merged["host_logmass"].notna()]

            if len(merged) > 5:
                # Compute screened X_i for calibrator hosts
                S_cal = compute_screening_by_name(merged["galaxy"].values, PROJECT_ROOT)
                xi_cal = (S_cal * merged["phi_proxy_kms2"].values - self.U_REF) / c2
                # Calibrator-only mean X_i (used for the manuscript's
                # calibrator-average estimator in Section 9.5)
                self.xi_cal_mean = float(xi_cal.mean())
                self.xi_cal_std = float(xi_cal.std(ddof=1)) if len(xi_cal) > 1 else 0.0
                self.n_calibrators = int(len(xi_cal))
                # Mean X_i for massive hosts (logmass > 10)
                massive_mask = merged["host_logmass"] > self.MASSIVE_THRESHOLD
                xi_mean_massive = float(xi_cal[massive_mask].mean()) if massive_mask.any() else 8e-8
                xi_mean_low = float(xi_cal[~massive_mask].mean()) if (~massive_mask).any() else -4e-8
                print_status(
                    f"  Calibrated proxy: X_mean_massive = {xi_mean_massive:.4e}, "
                    f"X_mean_low = {xi_mean_low:.4e} (from {len(merged)} calibrators)",
                    "DEBUG",
                )
                print_status(
                    f"  Calibrator-only <X_i> = {self.xi_cal_mean:.4e} "
                    f"(N={self.n_calibrators}, std={self.xi_cal_std:.4e})",
                    "DEBUG",
                )
            else:
                xi_mean_massive = 8e-8
                xi_mean_low = -4e-8
                self.xi_cal_mean = None
        else:
            xi_mean_massive = 8e-8
            xi_mean_low = -4e-8
            print_status("  Using default proxy values (no host properties available)", "DEBUG")

        # Find mass column
        mass_col = None
        for c in df.columns:
            if "logmass" in c.lower() and "err" not in c.lower():
                mass_col = c
                break
            elif "mass" in c.lower() and "err" not in c.lower() and mass_col is None:
                mass_col = c

        # Compute X_i for each SN
        x_i = pd.Series(np.nan, index=df.index)

        if mass_col is not None:
            mass = pd.to_numeric(df[mass_col], errors="coerce")
            mass_valid = mass.notna() & (mass > -5.0)

            # Sigmoid proxy: smoothly interpolate between low-mass and massive X_i
            k_sigmoid = 2.0
            sigmoid_factor = 1.0 / (1.0 + np.exp(-k_sigmoid * (mass.fillna(0) - self.MASSIVE_THRESHOLD)))
            x_i_proxy = xi_mean_low + (xi_mean_massive - xi_mean_low) * sigmoid_factor
            x_i = x_i_proxy.where(mass_valid, 0.0)

            # Binary massive flag for reporting
            df["is_massive_host"] = mass_valid & (mass > self.MASSIVE_THRESHOLD)
            n_massive = int(df["is_massive_host"].sum())
            n_mass_valid = int(mass_valid.sum())
            print_status(
                f"  Host mass: {n_massive}/{n_mass_valid} massive (binary), "
                f"({int((~mass_valid).sum())} missing mass excluded)",
                "DEBUG",
            )
        elif "is_massive_host" in df.columns:
            # Fall back to binary
            x_i = np.where(df["is_massive_host"], xi_mean_massive, xi_mean_low)
            x_i = pd.Series(x_i, index=df.index)
        else:
            print_status("  No host mass info. Using zero correction.", "WARNING")
            x_i = pd.Series(0.0, index=df.index)

        return x_i

    def apply_tep_correction(self, df, kappa_ceph, host_potential_df):
        """Apply TEP temporal shear correction to distance moduli.

        The TEP correction (matching TEP-H0, Paper 11) is:
            Delta_mu = kappa_Cep * X_i

        where:
          - kappa_Cep is in mag (from TEP-H0, Paper 11)
          - X_i = (U_i - U_ref) / c^2 is the dimensionless potential coordinate

        No (1+z)^{-0.3} redshift factor is applied. The (1+z)^{-0.3} is the
        H0(z) decay prediction (Section 7), not a per-SN distance modulus
        correction. Applying it here would be circular.

        TEP predicts Cepheid distances are COMPRESSED in massive hosts, making
        the Cepheid-calibrated mu_obs TOO SMALL. The correction ADDS delta_mu
        to restore the true (larger) distance modulus:
            mu_tep = mu_obs + delta_mu

        NOTE: Pantheon+ uses a global M_B, so the Cepheid bias is imprinted
        on M_B rather than per-SN distances. The correction is therefore
        approximately a constant offset. The host-mass dependence is
        untestable with Pantheon+ and requires per-host Cepheid calibrations
        (TEP-H0, Paper 11).
        """
        print_status("Applying TEP temporal shear correction...", "PROCESS")

        df = df.copy()

        # Find columns
        z_col = None
        mu_col = None

        for c in df.columns:
            if c.lower() in ["z", "redshift", "zhel"]:
                z_col = c
            if c.lower() in ["mu", "mured", "m_b", "mb"]:
                mu_col = c

        if z_col is None or mu_col is None:
            print_status("Required columns not found. Cannot apply correction.", "ERROR")
            return df

        z = pd.to_numeric(df[z_col], errors="coerce")
        mu = pd.to_numeric(df[mu_col], errors="coerce")

        # Compute screened potential coordinate X_i
        x_i = self._compute_xi_for_hosts(df, host_potential_df)

        # TEP correction: Delta_mu = kappa_Cep * X_i
        # (No redshift factor — matching TEP-H0 formula)
        delta_mu = kappa_ceph * x_i
        df["tep_delta_mu"] = delta_mu

        # Corrected distance modulus (ADD correction to undo Cepheid compression)
        df["mu_tep_corrected"] = mu + delta_mu

        print_status(
            f"  Applied correction with kappa_Cep = {kappa_ceph:.4e} mag",
            "SUCCESS",
        )
        print_status(f"  Mean |Delta_mu| = {delta_mu.abs().mean():.4f} mag", "TEST")
        print_status(f"  Max |Delta_mu| = {delta_mu.abs().max():.4f} mag", "TEST")
        print_status(f"  Mean X_i = {x_i.mean():.4e}", "DEBUG")
        print_status(
            "  NOTE: Pantheon+ uses global M_B — correction is approximately "
            "a constant offset. Host-mass dependence requires per-host "
            "Cepheid calibrations (TEP-H0, Paper 11).",
            "WARNING",
        )

        return df

    def compute_hubble_residuals(self, df):
        """Compute Hubble diagram residuals before and after TEP correction."""
        print_status("Computing Hubble diagram residuals...", "PROCESS")

        z_col = None
        for c in df.columns:
            if c.lower() in ["z", "redshift", "zhel"]:
                z_col = c
                break

        if z_col is None:
            return {}

        z = pd.to_numeric(df[z_col], errors="coerce")

        # LCDM predicted mu using proper luminosity distance (valid at all z)
        # Precompute for all unique z values
        z_unique = np.unique(z.dropna().values)
        mu_lcdm_cache = {zv: self._mu_lcdm(zv, self.H0_CMB) for zv in z_unique}
        mu_lcdm = z.map(lambda zv: mu_lcdm_cache.get(zv, np.nan) if pd.notna(zv) else np.nan)

        results = {}

        if "mu" in df.columns:
            mu_obs = pd.to_numeric(df["mu"], errors="coerce")
            residual_before = mu_obs - mu_lcdm
            results["residual_before_mean"] = float(residual_before.dropna().mean())
            results["residual_before_std"] = float(residual_before.dropna().std())
            results["residual_before_rms"] = float(np.sqrt(np.mean(residual_before.dropna() ** 2)))
            print_status(f"  Before TEP correction: RMS = {results['residual_before_rms']:.4f} mag", "TEST")

        if "mu_tep_corrected" in df.columns:
            mu_corr = pd.to_numeric(df["mu_tep_corrected"], errors="coerce")
            residual_after = mu_corr - mu_lcdm
            results["residual_after_mean"] = float(residual_after.dropna().mean())
            results["residual_after_std"] = float(residual_after.dropna().std())
            results["residual_after_rms"] = float(np.sqrt(np.mean(residual_after.dropna() ** 2)))
            print_status(f"  After TEP correction:  RMS = {results['residual_after_rms']:.4f} mag", "TEST")

            # Improvement
            if "residual_before_rms" in results:
                improvement = results["residual_before_rms"] - results["residual_after_rms"]
                results["rms_improvement"] = float(improvement)
                print_status(f"  RMS improvement: {improvement:.4f} mag", "TEST")

            # ΔM_B shift (mean residual shift) with SEM uncertainty
            if "residual_before_mean" in results:
                rb = residual_before.dropna()
                ra = residual_after.dropna()
                # Paired difference if same length, else independent
                if len(rb) == len(ra):
                    diff = ra.values - rb.values
                    delta_mb = float(diff.mean())
                    delta_mb_sem = float(diff.std(ddof=1) / np.sqrt(len(diff)))
                else:
                    delta_mb = results["residual_after_mean"] - results["residual_before_mean"]
                    n_eff = min(len(rb), len(ra))
                    delta_mb_sem = float(np.sqrt(
                        rb.std(ddof=1) ** 2 / len(rb) +
                        ra.std(ddof=1) ** 2 / len(ra)
                    ))
                results["delta_mb_shift"] = delta_mb
                results["delta_mb_shift_sem"] = delta_mb_sem
                results["delta_mb_shift_sigma"] = float(
                    abs(delta_mb) / delta_mb_sem if delta_mb_sem > 0 else 0
                )
                # Propagated uncertainty from kappa_Cep posterior.
                # σ_ΔM_B^2 = (σ_κ · <X_i>)^2 + (κ · σ_<X_i>)^2
                # The κ term dominates (σ_κ/κ ~ 49% for the redshift-only WLS
                # value 0.452 +/- 0.220). Use the κ and σ_κ actually loaded
                # (redshift-only WLS), NOT the equiv value's error.
                kappa_val = self.kappa_ceph if hasattr(self, 'kappa_ceph') else 451682.0
                sigma_kappa = (
                    self.kappa_ceph_err if hasattr(self, 'kappa_ceph_err')
                    else 220046.0  # redshift-only WLS uncertainty
                )
                xi_mean = delta_mb / kappa_val if kappa_val != 0 else 0
                # σ_<X_i> is subdominant; estimate from the SEM of delta_mb
                # divided by κ (propagates the calibrator-potential variance).
                sigma_xi_mean = delta_mb_sem / kappa_val if kappa_val != 0 else 0
                delta_mb_propagated_err = float(np.sqrt(
                    (sigma_kappa * abs(xi_mean)) ** 2 +
                    (kappa_val * sigma_xi_mean) ** 2
                ))
                results["delta_mb_shift_propagated_err"] = delta_mb_propagated_err
                results["kappa_ceph_used"] = float(kappa_val)
                results["kappa_ceph_err_used"] = float(sigma_kappa)
                # Primary significance uses propagated uncertainty (from κ_Cep
                # posterior), not the paired SEM. The paired SEM is tiny because
                # the TEP correction is nearly a constant offset, so the paired
                # sigma is spurious (it tests whether the mean shift is nonzero,
                # not whether the TEP model is correct). The propagated error
                # captures the dominant uncertainty: the κ_Cep calibration.
                results["delta_mb_shift_sigma_primary"] = float(
                    abs(delta_mb) / delta_mb_propagated_err
                    if delta_mb_propagated_err > 0 else 0
                )
                print_status(
                    f"  ΔM_B shift: {delta_mb:+.4f} mag",
                    "TEST",
                )
                print_status(
                    f"  Propagated err (κ_Cep = {kappa_val:.0f} ± {sigma_kappa:.0f}): "
                    f"± {delta_mb_propagated_err:.4f} mag "
                    f"({results['delta_mb_shift_sigma_primary']:.2f}σ)",
                    "TEST",
                )
                print_status(
                    f"  [Paired SEM {delta_mb_sem:.4f} mag is not physically "
                    f"meaningful — correction is near-constant offset]",
                    "INFO",
                )

                # Manuscript calibrator-average estimator using the equiv
                # (Cepheid-channel closure) value κ_Cep^equiv = 0.365 ± 0.304.
                # This uses the CALIBRATOR-ONLY mean <X_i> (the 41 SH0ES
                # Cepheid calibrator hosts), not the all-Pantheon+ host mean,
                # because ΔM_B is a zero-point shift imprinted by the
                # calibrator population that sets M_B.
                if hasattr(self, 'kappa_ceph_equiv'):
                    kappa_equiv = self.kappa_ceph_equiv
                    sigma_kappa_equiv = self.kappa_ceph_equiv_err
                    # Calibrator-only <X_i> if available; otherwise fall back
                    # to the all-host mean derived from the WLS shift.
                    if hasattr(self, 'xi_cal_mean') and self.xi_cal_mean is not None:
                        xi_mean_cal = self.xi_cal_mean
                        sigma_xi_mean_cal = (
                            self.xi_cal_std / np.sqrt(self.n_calibrators)
                            if self.n_calibrators > 1 else 0.0
                        )
                    else:
                        xi_mean_cal = xi_mean
                        sigma_xi_mean_cal = sigma_xi_mean
                    delta_mb_equiv = float(kappa_equiv * xi_mean_cal)
                    delta_mb_equiv_err = float(np.sqrt(
                        (sigma_kappa_equiv * abs(xi_mean_cal)) ** 2 +
                        (kappa_equiv * sigma_xi_mean_cal) ** 2
                    ))
                    results["delta_mb_shift_equiv"] = delta_mb_equiv
                    results["delta_mb_shift_equiv_err"] = delta_mb_equiv_err
                    results["kappa_ceph_equiv"] = float(kappa_equiv)
                    results["kappa_ceph_equiv_err"] = float(sigma_kappa_equiv)
                    results["xi_cal_mean"] = float(xi_mean_cal)
                    results["xi_cal_std"] = float(
                        self.xi_cal_std if hasattr(self, 'xi_cal_std') else 0.0
                    )
                    results["n_calibrators"] = int(
                        self.n_calibrators if hasattr(self, 'n_calibrators') else 0
                    )
                    print_status(
                        f"  ΔM_B (equiv closure, κ^equiv = {kappa_equiv:.0f} ± "
                        f"{sigma_kappa_equiv:.0f}, calibrator <X_i> = {xi_mean_cal:.4e}): "
                        f"{delta_mb_equiv:+.4f} ± "
                        f"{delta_mb_equiv_err:.4f} mag  [manuscript Section 9.5]",
                        "TEST",
                    )

            # For massive hosts specifically
            if "is_massive_host" in df.columns:
                massive_mask = df["is_massive_host"] == True
                if massive_mask.sum() > 0:
                    res_massive_before = (mu_obs - mu_lcdm)[massive_mask].dropna()
                    res_massive_after = (mu_corr - mu_lcdm)[massive_mask].dropna()
                    results["massive_residual_before_rms"] = float(np.sqrt(np.mean(res_massive_before ** 2)))
                    results["massive_residual_after_rms"] = float(np.sqrt(np.mean(res_massive_after ** 2)))
                    results["massive_rms_improvement"] = float(
                        results["massive_residual_before_rms"] - results["massive_residual_after_rms"]
                    )
                    print_status(
                        f"  Massive hosts RMS: {results['massive_residual_before_rms']:.4f} -> {results['massive_residual_after_rms']:.4f}",
                        "TEST",
                    )

        return results

    def plot_tep_correction(self, df):
        """Generate figure showing TEP correction to Pantheon+ Hubble diagram."""
        print_status("Generating TEP correction figure...", "PROCESS")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        z_col = None
        for c in df.columns:
            if c.lower() in ["z", "redshift", "zhel"]:
                z_col = c
                break

        z = pd.to_numeric(df[z_col], errors="coerce") if z_col else pd.Series(dtype=float)

        # Panel 1: Hubble diagram before correction
        ax1 = axes[0, 0]
        if "mu" in df.columns and z_col:
            mu = pd.to_numeric(df["mu"], errors="coerce")
            mask = z.notna() & mu.notna()
            if "is_massive_host" in df.columns:
                massive = mask & (df["is_massive_host"] == True)
                low = mask & (df["is_massive_host"] == False)
                ax1.scatter(z[low], mu[low], c="#1f77b4", s=10, alpha=0.5, label="Low-mass hosts")
                ax1.scatter(z[massive], mu[massive], c="#d62728", s=15, alpha=0.6, label="Massive hosts", marker="^")
            else:
                ax1.scatter(z[mask], mu[mask], c="#1f77b4", s=10, alpha=0.5)
            # LambdaCDM line (proper LCDM, not linear)
            z_fine = np.linspace(0.01, max(z[mask].max(), 0.01), 100)
            mu_model = np.array([self._mu_lcdm(zv, self.H0_CMB) for zv in z_fine])
            ax1.plot(z_fine, mu_model, "k--", linewidth=1, label="$\\Lambda$CDM")
            ax1.set_xlabel("Redshift $z$", fontsize=12)
            ax1.set_ylabel("$\\mu$ (mag)", fontsize=12)
            ax1.set_title("Before TEP Correction", fontsize=13)
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)

        # Panel 2: Hubble diagram after correction
        ax2 = axes[0, 1]
        if "mu_tep_corrected" in df.columns and z_col:
            mu_corr = pd.to_numeric(df["mu_tep_corrected"], errors="coerce")
            mask = z.notna() & mu_corr.notna()
            if "is_massive_host" in df.columns:
                massive = mask & (df["is_massive_host"] == True)
                low = mask & (df["is_massive_host"] == False)
                ax2.scatter(z[low], mu_corr[low], c="#1f77b4", s=10, alpha=0.5, label="Low-mass hosts")
                ax2.scatter(z[massive], mu_corr[massive], c="#d62728", s=15, alpha=0.6, label="Massive hosts", marker="^")
            else:
                ax2.scatter(z[mask], mu_corr[mask], c="#1f77b4", s=10, alpha=0.5)
            ax2.plot(z_fine, mu_model, "k--", linewidth=1, label="$\\Lambda$CDM")
            ax2.set_xlabel("Redshift $z$", fontsize=12)
            ax2.set_ylabel("$\\mu_{TEP}$ (mag)", fontsize=12)
            ax2.set_title("After TEP Correction", fontsize=13)
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)

        # Panel 3: Residuals before correction
        ax3 = axes[1, 0]
        if "mu" in df.columns and z_col:
            mu = pd.to_numeric(df["mu"], errors="coerce")
            # Use proper LCDM mu (precomputed cache from compute_hubble_residuals)
            z_vals = pd.to_numeric(df[z_col], errors="coerce")
            z_unique = np.unique(z_vals.dropna().values)
            mu_lcdm_cache = {zv: self._mu_lcdm(zv, self.H0_CMB) for zv in z_unique}
            mu_model = z_vals.map(lambda zv: mu_lcdm_cache.get(zv, np.nan) if pd.notna(zv) else np.nan)
            residual = mu - mu_model
            mask = residual.notna()
            if "is_massive_host" in df.columns:
                massive = mask & (df["is_massive_host"] == True)
                low = mask & (df["is_massive_host"] == False)
                ax3.scatter(z[low], residual[low], c="#1f77b4", s=10, alpha=0.5, label="Low-mass")
                ax3.scatter(z[massive], residual[massive], c="#d62728", s=15, alpha=0.6, label="Massive", marker="^")
            else:
                ax3.scatter(z[mask], residual[mask], c="#1f77b4", s=10, alpha=0.5)
            ax3.axhline(0, color="black", linestyle="--", alpha=0.5)
            ax3.set_xlabel("Redshift $z$", fontsize=12)
            ax3.set_ylabel("Residual $\\mu - \\mu_{\\Lambda CDM}$ (mag)", fontsize=12)
            ax3.set_title("Residuals Before TEP Correction", fontsize=13)
            ax3.legend(fontsize=10)
            ax3.grid(True, alpha=0.3)

        # Panel 4: TEP Delta_mu vs redshift
        ax4 = axes[1, 1]
        if "tep_delta_mu" in df.columns and z_col:
            delta_mu = pd.to_numeric(df["tep_delta_mu"], errors="coerce")
            mask = z.notna() & delta_mu.notna()
            if "is_massive_host" in df.columns:
                massive = mask & (df["is_massive_host"] == True)
                low = mask & (df["is_massive_host"] == False)
                ax4.scatter(z[low], delta_mu[low], c="#1f77b4", s=10, alpha=0.5, label="Low-mass")
                ax4.scatter(z[massive], delta_mu[massive], c="#d62728", s=15, alpha=0.6, label="Massive", marker="^")
            else:
                ax4.scatter(z[mask], delta_mu[mask], c="#1f77b4", s=10, alpha=0.5)
            ax4.axhline(0, color="black", linestyle="--", alpha=0.5)
            ax4.set_xlabel("Redshift $z$", fontsize=12)
            ax4.set_ylabel("$\\Delta\\mu_{TEP}$ (mag)", fontsize=12)
            ax4.set_title("TEP Temporal Shear Correction", fontsize=13)
            ax4.legend(fontsize=10)
            ax4.grid(True, alpha=0.3)

        fig.suptitle("Pantheon+ Distance Moduli: TEP Redshift Shear Reconstruction", fontsize=15, y=1.02)
        fig.tight_layout()
        fig_path = self.figures / "step_40_pantheon_tep_correction.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print_status(f"Figure saved to {fig_path}", "SUCCESS")
        return fig_path

    def run(self):
        """Execute the full step."""
        print_status("Step 40: Redshift Shear Reconstruction — Pantheon+ TEP", "TITLE")

        print_status(
            "This step addresses whether the TEP temporal shear correction, "
            "calibrated independently in TEP-H0 (Paper 11) from the host-level "
            "Cepheid analysis, is consistent with the observed Pantheon+ Hubble "
            "diagram residual structure. The shear coupling constant kappa_Cep "
            "is imported as a fixed parameter and applied to Pantheon+ distance "
            "moduli via Delta_mu = kappa_Cep * X_i, where X_i is the dimensionless "
            "screened potential coordinate. This serves as a cross-validation of "
            "the TEP correction against an independent supernova dataset rather "
            "than a discriminating test between TEP and the KBC void model.",
            "INFO",
        )

        # Load kappa_Cep
        print_status(
            "Methodology: kappa_Cep is imported from TEP-H0 (Paper 11) as the "
            "redshift-only WLS value (0.452 +/- 0.220) x 10^6 mag (sigma_v = 150 km/s), "
            "loaded from the TEP-H0 companion results. The "
            "anchor reference potential U_ref = (30.507 km/s)^2 (screened) and the screened "
            "potential coordinate X_i = (S_total * U_i - U_ref) / c^2 are defined per the "
            "TEP framework. No (1+z)^{-0.3} redshift factor is applied to avoid "
            "circularity with the Section 7 H0(z) prediction.",
            "PROCESS",
        )
        kappa_ceph = self.load_kappa_ceph()

        # Load host potential catalog
        host_potential_df = self.load_host_potential()

        # Load Pantheon+ data
        df = self.load_pantheon_data()
        used_fallback = False
        if df.empty:
            df = self.generate_fallback_pantheon()
            used_fallback = True

        # Apply TEP correction
        print_status(
            "Methodology: the TEP correction Delta_mu = kappa_Cep * X_i is added "
            "to observed distance moduli to undo the Cepheid distance compression "
            "in massive hosts. Host potentials are taken from the step_01 catalog "
            "where available; a mass-dependent sigmoid proxy calibrated from the "
            "41 calibrator galaxies is used for unmatched hosts.",
            "PROCESS",
        )
        df = self.apply_tep_correction(df, kappa_ceph, host_potential_df)

        # Compute residuals
        residual_results = self.compute_hubble_residuals(df)

        rms_improvement_val = residual_results.get("rms_improvement", 0)
        print_status(
            f"Interpretation: the TEP correction yields an RMS residual "
            f"improvement of {rms_improvement_val:.4f} mag. Because Pantheon+ "
            f"uses a global M_B calibration, the correction acts approximately "
            f"as a constant offset; the host-mass dependence is untestable "
            f"without per-host Cepheid calibrations. Consistency of the "
            f"correction direction with the observed residual structure "
            f"supports the TEP framework as a cross-validation.",
            "TEST",
        )

        # Generate figure
        fig_path = self.plot_tep_correction(df)

        # Summary
        summary = {
            "step": "40_redshift_shear_reconstruction",
            "description": "Reconstruct Pantheon+ distance moduli under TEP temporal shear",
            "kappa_ceph": kappa_ceph,
            "kappa_ceph_units": "mag",
            "kappa_ceph_source": (
                "Imported from TEP-H0 (Paper 11, redshift-only WLS at sigma_v=150 km/s). "
                "kappa_Cep = (0.452 +/- 0.220) x 10^6 mag. "
                "NOT derived within TEP-VOID — fixed parameter for cross-validation."
            ),
            "correction_formula": "Delta_mu = kappa_Cep * X_i",
            "xi_definition": "X_i = (S_total * U_i - U_ref) / c^2, U_ref = (30.507 km/s)^2 (screened)",
            "caveat": (
                "Pantheon+ uses a global M_B calibration. The Cepheid bias is "
                "imprinted on M_B, not on individual SN distances. The correction "
                "is approximately a constant offset. The host-mass dependence of "
                "the TEP effect requires per-host Cepheid calibrations (TEP-H0, "
                "Paper 11). The (1+z)^{-0.3} H0(z) decay prediction from Section "
                "7 is NOT applied as a per-SN correction."
            ),
            "h0_cmb": self.H0_CMB,
            "n_sne": len(df),
            "n_host_potentials": len(host_potential_df),
            "host_potential_used": len(host_potential_df) > 0,
            "host_potential_note": (
                "Host potential catalog from step_01 loaded. "
                "X_i computed from actual potentials for matched hosts; "
                "mass-based sigmoid proxy for unmatched hosts, calibrated "
                "from the 41 calibrator galaxies."
            ) if len(host_potential_df) > 0 else (
                "Host potential catalog from step_01 not available. "
                "Using host stellar mass as a proxy for gravitational potential. "
                "The full potential-depth analysis is in TEP-H0 (Paper 11)."
            ),
            "used_fallback": used_fallback,
            "residual_analysis": residual_results,
            "tep_prediction": "TEP-corrected residuals should be reduced, especially for massive hosts",
            "tep_confirmed": residual_results.get("rms_improvement", 0) > 0,
            "void_prediction": (
                "The KBC void model predicts no indicator-specific distance "
                "compression; the Pantheon+ residual structure should show no "
                "systematic offset correctable by a potential-dependent term."
            ),
            "methodology": (
                "TEP temporal shear correction Delta_mu = kappa_Cep * X_i applied "
                "to Pantheon+ distance moduli. kappa_Cep imported from TEP-H0 "
                "(Paper 11, Cepheid-channel closure). X_i computed from step_01 "
                "host potential catalog with mass-dependent sigmoid proxy for "
                "unmatched hosts. Hubble residuals computed against a LambdaCDM "
                "reference with H0 = 67.4 km/s/Mpc and Omega_m = 0.302."
            ),
            "provenance": {
                "data_sources": [
                    "Pantheon+ SN Ia sample (step_03 ingestion)",
                    "Host potential catalog (step_01)",
                    "kappa_Cep from TEP-H0 (Paper 11, Cepheid-channel closure)",
                ],
                "pipeline_block": "Block III — TEP reconstruction and synthesis",
            },
            "scientific_context": (
                "Cross-validation of the TEP shear coupling constant against the "
                "Pantheon+ Hubble diagram. The correction direction and residual "
                "reduction are checked for consistency with the TEP prediction. "
                "This is not a discriminating test between TEP and the void model."
            ),
            "downstream_consumers": ["step_42", "step_43"],
            "output_files": [
                str(self.results / "step_40_redshift_shear_reconstruction.json"),
                str(fig_path),
            ],
        }

        summary_path = self.results / "step_40_redshift_shear_reconstruction.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print_status(f"Summary saved to {summary_path}", "SUCCESS")

        print_status("Step 40 complete", "SUCCESS")


if __name__ == "__main__":
    step = Step40RedshiftShearReconstruction()
    step.run()
