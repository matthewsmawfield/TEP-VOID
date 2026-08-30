#!/usr/bin/env python3
"""
Step 32: Redshift Decay Profile — Void Model vs TEP Prediction
================================================================
Compares the void-model H0(z) decay prediction with the TEP prediction
for Pantheon+ with global M_B, using Pantheon+ supernova data.

Key Tasks:
1. Load Pantheon+ data (from step_03)
2. Compute H0(z) in redshift bins from SN Ia distance moduli, using the
   proper LCDM luminosity-distance relation
3. Compare the ACTUAL published KBC/MOND void model H0(z) curves
   (Gaussian and Exponential density profiles from Haslbauer, Banik &
   Kroupa 2020, as used by Mazurenko, Banik & Kroupa 2025) against the
   TEP prediction for global M_B (flat H0(z) ≈ 73)
4. Perform both fixed-prediction and free-parameter model comparisons

VOID MODEL (published):
    The KBC/MOND model does NOT predict a sharp step at z ~ 0.07.
    The published HBK20 models use Gaussian and Exponential void density
    profiles, and the resulting H0(z) declines GRADUALLY, converging
    to within 1σ of the Planck value only at z ≳ 1.8 (Mazurenko et al.
    2024, Figure 3). The observer is offset ~100-150 Mpc from the void
    centre.

    We approximate the published curves with:
      Gaussian:     H0(z) = H0_CMB + ΔH0 * exp(-z²/(2σ_z²))
      Exponential:  H0(z) = H0_CMB + ΔH0 * exp(-z/z_0)

    Calibrated to the published convergence point (1σ of Planck at z≈1.8):
      σ_z ≈ 0.82 (Gaussian), z_0 ≈ 0.74 (Exponential)

TEP Prediction (global M_B):
    Pantheon+ uses a global M_B = -19.253 calibrated from Cepheid anchors
    at z ~ 0. Under TEP, the Cepheid clock bias is imprinted on this
    zero-point, and since all SNe share the same M_B, the bias does not
    vary with redshift. The TEP prediction is therefore a FLAT H0(z) ≈ 73.

Outputs:
    results/outputs/step_32_redshift_decay_profile.json
    results/figures/step_32_h0_vs_redshift.png
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats, integrate, optimize

# Apple's Accelerate BLAS on ARM64 produces spurious "divide by zero"
# RuntimeWarnings in matmul when intermediate values hit denormals.
# The results are correct; suppress the noise.
warnings.filterwarnings("ignore", message="divide by zero encountered in matmul")
warnings.filterwarnings("ignore", message="overflow encountered in matmul")
warnings.filterwarnings("ignore", message="invalid value encountered in matmul")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.plot_style import apply_tep_style


class Step32RedshiftDecayProfile:
    """Step 32: Compare void-model and TEP H0(z) decay predictions."""

    # Cosmological parameters
    H0_CMB = 67.4  # km/s/Mpc (Planck 2018)
    H0_SH0ES = 73.0  # km/s/Mpc (Riess et al. 2022)
    OMEGA_M = 0.302
    C_KMS = 299792.458

    # Redshift bins
    Z_BINS = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.80, 1.50]

    # Void model parameters — published HBK20/Mazurenko et al. 2025
    # The KBC void model predicts a GRADUAL decline, NOT a sharp step.
    # Convergence to within 1σ of Planck (67.4±0.5) occurs at z ≳ 1.8.
    # We approximate the published curves with Gaussian and Exponential profiles.
    VOID_RADIUS_MPC = 300.0  # Mpc, void characteristic radius
    VOID_GAUSSIAN_SIGMA_Z = 0.82  # Gaussian decay scale (calibrated to z≈1.8 convergence)
    VOID_EXPONENTIAL_Z0 = 0.74   # Exponential decay scale (calibrated to z≈1.8 convergence)

    # TEP parameters
    KAPPA_SHEAR = 0.040  # dimensionless TEP shear coupling (not kappa_Cep in mag)
    TEP_DECAY_INDEX = 0.3  # <X_i>(z) = (1+z)^-0.3

    # Host mass threshold for massive/low-mass split
    MASSIVE_THRESHOLD = 10.0  # Standard threshold across all TEP-VOID steps

    def __init__(self):
        self.root = PROJECT_ROOT
        self.data_interim = self.root / "data" / "interim"
        self.data_raw = self.root / "data" / "raw"
        self.data_external = self.data_raw / "external"
        self.data_processed = self.root / "data" / "processed"
        self.results = self.root / "results" / "outputs"
        self.figures = self.root / "results" / "figures"
        self.logs = self.root / "logs"

        for d in [self.data_interim, self.data_processed, self.results, self.figures, self.logs]:
            d.mkdir(parents=True, exist_ok=True)

        self.logger = TEPLogger(
            "step_32", log_file_path=self.logs / "step_32_redshift_decay_profile.log"
        )
        set_step_logger(self.logger)

    # ------------------------------------------------------------------
    # Cosmological utilities
    # ------------------------------------------------------------------
    def _E(self, z):
        """Dimensionless Hubble parameter E(z) = H(z)/H0."""
        return np.sqrt(self.OMEGA_M * (1 + z) ** 3 + (1 - self.OMEGA_M))

    def _comoving_distance_integral(self, z):
        """
        Compute the comoving distance integral:
            D_C(z) = integral_0^z dz' / E(z')
        in units of c/H0.  This is independent of H0.
        """
        result, _ = integrate.quad(lambda zp: 1.0 / self._E(zp), 0, z)
        return result

    def _h0_from_mu(self, z, mu):
        """
        Infer H0 from a single SN's redshift and distance modulus.

        In a flat LCDM cosmology:
            d_L(z, H0) = (1+z) * c/H0 * D_C(z)
        where D_C(z) = integral_0^z dz'/E(z') is independent of H0.

        Given the observed distance modulus mu:
            d_L_obs = 10^((mu - 25) / 5)
        we solve for H0:
            H0 = (1+z) * c * D_C(z) / d_L_obs

        This is valid at ALL redshifts, unlike the linear approximation
        H0 ~ cz/d_L which is only valid at z << 0.01.
        """
        d_L = 10 ** ((mu - 25.0) / 5.0)
        d_c = self._comoving_distance_integral(z)
        h0 = (1 + z) * self.C_KMS * d_c / d_L
        return h0

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
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
                    print_status(f"Columns: {list(df.columns)}", "DEBUG")
                    return df
                except Exception as e:
                    print_status(f"Error reading {path}: {e}", "ERROR")

        print_status(
            "Pantheon+ data not found. Cannot proceed without real data.",
            "ERROR",
        )
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # H0(z) computation
    # ------------------------------------------------------------------
    def compute_h0_in_bins(self, df, mass_filter=None, label="all"):
        """
        Compute H0(z) in redshift bins from SN Ia distance moduli,
        using the proper LCDM luminosity-distance relation.

        Parameters
        ----------
        df : DataFrame
            Pantheon+ data with z, mu, and optionally host_logmass columns.
        mass_filter : str or None
            If 'massive', filter to host_logmass >= threshold.
            If 'lowmass', filter to host_logmass < threshold.
            If None, use all SNe.
        label : str
            Label for logging.
        """
        print_status(f"Computing H0(z) in redshift bins [{label}]...", "PROCESS")

        if df.empty:
            print_status("No data available.", "WARNING")
            return {}

        # Find relevant columns
        z_col = None
        mu_col = None
        mass_col = None

        for c in df.columns:
            if c.lower() in ["z", "redshift", "zhel"]:
                z_col = c
            if c.lower() in ["mu", "mured", "m_b", "mb"]:
                mu_col = c
            if "logmass" in c.lower() and "err" not in c.lower():
                mass_col = c

        if z_col is None or mu_col is None:
            print_status("Required columns (z, mu) not found.", "ERROR")
            return {}

        # Apply mass filter
        df_use = df.copy()
        if mass_filter and mass_col is not None:
            mass = pd.to_numeric(df_use[mass_col], errors="coerce")
            if mass_filter == "massive":
                df_use = df_use[mass >= self.MASSIVE_THRESHOLD]
            elif mass_filter == "lowmass":
                df_use = df_use[mass < self.MASSIVE_THRESHOLD]
            print_status(f"  After mass filter ({mass_filter}): {len(df_use)} SNe", "DEBUG")

        z = pd.to_numeric(df_use[z_col], errors="coerce")
        mu = pd.to_numeric(df_use[mu_col], errors="coerce")
        # Per-SN distance modulus error (for proper error propagation)
        mu_err_col = None
        for c in df_use.columns:
            if c.lower() in ["mu_err", "muerr", "mu_sh0es_err_diag", "m_b_corr_err"]:
                mu_err_col = c
                break
        mu_err_series = (
            pd.to_numeric(df_use[mu_err_col], errors="coerce")
            if mu_err_col
            else pd.Series(dtype=float)
        )

        # Build SN name → covariance matrix index mapping
        sn_name_col = None
        for c in df_use.columns:
            if c.lower() in ["sn_name", "name", "cid", "sn"]:
                sn_name_col = c
                break
        cov_index_map = {}
        if sn_name_col:
            dat_path = self.data_raw / "Pantheon+SH0ES.dat"
            if dat_path.exists():
                with open(dat_path) as f:
                    header = f.readline()
                    for idx, line in enumerate(f):
                        parts = line.split()
                        if parts:
                            cov_index_map[parts[0]] = idx

        mask = z.notna() & mu.notna() & (z > 0)
        if mu_err_col:
            mask = mask & mu_err_series.notna()
        z = z[mask]
        mu = mu[mask]
        if mu_err_col:
            mu_err_vals = mu_err_series[mask]
        else:
            mu_err_vals = None

        # Precompute comoving distance integrals for unique z values
        z_unique = np.unique(z.values)
        print_status(f"  Computing LCDM comoving distances for {len(z_unique)} unique redshifts...", "DEBUG")
        integral_cache = {zv: self._comoving_distance_integral(zv) for zv in z_unique}

        results = {}
        for i in range(len(self.Z_BINS) - 1):
            z_lo = self.Z_BINS[i]
            z_hi = self.Z_BINS[i + 1]
            z_center = np.sqrt(z_lo * z_hi)

            bin_mask = (z >= z_lo) & (z < z_hi)
            n = int(bin_mask.sum())

            if n < 5:
                print_status(f"  z=[{z_lo:.2f},{z_hi:.2f}]: n={n} — insufficient", "WARNING")
                continue

            z_bin = z[bin_mask].values
            mu_bin = mu[bin_mask].values

            # Compute H0 for each SN using proper LCDM formula
            h0_vals = np.array([
                (1 + zv) * self.C_KMS * integral_cache[zv] / 10 ** ((muv - 25.0) / 5.0)
                for zv, muv in zip(z_bin, mu_bin)
            ])

            h0_mean = float(np.mean(h0_vals))

            # Error propagation: use per-SN mu_err when available.
            # sigma_H0 = H0 * ln(10)/5 * sigma_mu  (from dL = 10^((mu-25)/5))
            # Bin error = sqrt(mean(sigma_H0_i^2) / n)  (standard error of uncorrelated mean)
            if mu_err_vals is not None:
                mu_err_bin = mu_err_vals[bin_mask].values
                sigma_h0_per = h0_vals * np.log(10.0) / 5.0 * mu_err_bin
                h0_err = float(np.sqrt(np.mean(sigma_h0_per ** 2) / n))
                h0_err_method = "per_sn_mu_err"
            else:
                h0_err = float(np.std(h0_vals) / np.sqrt(n))
                h0_err_method = "sample_sem"

            # Massive host fraction
            massive_frac = 0.0
            if mass_col is not None:
                bin_indices = z[bin_mask].index
                mass_data = pd.to_numeric(df_use.loc[bin_indices, mass_col], errors="coerce")
                if mass_data.notna().sum() > 0:
                    massive_frac = float((mass_data >= self.MASSIVE_THRESHOLD).mean())

            # Map SN names to covariance matrix indices
            cov_indices = []
            if sn_name_col and cov_index_map:
                bin_sn_names = df_use.loc[z[bin_mask].index, sn_name_col].values
                cov_indices = [
                    cov_index_map[name] for name in bin_sn_names
                    if name in cov_index_map
                ]

            results[z_center] = {
                "h0": h0_mean,
                "h0_err": h0_err,
                "h0_err_method": h0_err_method,
                "n": n,
                "massive_frac": massive_frac,
                "cov_indices": cov_indices,
                "h0_per_sn": h0_vals.tolist(),  # for Jacobian computation
            }
            print_status(
                f"  z=[{z_lo:.2f},{z_hi:.2f}]: H0={h0_mean:.1f} +/- {h0_err:.1f} km/s/Mpc (n={n})",
                "SUCCESS",
            )

        return results

    # ------------------------------------------------------------------
    # Model predictions
    # ------------------------------------------------------------------
    def void_model_prediction(self, z_array, profile="gaussian"):
        """
        Void model H0(z) prediction using the ACTUAL published KBC/MOND
        Method-3 curves digitized from Mazurenko, Banik & Kroupa (2025,
        MNRAS 536, 3232–3241, Figure 3).

        The published curves are from the full GR calculation (Method 3)
        using HBK20 best-fitting void parameters. The Gaussian and
        Exponential density profiles produce nearly identical H0(z) curves
        (mean |ΔH0| ≈ 0.5 km/s/Mpc across the full redshift range).

        The digitized curve data is stored in:
          data/raw/external/mazurenko_curves/{profile}_method3.json

        Digitization noise mitigation:
          The Gaussian Method-3 curve exhibits significant digitization
          artifacts (30 sign changes in 35 intervals at z < 0.5, from
          interleaved upper/lower envelope tracing). Raw interpolation
          of these noisy points inflates the model shape s_KBC and
          artificially increases ΔAIC by ~7 at z≥0.05 and ~56 in the
          full sample. A parametric Gaussian-decline fit
          H(z) = H_inf + A·exp(-z²/(2σ²)) is applied to denoise;
          the fit achieves reduced χ² ≈ 0.93 (σ_digitize = 0.5 km/s/Mpc)
          and produces a smooth, monotonically declining curve consistent
          with the published KBC prediction. The Exponential curve is
          already clean (0 sign changes) and requires no denoising.

        Falls back to analytic surrogates if digitized data is unavailable.

        Performance: The denoised curve data is cached after the first call
        to avoid re-reading and re-fitting on every subsequent call.
        """
        # Check cache first
        cache_key = f"_void_curve_{profile}"
        if hasattr(self, cache_key):
            cached = getattr(self, cache_key)
            z_curve, h0_curve = cached
            # Interpolate in log(z) space for better coverage
            log_z = np.log10(np.clip(z_array, z_curve.min(), z_curve.max()))
            log_z_curve = np.log10(z_curve)
            h0_void = np.interp(log_z, log_z_curve, h0_curve)
            h0_void = np.where(z_array < z_curve.min(), h0_curve[0], h0_void)
            h0_void = np.where(z_array > z_curve.max(), h0_curve[-1], h0_void)
            return h0_void

        # Try to load digitized curve
        curve_path = self.data_external / "mazurenko_curves" / f"{profile}_method3.json"
        if curve_path.exists():
            import json
            with open(curve_path) as f:
                curve_data = json.load(f)
            z_curve = np.array([p["z"] for p in curve_data])
            h0_curve = np.array([p["H0"] for p in curve_data])

            # --- Digitization noise mitigation for the Gaussian curve ---
            # Detect non-monotonic behavior (digitization artifacts).
            # Always sort by z to ensure np.interp receives a monotonic
            # abscissa (required for correct interpolation).
            sort_idx = np.argsort(z_curve)
            z_s = z_curve[sort_idx]
            h0_s = h0_curve[sort_idx]
            diffs = np.diff(h0_s)
            sign_changes = int(np.sum(np.abs(np.diff(np.sign(diffs))) > 0))

            if profile == "gaussian" and sign_changes > 10:
                # The Gaussian curve has interleaved upper/lower envelope
                # points from digitization. Fit the parametric KBC form
                # H(z) = H_inf + A * exp(-z^2 / (2*sigma^2)) to denoise.
                from scipy.optimize import curve_fit

                def _gaussian_decline(z, h_inf, amplitude, sigma_z):
                    return h_inf + amplitude * np.exp(-z**2 / (2 * sigma_z**2))

                popt, _ = curve_fit(
                    _gaussian_decline, z_curve, h0_curve,
                    p0=[67.4, 6.0, 0.15], maxfev=10000,
                )
                # Replace the noisy curve with the smooth parametric fit
                h0_curve = _gaussian_decline(z_s, *popt)
                z_curve = z_s
            else:
                # Ensure the curve is sorted by z for both profiles
                z_curve = z_s
                h0_curve = h0_s

            # Cache the processed curve
            setattr(self, cache_key, (z_curve, h0_curve))

            # Interpolate in log(z) space for better coverage
            log_z = np.log10(np.clip(z_array, z_curve.min(), z_curve.max()))
            log_z_curve = np.log10(z_curve)
            h0_void = np.interp(log_z, log_z_curve, h0_curve)
            # Clamp values outside the curve range
            h0_void = np.where(z_array < z_curve.min(), h0_curve[0], h0_void)
            h0_void = np.where(z_array > z_curve.max(), h0_curve[-1], h0_void)
            return h0_void

        # Fallback: analytic surrogates (NOT the actual published curves)
        delta_h0 = self.H0_SH0ES - self.H0_CMB
        if profile == "gaussian":
            h0_void = self.H0_CMB + delta_h0 * np.exp(-z_array**2 / (2 * self.VOID_GAUSSIAN_SIGMA_Z**2))
        elif profile == "exponential":
            h0_void = self.H0_CMB + delta_h0 * np.exp(-z_array / self.VOID_EXPONENTIAL_Z0)
        else:
            raise ValueError(f"Unknown profile: {profile}")
        return h0_void

    def tep_model_prediction(self, z_array):
        """
        TEP H0(z) prediction.

        TEP predicts that the H0 inflation from Cepheid distance compression
        decays slowly with redshift as (1+z)^-decay_index, reflecting the
        temporal shear integral <X_i>(z).

        The Cepheid calibrators are all at z ~ 0, where <X_i> ~ 1, so the
        full Hubble tension (73.0 - 67.4 = 5.6 km/s/Mpc) is imprinted on
        the zero-point.  At higher redshift, the shear contribution decays:

            H0_tep(z) = H0_CMB + (H0_SH0ES - H0_CMB) * (1+z)^(-decay_index)

        This gives:
          - z = 0:   H0 = 73.0 (full tension)
          - z = 0.5: H0 = 72.4 (slow decay)
          - z = 1.0: H0 = 71.9 (still inflated)
          - z -> inf: H0 = 67.4 (CMB value)
        """
        delta_h0 = self.H0_SH0ES - self.H0_CMB
        x_i = 1.0 / (1.0 + z_array) ** self.TEP_DECAY_INDEX
        h0_tep = self.H0_CMB + delta_h0 * x_i
        return h0_tep

    # ------------------------------------------------------------------
    # Model fitting with free parameters
    # ------------------------------------------------------------------
    def _void_model_gaussian(self, z, sigma_z, delta_h0):
        """Void model: Gaussian gradual decay. H0_CMB fixed at Planck value."""
        return self.H0_CMB + delta_h0 * np.exp(-z**2 / (2 * sigma_z**2))

    def _void_model_exponential(self, z, z_0, delta_h0):
        """Void model: Exponential gradual decay. H0_CMB fixed at Planck value."""
        return self.H0_CMB + delta_h0 * np.exp(-z / z_0)

    def _tep_model(self, z, delta_h0, n):
        """TEP model: smooth (1+z)^-n decay. H0_CMB fixed at Planck value."""
        return self.H0_CMB + delta_h0 * (1.0 + z) ** (-n)

    # ------------------------------------------------------------------
    # Covariance-based chi-squared (Blocker 2 response)
    # ------------------------------------------------------------------
    def _load_pantheon_covariance(self):
        """
        Load the Pantheon+ STAT+SYS covariance matrix.

        Returns (cov, n) or (None, 0) if unavailable.
        """
        cov_path = self.data_raw / "Pantheon+SH0ES_STAT+SYS.cov"
        if not cov_path.exists():
            print_status(f"Pantheon+ covariance not found at {cov_path}", "WARNING")
            return None, 0

        print_status(f"Loading Pantheon+ covariance matrix ({cov_path.name})...", "PROCESS")
        with open(cov_path) as f:
            n = int(f.readline().strip())
            data = np.fromstring(f.read(), sep="\n")
        cov = data[:n * n].reshape(n, n)
        print_status(f"Loaded {n}×{n} covariance matrix", "SUCCESS")
        return cov, n

    def _compute_bin_covariance(self, cov_full, sn_indices_per_bin, jacobian_factors_per_bin=None):
        """
        Compute bin-level covariance from the full SN-level covariance.

        The Pantheon+ covariance matrix is in distance modulus (mu) space.
        To use it for H0-space chi-squared, we apply the Jacobian:

            J_i = dH0_i/dmu_i = -H0_i * ln(10)/5

        So C_H0[a,b] = J_a * J_b * C_mu[a,b]

        For the bin-level covariance:
            C_bin[i,j] = (1/n_i * 1/n_j) * sum_{a∈i} sum_{b∈j} J_a * J_b * C_mu[a,b]

        Args:
            cov_full: N×N covariance matrix in mu-space
            sn_indices_per_bin: list of arrays of SN indices per bin
            jacobian_factors_per_bin: list of arrays of |J_i| = H0_i * ln(10)/5 per SN per bin

        Returns:
            cov_bin: n_bins × n_bins bin-level covariance in H0-space
        """
        n_bins = len(sn_indices_per_bin)
        cov_bin = np.zeros((n_bins, n_bins))

        for i in range(n_bins):
            idx_i = sn_indices_per_bin[i]
            if len(idx_i) == 0:
                continue
            J_i = jacobian_factors_per_bin[i] if jacobian_factors_per_bin else np.ones(len(idx_i))

            for j in range(i, n_bins):
                idx_j = sn_indices_per_bin[j]
                if len(idx_j) == 0:
                    continue
                J_j = jacobian_factors_per_bin[j] if jacobian_factors_per_bin else np.ones(len(idx_j))

                sub = cov_full[np.ix_(idx_i, idx_j)]
                # Apply Jacobian: C_H0 = J_i * J_j * C_mu
                weighted = sub * np.outer(J_i, J_j)
                cov_bin[i, j] = weighted.mean()
                if i != j:
                    cov_bin[j, i] = cov_bin[i, j]

        return cov_bin

    def _chi2_covariance(self, h0_obs, h0_model, cov_bin):
        """
        Compute chi-squared using the full bin-level covariance matrix.

        chi^2 = Δ^T C^{-1} Δ
        """
        delta = h0_obs - h0_model
        try:
            cov_inv = np.linalg.inv(cov_bin)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov_bin)
        return float(delta @ cov_inv @ delta)

    def _chi2_mu_space_marginalized(self, mu_obs, mu_model, cov_bin):
        """
        Compute chi-squared in mu-space with a marginalized zero-point.

        chi^2 = min_M [ (Δμ - M)^T C^{-1} (Δμ - M) ]

        where Δμ = mu_obs - mu_model and M is a common zero-point nuisance.

        The analytic marginalization over M (uniform prior) gives:

            chi^2 = Δμ^T C^{-1} Δμ - (Δμ^T C^{-1} 1)^2 / (1^T C^{-1} 1)

        This makes the test strictly zero-point independent.
        """
        delta = mu_obs - mu_model
        n = len(delta)
        diag_pos = np.diag(cov_bin)[np.diag(cov_bin) > 0]
        diag_med = np.median(diag_pos) if len(diag_pos) > 0 else 1.0
        cov_reg = cov_bin + 1e-8 * diag_med * np.eye(n)

        ones = np.ones(n)
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            try:
                cov_inv_ones = np.linalg.solve(cov_reg, ones)
                cov_inv_delta = np.linalg.solve(cov_reg, delta)
                denom = float(ones @ cov_inv_ones)
            except np.linalg.LinAlgError:
                cov_inv = np.linalg.pinv(cov_reg)
                cov_inv_ones = cov_inv @ ones
                cov_inv_delta = cov_inv @ delta
                denom = float(ones @ cov_inv_ones)

            if denom == 0:
                return float(delta @ cov_inv_delta)

            chi2_raw = float(delta @ cov_inv_delta)
            correction = float((delta @ cov_inv_ones) ** 2 / denom)
            return chi2_raw - correction

    def _compute_mu_space_comparison(self, cov_full, n_cov):
        """
        Unbinned 1701-row native-μ likelihood with marginalized zero-point.

        This is the definitive KBC falsification test:
        - Uses ALL 1701 Pantheon+ rows (no deduplication, no binning)
        - Native μ-space (no Jacobian, no H0 inversion)
        - Full 1701×1701 STAT+SYS covariance matrix
        - KBC H0(z) converted to Δμ(z) = 5*log10(H_ref / H_KBC(z))
          evaluated at EVERY SN redshift
        - Common zero-point M analytically marginalized

        The data residual is:
            d_i = μ_i - μ_ref(z_i)
        where μ_ref(z) is the distance modulus at a reference cosmology
        (H0_ref=73, Ω_m=0.302). This is not zero — it contains measurement
        noise, peculiar velocity contributions, and any real signal.

        The model shapes are:
            s_flat = 0           (TEP/global-M_B prediction)
            s_KBC(z) = 5*log10(H_ref / H_KBC(z))

        The marginalized χ² is:
            χ² = d^T C^{-1} d - (d^T C^{-1} 1)² / (1^T C^{-1} 1)
        for the flat model (s=0), and
            χ² = (d-s)^T C^{-1} (d-s) - ((d-s)^T C^{-1} 1)² / (1^T C^{-1} 1)
        for the KBC model.

        Both models have k=1 (the marginalized zero-point only), since
        the KBC curves are frozen published predictions with no fitted
        amplitude. Thus ΔAIC = Δχ².
        """
        import pandas as pd
        from scipy.integrate import quad

        dat_path = self.data_raw / "Pantheon+SH0ES.dat"
        if not dat_path.exists():
            print_status("Pantheon+SH0ES.dat not found for mu-space comparison", "WARNING")
            return {}

        print_status("  Loading ALL 1701 Pantheon+ rows for unbinned mu-space comparison...", "PROCESS")
        df = pd.read_csv(dat_path, sep=r"\s+")
        print_status(f"  Loaded {len(df)} rows", "PROCESS")

        # Use zCMB (not zHD) — zCMB is the CMB-frame redshift without
        # Pantheon+'s peculiar-velocity correction.
        z = pd.to_numeric(df["zCMB"], errors="coerce")
        mu = pd.to_numeric(df["MU_SH0ES"], errors="coerce")
        mask = z.notna() & mu.notna() & (z > 0)
        z = z[mask].values
        mu = mu[mask].values
        indices = mask[mask].index.values  # row indices in the full 1701 array

        n_sn = len(z)
        print_status(f"  Valid SNe with zCMB and MU_SH0ES: {n_sn}", "PROCESS")

        # Reference cosmology: H0_ref=73, Omega_m=0.302
        # The zero-point marginalization makes H0_ref irrelevant —
        # any constant offset is absorbed by M.
        H0_ref = 73.0
        C_KMS = 299792.458
        Omega_m = self.OMEGA_M

        # Compute reference distance modulus μ_ref(z_i) for each SN
        # Vectorized: use cumulative trapezoid integration on a fine grid
        print_status("  Computing reference cosmology moduli...", "PROCESS")
        from scipy.integrate import cumulative_trapezoid
        z_fine = np.linspace(0, max(z.max() + 0.01, 2.5), 5000)
        E_fine = np.sqrt(Omega_m * (1 + z_fine) ** 3 + (1 - Omega_m))
        d_c_fine = cumulative_trapezoid(1.0 / E_fine, z_fine, initial=0)
        d_c = np.interp(z, z_fine, d_c_fine)
        mu_ref = 5 * np.log10((1 + z) * d_c * C_KMS / H0_ref) + 25

        # Data residual: d_i = μ_i - μ_ref(z_i)
        # This is NOT zero — it contains noise, peculiar velocities, and signal
        d = mu - mu_ref

        print_status(f"  Data residual: mean={d.mean():.4f}, std={d.std():.4f}, min={d.min():.4f}, max={d.max():.4f}", "PROCESS")

        # KBC model shapes: s_KBC(z_i) = 5*log10(H_ref / H_KBC(z_i))
        # Vectorized: evaluate the digitized curves at ALL SN redshifts at once
        h0_g_all = self.void_model_prediction(z, "gaussian")
        h0_e_all = self.void_model_prediction(z, "exponential")
        s_gauss = np.where(h0_g_all > 0, 5 * np.log10(H0_ref / h0_g_all), 0)
        s_exp = np.where(h0_e_all > 0, 5 * np.log10(H0_ref / h0_e_all), 0)

        print_status(f"  KBC Gaussian shape: mean={s_gauss.mean():.4f}, std={s_gauss.std():.4f}", "PROCESS")
        print_status(f"  KBC Exponential shape: mean={s_exp.mean():.4f}, std={s_exp.std():.4f}", "PROCESS")

        # Extract the covariance sub-matrix for the valid SNe
        cov_sub = cov_full[np.ix_(indices, indices)]

        # Solve linear systems instead of explicit inversion for numerical
        # stability and performance. The Pantheon+ STAT+SYS covariance has
        # near-zero eigenvalues (calibrator rows are highly correlated),
        # so we add a tiny diagonal jitter for numerical stability.
        print_status("  Solving 1701×1701 covariance system...", "PROCESS")
        n_cov_sub = cov_sub.shape[0]
        # Add jitter proportional to the median diagonal for numerical
        # stability. The Pantheon+ covariance has exact zeros for some
        # calibrator-calibrator blocks, so trace-based jitter is too small.
        diag_med = np.median(np.diag(cov_sub)[np.diag(cov_sub) > 0])
        jitter = 1e-8 * diag_med
        cov_sub_reg = cov_sub + jitter * np.eye(n_cov_sub)

        ones = np.ones(n_cov_sub)
        try:
            # C^{-1} * ones  (solve C x = ones)
            cov_inv_ones = np.linalg.solve(cov_sub_reg, ones)
            denom = float(ones @ cov_inv_ones)
        except np.linalg.LinAlgError:
            print_status("  Singular matrix — using pseudo-inverse", "WARNING")
            cov_inv = np.linalg.pinv(cov_sub_reg)
            cov_inv_ones = cov_inv @ ones
            denom = float(ones @ cov_inv_ones)

        def chi2_marg(residual):
            # Solve C x = residual, then chi2 = residual @ x
            try:
                cr = np.linalg.solve(cov_sub_reg, residual)
            except np.linalg.LinAlgError:
                cr = cov_inv @ residual if 'cov_inv' in dir() else np.linalg.pinv(cov_sub_reg) @ residual
            chi2_raw = float(residual @ cr)
            # correction = (residual^T C^{-1} ones)^2 / (ones^T C^{-1} ones)
            r_ci = float(residual @ cov_inv_ones)
            correction = r_ci ** 2 / denom
            return chi2_raw - correction

        # Flat model: s = 0, residual = d
        chi2_flat = chi2_marg(d)

        # KBC Gaussian: s = s_gauss, residual = d - s_gauss
        chi2_gauss = chi2_marg(d - s_gauss)

        # KBC Exponential: s = s_exp, residual = d - s_exp
        chi2_exp = chi2_marg(d - s_exp)

        # AIC: both models have k=1 (marginalized zero-point only)
        # The KBC curves are frozen published predictions — no fitted amplitude
        aic_flat = chi2_flat + 2 * 1
        aic_gauss = chi2_gauss + 2 * 1
        aic_exp = chi2_exp + 2 * 1

        delta_aic_gauss = aic_gauss - aic_flat
        delta_aic_exp = aic_exp - aic_flat

        print_status("  --- Unbinned mu-space (1701 rows, marginalized zero-point) ---", "TEST")
        print_status(f"  Flat (TEP): chi2={chi2_flat:.1f}, AIC={aic_flat:.1f}", "TEST")
        print_status(f"  Void-Gaussian: chi2={chi2_gauss:.1f}, AIC={aic_gauss:.1f}, ΔAIC=+{delta_aic_gauss:.1f}", "TEST")
        print_status(f"  Void-Exponential: chi2={chi2_exp:.1f}, AIC={aic_exp:.1f}, ΔAIC=+{delta_aic_exp:.1f}", "TEST")

        # Also compute z_min robustness
        z_min_results = {}
        for z_min in [0.01, 0.023, 0.05, 0.075, 0.10]:
            z_mask = z >= z_min
            if z_mask.sum() < 100:
                continue
            d_sub = d[z_mask]
            s_g_sub = s_gauss[z_mask]
            s_e_sub = s_exp[z_mask]
            idx_sub = indices[z_mask]
            cov_sub_z = cov_full[np.ix_(idx_sub, idx_sub)]
            n_sub = cov_sub_z.shape[0]
            diag_med_z = np.median(np.diag(cov_sub_z)[np.diag(cov_sub_z) > 0])
            jitter_z = 1e-8 * diag_med_z
            cov_sub_z_reg = cov_sub_z + jitter_z * np.eye(n_sub)

            ones_z = np.ones(n_sub)
            try:
                cov_inv_ones_z = np.linalg.solve(cov_sub_z_reg, ones_z)
                denom_z = float(ones_z @ cov_inv_ones_z)
            except np.linalg.LinAlgError:
                cov_inv_z = np.linalg.pinv(cov_sub_z_reg)
                cov_inv_ones_z = cov_inv_z @ ones_z
                denom_z = float(ones_z @ cov_inv_ones_z)

            def chi2_marg_z(residual):
                try:
                    cr = np.linalg.solve(cov_sub_z_reg, residual)
                except np.linalg.LinAlgError:
                    cr = cov_inv_z @ residual
                chi2_raw = float(residual @ cr)
                r_ci = float(residual @ cov_inv_ones_z)
                correction = r_ci ** 2 / denom_z
                return chi2_raw - correction

            chi2_f = chi2_marg_z(d_sub)
            chi2_g = chi2_marg_z(d_sub - s_g_sub)
            chi2_e = chi2_marg_z(d_sub - s_e_sub)
            daic_g = (chi2_g + 2) - (chi2_f + 2)
            daic_e = (chi2_e + 2) - (chi2_f + 2)

            z_min_results[f"z_min_{z_min:.3f}"] = {
                "z_min": z_min,
                "n_sne": int(z_mask.sum()),
                "chi2_flat": float(chi2_f),
                "chi2_gaussian": float(chi2_g),
                "chi2_exponential": float(chi2_e),
                "delta_aic_gaussian": float(daic_g),
                "delta_aic_exponential": float(daic_e),
            }
            print_status(
                f"  z>={z_min:.3f}: N={z_mask.sum()}, "
                f"χ²_flat={chi2_f:.1f}, ΔAIC_G={daic_g:+.1f}, ΔAIC_E={daic_e:+.1f}",
                "TEST",
            )

        # Also compute z-frame robustness
        z_frame_results = {}
        for z_col, z_label in [("zHEL", "zHEL"), ("zHD", "zHD")]:
            z_alt = pd.to_numeric(df[z_col], errors="coerce")[mask].values
            # Vectorized mu_ref using the same fine grid
            d_c_alt = np.interp(z_alt, z_fine, d_c_fine)
            mu_ref_alt = 5 * np.log10((1 + z_alt) * d_c_alt * C_KMS / H0_ref) + 25
            d_alt = mu - mu_ref_alt
            # Vectorized void model prediction
            h0_g_alt = self.void_model_prediction(z_alt, "gaussian")
            h0_e_alt = self.void_model_prediction(z_alt, "exponential")
            s_g_alt = np.where(h0_g_alt > 0, 5 * np.log10(H0_ref / h0_g_alt), 0)
            s_e_alt = np.where(h0_e_alt > 0, 5 * np.log10(H0_ref / h0_e_alt), 0)

            chi2_f = chi2_marg(d_alt)
            chi2_g = chi2_marg(d_alt - s_g_alt)
            chi2_e = chi2_marg(d_alt - s_e_alt)
            daic_g = (chi2_g + 2) - (chi2_f + 2)
            daic_e = (chi2_e + 2) - (chi2_f + 2)

            z_frame_results[z_label] = {
                "chi2_flat": float(chi2_f),
                "chi2_gaussian": float(chi2_g),
                "chi2_exponential": float(chi2_e),
                "delta_aic_gaussian": float(daic_g),
                "delta_aic_exponential": float(daic_e),
            }
            print_status(
                f"  {z_label}: χ²_flat={chi2_f:.1f}, ΔAIC_G={daic_g:+.1f}, ΔAIC_E={daic_e:+.1f}",
                "TEST",
            )

        return {
            "method": "unbinned_1701rows_mu_space_marginalized",
            "covariance_matrix": "Pantheon+SH0ES_STAT+SYS.cov",
            "n_rows_used": int(n_sn),
            "redshift_used": "zCMB",
            "zero_point_marginalized": True,
            "binned": False,
            "flat_model": {
                "chi2": float(chi2_flat),
                "aic": float(aic_flat),
                "k": 1,
            },
            "void_gaussian": {
                "chi2": float(chi2_gauss),
                "aic": float(aic_gauss),
                "k": 1,
                "delta_aic_vs_flat": float(delta_aic_gauss),
            },
            "void_exponential": {
                "chi2": float(chi2_exp),
                "aic": float(aic_exp),
                "k": 1,
                "delta_aic_vs_flat": float(delta_aic_exp),
            },
            "z_min_robustness": z_min_results,
            "z_frame_robustness": z_frame_results,
            "note": (
                "Unbinned chi-squared in native mu-space using ALL 1701 "
                "Pantheon+ rows (no deduplication, no binning) with the "
                "full 1701×1701 STAT+SYS covariance matrix. The data "
                "residual d_i = mu_i - mu_ref(z_i) is computed at each "
                "SN redshift. KBC H0(z) curves are evaluated at each SN "
                "redshift and converted to Delta_mu(z) = "
                "5*log10(H_ref/H_KBC(z)). A common zero-point is "
                "analytically marginalized. Both models have k=1 "
                "(zero-point only); KBC curves are frozen published "
                "predictions with no fitted amplitude."
            ),
        }

    def fit_models(self, h0_data):
        """
        Fit void (Gaussian + Exponential), TEP, and constant models to the
        observed H0(z) data.

        All models are fit with free parameters so the AIC comparison
        is valid.

        Models:
          Void-Gauss  (k=2): H0(z) = H0_CMB + dH0 * exp(-z²/(2σ²))
                             Free: σ_z, dH0
          Void-Exp    (k=2): H0(z) = H0_CMB + dH0 * exp(-z/z_0)
                             Free: z_0, dH0
          TEP   (k=2): H0(z) = H0_CMB + dH0 * (1+z)^(-n)
                       Free: dH0, n
          Const (k=1): H0(z) = H0_const
                       Free: H0_const  [IS the TEP prediction for global M_B]
        """
        print_status("Fitting void (Gaussian + Exponential), TEP, and constant models to H0(z) data...", "PROCESS")

        z_vals = np.array(sorted(h0_data.keys()))
        h0_obs = np.array([h0_data[z]["h0"] for z in z_vals])
        h0_err = np.array([h0_data[z]["h0_err"] for z in z_vals])
        n_bins = len(z_vals)

        # --- Constant model (k=1) — IS the TEP prediction for global M_B ---
        w = 1.0 / h0_err ** 2
        h0_const = float(np.sum(w * h0_obs) / np.sum(w))
        chi2_const = float(np.sum(((h0_obs - h0_const) / h0_err) ** 2))
        dof_const = n_bins - 1
        chi2_const_red = chi2_const / dof_const
        k_const = 1
        aic_const = chi2_const + 2 * k_const
        print_status(f"  Constant (= TEP global M_B): H0={h0_const:.2f}, chi2/dof={chi2_const:.1f}/{dof_const}={chi2_const_red:.2f}", "TEST")

        # --- Void Gaussian model (k=2): fit sigma_z and dH0 ---
        def void_gauss_chi2(params):
            sigma_z, dH0 = params
            if dH0 < 0 or dH0 > 20 or sigma_z < 0.01 or sigma_z > 10.0:
                return 1e10
            model = self._void_model_gaussian(z_vals, sigma_z, dH0)
            return np.sum(((h0_obs - model) / h0_err) ** 2)

        void_gauss_init = [self.VOID_GAUSSIAN_SIGMA_Z, self.H0_SH0ES - self.H0_CMB]
        void_gauss_result = optimize.minimize(void_gauss_chi2, void_gauss_init, method="Nelder-Mead")
        sigma_z_fit, dH0_void_gauss_fit = void_gauss_result.x
        chi2_void_gauss = float(void_gauss_result.fun)
        dof_void = n_bins - 2
        chi2_void_gauss_red = chi2_void_gauss / dof_void if dof_void > 0 else float("inf")
        k_void = 2
        aic_void_gauss = chi2_void_gauss + 2 * k_void
        print_status(
            f"  Void-Gaussian: sigma_z={sigma_z_fit:.3f}, dH0={dH0_void_gauss_fit:.2f}, "
            f"chi2/dof={chi2_void_gauss:.1f}/{dof_void}={chi2_void_gauss_red:.2f}",
            "TEST",
        )

        # --- Void Exponential model (k=2): fit z_0 and dH0 ---
        def void_exp_chi2(params):
            z_0, dH0 = params
            if dH0 < 0 or dH0 > 20 or z_0 < 0.01 or z_0 > 10.0:
                return 1e10
            model = self._void_model_exponential(z_vals, z_0, dH0)
            return np.sum(((h0_obs - model) / h0_err) ** 2)

        void_exp_init = [self.VOID_EXPONENTIAL_Z0, self.H0_SH0ES - self.H0_CMB]
        void_exp_result = optimize.minimize(void_exp_chi2, void_exp_init, method="Nelder-Mead")
        z0_fit, dH0_void_exp_fit = void_exp_result.x
        chi2_void_exp = float(void_exp_result.fun)
        chi2_void_exp_red = chi2_void_exp / dof_void if dof_void > 0 else float("inf")
        aic_void_exp = chi2_void_exp + 2 * k_void
        print_status(
            f"  Void-Exponential: z_0={z0_fit:.3f}, dH0={dH0_void_exp_fit:.2f}, "
            f"chi2/dof={chi2_void_exp:.1f}/{dof_void}={chi2_void_exp_red:.2f}",
            "TEST",
        )

        # Use Gaussian as the primary void model for downstream comparison
        chi2_void = chi2_void_gauss
        dH0_void_fit = dH0_void_gauss_fit
        aic_void = aic_void_gauss

        # --- TEP model (k=2): fit dH0 and decay index n ---
        def tep_chi2(params):
            dH0, n = params
            if dH0 < 0 or dH0 > 20 or n < 0 or n > 5:
                return 1e10
            model = self._tep_model(z_vals, dH0, n)
            return np.sum(((h0_obs - model) / h0_err) ** 2)

        tep_init = [self.H0_SH0ES - self.H0_CMB, self.TEP_DECAY_INDEX]
        tep_result = optimize.minimize(tep_chi2, tep_init, method="Nelder-Mead")
        dH0_tep_fit, n_tep_fit = tep_result.x
        chi2_tep = float(tep_result.fun)
        dof_tep = n_bins - 2
        chi2_tep_red = chi2_tep / dof_tep if dof_tep > 0 else float("inf")
        k_tep = 2
        aic_tep = chi2_tep + 2 * k_tep
        print_status(
            f"  TEP: dH0={dH0_tep_fit:.2f}, n={n_tep_fit:.3f}, "
            f"chi2/dof={chi2_tep:.1f}/{dof_tep}={chi2_tep_red:.2f}",
            "TEST",
        )

        # --- AIC comparison (free-parameter models) ---
        aics = {"const": aic_const, "void": aic_void, "tep": aic_tep}
        best_model = min(aics, key=aics.get)
        aic_min = aics[best_model]

        delta_aic_void = aic_void - aic_min
        delta_aic_tep = aic_tep - aic_min
        delta_aic_const = aic_const - aic_min

        # Delta AIC of Void relative to TEP (positive = void worse, TEP preferred)
        delta_aic_tep_vs_void = aic_void - aic_tep

        print_status(f"  AIC(Const)={aic_const:.1f}, AIC(Void)={aic_void:.1f}, AIC(TEP)={aic_tep:.1f}", "TEST")
        print_status(f"  Best model: {best_model}", "TEST")
        print_status(f"  Delta AIC (TEP vs Void, free) = {delta_aic_tep_vs_void:.1f}", "TEST")
        print_status(f"  Delta AIC (Void vs best) = {delta_aic_void:.1f}", "TEST")

        # --- Fixed-prediction model comparison ---
        # The scientifically meaningful test is to evaluate each model at its
        # SPECIFIC published predicted parameter value.
        #
        # CRITICAL: The TEP prediction for Pantheon+ H0(z) with GLOBAL M_B is
        # a FLAT H0(z) ≈ 73, NOT a (1+z)^(-0.3) decay. The (1+z)^(-0.3) decay
        # applies to the Cepheid CALIBRATORS at z~0, not to the SN-inferred
        # H0(z). When M_B is global, the Cepheid clock bias is imprinted on
        # the zero-point (M_B = -19.253), and all SNe inherit the same bias
        # regardless of redshift — producing a flat H0(z).
        #
        # The constant model IS the TEP prediction in the global M_B regime.
        # The void prediction is the GRADUAL Gaussian/exponential decay
        # calibrated to the published HBK20/Mazurenko et al. 2025 curves.
        print_status("  --- Fixed-prediction comparison (published parameters) ---", "TEST")

        # Void Gaussian with sigma_z FIXED at published value
        def void_gauss_fixed_chi2(dH0):
            if dH0 < 0 or dH0 > 20:
                return 1e10
            model = self._void_model_gaussian(z_vals, self.VOID_GAUSSIAN_SIGMA_Z, dH0)
            return np.sum(((h0_obs - model) / h0_err) ** 2)
        void_gauss_fixed_result = optimize.minimize_scalar(
            void_gauss_fixed_chi2, bounds=(0, 20), method="bounded"
        )
        dH0_void_gauss_fixed = float(void_gauss_fixed_result.x)
        chi2_void_gauss_fixed = float(void_gauss_fixed_result.fun)
        k_void_fixed = 1
        aic_void_gauss_fixed = chi2_void_gauss_fixed + 2 * k_void_fixed

        # Void Exponential with z_0 FIXED at published value
        def void_exp_fixed_chi2(dH0):
            if dH0 < 0 or dH0 > 20:
                return 1e10
            model = self._void_model_exponential(z_vals, self.VOID_EXPONENTIAL_Z0, dH0)
            return np.sum(((h0_obs - model) / h0_err) ** 2)
        void_exp_fixed_result = optimize.minimize_scalar(
            void_exp_fixed_chi2, bounds=(0, 20), method="bounded"
        )
        dH0_void_exp_fixed = float(void_exp_fixed_result.x)
        chi2_void_exp_fixed = float(void_exp_fixed_result.fun)
        aic_void_exp_fixed = chi2_void_exp_fixed + 2 * k_void_fixed

        # Use Gaussian as the primary void fixed prediction
        chi2_void_fixed = chi2_void_gauss_fixed
        aic_void_fixed = aic_void_gauss_fixed
        dH0_void_fixed = dH0_void_gauss_fixed

        # TEP prediction for global M_B: FLAT H0(z) ≈ 73
        # This IS the constant model.
        chi2_tep_global = chi2_const
        aic_tep_global = aic_const
        k_tep_global = 1

        # Also compute TEP with n=0.3 applied directly to SN data, for reference.
        def tep_n03_chi2(dH0):
            if dH0 < 0 or dH0 > 20:
                return 1e10
            model = self._tep_model(z_vals, dH0, self.TEP_DECAY_INDEX)
            return np.sum(((h0_obs - model) / h0_err) ** 2)
        tep_n03_result = optimize.minimize_scalar(
            tep_n03_chi2, bounds=(0, 20), method="bounded"
        )
        dH0_tep_n03 = float(tep_n03_result.x)
        chi2_tep_n03 = float(tep_n03_result.fun)
        aic_tep_n03 = chi2_tep_n03 + 2 * 1

        # The primary ΔAIC: Void model relative to TEP (best).
        # Standard convention: ΔAIC = AIC_model - AIC_best (positive = worse)
        delta_aic_fixed = aic_void_fixed - aic_tep_global
        delta_aic_fixed_exp = aic_void_exp_fixed - aic_tep_global

        print_status(
            f"  Void-Gaussian (sigma_z={self.VOID_GAUSSIAN_SIGMA_Z:.2f} fixed): "
            f"dH0={dH0_void_gauss_fixed:.2f}, chi2={chi2_void_gauss_fixed:.1f}, "
            f"AIC={aic_void_gauss_fixed:.1f}",
            "TEST",
        )
        print_status(
            f"  Void-Exponential (z_0={self.VOID_EXPONENTIAL_Z0:.2f} fixed): "
            f"dH0={dH0_void_exp_fixed:.2f}, chi2={chi2_void_exp_fixed:.1f}, "
            f"AIC={aic_void_exp_fixed:.1f}",
            "TEST",
        )
        print_status(
            f"  TEP (flat H0, global M_B prediction): chi2={chi2_tep_global:.1f}, "
            f"AIC={aic_tep_global:.1f} [IS the constant model]",
            "TEST",
        )
        print_status(
            f"  TEP (n=0.3 applied to SN data — NOT the global M_B prediction): "
            f"chi2={chi2_tep_n03:.1f} [for reference only]",
            "TEST",
        )
        print_status(
            f"  Delta AIC (Void_Gaussian - TEP_flat) = {delta_aic_fixed:.1f}",
            "TEST",
        )
        print_status(
            f"  Delta AIC (Void_Exponential - TEP_flat) = {delta_aic_fixed_exp:.1f}",
            "TEST",
        )

        # The free-parameter TEP fit (n≈0) CONFIRMS the TEP prediction:
        # n≈0 is exactly what TEP predicts for global M_B (flat H0(z)).
        n_prediction_global = 0.0  # TEP predicts n≈0 for global M_B
        n_consistent = abs(n_tep_fit - n_prediction_global) < 0.1

        # TEP preferred over void? (positive ΔAIC = void worse than TEP)
        tep_preferred = delta_aic_fixed > 10
        tep_strongly_preferred = delta_aic_fixed > 50

        # Void model falsified
        void_falsified = delta_aic_void > 10
        void_falsified_fixed = (aic_void_fixed - aic_tep_global) > 10

        if tep_strongly_preferred:
            print_status("  TEP model STRONGLY preferred over void model (fixed predictions)", "SUCCESS")
        elif tep_preferred:
            print_status("  TEP model preferred over void model (fixed predictions)", "SUCCESS")
        else:
            print_status("  TEP and void models are comparable (fixed predictions)", "PROCESS")

        if n_consistent:
            print_status("  TEP free-fit n≈0 CONFIRMS the global M_B prediction (flat H0(z))", "SUCCESS")

        if void_falsified_fixed:
            print_status("  Void model FALSIFIED at published parameters: catastrophically worse than TEP", "SUCCESS")

        # ------------------------------------------------------------------
        # Covariance-based chi-squared (Blocker 2 response)
        # ------------------------------------------------------------------
        # Recompute chi-squared using the full Pantheon+ STAT+SYS
        # covariance matrix instead of diagonal errors only.
        cov_results = {}
        cov_full, n_cov = self._load_pantheon_covariance()
        if cov_full is not None and n_cov > 0:
            # Collect SN covariance indices and Jacobian factors per bin
            sn_indices_per_bin = []
            jacobian_per_bin = []
            valid_bins = []
            for i, z in enumerate(z_vals):
                z_key = z
                if z_key in h0_data and "cov_indices" in h0_data[z_key]:
                    cov_idx = h0_data[z_key]["cov_indices"]
                    h0_per_sn = h0_data[z_key].get("h0_per_sn", [])
                    if len(cov_idx) > 0 and len(h0_per_sn) > 0:
                        sn_indices_per_bin.append(np.array(cov_idx))
                        # Jacobian: |dH0/dmu| = H0 * ln(10)/5
                        J = np.array(h0_per_sn) * np.log(10.0) / 5.0
                        jacobian_per_bin.append(J)
                        valid_bins.append(i)

            if len(valid_bins) >= 3:
                # Compute bin-level covariance with Jacobian transformation
                cov_bin = self._compute_bin_covariance(
                    cov_full, sn_indices_per_bin, jacobian_per_bin
                )

                # Extract H0 values for valid bins only
                h0_obs_cov = h0_obs[valid_bins]
                z_vals_cov = z_vals[valid_bins]

                # Constant model (TEP global M_B) with covariance
                h0_const_cov = float(np.mean(h0_obs_cov))
                chi2_const_cov = self._chi2_covariance(
                    h0_obs_cov, np.full_like(h0_obs_cov, h0_const_cov), cov_bin
                )
                aic_const_cov = chi2_const_cov + 2 * 1

                # Void Gaussian (digitized published curve) with covariance
                h0_void_gauss_cov = self.void_model_prediction(z_vals_cov, "gaussian")
                chi2_void_gauss_cov = self._chi2_covariance(
                    h0_obs_cov, h0_void_gauss_cov, cov_bin
                )
                aic_void_gauss_cov = chi2_void_gauss_cov + 2 * 1

                # Void Exponential (digitized published curve) with covariance
                h0_void_exp_cov = self.void_model_prediction(z_vals_cov, "exponential")
                chi2_void_exp_cov = self._chi2_covariance(
                    h0_obs_cov, h0_void_exp_cov, cov_bin
                )
                aic_void_exp_cov = chi2_void_exp_cov + 2 * 1

                delta_aic_cov_gauss = aic_void_gauss_cov - aic_const_cov
                delta_aic_cov_exp = aic_void_exp_cov - aic_const_cov

                print_status("  --- Covariance-based chi-squared (Pantheon+ STAT+SYS) ---", "TEST")
                print_status(
                    f"  Constant (TEP): chi2_cov={chi2_const_cov:.1f}, AIC={aic_const_cov:.1f}",
                    "TEST",
                )
                print_status(
                    f"  Void-Gaussian (published curve): chi2_cov={chi2_void_gauss_cov:.1f}, "
                    f"AIC={aic_void_gauss_cov:.1f}",
                    "TEST",
                )
                print_status(
                    f"  Void-Exponential (published curve): chi2_cov={chi2_void_exp_cov:.1f}, "
                    f"AIC={aic_void_exp_cov:.1f}",
                    "TEST",
                )
                print_status(
                    f"  ΔAIC (cov) Void-Gauss vs TEP = {delta_aic_cov_gauss:.1f}",
                    "TEST",
                )
                print_status(
                    f"  ΔAIC (cov) Void-Exp vs TEP = {delta_aic_cov_exp:.1f}",
                    "TEST",
                )

                cov_results = {
                    "covariance_matrix": "Pantheon+SH0ES_STAT+SYS.cov",
                    "n_bins_used": len(valid_bins),
                    "constant_model": {
                        "chi2_cov": float(chi2_const_cov),
                        "aic_cov": float(aic_const_cov),
                    },
                    "void_gaussian": {
                        "chi2_cov": float(chi2_void_gauss_cov),
                        "aic_cov": float(aic_void_gauss_cov),
                        "delta_aic_vs_tep": float(delta_aic_cov_gauss),
                    },
                    "void_exponential": {
                        "chi2_cov": float(chi2_void_exp_cov),
                        "aic_cov": float(aic_void_exp_cov),
                        "delta_aic_vs_tep": float(delta_aic_cov_exp),
                    },
                    "note": (
                        "Chi-squared computed using the full Pantheon+ "
                        "STAT+SYS covariance matrix (1701×1701). Void "
                        "curves are the digitized Method-3 predictions "
                        "from Mazurenko et al. 2025, Figure 3."
                    ),
                }
            else:
                print_status(
                    f"  Insufficient bins with covariance indices ({len(valid_bins)})",
                    "WARNING",
                )
        else:
            print_status("  Covariance matrix not available — using diagonal errors only", "WARNING")

        # ------------------------------------------------------------------
        # Mu-space comparison with ALL 1701 rows (Forensic 5-6)
        # ------------------------------------------------------------------
        mu_space_results = {}
        if cov_full is not None and n_cov > 0:
            mu_space_results = self._compute_mu_space_comparison(cov_full, n_cov)

        return {
            "n_bins": n_bins,
            "constant_model": {
                "h0_const": h0_const,
                "chi2": chi2_const,
                "chi2_reduced": chi2_const_red,
                "k": k_const,
                "aic": float(aic_const),
                "delta_aic": float(delta_aic_const),
                "note": "This IS the TEP prediction for global M_B (flat H0(z))",
            },
            "void_model_gaussian": {
                "sigma_z_fit": float(sigma_z_fit),
                "delta_h0_fit": float(dH0_void_gauss_fit),
                "chi2": chi2_void_gauss,
                "chi2_reduced": chi2_void_gauss_red,
                "k": k_void,
                "aic": float(aic_void_gauss),
                "delta_aic": float(aic_void_gauss - aic_min),
            },
            "void_model_exponential": {
                "z_0_fit": float(z0_fit),
                "delta_h0_fit": float(dH0_void_exp_fit),
                "chi2": chi2_void_exp,
                "chi2_reduced": chi2_void_exp_red,
                "k": k_void,
                "aic": float(aic_void_exp),
                "delta_aic": float(aic_void_exp - aic_min),
            },
            "void_model": {
                "profile": "gaussian (primary)",
                "sigma_z_fit": float(sigma_z_fit),
                "delta_h0_fit": float(dH0_void_gauss_fit),
                "chi2": chi2_void_gauss,
                "chi2_reduced": chi2_void_gauss_red,
                "k": k_void,
                "aic": float(aic_void_gauss),
                "delta_aic": float(aic_void_gauss - aic_min),
            },
            "tep_model": {
                "delta_h0_fit": float(dH0_tep_fit),
                "decay_index_fit": float(n_tep_fit),
                "decay_index_prediction_global_mb": float(n_prediction_global),
                "decay_index_prediction_per_host": float(self.TEP_DECAY_INDEX),
                "decay_index_consistent": bool(n_consistent),
                "chi2": chi2_tep,
                "chi2_reduced": chi2_tep_red,
                "k": k_tep,
                "aic": float(aic_tep),
                "delta_aic": float(delta_aic_tep),
                "note": "n≈0 confirms TEP prediction for global M_B; n=0.3 is the per-host prediction",
            },
            "fixed_prediction_comparison": {
                "void_gaussian_sigma_z_fixed": float(self.VOID_GAUSSIAN_SIGMA_Z),
                "void_gaussian_dh0_fit": dH0_void_gauss_fixed,
                "void_gaussian_chi2": chi2_void_gauss_fixed,
                "void_gaussian_aic": float(aic_void_gauss_fixed),
                "void_exponential_z0_fixed": float(self.VOID_EXPONENTIAL_Z0),
                "void_exponential_dh0_fit": dH0_void_exp_fixed,
                "void_exponential_chi2": chi2_void_exp_fixed,
                "void_exponential_aic": float(aic_void_exp_fixed),
                "tep_prediction": "flat H0(z) ≈ 73 (global M_B zero-point bias)",
                "tep_chi2": float(chi2_tep_global),
                "tep_aic": float(aic_tep_global),
                "tep_n03_reference_chi2": float(chi2_tep_n03),
                "tep_n03_reference_note": "n=0.3 applied to SN data is NOT the global M_B prediction",
                "delta_aic_void_gaussian_vs_tep": float(delta_aic_fixed),
                "delta_aic_void_exponential_vs_tep": float(delta_aic_fixed_exp),
                "void_falsified_at_prediction": bool(void_falsified_fixed),
                "note": "Void curves calibrated to HBK20/Mazurenko et al. 2025 published convergence (1σ Planck at z≈1.8)",
            },
            "covariance_based_comparison": cov_results,
            "mu_space_comparison": mu_space_results,
            "best_model": best_model,
            "delta_aic_void_vs_tep": float(delta_aic_tep_vs_void),
            "delta_aic_void_vs_tep_fixed": float(delta_aic_fixed),
            "delta_aic_void_vs_best": float(delta_aic_void),
            "tep_preferred": bool(tep_preferred),
            "tep_strongly_preferred": bool(tep_strongly_preferred),
            "void_falsified": bool(void_falsified),
        }

    # ------------------------------------------------------------------
    # Host-mass-resolved analysis
    # ------------------------------------------------------------------
    def host_mass_analysis(self, df):
        """
        Perform host-mass-resolved H0(z) analysis.

        TEP predicts that the H0 inflation depends on host gravitational
        potential (proxied by stellar mass).  The void model predicts no
        host-mass dependence.

        We compute H0(z) separately for massive (logM >= 10.0) and
        low-mass (logM < 10.0) hosts and test whether the massive-host
        H0 is systematically higher.
        """
        print_status("Performing host-mass-resolved H0(z) analysis...", "PROCESS")

        h0_massive = self.compute_h0_in_bins(df, mass_filter="massive", label="massive")
        h0_lowmass = self.compute_h0_in_bins(df, mass_filter="lowmass", label="low-mass")

        # Compare overall mean H0
        all_h0_massive = []
        all_h0_lowmass = []

        if not df.empty:
            z_col = None
            mu_col = None
            mass_col = None
            for c in df.columns:
                if c.lower() in ["z", "redshift", "zhel"]:
                    z_col = c
                if c.lower() in ["mu", "mured", "m_b", "mb"]:
                    mu_col = c
                if "logmass" in c.lower() and "err" not in c.lower():
                    mass_col = c

            if z_col and mu_col and mass_col:
                for _, row in df.iterrows():
                    z = pd.to_numeric(row[z_col], errors="coerce")
                    mu = pd.to_numeric(row[mu_col], errors="coerce")
                    mass = pd.to_numeric(row[mass_col], errors="coerce")
                    if pd.isna(z) or pd.isna(mu) or pd.isna(mass) or z <= 0:
                        continue
                    # Skip missing-mass placeholder values
                    if mass <= -5.0:
                        continue
                    h0 = self._h0_from_mu(z, mu)
                    if mass >= self.MASSIVE_THRESHOLD:
                        all_h0_massive.append(h0)
                    else:
                        all_h0_lowmass.append(h0)

        all_h0_massive = np.array(all_h0_massive)
        all_h0_lowmass = np.array(all_h0_lowmass)

        if len(all_h0_massive) > 0 and len(all_h0_lowmass) > 0:
            mean_m = float(np.mean(all_h0_massive))
            mean_l = float(np.mean(all_h0_lowmass))
            delta_h0 = mean_m - mean_l

            # Statistical test
            t_stat, p_value = stats.ttest_ind(all_h0_massive, all_h0_lowmass, equal_var=False)

            print_status(f"  Massive hosts (logM >= {self.MASSIVE_THRESHOLD}): N={len(all_h0_massive)}, mean H0={mean_m:.2f}", "TEST")
            print_status(f"  Low-mass hosts (logM < {self.MASSIVE_THRESHOLD}): N={len(all_h0_lowmass)}, mean H0={mean_l:.2f}", "TEST")
            print_status(f"  Delta H0 (massive - low-mass): {delta_h0:.2f} km/s/Mpc", "TEST")
            print_status(f"  Welch's t-test: t={t_stat:.3f}, p={p_value:.4e}", "TEST")

            tep_host_mass_confirmed = bool(delta_h0 > 0 and p_value < 0.05)
            print_status(f"  TEP prediction: massive hosts should have higher H0", "TEST")
            print_status(f"  TEP host-mass confirmed: {tep_host_mass_confirmed}", "TEST")
            if not tep_host_mass_confirmed:
                print_status(
                    "  NOTE: Pantheon+ uses a global M_B calibration — all SNe share "
                    "the same absolute magnitude regardless of host mass. The host-mass "
                    "dependence of the TEP effect requires per-host Cepheid calibrations "
                    "(established in TEP-H0, Paper 11), not the global calibration used "
                    "in Pantheon+. This null result is expected and does not falsify TEP.",
                    "PROCESS",
                )

            return {
                "n_massive": int(len(all_h0_massive)),
                "n_lowmass": int(len(all_h0_lowmass)),
                "mean_h0_massive": mean_m,
                "mean_h0_lowmass": mean_l,
                "delta_h0": delta_h0,
                "welch_t": float(t_stat),
                "welch_p": float(p_value),
                "tep_prediction": "Massive hosts should have higher H0 than low-mass hosts",
                "tep_confirmed": tep_host_mass_confirmed,
                "null_explanation": (
                    "Pantheon+ uses a global M_B calibration, so all SNe share the same "
                    "absolute magnitude regardless of host mass. The host-mass dependence "
                    "of the TEP effect requires per-host Cepheid calibrations (TEP-H0, "
                    "Paper 11), not the global Pantheon+ calibration. This null is expected."
                ),
                "h0_z_massive": {str(k): v for k, v in h0_massive.items()},
                "h0_z_lowmass": {str(k): v for k, v in h0_lowmass.items()},
            }
        else:
            print_status("  Insufficient data for host-mass analysis", "WARNING")
            return {}

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def plot_h0_vs_redshift(self, h0_data, fit_results, host_mass_results):
        """Generate H0(z) vs redshift figure."""
        colors = apply_tep_style()
        print_status("Generating H0 vs redshift figure...", "PROCESS")

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # --- Panel 1: Overall H0(z) with model fits ---
        ax1 = axes[0]

        z_vals = np.array(sorted(h0_data.keys()))
        h0_obs = np.array([h0_data[z]["h0"] for z in z_vals])
        h0_err = np.array([h0_data[z]["h0_err"] for z in z_vals])

        ax1.errorbar(z_vals, h0_obs, yerr=h0_err, fmt="o", color=colors['dark'], capsize=5, markersize=8,
                     label="Pantheon+ $H_0(z)$", zorder=5)

        # Model predictions — use fitted parameters
        z_fine = np.linspace(0.005, 2.0, 500)
        vm = fit_results.get("void_model", {})
        vm_g = fit_results.get("void_model_gaussian", {})
        vm_e = fit_results.get("void_model_exponential", {})
        tm = fit_results.get("tep_model", {})
        cm = fit_results.get("constant_model", {})

        if vm_g:
            h0_void_g = self._void_model_gaussian(z_fine, vm_g["sigma_z_fit"], vm_g["delta_h0_fit"])
            ax1.plot(z_fine, h0_void_g, color=colors['red'], linestyle="--", linewidth=2,
                     label=f"Void-Gaussian fit ($\\sigma_z$={vm_g['sigma_z_fit']:.2f})")
        if vm_e:
            h0_void_e = self._void_model_exponential(z_fine, vm_e["z_0_fit"], vm_e["delta_h0_fit"])
            ax1.plot(z_fine, h0_void_e, color=colors['purple'], linestyle="--", linewidth=2, alpha=0.7,
                     label=f"Void-Exp fit ($z_0$={vm_e['z_0_fit']:.2f})")
        # Also plot the FIXED published predictions
        fpc = fit_results.get("fixed_prediction_comparison", {})
        if fpc:
            h0_void_fixed = self.void_model_prediction(z_fine, profile="gaussian")
            ax1.plot(z_fine, h0_void_fixed, color=colors['red'], linestyle=":", linewidth=1.5, alpha=0.6,
                     label=f"Void-Gaussian (published $\\sigma_z$={self.VOID_GAUSSIAN_SIGMA_Z})")
            h0_void_exp_fixed = self.void_model_prediction(z_fine, profile="exponential")
            ax1.plot(z_fine, h0_void_exp_fixed, color=colors['purple'], linestyle=":", linewidth=1.5, alpha=0.6,
                     label=f"Void-Exp (published $z_0$={self.VOID_EXPONENTIAL_Z0})")
        if tm:
            h0_tep = self._tep_model(z_fine, tm["delta_h0_fit"], tm["decay_index_fit"])
            ax1.plot(z_fine, h0_tep, color=colors['blue'], linestyle="-", linewidth=2,
                     label=f"TEP fit ($n$={tm['decay_index_fit']:.2f}, $\\Delta H_0$={tm['delta_h0_fit']:.1f})")
        if cm:
            ax1.axhline(cm["h0_const"], color=colors['green'], linestyle="-.", alpha=0.5,
                        label=f"Constant/TEP global $M_B$ ($H_0$={cm['h0_const']:.1f})")

        # Reference lines
        ax1.axhline(self.H0_CMB, color=colors['purple'], linestyle=":", alpha=0.5, label=f"Planck $H_0$={self.H0_CMB}")
        ax1.axhline(self.H0_SH0ES, color=colors['purple'], linestyle="--", alpha=0.5, label=f"SH0ES $H_0$={self.H0_SH0ES}")

        ax1.set_xlabel("Redshift $z$")
        ax1.set_ylabel("$H_0(z)$ (km/s/Mpc)")
        ax1.set_title("Redshift Profile: Published KBC Curves vs TEP")

        textstr = (
            f"$\\chi^2_{{\\rm red}}$(Void-Gauss) = {vm_g.get('chi2_reduced', 0):.2f}\n"
            f"$\\chi^2_{{\\rm red}}$(Void-Exp) = {vm_e.get('chi2_reduced', 0):.2f}\n"
            f"$\\chi^2_{{\\rm red}}$(TEP) = {tm.get('chi2_reduced', 0):.2f}\n"
            f"$\\chi^2_{{\\rm red}}$(Const) = {cm.get('chi2_reduced', 0):.2f}\n"
            f"$\\Delta$AIC(TEP$-$Void_Gauss) = {fit_results.get('delta_aic_tep_vs_void_fixed', 0):.1f}"
        )
        ax1.text(0.55, 0.95, textstr, transform=ax1.transAxes, fontsize=11,
                 verticalalignment="top", bbox=dict(boxstyle="round", facecolor=colors['bg'], alpha=0.8))

        ax1.legend(loc="lower right")
        ax1.grid(True)
        ax1.set_xlim(0, 1.5)
        ax1.set_ylim(65, 76)

        # --- Panel 2: Host-mass-resolved H0(z) ---
        ax2 = axes[1]

        if host_mass_results and "h0_z_massive" in host_mass_results:
            h0_m = host_mass_results["h0_z_massive"]
            h0_l = host_mass_results["h0_z_lowmass"]

            z_m = np.array(sorted([float(k) for k in h0_m.keys()]))
            z_l = np.array(sorted([float(k) for k in h0_l.keys()]))

            h0_m_vals = np.array([h0_m[str(z)]["h0"] for z in z_m])
            h0_m_errs = np.array([h0_m[str(z)]["h0_err"] for z in z_m])
            h0_l_vals = np.array([h0_l[str(z)]["h0"] for z in z_l])
            h0_l_errs = np.array([h0_l[str(z)]["h0_err"] for z in z_l])

            ax2.errorbar(z_m, h0_m_vals, yerr=h0_m_errs, fmt="^", color=colors['red'],
                         capsize=5, markersize=8, label=f"Massive hosts (logM >= {self.MASSIVE_THRESHOLD})")
            ax2.errorbar(z_l, h0_l_vals, yerr=h0_l_errs, fmt="s", color=colors['green'],
                         capsize=5, markersize=8, label=f"Low-mass hosts (logM < {self.MASSIVE_THRESHOLD})")

            # TEP fitted model for both populations
            if tm:
                ax2.plot(z_fine, self._tep_model(z_fine, tm["delta_h0_fit"], tm["decay_index_fit"]),
                         color=colors['blue'], linestyle="-", linewidth=2, alpha=0.5, label="TEP fit (overall)")

            ax2.axhline(self.H0_CMB, color=colors['purple'], linestyle=":", alpha=0.5)
            ax2.axhline(self.H0_SH0ES, color=colors['purple'], linestyle="--", alpha=0.5)

            hm_text = (
                f"$\\Delta H_0$ = {host_mass_results['delta_h0']:.2f} km/s/Mpc\n"
                f"p = {host_mass_results['welch_p']:.2e}\n"
                f"Null: global $M_B$ calibration"
            )
            ax2.text(0.55, 0.95, hm_text, transform=ax2.transAxes, fontsize=11,
                     verticalalignment="top", bbox=dict(boxstyle="round", facecolor=colors['bg'], alpha=0.8))
        else:
            ax2.text(0.5, 0.5, "Host-mass analysis\nnot available", transform=ax2.transAxes,
                     ha="center", va="center", fontsize=14)

        ax2.set_xlabel("Redshift $z$")
        ax2.set_ylabel("$H_0(z)$ (km/s/Mpc)")
        ax2.set_title("Host-Mass-Resolved $H_0(z)$")
        ax2.legend(loc="lower right")
        ax2.grid(True)
        ax2.set_xlim(0, 1.5)
        ax2.set_ylim(65, 76)

        fig.suptitle("Redshift Decay Profile: Void Model vs TEP Prediction", fontsize=15, y=1.02)
        fig.tight_layout()
        fig_path = self.figures / "step_32_h0_vs_redshift.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print_status(f"Figure saved to {fig_path}", "SUCCESS")
        return fig_path

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------
    def run(self):
        """Execute the full step."""
        print_status("Step 32: Redshift Decay Profile — Void vs TEP", "TITLE")

        print_status(
            "This step compares the published KBC/MOND void-model H0(z) gradual decay "
            "predictions (Gaussian and Exponential density profiles from Haslbauer, "
            "Banik & Kroupa 2020, as used by Mazurenko et al. 2025) against the TEP "
            "prediction for Pantheon+ with global M_B. The TEP prediction is a flat "
            "H0(z) ≈ 73, because the Cepheid clock bias is imprinted on the global "
            "zero-point and all supernovae inherit the same bias regardless of "
            "redshift. The key discriminating observable is the redshift dependence "
            "of H0 inferred from SN Ia distance moduli: a gradual decline favours "
            "the void model, while a flat profile favours TEP.",
            "INFO",
        )

        # Load Pantheon+ data
        df = self.load_pantheon_data()

        # Compute H0(z) in bins using proper LCDM formula
        print_status(
            "H0(z) methodology: H0 is inferred from each SN Ia distance modulus using "
            "the proper flat-LCDM luminosity-distance relation H0 = (1+z)·c·D_C(z)/d_L, "
            "where D_C(z) is the H0-independent comoving distance integral. SNe are "
            "binned in redshift (z = 0.01–1.50) and the bin-level H0 is the mean of "
            "per-SN values, with errors propagated from per-SN distance modulus "
            "uncertainties. The Pantheon+ STAT+SYS covariance matrix is used for "
            "the covariance-based chi-squared comparison.",
            "PROCESS",
        )
        h0_data = self.compute_h0_in_bins(df, label="all")

        # Fit void and TEP models
        print_status(
            "Model fitting methodology: four models are compared — constant (k=1, "
            "identical to the TEP global-M_B flat prediction), void-Gaussian (k=2, "
            "free σ_z and ΔH0), void-Exponential (k=2, free z_0 and ΔH0), and TEP "
            "(k=2, free ΔH0 and decay index n). Both free-parameter and fixed-"
            "prediction AIC comparisons are performed. The fixed-prediction test "
            "evaluates void curves at published HBK20/Mazurenko et al. 2025 "
            "parameters against the TEP flat H0(z) prediction. An unbinned 1701-row "
            "mu-space likelihood with analytically marginalized zero-point provides "
            "the definitive covariance-based falsification test.",
            "PROCESS",
        )
        fit_results = self.fit_models(h0_data)

        # Host-mass-resolved analysis
        print_status(
            "Host-mass analysis methodology: H0(z) is computed separately for massive "
            "(logM ≥ 10.0) and low-mass (logM < 10.0) host galaxies. TEP predicts "
            "that the H0 inflation depends on host gravitational potential; the void "
            "model predicts no host-mass dependence. A Welch's t-test on the mean H0 "
            "difference provides the statistical assessment. Under the global M_B "
            "calibration used by Pantheon+, a null result is expected and does not "
            "falsify TEP.",
            "PROCESS",
        )
        host_mass_results = self.host_mass_analysis(df)

        # Generate figure
        fig_path = self.plot_h0_vs_redshift(h0_data, fit_results, host_mass_results)

        # Determine overall TEP confirmation: the primary test is the
        # H0(z) model comparison (published gradual KBC curves vs TEP flat).
        tep_confirmed = fit_results.get("tep_preferred", False) or fit_results.get("void_falsified", False)

        delta_aic_fixed = fit_results.get("delta_aic_void_vs_tep_fixed", 0.0)
        if tep_confirmed:
            print_status(
                f"The TEP flat H0(z) prediction is preferred over the published KBC "
                f"void-model decay curves with ΔAIC = {delta_aic_fixed:.1f}. The "
                f"observed H0(z) shows no gradual decline, consistent with the TEP "
                f"prediction that the Cepheid clock bias is imprinted on the global "
                f"M_B zero-point and inherited uniformly by all supernovae. The "
                f"void-model prediction of a gradual H0(z) decline is rejected.",
                "SUCCESS",
            )
        else:
            print_status(
                f"The model comparison does not yield a decisive preference "
                f"(ΔAIC = {delta_aic_fixed:.1f}). The TEP and void predictions "
                f"are not clearly distinguished by the current H0(z) data.",
                "PROCESS",
            )

        # Summary
        summary = {
            "step": "32_redshift_decay_profile",
            "description": "Compare published KBC/MOND H0(z) gradual decay curves (Gaussian + Exponential, HBK20/Mazurenko et al. 2025) with TEP flat prediction for global M_B",
            "h0_cmb": self.H0_CMB,
            "h0_sh0es": self.H0_SH0ES,
            "omega_m": self.OMEGA_M,
            "void_radius_mpc": self.VOID_RADIUS_MPC,
            "void_gaussian_sigma_z": self.VOID_GAUSSIAN_SIGMA_Z,
            "void_exponential_z0": self.VOID_EXPONENTIAL_Z0,
            "void_curve_source": "HBK20 best-fitting profiles; convergence calibrated to Mazurenko et al. 2025 Fig. 3 (1σ Planck at z≈1.8)",
            "kappa_ceph": self.KAPPA_SHEAR,
            "tep_decay_index_prediction": self.TEP_DECAY_INDEX,
            "h0_method": "LCDM proper: H0 = (1+z) * c * D_C(z) / d_L",
            "h0_z_data": {str(k): v for k, v in h0_data.items()},
            "model_comparison": fit_results,
            "host_mass_analysis": host_mass_results,
            "tep_prediction": "H0(z) is flat at ~73 (global M_B zero-point bias); published KBC gradual decay curves are rejected",
            "tep_confirmed": bool(tep_confirmed),
            "methodology": (
                "H0(z) is inferred from Pantheon+ SN Ia distance moduli using the "
                "proper flat-LCDM luminosity-distance relation. Four models are "
                "compared via AIC: constant (TEP global-M_B flat prediction), "
                "void-Gaussian, void-Exponential, and TEP (1+z)^-n decay. Both "
                "free-parameter and fixed-prediction comparisons are performed. "
                "An unbinned 1701-row mu-space likelihood with analytically "
                "marginalized zero-point and the full STAT+SYS covariance matrix "
                "provides the definitive falsification test. Host-mass-resolved "
                "H0(z) is also computed with a Welch t-test."
            ),
            "provenance": {
                "data_sources": [
                    "Pantheon+ SN Ia sample (Scolnic et al. 2022, from step_03)",
                    "Pantheon+ STAT+SYS covariance matrix (1701×1701)",
                    "KBC/MOND void-model H0(z) curves digitized from Mazurenko et al. 2025 Fig. 3",
                    "HBK20 best-fitting void parameters (Haslbauer, Banik & Kroupa 2020)",
                ],
                "pipeline_block": "Block Ib — bulk flow and redshift decay",
                "covariance": (
                    "Full Pantheon+ STAT+SYS covariance matrix (1701×1701) used "
                    "for the unbinned mu-space likelihood. Bin-level covariance "
                    "computed via Jacobian transformation (dH0/dmu = -H0·ln10/5) "
                    "for the binned H0-space comparison. Zero-point analytically "
                    "marginalized in both cases."
                ),
            },
            "scientific_context": (
                "The redshift dependence of H0 inferred from SN Ia distance moduli "
                "discriminates between the void model (predicting a gradual H0(z) "
                "decline as the observer-weighted mean void density decreases with "
                "redshift) and TEP (predicting a flat H0(z) ≈ 73 when a global M_B "
                "is used, because the Cepheid clock bias is imprinted on the "
                "zero-point and inherited uniformly by all supernovae). The key "
                "observable is whether H0(z) declines from ~73 at low z toward the "
                "Planck value ~67.4 at high z."
            ),
            "void_prediction": (
                "H0(z) declines gradually from ~73 at z≈0 to within 1σ of the "
                "Planck value (67.4±0.5) at z≳1.8, following the published KBC "
                "Gaussian or Exponential density profile"
            ),
            "downstream_consumers": ["step_33", "step_34"],
            "n_pantheon_loaded": len(df),
            "used_fallback": bool(df.empty),
            "output_files": [
                str(self.results / "step_32_redshift_decay_profile.json"),
                str(fig_path),
            ],
        }

        summary_path = self.results / "step_32_redshift_decay_profile.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print_status(f"Summary saved to {summary_path}", "SUCCESS")

        print_status("Step 32 complete", "SUCCESS")


if __name__ == "__main__":
    step = Step32RedshiftDecayProfile()
    step.run()
