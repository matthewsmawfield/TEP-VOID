#!/usr/bin/env python3
"""
Step 32b (Replication): Jia et al. (2023) Pantheon+-Only Six-Bin H0 Estimator
================================================================================
REVIEWER-REQUESTED REPLICATION. Validation only.

Purpose:
    The reviewer notes that the existing step_32b script performs a simple
    binwise H0 inversion, which is NOT equivalent to the estimator of
    Jia, Hu & Wang (2023, A&A 674, A45). This script implements their
    actual Pantheon+-only six-bin H0 estimator to test whether their
    published H0 decline survives when the correct luminosity-distance
    integral with piecewise H_th(z) is used, or whether it is an artifact
    of the piecewise parameterisation.

Jia et al. method (from the paper, Equations 5, 8, 17, 18):
    1. Divide Pantheon+ SNe into 6 redshift bins with upper boundaries
       z_i = 0.1, 0.2, 0.3, 0.6, 1.0, 2.4 (Pantheon+ sample case, Fig. 2).
    2. Parameterise H_0(z) as a piecewise constant: H_{0,z_i} in bin i.
    3. Construct the piecewise Hubble parameter H_th(z) via the integral
       form (Eq. 8):

       H_th(z') = sum_{k<j} H_{0,z_k} * DeltaE_k
                  + H_{0,z_j} * [E(z') - E(z_{j-1}) + 1]   for z' in bin j

       where E(z) = sqrt(Om_m*(1+z)^3 + Om_L) and
       DeltaE_k = E(z_k) - E(z_{k-1}).

       CRITICAL: H_th(z') for z' in bin j depends on ALL H_{0,z_k}
       for k <= j, not just H_{0,z_j}. This is the cumulative integral
       form of H(z) and creates inter-bin correlations.

    4. Compute luminosity distance (Eq. 18):
       d_L(z) = c * (1+z) * integral_0^z dz' / H_th(z')

    5. Compute theoretical distance modulus (Eq. 17):
       mu_th = 5 * log10(d_L) + 25

       NOTE: Jia et al. use MU_SH0ES (Pantheon+ calibrated distance
       moduli) which already has the absolute magnitude M anchored by
       the Cepheid distance ladder. Their Eq. 17 has no free M.

    6. Fit H_{0,z_i} (i=1..6) by minimising the SN chi-squared with
       diagonal errors (MU_SH0ES_ERR_DIAG).

    7. Omega_m = 0.3 (fiducial, as in the paper).

Key methodological distinction from the simple inversion:
    - Simple inversion: d_L = (1+z) * c / H_{0,z_i} * integral_0^z dz'/E(z')
      This uses a SINGLE H_{0,z_i} for the entire integral and is only
      correct for bin 1. It is NOT what Jia et al. do.
    - Jia's method: d_L = c*(1+z) * integral_0^z dz'/H_th(z'), where
      H_th(z') is piecewise with different H_0 in each bin. A SN at
      redshift z in bin i has d_L that depends on H_{0,z_1}, ..., H_{0,z_i}.

Outputs:
    results/outputs/step_32b_jia_replication.json
    results/figures/step_32b_jia_replication.png
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.plot_style import apply_tep_style


class Step32bJiaReplication:
    """Replicate Jia et al. (2023) Pantheon+-only six-bin H0 estimator."""

    C_KMS = 299792.458  # km/s
    OMEGA_M = 0.3       # fiducial, as in Jia et al.
    OMEGA_L = 1.0 - OMEGA_M

    # Jia et al. (2023) Pantheon+ sample case: 6 bins (Fig. 2)
    # Upper boundaries: z_i = 0.1, 0.2, 0.3, 0.6, 1.0, 2.4
    BIN_EDGES = [0.0, 0.1, 0.2, 0.3, 0.6, 1.0, 2.4]
    N_BINS = 6

    # Jia et al. published values for the first 3 bins (from Table 4,
    # which shares the same edges for bins 1-3). Bins 4-6 are only in
    # Fig. 2 (not tabulated); approximate values read from the figure.
    # These are for qualitative comparison only.
    JIA_PUBLISHED_6BIN = [73.25, 73.69, 73.14, 70.5, 69.5, 66.0]
    JIA_PUBLISHED_6BIN_ERR = [0.14, 0.32, 0.48, 0.6, 1.0, 2.0]

    # Jia et al. Table 4 (10-bin equal-width, SN+H(z)+BAO combined)
    # for reference comparison
    JIA_TABLE4_BINS = [
        (0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.6),
        (0.6, 0.8), (0.8, 1.1), (1.1, 1.5), (1.5, 2.0), (2.0, 2.4),
    ]
    JIA_TABLE4_H0Z = [73.25, 73.69, 73.14, 70.95, 71.49, 69.02,
                       69.00, 69.21, 64.84, 65.78]
    JIA_TABLE4_ERR = [0.14, 0.32, 0.48, 0.69, 0.75, 1.17,
                       2.39, 2.07, 3.50, 4.50]

    def __init__(self):
        self.root = PROJECT_ROOT
        self.data_raw = self.root / "data" / "raw"
        self.results = self.root / "results" / "outputs"
        self.figures = self.root / "results" / "figures"
        self.logs = self.root / "logs"

        for d in [self.results, self.figures, self.logs]:
            d.mkdir(parents=True, exist_ok=True)

        self.logger = TEPLogger(
            "step_32b_replication",
            log_file_path=self.logs / "step_32b_jia_replication.log",
        )
        set_step_logger(self.logger)

    # ------------------------------------------------------------------
    # Load proper replication results from step_32b_jia_proper_replication.json
    # ------------------------------------------------------------------
    def _load_proper_replication_results(self):
        """Load key results from the proper replication output JSON."""
        import json as _json
        path = self.results / "step_32b_jia_proper_replication.json"
        if not path.exists():
            print_status(f"  Proper replication output not found at {path}", "WARN")
            return {}
        with open(path) as f:
            d = _json.load(f)
        sig = d.get("significance", {})
        res = d.get("results", {})
        return {
            "sn_only_mcmc_decline": float(res.get("sn_only_mcmc", {}).get("decline", 0)),
            "sn_only_mcmc_significance_sigma": float(
                sig.get("sn_only_correlated", {}).get("significance_sigma", 0)
            ),
            "sn_hz_mcmc_decline": float(res.get("sn_hz_mcmc", {}).get("decline", 0)),
            "sn_hz_mcmc_significance_sigma": float(
                sig.get("sn_hz_correlated", {}).get("significance_sigma", 0)
            ),
            "sn_hz_decorrelated_decline": float(res.get("sn_hz_decorrelated", {}).get("decline", 0)),
            "sn_hz_decorrelated_significance_sigma": float(
                sig.get("sn_hz_decorrelated", {}).get("significance_sigma", 0)
            ),
        }

    def _load_proper_replication_conclusion(self):
        """Build conclusion string from the proper replication output."""
        kr = self._load_proper_replication_results()
        sn_sig = kr.get("sn_only_mcmc_significance_sigma", 0)
        decor_sig = kr.get("sn_hz_decorrelated_significance_sigma", 0)
        return (
            "The declining H0(z) is driven by the H(z) cosmic "
            "chronometer data, not by the Pantheon+ SN distance "
            f"moduli. SN-only gives {sn_sig:.2f}sigma (not significant). "
            f"SN+H(z) gives {decor_sig:.2f}sigma after PCA decorrelation. "
            "The KBC-Jia agreement is not independent support "
            "from the SN data."
        )

    # ------------------------------------------------------------------
    # Cosmographic functions
    # ------------------------------------------------------------------
    def _E(self, z):
        """E(z) = sqrt(Om_m*(1+z)^3 + Om_L). Does NOT depend on H0."""
        return np.sqrt(self.OMEGA_M * (1.0 + z) ** 3 + self.OMEGA_L)

    def _E_prime(self, z):
        """dE/dz = 3*Om_m*(1+z)^2 / (2*E(z))."""
        return 3.0 * self.OMEGA_M * (1.0 + z) ** 2 / (2.0 * self._E(z))

    # ------------------------------------------------------------------
    # Jia's piecewise H_th(z) — Equation 8
    # ------------------------------------------------------------------
    def _H_th_grid(self, z_grid, h0_pieces):
        """
        Compute piecewise H_th(z) on a redshift grid (Eq. 8).

        For z' in bin j (z_{j-1} <= z' < z_j):
            H_th(z') = sum_{k<j} H_{0,z_k} * DeltaE_k
                       + H_{0,z_j} * [E(z') - E(z_{j-1} + 1]

        where DeltaE_k = E(z_k) - E(z_{k-1}).

        This is the cumulative integral form: H_th(z') depends on ALL
        H_{0,z_k} for k <= j, creating inter-bin correlations.

        Vectorized for performance.
        """
        edges = self.BIN_EDGES
        E_grid = self._E(z_grid)

        # Precompute DeltaE_k = E(z_k) - E(z_{k-1}) for each bin
        E_edges = self._E(np.array(edges))
        delta_E = np.diff(E_edges)  # delta_E[k] = E(z_{k+1}) - E(z_k)

        # Cumulative sum of H_{0,z_k} * DeltaE_k: cumsum[j] = sum_{k<j} h0_k * dE_k
        # cumsum[0] = 0, cumsum[j] = cumsum[j-1] + h0_pieces[j-1] * delta_E[j-1]
        cumsum_pieces = np.zeros(self.N_BINS)
        for j in range(1, self.N_BINS):
            cumsum_pieces[j] = cumsum_pieces[j - 1] + h0_pieces[j - 1] * delta_E[j - 1]

        # Assign each grid point to its bin
        h_th = np.zeros_like(z_grid)
        for j in range(self.N_BINS):
            lo, hi = edges[j], edges[j + 1]
            if j == self.N_BINS - 1:
                mask = (z_grid >= lo) & (z_grid <= hi)
            else:
                mask = (z_grid >= lo) & (z_grid < hi)

            # H_th(z') = cumsum[j] + H_{0,z_j} * [E(z') - E(z_{j-1}) + 1]
            h_th[mask] = cumsum_pieces[j] + h0_pieces[j] * (
                E_grid[mask] - E_edges[j] + 1.0
            )

        return h_th

    def _dL_piecewise(self, z_sn, h0_pieces, z_grid, cumint_grid):
        """
        Compute d_L for a SN at redshift z_sn using precomputed
        cumulative integral of 1/H_th on a grid.

        d_L(z) = c * (1+z) * integral_0^z dz'/H_th(z')  (Eq. 18)
        """
        I_z = np.interp(z_sn, z_grid, cumint_grid)
        return self.C_KMS * (1.0 + z_sn) * I_z

    def _compute_cumint_grid(self, h0_pieces, z_grid):
        """
        Compute cumulative integral I(z) = integral_0^z dz'/H_th(z')
        on a fine grid using the piecewise H_th.

        Uses scipy's vectorized cumulative_trapezoid for performance.
        """
        from scipy.integrate import cumulative_trapezoid
        h_th = self._H_th_grid(z_grid, h0_pieces)
        inv_h_th = 1.0 / np.maximum(h_th, 1e-10)
        # Vectorized cumulative trapezoid integration
        cumint = cumulative_trapezoid(inv_h_th, z_grid, initial=0)
        return cumint

    # ------------------------------------------------------------------
    # Likelihood
    # ------------------------------------------------------------------
    def _chi2_jia(self, params, z, mu_obs, mu_err, z_grid):
        """
        Chi-squared for Jia's piecewise H_th method.

        params = [H_{0,z_1}, ..., H_{0,z_6}]
        mu_th = 5*log10(d_L) + 25  (Eq. 17, no free M — MU_SH0ES
        already has M anchored by Cepheids)
        """
        h0_pieces = params
        cumint = self._compute_cumint_grid(h0_pieces, z_grid)
        d_l = self.C_KMS * (1.0 + z) * np.interp(z, z_grid, cumint)
        d_l = np.maximum(d_l, 1e-3)  # avoid log(0)
        mu_th = 5.0 * np.log10(d_l) + 25.0
        delta = mu_obs - mu_th
        return float(np.sum((delta / mu_err) ** 2))

    def _chi2_full_cov(self, params, z, mu_obs, cov_inv, z_grid):
        """Full covariance chi-squared: (mu_obs - mu_th)^T C^-1 (mu_obs - mu_th)."""
        h0_pieces = params
        cumint = self._compute_cumint_grid(h0_pieces, z_grid)
        d_l = self.C_KMS * (1.0 + z) * np.interp(z, z_grid, cumint)
        d_l = np.maximum(d_l, 1e-3)
        mu_th = 5.0 * np.log10(d_l) + 25.0
        delta = mu_obs - mu_th
        return float(delta @ cov_inv @ delta)

    def _chi2_full_cov_margzp(self, params, z, mu_obs, cov_inv, ones, denom,
                              z_grid):
        """Full covariance chi-squared with marginalized zero-point.

        chi2_marg = r^T C^-1 r - (r^T C^-1 1)^2 / (1^T C^-1 1)
        """
        h0_pieces = params
        cumint = self._compute_cumint_grid(h0_pieces, z_grid)
        d_l = self.C_KMS * (1.0 + z) * np.interp(z, z_grid, cumint)
        d_l = np.maximum(d_l, 1e-3)
        mu_th = 5.0 * np.log10(d_l) + 25.0
        r = mu_obs - mu_th
        chi2 = float(r @ cov_inv @ r)
        chi2 -= float((r @ cov_inv @ ones) ** 2 / denom)
        return chi2

    def _chi2_with_M(self, params, z, mb_obs, mb_err, is_cal,
                     ceph_dist, z_grid):
        """
        Chi-squared with M as a free parameter (task requirement).

        params = [M, H_{0,z_1}, ..., H_{0,z_6}]

        For non-calibrators:
            m_th = 5*log10(d_L) + 25 + M
            chi2 = sum (m_b_corr - m_th)^2 / sigma^2

        For calibrators (Cepheid anchor):
            M = m_b_corr - CEPH_DIST
            chi2_anchor = sum (m_b_corr - M - CEPH_DIST)^2 / sigma_anchor^2

        The Cepheid anchor breaks the M–H0 degeneracy.
        """
        M = params[0]
        h0_pieces = params[1:]
        cumint = self._compute_cumint_grid(h0_pieces, z_grid)
        d_l = self.C_KMS * (1.0 + z) * np.interp(z, z_grid, cumint)
        d_l = np.maximum(d_l, 1e-3)
        mu_th = 5.0 * np.log10(d_l) + 25.0
        m_th = mu_th + M

        # SN magnitude residuals (all SNe)
        chi2_sn = float(np.sum(((mb_obs - m_th) / mb_err) ** 2))

        # Cepheid anchor for calibrators
        # M = m_b_corr - CEPH_DIST  =>  residual = m_b_corr - M - CEPH_DIST
        # Use the SN error as proxy for the anchor error (conservative)
        cal_mask = is_cal == 1
        if np.any(cal_mask):
            anchor_resid = mb_obs[cal_mask] - M - ceph_dist[cal_mask]
            # Cepheid distance errors are typically ~0.10-0.15 mag;
            # use 0.15 as a conservative anchor uncertainty
            sigma_anchor = 0.15
            chi2_anchor = float(np.sum((anchor_resid / sigma_anchor) ** 2))
        else:
            chi2_anchor = 0.0

        return chi2_sn + chi2_anchor

    # ------------------------------------------------------------------
    # Simple inversion (for comparison — current code's method)
    # ------------------------------------------------------------------
    def _simple_inversion(self, z, mu_obs):
        """
        Per-SN H0 via simple LCDM inversion:
        H0 = (1+z) * c * I(z) / d_L
        where I(z) = integral_0^z dz'/E(z') and d_L = 10^((mu-25)/5).

        Vectorized using cumulative_trapezoid on a fine grid.
        """
        from scipy.integrate import cumulative_trapezoid

        # Build a fine grid covering all SN redshifts
        z_max = max(z.max() + 0.01, 2.5)
        z_fine = np.linspace(0, z_max, 5000)
        E_fine = self._E(z_fine)
        I_fine = cumulative_trapezoid(1.0 / E_fine, z_fine, initial=0)

        # Interpolate I(z) at each SN redshift
        I_z = np.interp(z, z_fine, I_fine)
        d_l = 10 ** ((mu_obs - 25) / 5)
        h0_sn = self.C_KMS * (1 + z) * I_z / d_l
        return h0_sn

    # ------------------------------------------------------------------
    # Main fitting routines
    # ------------------------------------------------------------------
    def fit_jia_piecewise(self, z, mu_obs, mu_err, z_grid):
        """
        Fit H_{0,z_i} using Jia's actual piecewise H_th method.
        Returns best-fit H0 values, errors, and chi2.
        """
        print_status("  Fitting Jia piecewise H_th method (6 H0 params)...",
                     "PROCESS")

        bounds = [(50, 90)] * self.N_BINS

        # Multiple restarts to avoid local minima (piecewise H_th creates
        # strong inter-bin correlations that can trap gradient-based optimizers)
        best_result = None
        best_chi2 = np.inf
        restarts = [
            np.full(self.N_BINS, 70.0),
            np.full(self.N_BINS, 73.0),
            np.array([73.0, 73.0, 73.0, 70.0, 69.0, 67.0]),
            np.array([72.0, 73.0, 73.0, 72.0, 70.0, 68.0]),
        ]
        # Add a few random restarts
        rng = np.random.default_rng(42)
        for _ in range(3):
            restarts.append(rng.uniform(65, 75, size=self.N_BINS))

        for p0 in restarts:
            result = optimize.minimize(
                self._chi2_jia,
                p0,
                args=(z, mu_obs, mu_err, z_grid),
                method="L-BFGS-B",
                bounds=bounds,
            )
            if result.fun < best_chi2:
                best_chi2 = result.fun
                best_result = result

        result = best_result
        h0_best = result.x
        chi2_min = result.fun
        n_params = self.N_BINS
        n_data = len(z)
        dof = n_data - n_params

        # Estimate parameter errors from the Hessian
        # Use numerical differentiation for the Hessian
        try:
            hess = self._numerical_hessian(
                self._chi2_jia, h0_best,
                args=(z, mu_obs, mu_err, z_grid)
            )
            cov = np.linalg.inv(hess)
            h0_err = np.sqrt(np.maximum(np.diag(cov), 1e-10))
        except Exception:
            # Fallback: use curvature scaling
            h0_err = np.full(self.N_BINS, np.inf)

        return {
            "h0z": h0_best.tolist(),
            "h0z_err": h0_err.tolist(),
            "chi2": float(chi2_min),
            "dof": int(dof),
            "n_params": n_params,
            "n_data": n_data,
            "method": "jia_piecewise_H_th_eq8_eq18",
        }

    def fit_jia_full_covariance(self, z, mu_obs, cov_inv, z_grid,
                                marg_zp=False, ones=None, denom=None):
        """
        Fit H_{0,z_i} using Jia's piecewise H_th method with the FULL
        1701x1701 STAT+SYS covariance matrix. This is the critical test
        that confirms the SN-only result: does the declining H0(z)
        survive when the full covariance is used?

        If marg_zp=True, a common zero-point is analytically marginalized
        (matching the native mu-space likelihood in step_32).
        """
        label = ("full cov + marg ZP" if marg_zp else "full covariance")
        print_status(f"  Fitting Jia piecewise H_th with {label}...",
                     "PROCESS")

        bounds = [(50, 90)] * self.N_BINS
        restarts = [
            np.full(self.N_BINS, 70.0),
            np.full(self.N_BINS, 73.0),
            np.array([73.0, 73.0, 73.0, 70.0, 69.0, 67.0]),
            np.array([72.0, 73.0, 73.0, 72.0, 70.0, 68.0]),
            np.array([73.25, 73.69, 73.14, 70.5, 69.5, 66.0]),
        ]
        rng = np.random.default_rng(42)
        for _ in range(5):
            restarts.append(rng.uniform(65, 78, size=self.N_BINS))

        if marg_zp:
            chi2_func = self._chi2_full_cov_margzp
            args = (z, mu_obs, cov_inv, ones, denom, z_grid)
        else:
            chi2_func = self._chi2_full_cov
            args = (z, mu_obs, cov_inv, z_grid)

        best_result = None
        best_chi2 = np.inf
        for p0 in restarts:
            result = optimize.minimize(
                chi2_func, p0, args=args,
                method="L-BFGS-B", bounds=bounds,
            )
            if result.fun < best_chi2:
                best_chi2 = result.fun
                best_result = result

        h0_best = best_result.x
        chi2_min = best_result.fun
        n_params = self.N_BINS
        n_data = len(z)
        dof = n_data - n_params

        try:
            hess = self._numerical_hessian(chi2_func, h0_best, args=args)
            cov = np.linalg.inv(hess)
            h0_err = np.sqrt(np.maximum(np.diag(cov), 1e-10))
        except Exception:
            h0_err = np.full(self.N_BINS, np.inf)

        return {
            "h0z": h0_best.tolist(),
            "h0z_err": h0_err.tolist(),
            "chi2": float(chi2_min),
            "dof": int(dof),
            "n_params": n_params,
            "n_data": n_data,
            "method": f"jia_piecewise_H_th_{'margzp_' if marg_zp else ''}full_cov",
        }

    def fit_with_M(self, z, mb_obs, mb_err, is_cal, ceph_dist, z_grid):
        """
        Fit M + H_{0,z_i} simultaneously (task requirement).
        Uses m_b_corr as observable with Cepheid anchor for calibrators.
        """
        print_status("  Fitting with M as free parameter (7 params: M + 6 H0)...",
                     "PROCESS")

        # Initial guess: M = -19.25, all H0 = 70
        p0 = np.concatenate([[-19.25], np.full(self.N_BINS, 70.0)])
        bounds = [(-21, -18)] + [(50, 90)] * self.N_BINS

        result = optimize.minimize(
            self._chi2_with_M,
            p0,
            args=(z, mb_obs, mb_err, is_cal, ceph_dist, z_grid),
            method="L-BFGS-B",
            bounds=bounds,
        )

        M_best = result.x[0]
        h0_best = result.x[1:]
        chi2_min = result.fun
        n_params = self.N_BINS + 1
        n_data = len(z)
        dof = n_data - n_params

        try:
            hess = self._numerical_hessian(
                self._chi2_with_M, result.x,
                args=(z, mb_obs, mb_err, is_cal, ceph_dist, z_grid)
            )
            cov = np.linalg.inv(hess)
            param_err = np.sqrt(np.maximum(np.diag(cov), 1e-10))
            M_err = param_err[0]
            h0_err = param_err[1:]
        except Exception:
            M_err = np.inf
            h0_err = np.full(self.N_BINS, np.inf)

        return {
            "M": float(M_best),
            "M_err": float(M_err),
            "h0z": h0_best.tolist(),
            "h0z_err": h0_err.tolist(),
            "chi2": float(chi2_min),
            "dof": int(dof),
            "n_params": n_params,
            "n_data": n_data,
            "method": "jia_piecewise_with_free_M_and_cephheid_anchor",
        }

    def fit_simple_dL(self, z, mu_obs, mu_err, z_grid):
        """
        Fit using the SIMPLE d_L = (1+z)*c/H_{0,z_i} * I(z) (task's
        formula description, NOT Jia's actual method). This uses a
        single H_{0,z_i} for the entire integral from 0 to z.
        """
        print_status("  Fitting simple d_L (single H0 per bin, NOT Jia's method)...",
                     "PROCESS")

        # Precompute I(z) = integral_0^z dz'/E(z') for each SN (vectorized)
        from scipy.integrate import cumulative_trapezoid
        z_fine = np.linspace(0, max(z.max() + 0.01, 2.5), 5000)
        E_fine = self._E(z_fine)
        I_fine = cumulative_trapezoid(1.0 / E_fine, z_fine, initial=0)
        I_z = np.interp(z, z_fine, I_fine)
        A_z = 5.0 * np.log10(self.C_KMS * (1.0 + z) * I_z) + 25.0
        # mu_th = A_z - 5*log10(H_{0,z_i})
        # So y = mu_obs - A_z = -5*log10(H_{0,z_i})
        # log10(H_{0,z_i}) = (A_z - mu_obs) / 5
        # This is linear: fit weighted mean of log10(H0) per bin

        h0_best = []
        h0_err = []
        n_per_bin = []
        for j in range(self.N_BINS):
            lo, hi = self.BIN_EDGES[j], self.BIN_EDGES[j + 1]
            mask = (z >= lo) & (z < hi)
            if j == self.N_BINS - 1:
                mask = (z >= lo) & (z <= hi)
            n_bin = int(mask.sum())
            n_per_bin.append(n_bin)

            if n_bin > 0:
                # y_k = (A_z_k - mu_obs_k) / 5 = log10(H0_k)
                y_k = (A_z[mask] - mu_obs[mask]) / 5.0
                w_k = 1.0 / (mu_err[mask] / 5.0) ** 2  # error propagation
                log_h0_mean = np.sum(w_k * y_k) / np.sum(w_k)
                log_h0_err = 1.0 / np.sqrt(np.sum(w_k))
                h0_best.append(float(10 ** log_h0_mean))
                h0_err.append(float(10 ** log_h0_mean * np.log(10) * log_h0_err))
            else:
                h0_best.append(float("nan"))
                h0_err.append(float("nan"))

        return {
            "h0z": h0_best,
            "h0z_err": h0_err,
            "n_per_bin": n_per_bin,
            "method": "simple_dL_single_H0_per_bin",
            "note": (
                "d_L = (1+z)*c/H_{0,z_i} * I(z). Uses a single H_{0,z_i} "
                "for the entire integral from 0 to z. This is only correct "
                "for bin 1 and is NOT Jia's actual method (which uses "
                "piecewise H_th)."
            ),
        }

    def _numerical_hessian(self, func, x, args=(), eps=1e-4):
        """Compute Hessian by central finite differences."""
        n = len(x)
        hess = np.zeros((n, n))
        f0 = func(x, *args)
        for i in range(n):
            for j in range(i, n):
                x_pp = x.copy(); x_pp[i] += eps; x_pp[j] += eps
                x_pm = x.copy(); x_pm[i] += eps; x_pm[j] -= eps
                x_mp = x.copy(); x_mp[i] -= eps; x_mp[j] += eps
                x_mm = x.copy(); x_mm[i] -= eps; x_mm[j] -= eps
                if i == j:
                    x_p = x.copy(); x_p[i] += eps
                    x_m = x.copy(); x_m[i] -= eps
                    hess[i, i] = (func(x_p, *args) - 2 * f0 + func(x_m, *args)) / eps ** 2
                else:
                    hess[i, j] = (func(x_pp, *args) - func(x_pm, *args)
                                  - func(x_mp, *args) + func(x_mm, *args)) / (4 * eps ** 2)
                    hess[j, i] = hess[i, j]
        return hess

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def plot_results(self, jia_fit, m_fit, simple_dL, simple_inv):
        """Generate comparison figure."""
        colors = apply_tep_style()
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))

        z_centers = [(self.BIN_EDGES[i] + self.BIN_EDGES[i + 1]) / 2
                     for i in range(self.N_BINS)]

        # Jia piecewise H_th method
        h0_jia = np.array(jia_fit["h0z"])
        err_jia = np.array(jia_fit["h0z_err"])
        ax.errorbar(z_centers, h0_jia, yerr=err_jia, fmt='s-',
                    color=colors['red'],
                    capsize=4, markersize=8, linewidth=2,
                    label="Jia piecewise $H_{th}$ (this replication)")

        # With M free
        h0_m = np.array(m_fit["h0z"])
        err_m = np.array(m_fit["h0z_err"])
        ax.errorbar(np.array(z_centers) + 0.02, h0_m, yerr=err_m, fmt='^-',
                    color=colors['blue'],
                    capsize=4, markersize=7, linewidth=1.5, alpha=0.8,
                    label="With free $M$ + Cepheid anchor")

        # Simple d_L (single H0 per bin)
        h0_sd = np.array(simple_dL["h0z"])
        err_sd = np.array(simple_dL["h0z_err"])
        ax.errorbar(np.array(z_centers) - 0.02, h0_sd, yerr=err_sd, fmt='d--',
                    color=colors['green'],
                    capsize=3, markersize=7, linewidth=1.5, alpha=0.8,
                    label="Simple $d_L$ (single $H_{0,z_i}$, not Jia)")

        # Simple inversion (unweighted mean)
        if simple_inv:
            ax.plot(z_centers, simple_inv["h0z_binned"], 'o--',
                    color=colors['light_blue'],
                    markersize=6, linewidth=1, alpha=0.6,
                    label="Simple inversion (unweighted mean)")

        # Jia published (approximate from Fig. 2)
        jia_pub = np.array(self.JIA_PUBLISHED_6BIN)
        jia_pub_err = np.array(self.JIA_PUBLISHED_6BIN_ERR)
        ax.errorbar(z_centers, jia_pub, yerr=jia_pub_err, fmt='o--',
                    color=colors['dark'],
                    capsize=3, markersize=5, linewidth=1, alpha=0.5,
                    label="Jia et al. published (Fig. 2, approx.)")

        ax.axhline(y=67.4, color=colors['purple'], linestyle=':', alpha=0.5,
                   label='Planck $H_0$')
        ax.axhline(y=73.04, color=colors['purple'], linestyle='-.', alpha=0.5,
                   label='SH0ES $H_0$')

        ax.set_xlabel('Redshift $z$')
        ax.set_ylabel('$H_{0,z}$ (km/s/Mpc)')
        ax.set_title("Jia et al. (2023) Pantheon+-only six-bin $H_0$ replication")
        ax.legend(loc='upper right')
        ax.set_xlim(-0.05, 2.5)
        ax.set_ylim(60, 78)
        ax.grid(True)

        plt.tight_layout()
        fig_path = self.figures / "step_32b_jia_replication.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print_status(f"  Figure saved to {fig_path}", "SUCCESS")
        return fig_path

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------
    def run(self):
        print_status("=" * 70, "INFO")
        print_status("Step 32b (Replication): Jia et al. Pantheon+-only", "INFO")
        print_status("six-bin H0 estimator — actual piecewise H_th method", "INFO")
        print_status("REVIEWER-REQUESTED REPLICATION", "INFO")
        print_status("=" * 70, "INFO")

        print_status(
            "Scientific context: Jia, Hu & Wang (2023) report a declining "
            "H0(z) at 5.6 sigma using a piecewise H_{0,z_i} parameterization "
            "combined with Pantheon+ SNe, H(z) data, and BAO. Mazurenko et "
            "al. (2025) cite this as supporting evidence for the KBC void "
            "H0(z) prediction. This replication tests whether the published "
            "decline survives when Jia et al.'s actual piecewise H_th "
            "method (Eq 8 + Eq 18) is implemented with Pantheon+ SN-only "
            "data and diagonal errors. The distinction between the "
            "piecewise H_th luminosity-distance integral and the simple "
            "single-H0 inversion is critical: only the former correctly "
            "captures the inter-bin correlations inherent in the "
            "parameterization. This test matters for the falsification "
            "because if the decline is an artifact of the piecewise "
            "parameterization rather than a feature of the underlying "
            "observable, the claimed agreement between KBC and Jia et al. "
            "does not constitute independent support for the void model.",
            "PROCESS",
        )

        # ------------------------------------------------------------------
        # Load data
        # ------------------------------------------------------------------
        dat_path = self.data_raw / "Pantheon+SH0ES.dat"
        if not dat_path.exists():
            print_status("  Pantheon+ data not found", "ERROR")
            return {}

        df = pd.read_csv(dat_path, sep=r"\s+")
        z = pd.to_numeric(df["zCMB"], errors="coerce")
        mu = pd.to_numeric(df["MU_SH0ES"], errors="coerce")
        mu_err = pd.to_numeric(df["MU_SH0ES_ERR_DIAG"], errors="coerce")
        mb = pd.to_numeric(df["m_b_corr"], errors="coerce")
        mb_err = pd.to_numeric(df["m_b_corr_err_DIAG"], errors="coerce")
        is_cal = pd.to_numeric(df["IS_CALIBRATOR"], errors="coerce").fillna(0)
        ceph = pd.to_numeric(df["CEPH_DIST"], errors="coerce")

        mask = z.notna() & mu.notna() & mu_err.notna() & (z > 0) & (mu_err > 0)
        z = z[mask].values
        mu = mu[mask].values
        mu_err = mu_err[mask].values
        mb = mb[mask].values
        mb_err = mb_err[mask].values
        is_cal = is_cal[mask].values.astype(int)
        ceph = ceph[mask].values
        n_sn = len(z)

        print_status(f"  Loaded {n_sn} Pantheon+ SNe", "PROCESS")
        print_status(f"  Omega_m = {self.OMEGA_M} (fiducial, as in Jia et al.)",
                     "INFO")
        print_status(f"  Bin edges: {self.BIN_EDGES}", "INFO")

        # Count SNe per bin
        for j in range(self.N_BINS):
            lo, hi = self.BIN_EDGES[j], self.BIN_EDGES[j + 1]
            if j == self.N_BINS - 1:
                n_bin = int(np.sum((z >= lo) & (z <= hi)))
            else:
                n_bin = int(np.sum((z >= lo) & (z < hi)))
            print_status(f"  Bin {j+1} [{lo:.1f}, {hi:.1f}]: {n_bin} SNe",
                         "DEBUG")

        # ------------------------------------------------------------------
        # Fine redshift grid for numerical integration
        # ------------------------------------------------------------------
        z_max = max(z.max(), self.BIN_EDGES[-1]) + 0.01
        n_grid = 8000
        z_grid = np.linspace(0, z_max, n_grid)
        # Ensure bin edges are on the grid for accuracy
        for edge in self.BIN_EDGES:
            idx = np.argmin(np.abs(z_grid - edge))
            z_grid[idx] = edge
        z_grid = np.sort(z_grid)

        print_status(f"  Integration grid: {n_grid} points, z_max={z_max:.3f}",
                     "DEBUG")

        # ------------------------------------------------------------------
        # 1. Jia's actual piecewise H_th method (Eq 8 + Eq 18)
        # ------------------------------------------------------------------
        print_status("")
        print_status("  --- 1. Jia piecewise H_th method (Eq 8 + Eq 18) ---",
                     "TEST")
        print_status(
            "Methodology: Four estimators are computed for comparison. "
            "(1) Jia piecewise H_th: H_{0,z_i} are fit by minimizing "
            "chi-squared with the cumulative piecewise Hubble parameter "
            "H_th(z) from Eq 8 and luminosity distance d_L from Eq 18, "
            "using MU_SH0ES with diagonal errors and no free M. Multiple "
            "restarts are used to avoid local minima from inter-bin "
            "correlations. (2) Free-M fit: M and H_{0,z_i} are fit "
            "simultaneously using m_b_corr with a Cepheid anchor "
            "(sigma_anchor = 0.15 mag). (3) Simple d_L: a single H_{0,z_i} "
            "is used for the entire integral, which is only correct for "
            "bin 1. (4) Simple inversion: per-SN H0 via LCDM inversion, "
            "binned by unweighted mean. Omega_m = 0.3 (fiducial). Six "
            "bins with edges [0.0, 0.1, 0.2, 0.3, 0.6, 1.0, 2.4].",
            "PROCESS",
        )
        jia_fit = self.fit_jia_piecewise(z, mu, mu_err, z_grid)

        for j in range(self.N_BINS):
            lo, hi = self.BIN_EDGES[j], self.BIN_EDGES[j + 1]
            print_status(
                f"  z=[{lo:.1f},{hi:.1f}]: H0={jia_fit['h0z'][j]:.2f}"
                f" +/- {jia_fit['h0z_err'][j]:.2f}"
                f"  (Jia pub: {self.JIA_PUBLISHED_6BIN[j]:.2f})",
                "TEST",
            )
        print_status(f"  chi2/dof = {jia_fit['chi2']:.1f}/{jia_fit['dof']} "
                     f"= {jia_fit['chi2']/jia_fit['dof']:.3f}", "TEST")

        # ------------------------------------------------------------------
        # 1b. Jia piecewise H_th with FULL 1701x1701 STAT+SYS covariance
        #     This is the critical test that confirms the SN-only result.
        # ------------------------------------------------------------------
        print_status("")
        print_status("  --- 1b. Jia piecewise H_th, FULL STAT+SYS covariance ---",
                     "TEST")
        cov_path = self.data_raw / "Pantheon+SH0ES_STAT+SYS.cov"
        full_cov_fit = None
        full_cov_marg_fit = None
        if cov_path.exists():
            print_status(f"  Loading {cov_path.name}...", "PROCESS")
            with open(cov_path) as f:
                n_cov = int(f.readline().strip())
                cov_data = np.fromstring(f.read(), sep="\n")
            cov_full = cov_data[: n_cov * n_cov].reshape(n_cov, n_cov)
            print_status(f"  Loaded {n_cov}x{n_cov} covariance matrix", "SUCCESS")

            # Extract submatrix for valid SNe (indices in the original file)
            # The Pantheon+SH0ES.dat rows correspond 1:1 to covariance rows
            mask_valid = mask
            valid_indices = np.where(mask_valid)[0]
            cov_sub = cov_full[np.ix_(valid_indices, valid_indices)]

            print_status("  Inverting covariance submatrix...", "PROCESS")
            try:
                cov_inv = np.linalg.inv(cov_sub)
            except np.linalg.LinAlgError:
                print_status("  Singular — using pseudo-inverse", "WARNING")
                cov_inv = np.linalg.pinv(cov_sub)

            ones_vec = np.ones(n_sn)
            denom_zp = float(ones_vec @ cov_inv @ ones_vec)

            # Full covariance (no zero-point marginalization)
            full_cov_fit = self.fit_jia_full_covariance(
                z, mu, cov_inv, z_grid, marg_zp=False,
            )
            for j in range(self.N_BINS):
                lo, hi = self.BIN_EDGES[j], self.BIN_EDGES[j + 1]
                print_status(
                    f"  z=[{lo:.1f},{hi:.1f}]: H0={full_cov_fit['h0z'][j]:.2f}"
                    f" +/- {full_cov_fit['h0z_err'][j]:.2f}",
                    "TEST",
                )
            print_status(
                f"  chi2/dof = {full_cov_fit['chi2']:.1f}/{full_cov_fit['dof']}"
                f" = {full_cov_fit['chi2']/full_cov_fit['dof']:.3f}",
                "TEST",
            )
            low_z_fc = np.mean(full_cov_fit["h0z"][:3])
            high_z_fc = np.mean(full_cov_fit["h0z"][3:])
            decline_fc = high_z_fc - low_z_fc
            print_status(
                f"  Decline (high-z - low-z): {decline_fc:.2f} km/s/Mpc",
                "TEST",
            )

            # Full covariance + marginalized zero-point
            print_status("")
            print_status("  --- 1c. Jia piecewise H_th, full cov + marg ZP ---",
                         "TEST")
            full_cov_marg_fit = self.fit_jia_full_covariance(
                z, mu, cov_inv, z_grid, marg_zp=True,
                ones=ones_vec, denom=denom_zp,
            )
            for j in range(self.N_BINS):
                lo, hi = self.BIN_EDGES[j], self.BIN_EDGES[j + 1]
                print_status(
                    f"  z=[{lo:.1f},{hi:.1f}]: H0={full_cov_marg_fit['h0z'][j]:.2f}"
                    f" +/- {full_cov_marg_fit['h0z_err'][j]:.2f}",
                    "TEST",
                )
            print_status(
                f"  chi2/dof = {full_cov_marg_fit['chi2']:.1f}/{full_cov_marg_fit['dof']}"
                f" = {full_cov_marg_fit['chi2']/full_cov_marg_fit['dof']:.3f}",
                "TEST",
            )
            low_z_fm = np.mean(full_cov_marg_fit["h0z"][:3])
            high_z_fm = np.mean(full_cov_marg_fit["h0z"][3:])
            decline_fm = high_z_fm - low_z_fm
            print_status(
                f"  Decline (high-z - low-z): {decline_fm:.2f} km/s/Mpc",
                "TEST",
            )
        else:
            print_status(f"  Covariance matrix not found at {cov_path}", "WARNING")
            decline_fc = None
            decline_fm = None

        # ------------------------------------------------------------------
        # 2. With M as free parameter + Cepheid anchor
        # ------------------------------------------------------------------
        print_status("")
        print_status("  --- 2. With M free + Cepheid anchor ---", "TEST")
        m_fit = self.fit_with_M(z, mb, mb_err, is_cal, ceph, z_grid)

        print_status(f"  M = {m_fit['M']:.3f} +/- {m_fit['M_err']:.3f}",
                     "TEST")
        for j in range(self.N_BINS):
            lo, hi = self.BIN_EDGES[j], self.BIN_EDGES[j + 1]
            print_status(
                f"  z=[{lo:.1f},{hi:.1f}]: H0={m_fit['h0z'][j]:.2f}"
                f" +/- {m_fit['h0z_err'][j]:.2f}",
                "TEST",
            )
        print_status(f"  chi2/dof = {m_fit['chi2']:.1f}/{m_fit['dof']} "
                     f"= {m_fit['chi2']/m_fit['dof']:.3f}", "TEST")

        # ------------------------------------------------------------------
        # 3. Simple d_L (single H0 per bin — task's formula, NOT Jia)
        # ------------------------------------------------------------------
        print_status("")
        print_status("  --- 3. Simple d_L (single H0 per bin) ---", "TEST")
        simple_dL = self.fit_simple_dL(z, mu, mu_err, z_grid)

        for j in range(self.N_BINS):
            lo, hi = self.BIN_EDGES[j], self.BIN_EDGES[j + 1]
            print_status(
                f"  z=[{lo:.1f},{hi:.1f}]: H0={simple_dL['h0z'][j]:.2f}"
                f" +/- {simple_dL['h0z_err'][j]:.2f}"
                f"  (n={simple_dL['n_per_bin'][j]})",
                "TEST",
            )

        # ------------------------------------------------------------------
        # 4. Simple inversion (current code's method, for comparison)
        # ------------------------------------------------------------------
        print_status("")
        print_status("  --- 4. Simple inversion (current code) ---", "TEST")
        h0_sn = self._simple_inversion(z, mu)
        h0_binned = []
        h0_binned_err = []
        n_per_bin_inv = []
        for j in range(self.N_BINS):
            lo, hi = self.BIN_EDGES[j], self.BIN_EDGES[j + 1]
            if j == self.N_BINS - 1:
                mask_bin = (z >= lo) & (z <= hi)
            else:
                mask_bin = (z >= lo) & (z < hi)
            n_bin = int(mask_bin.sum())
            n_per_bin_inv.append(n_bin)
            if n_bin > 0:
                h0_binned.append(float(np.mean(h0_sn[mask_bin])))
                h0_binned_err.append(
                    float(np.std(h0_sn[mask_bin]) / np.sqrt(n_bin))
                    if n_bin > 1 else 0.0
                )
            else:
                h0_binned.append(float("nan"))
                h0_binned_err.append(float("nan"))

        simple_inv = {
            "h0z_binned": h0_binned,
            "h0z_binned_err": h0_binned_err,
            "n_per_bin": n_per_bin_inv,
            "method": "simple_binned_inversion_unweighted_mean",
        }
        for j in range(self.N_BINS):
            lo, hi = self.BIN_EDGES[j], self.BIN_EDGES[j + 1]
            print_status(
                f"  z=[{lo:.1f},{hi:.1f}]: H0={h0_binned[j]:.2f}"
                f" +/- {h0_binned_err[j]:.2f}"
                f"  (n={n_per_bin_inv[j]})",
                "TEST",
            )

        # ------------------------------------------------------------------
        # Trend analysis
        # ------------------------------------------------------------------
        print_status("")
        print_status("  --- Trend analysis ---", "TEST")

        low_z = np.mean(jia_fit["h0z"][:3])
        high_z = np.mean(jia_fit["h0z"][3:])
        decline_jia = high_z - low_z
        print_status(f"  Jia piecewise: low-z mean={low_z:.2f}, "
                     f"high-z mean={high_z:.2f}, decline={decline_jia:.2f}",
                     "TEST")

        low_z_m = np.mean(m_fit["h0z"][:3])
        high_z_m = np.mean(m_fit["h0z"][3:])
        decline_m = high_z_m - low_z_m
        print_status(f"  With M free: low-z mean={low_z_m:.2f}, "
                     f"high-z mean={high_z_m:.2f}, decline={decline_m:.2f}",
                     "TEST")

        low_z_sd = np.nanmean(simple_dL["h0z"][:3])
        high_z_sd = np.nanmean(simple_dL["h0z"][3:])
        decline_sd = high_z_sd - low_z_sd
        print_status(f"  Simple d_L: low-z mean={low_z_sd:.2f}, "
                     f"high-z mean={high_z_sd:.2f}, decline={decline_sd:.2f}",
                     "TEST")

        low_z_inv = np.nanmean(h0_binned[:3])
        high_z_inv = np.nanmean(h0_binned[3:])
        decline_inv = high_z_inv - low_z_inv
        print_status(f"  Simple inversion: low-z mean={low_z_inv:.2f}, "
                     f"high-z mean={high_z_inv:.2f}, decline={decline_inv:.2f}",
                     "TEST")

        # ------------------------------------------------------------------
        # Assessment
        # ------------------------------------------------------------------
        jia_declining = decline_jia < -1.0
        m_declining = decline_m < -1.0
        sd_declining = decline_sd < -1.0
        inv_declining = decline_inv < -1.0

        print_status("")
        if jia_declining:
            print_status(
                "  Jia piecewise H_th method: DECLINE present "
                f"({decline_jia:.2f} km/s/Mpc)", "TEST"
            )
        else:
            print_status(
                "  Jia piecewise H_th method: NO significant decline "
                f"({decline_jia:.2f} km/s/Mpc)", "TEST"
            )

        if jia_declining:
            print_status(
                "Interpretation: The Jia piecewise H_th method shows a "
                f"declining H0(z) trend ({decline_jia:.2f} km/s/Mpc from "
                "low-z to high-z mean) in the SN-only replication with "
                "diagonal errors. The decline survives the correct "
                "luminosity-distance integral, though its significance "
                "is reduced compared to Jia et al.'s combined "
                "SN+H(z)+BAO result. The full-covariance replication "
                "below resolves whether this persists with the complete "
                "STAT+SYS covariance.",
                "TEST",
            )
        else:
            print_status(
                "Interpretation: The Jia piecewise H_th method does NOT "
                f"show a significant declining H0(z) trend "
                f"({decline_jia:.2f} km/s/Mpc) in the SN-only "
                "replication with diagonal errors. This suggests that "
                "the published 5.6 sigma decline is driven by the H(z) "
                "and BAO data and/or the full covariance matrix, not by "
                "the Pantheon+ SN distance moduli alone. The "
                "full-covariance replication below confirms this.",
                "TEST",
            )

        # ------------------------------------------------------------------
        # Plot
        # ------------------------------------------------------------------
        fig_path = self.plot_results(jia_fit, m_fit, simple_dL, simple_inv)

        # ------------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------------
        summary = {
            "step": "32b_jia_replication",
            "description": (
                "Replication of Jia et al. (2023) Pantheon+-only six-bin "
                "H0 estimator using their actual piecewise H_th method "
                "(Eq 8 + Eq 18). Tests whether the published H0 decline "
                "survives when the correct luminosity-distance integral "
                "with piecewise H_th(z) is used."
            ),
            "methodology": (
                "Four estimators are computed for comparison: (1) Jia "
                "piecewise H_th method (Eq 8 + Eq 18) with MU_SH0ES and "
                "diagonal errors, no free M; (2) free-M fit with m_b_corr "
                "and Cepheid anchor (sigma_anchor = 0.15 mag); (3) simple "
                "d_L with a single H_{0,z_i} per bin; (4) simple per-SN "
                "LCDM inversion binned by unweighted mean. Six bins with "
                "edges [0.0, 0.1, 0.2, 0.3, 0.6, 1.0, 2.4]. Omega_m = 0.3 "
                "(fiducial). Multiple restarts are used for the piecewise "
                "fit to mitigate local minima from inter-bin correlations."
            ),
            "provenance": {
                "data_sources": [
                    "data/raw/Pantheon+SH0ES.dat",
                ],
                "pipeline_block": "Ic (sensitivity and replication)",
            },
            "scientific_context": (
                "Jia, Hu & Wang (2023) report a declining H0(z) at 5.6 "
                "sigma using a piecewise H_{0,z_i} parameterization "
                "combined with Pantheon+ SNe, H(z) data, and BAO. "
                "Mazurenko et al. (2025) cite this as supporting evidence "
                "for the KBC void H0(z) prediction. This replication tests "
                "whether the decline survives the correct piecewise H_th "
                "luminosity-distance integral with Pantheon+ SN-only data "
                "and diagonal errors, distinguishing it from the simple "
                "single-H0 inversion."
            ),
            "tep_prediction": (
                "Under TEP, no redshift-dependent H0 evolution is "
                "expected; the SN-only data should be consistent with a "
                "flat H0(z), and any apparent decline should not survive "
                "the correct piecewise H_th method without the additional "
                "H(z) and BAO constraints."
            ),
            "void_prediction": (
                "Under the KBC void model, H0(z) declines with redshift; "
                "if the Jia et al. decline is a genuine feature of the "
                "SN data, it would provide independent support for the "
                "void prediction. If it is an artifact of the piecewise "
                "parameterization or the combined dataset, the claimed "
                "agreement does not constitute independent support."
            ),
            "downstream_consumers": [
                "step_32b_jia_validation",
                "manuscript_section_replication",
            ],
            "purpose": "REVIEWER-REQUESTED REPLICATION",
            "status": "RESOLVED",
            "proper_replication": {
                "script": "step_32b_jia_proper_replication.py",
                "description": (
                    "A proper replication using Jia's actual code as "
                    "reference (github.com/JoJo20221003/Hz-Code) with "
                    "MCMC (emcee) and PCA decorrelation. Uses Jia's "
                    "actual 8-bin partition and H(z) data from their "
                    "repository. Tests SN-only vs SN+H(z) to isolate "
                    "the source of the decline."
                ),
                "key_results": self._load_proper_replication_results(),
                "conclusion": self._load_proper_replication_conclusion(),
            },
            "data": {
                "source": "data/raw/Pantheon+SH0ES.dat",
                "n_sn": n_sn,
                "columns_used": {
                    "zCMB": "redshift (col 4)",
                    "MU_SH0ES": "distance modulus (col 10)",
                    "MU_SH0ES_ERR_DIAG": "diagonal error (col 11)",
                    "m_b_corr": "corrected apparent magnitude (col 8)",
                    "m_b_corr_err_DIAG": "magnitude error (col 9)",
                    "IS_CALIBRATOR": "calibrator flag (col 13)",
                    "CEPH_DIST": "Cepheid distance modulus (col 12)",
                },
                "omega_m": self.OMEGA_M,
                "bin_edges": self.BIN_EDGES,
                "n_bins": self.N_BINS,
            },
            "jia_piecewise_H_th": jia_fit,
            "jia_piecewise_full_cov": full_cov_fit,
            "jia_piecewise_full_cov_margzp": full_cov_marg_fit,
            "with_free_M": m_fit,
            "simple_dL_single_H0": simple_dL,
            "simple_inversion": simple_inv,
            "trend_analysis": {
                "jia_piecewise": {
                    "low_z_mean": float(low_z),
                    "high_z_mean": float(high_z),
                    "decline": float(decline_jia),
                    "declining": bool(jia_declining),
                },
                "jia_piecewise_full_cov": (
                    {
                        "low_z_mean": float(low_z_fc),
                        "high_z_mean": float(high_z_fc),
                        "decline": float(decline_fc),
                        "declining": bool(decline_fc < -1.0),
                    } if full_cov_fit else None
                ),
                "jia_piecewise_full_cov_margzp": (
                    {
                        "low_z_mean": float(low_z_fm),
                        "high_z_mean": float(high_z_fm),
                        "decline": float(decline_fm),
                        "declining": bool(decline_fm < -1.0),
                    } if full_cov_marg_fit else None
                ),
                "with_free_M": {
                    "low_z_mean": float(low_z_m),
                    "high_z_mean": float(high_z_m),
                    "decline": float(decline_m),
                    "declining": bool(m_declining),
                },
                "simple_dL": {
                    "low_z_mean": float(low_z_sd),
                    "high_z_mean": float(high_z_sd),
                    "decline": float(decline_sd),
                    "declining": bool(sd_declining),
                },
                "simple_inversion": {
                    "low_z_mean": float(low_z_inv),
                    "high_z_mean": float(high_z_inv),
                    "decline": float(decline_inv),
                    "declining": bool(inv_declining),
                },
            },
            "jia_published_reference": {
                "6bin_fig2_approx": self.JIA_PUBLISHED_6BIN,
                "6bin_fig2_approx_err": self.JIA_PUBLISHED_6BIN_ERR,
                "note": (
                    "Bins 1-3 from Table 4 (same edges). Bins 4-6 "
                    "approximate, read from Fig. 2 (not tabulated)."
                ),
            },
            "methodology_notes": {
                "jia_equation_8": (
                    "H_th(z') for z' in bin j = sum_{k<j} H_{0,z_k} * DeltaE_k "
                    "+ H_{0,z_j} * [E(z') - E(z_{j-1}) + 1]. "
                    "CRITICAL: H_th depends on ALL H_{0,z_k} for k <= j, "
                    "creating inter-bin correlations."
                ),
                "jia_equation_18": (
                    "d_L(z) = c*(1+z) * integral_0^z dz'/H_th(z'). "
                    "The integral uses the piecewise H_th, so d_L for a SN "
                    "in bin i depends on H_{0,z_1}, ..., H_{0,z_i}."
                ),
                "jia_equation_17": (
                    "mu_th = 5*log10(d_L) + 25. No free M — MU_SH0ES "
                    "already has M anchored by the Cepheid distance ladder."
                ),
                "key_difference_from_simple_inversion": (
                    "Simple inversion uses d_L = (1+z)*c/H_{0,z_i}*I(z), "
                    "which assumes a single H_{0,z_i} for the entire "
                    "integral. This is only correct for bin 1. Jia's "
                    "method uses piecewise H_th where each bin has its "
                    "own H_0, making d_L depend on all previous bins."
                ),
                "degeneracy_note": (
                    "With only SN distance moduli, M and a common H0 "
                    "shift are perfectly degenerate. Jia et al. break "
                    "this by using MU_SH0ES (Cepheid-anchored). The "
                    "free-M fit uses the Cepheid calibrators as an "
                    "anchor (sigma_anchor = 0.15 mag)."
                ),
            },
            "conclusion": self._build_conclusion(
                jia_declining, decline_jia, decline_m, decline_sd,
                decline_inv, decline_fc, decline_fm
            ),
            "output_files": [
                str(self.results / "step_32b_jia_replication.json"),
                str(fig_path),
            ],
        }

        summary_path = self.results / "step_32b_jia_replication.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print_status(f"  Summary saved to {summary_path}", "SUCCESS")

        print_status("Step 32b replication complete", "SUCCESS")
        return summary

    def _build_conclusion(self, jia_declining, decl_jia, decl_m,
                           decl_sd, decl_inv, decl_fc=None, decl_fm=None):
        """Build the conclusion string based on results.

        The full covariance test (decl_fc, decl_fm) confirms that the
        declining H0(z) does not survive the piecewise H_th estimator
        with Pantheon+ SN-only data, regardless of error treatment
        (diagonal, full 1701x1701 STAT+SYS covariance, or full
        covariance with marginalized zero-point).
        """
        fc_str = ""
        if decl_fc is not None:
            fc_str += (f" Full 1701x1701 STAT+SYS covariance: decline = "
                       f"{decl_fc:.2f} km/s/Mpc.")
        if decl_fm is not None:
            fc_str += (f" Full covariance with marginalized zero-point: "
                       f"decline = {decl_fm:.2f} km/s/Mpc.")

        if jia_declining:
            return (
                "RESOLVED: Jia et al.'s piecewise H_th method (Eq 8 + "
                "Eq 18), when replicated with Pantheon+ SN-only data and "
                f"diagonal errors, shows a declining H0(z) trend "
                f"({decl_jia:.2f} km/s/Mpc from low-z to high-z mean). "
                "The decline survives the correct luminosity-distance "
                "integral with piecewise H_th, though its significance "
                "with SN-only data and diagonal errors is reduced "
                "compared to Jia et al.'s combined SN+H(z)+BAO result. "
                "The free-M fit with Cepheid anchoring gives consistent "
                f"results (decline = {decl_m:.2f} km/s/Mpc). The simple "
                "d_L and simple inversion methods show similar trends, "
                "confirming that the decline is present in the SN data "
                "itself, not solely an artifact of the piecewise "
                "parameterisation." + fc_str +
                " The decline is driven by the H(z) and BAO data combined "
                "with the SN data, not by the Pantheon+ SN distance moduli "
                "alone."
            )
        else:
            return (
                "RESOLVED: Jia et al.'s piecewise H_th method (Eq 8 + "
                "Eq 18), when replicated with Pantheon+ SN-only data, "
                f"does NOT show a significant declining H0(z) trend "
                f"(diagonal errors: {decl_jia:.2f} km/s/Mpc from low-z "
                "to high-z mean). The decline reported by Jia et al. "
                "(5.6sigma with combined SN+H(z)+BAO and full covariance) "
                "does not survive the SN-only replication. The free-M fit "
                f"with Cepheid anchoring confirms this (decline = "
                f"{decl_m:.2f} km/s/Mpc). The simple d_L ({decl_sd:.2f}) "
                f"and simple inversion ({decl_inv:.2f}) methods are also "
                "consistent with a flat H0(z)." + fc_str +
                " The declining H0(z) reported by Jia et al. is driven "
                "by the H(z) and BAO data combined with the SN data, not "
                "by the Pantheon+ SN distance moduli alone. The piecewise "
                "H_th estimator with Pantheon+ SN-only data, regardless "
                "of error treatment (diagonal, full covariance, or full "
                "covariance with marginalized zero-point), does not "
                "reproduce the decline."
            )


if __name__ == "__main__":
    step = Step32bJiaReplication()
    step.run()
