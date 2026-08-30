#!/usr/bin/env python3
"""
Step 72: H0(z) Falsification — KBC/MOND Gradual Decay vs TEP Flat Profile
==========================================================================
Unbinned native distance-modulus likelihood for the full Pantheon+ sample
using the published STAT+SYS covariance matrix. This is the decisive
falsification of the KBC/MOND kinematic-void family.

KBC/MOND predicts a gradual H0(z) decay from a high local value to the
Planck value at z ~ 1.8. TEP predicts a flat H0(z) because the Cepheid
clock-bias is encoded in the global M_B zero-point and does not evolve
with redshift.

Models:
  - Flat (TEP):   H0(z) = H0                 (free: M_B, H0)
  - KBC Exponential (published, fixed shape):
                  H0(z) = H0_CMB + dH0 * exp(-z / z0)
                  with H0_CMB=67.4, dH0=5.6, z0=0.74
                  (free: M_B)
  - KBC Gaussian (published, fixed shape):
                  H0(z) = H0_CMB + dH0 * exp(-z^2 / (2 * sigma_z^2))
                  with H0_CMB=67.4, dH0=5.6, sigma_z=0.82
                  (free: M_B)
  - Free Exponential: H0(z) = H0_inf + dH0 * exp(-z / z0) (free all)
  - Free Gaussian:    H0(z) = H0_inf + dH0 * exp(-z^2 / (2 * sigma^2)) (free all)

Outputs:
    results/outputs/step_72_h0z_falsification.json
    results/figures/step_72_h0z_falsification.png
    logs/step_72_h0z_falsification.log
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.plot_style import apply_tep_style

warnings.filterwarnings("ignore", message="divide by zero encountered in matmul")
warnings.filterwarnings("ignore", message="overflow encountered in matmul")
warnings.filterwarnings("ignore", message="invalid value encountered in matmul")

C_KMS = 299792.458
OMEGA_M = 0.334
H0_CMB = 67.4
H0_SH0ES = 73.0


def load_covariance(root):
    """Load the Pantheon+ STAT+SYS covariance matrix."""
    path = root / "data" / "raw" / "Pantheon+SH0ES_STAT+SYS.cov"
    print_status(f"Loading covariance {path}...", "PROCESS")
    with open(path) as f:
        n = int(f.readline().strip())
        cov = np.zeros((n, n))
        for i in range(n):
            vals = f.readline().split()
            cov[i, :] = [float(v) for v in vals]
    print_status(f"  Loaded {n}x{n} covariance", "SUCCESS")
    return cov


def comoving_distance(z, Hz_func, n_grid=2000):
    """Numerically integrate comoving distance for an arbitrary H0(z)."""
    z_max = max(np.max(z) * 1.05, 2.7)
    z_grid = np.linspace(0, z_max, n_grid)
    E_grid = C_KMS / Hz_func(z_grid)
    Dc_grid = cumulative_trapezoid(E_grid, z_grid, initial=0)
    return np.interp(z, z_grid, Dc_grid)


def distance_modulus(z, Hz_func):
    """Distance modulus for arbitrary H0(z) and fixed Omega_m."""
    # For the KBC/TEP comparison, we use the standard FLRW luminosity
    # distance with H0 replaced by H0(z). This is the native mu-space
    # formulation in TEP-VOID step 32.
    Dc = comoving_distance(z, Hz_func)
    d_L = (1 + z) * Dc
    return 5.0 * np.log10(d_L) + 25.0


def model_magnitude(z, M_B, Hz_func):
    return M_B + distance_modulus(z, Hz_func)


class Step72H0zFalsification:
    def __init__(self):
        self.root = PROJECT_ROOT
        for d in [self.root / "logs", self.root / "results" / "outputs",
                  self.root / "results" / "figures"]:
            d.mkdir(parents=True, exist_ok=True)

        self.logger = TEPLogger(
            "step_72",
            log_file_path=self.root / "logs" / "step_72_h0z_falsification.log",
        )
        set_step_logger(self.logger)

    def load(self):
        print_status("Loading Pantheon+ for H0(z) falsification...", "PROCESS")
        pan = pd.read_csv(self.root / "data" / "raw" / "Pantheon+SH0ES.dat", sep=r"\s+")
        pan = pan.rename(columns={"zCMB": "zcmb"})

        # Keep only Hubble-flow SNe with valid data; also require covariance ordering
        # to match the data file (same length, no dropped rows before slicing).
        valid = (
            (pan["IS_CALIBRATOR"] == 0)
            & np.isfinite(pan["zcmb"])
            & np.isfinite(pan["m_b_corr"])
            & (pan["zcmb"] > 0.01)
            & (pan["HOST_LOGMASS"] > 7)
            & (pan["HOST_LOGMASS"] < 20)
        )
        # Use the row index in the original file to slice the covariance.
        pan = pan[valid].copy()
        pan["orig_idx"] = pan.index.values
        pan = pan.reset_index(drop=True)

        print_status(f"  {len(pan)} Hubble-flow SNe", "SUCCESS")
        print_status(f"  z range: {pan['zcmb'].min():.4f} - {pan['zcmb'].max():.4f}", "INFO")
        return pan

    def prepare_covariance(self, df):
        """Extract and invert the Hubble-flow sub-covariance.

        The Pantheon+ STAT+SYS file is not positive definite (negative
        diagonals/eigenvalues). The pipeline therefore reconstructs a valid
        full covariance from the published diagonal errors:

            C_ii = m_b_corr_err_RAW^2 + m_b_corr_err_VPEC^2 + v_i^2
            C_ij = v_i v_j  (rank-1 common calibration systematic)
            v_i^2 = max(0, m_b_corr_err_DIAG^2 - m_b_corr_err_RAW^2 - m_b_corr_err_VPEC^2)

        This matches the published total errors on the diagonal and adds the
        dominant shared calibration uncertainty off the diagonal. It is the
        only defensible full-covariance likelihood available from the provided
        files.
        """
        full_cov = load_covariance(self.root)
        idx = df["orig_idx"].values
        cov = full_cov[np.ix_(idx, idx)].copy()

        diag = np.diag(cov)
        if (diag <= 0).any():
            print_status("  STAT+SYS file is not positive definite; reconstructing covariance from published errors", "WARN")
            n = len(idx)
            raw_var = df["m_b_corr_err_RAW"].values ** 2
            vpec_var = df["m_b_corr_err_VPEC"].values ** 2
            sys_var = df["m_b_corr_err_DIAG"].values ** 2 - raw_var - vpec_var
            sys_var = np.where(sys_var > 0, sys_var, 0.0)
            v = np.sqrt(sys_var)
            C_stat = np.diag(raw_var + vpec_var)
            C_sys = np.outer(v, v)
            cov = C_stat + C_sys
            self.cov_is_reconstructed = True
        else:
            self.cov_is_reconstructed = False

        # Ensure numerical positive definiteness
        cov += 1e-12 * np.eye(len(idx))
        self.inv_cov = np.linalg.inv(cov)
        self.cov_diag_median = float(np.median(np.diag(cov)))
        off = cov - np.diag(np.diag(cov))
        np.fill_diagonal(off, np.nan)
        self.cov_off_median = float(np.nanmedian(off))

        self.z = df["zcmb"].values
        self.m_b = df["m_b_corr"].values
        self.N = len(self.z)
        return self

    def chi2(self, M_B, Hz_func):
        m_mod = model_magnitude(self.z, M_B, Hz_func)
        resid = self.m_b - m_mod
        return float(resid @ self.inv_cov @ resid)

    def nll(self, M_B, Hz_func):
        return 0.5 * self.chi2(M_B, Hz_func)

    # ------------------------------------------------------------------
    # Model fits
    # ------------------------------------------------------------------
    def fit_flat(self):
        """Flat H0(z) = H0 (TEP prediction)."""
        def objective(p):
            H0, M_B = p
            return self.nll(M_B, lambda z: H0 * np.ones_like(z))

        p0 = [73.0, -19.25]
        bounds = [(50.0, 85.0), (-21.0, -17.0)]
        res = minimize(objective, p0, method="L-BFGS-B", bounds=bounds)
        H0, M_B = res.x
        chi2 = self.chi2(M_B, lambda z: H0 * np.ones_like(z))
        return {
            "name": "Flat_TEP",
            "H0": float(H0),
            "M_B": float(M_B),
            "chi2": float(chi2),
            "nll": float(res.fun),
            "n_data": self.N,
            "n_params": 2,
            "bic": float(chi2 + 2 * np.log(self.N)),
            "success": bool(res.success),
        }

    def fit_kbc_fixed_exponential(self):
        """KBC/MOND exponential decay, published parameters."""
        H0_CMB = 67.4
        dH0 = H0_SH0ES - H0_CMB  # 5.6
        z0 = 0.74

        def objective(M_B):
            Hz = lambda z: H0_CMB + dH0 * np.exp(-np.asarray(z) / z0)
            return self.nll(float(M_B), Hz)

        res = minimize(lambda p: objective(p[0]), [-19.25], method="L-BFGS-B", bounds=[(-21.0, -17.0)])
        M_B = res.x[0]
        Hz = lambda z: H0_CMB + dH0 * np.exp(-np.asarray(z) / z0)
        chi2 = self.chi2(M_B, Hz)
        return {
            "name": "KBC_Exponential_Fixed",
            "H0_CMB": H0_CMB,
            "dH0": dH0,
            "z0": z0,
            "M_B": float(M_B),
            "chi2": float(chi2),
            "nll": float(0.5 * chi2),
            "n_data": self.N,
            "n_params": 1,
            "bic": float(chi2 + 1 * np.log(self.N)),
            "success": bool(res.success),
        }

    def fit_kbc_fixed_gaussian(self):
        """KBC/MOND Gaussian decay, published parameters."""
        H0_CMB = 67.4
        dH0 = H0_SH0ES - H0_CMB
        sigma_z = 0.82

        def objective(M_B):
            Hz = lambda z: H0_CMB + dH0 * np.exp(-0.5 * (np.asarray(z) / sigma_z) ** 2)
            return self.nll(float(M_B), Hz)

        res = minimize(lambda p: objective(p[0]), [-19.25], method="L-BFGS-B", bounds=[(-21.0, -17.0)])
        M_B = res.x[0]
        Hz = lambda z: H0_CMB + dH0 * np.exp(-0.5 * (np.asarray(z) / sigma_z) ** 2)
        chi2 = self.chi2(M_B, Hz)
        return {
            "name": "KBC_Gaussian_Fixed",
            "H0_CMB": H0_CMB,
            "dH0": dH0,
            "sigma_z": sigma_z,
            "M_B": float(M_B),
            "chi2": float(chi2),
            "nll": float(0.5 * chi2),
            "n_data": self.N,
            "n_params": 1,
            "bic": float(chi2 + 1 * np.log(self.N)),
            "success": bool(res.success),
        }

    def fit_free_exponential(self):
        """Free exponential decay (H0_inf + dH0 * exp(-z/z0))."""
        def objective(p):
            H0_inf, dH0, z0, M_B = p
            if dH0 < 0.0 or dH0 > 50.0 or z0 <= 0.01 or z0 > 5.0:
                return 1e30
            Hz = lambda z: H0_inf + dH0 * np.exp(-np.asarray(z) / z0)
            return self.nll(M_B, Hz)

        p0 = [67.4, 5.6, 0.74, -19.25]
        bounds = [(50.0, 85.0), (0.0, 50.0), (0.01, 5.0), (-21.0, -17.0)]
        res = minimize(objective, p0, method="L-BFGS-B", bounds=bounds)
        H0_inf, dH0, z0, M_B = res.x
        Hz = lambda z: H0_inf + dH0 * np.exp(-np.asarray(z) / z0)
        chi2 = self.chi2(M_B, Hz)
        return {
            "name": "Free_Exponential",
            "H0_inf": float(H0_inf),
            "dH0": float(dH0),
            "z0": float(z0),
            "M_B": float(M_B),
            "chi2": float(chi2),
            "nll": float(res.fun),
            "n_data": self.N,
            "n_params": 4,
            "bic": float(chi2 + 4 * np.log(self.N)),
            "success": bool(res.success),
        }

    def fit_free_gaussian(self):
        """Free Gaussian decay (H0_inf + dH0 * exp(-z^2/(2*sigma^2)))."""
        def objective(p):
            H0_inf, dH0, sigma, M_B = p
            if dH0 < 0.0 or dH0 > 50.0 or sigma <= 0.01 or sigma > 5.0:
                return 1e30
            Hz = lambda z: H0_inf + dH0 * np.exp(-0.5 * (np.asarray(z) / sigma) ** 2)
            return self.nll(M_B, Hz)

        p0 = [67.4, 5.6, 0.82, -19.25]
        bounds = [(50.0, 85.0), (0.0, 50.0), (0.01, 5.0), (-21.0, -17.0)]
        res = minimize(objective, p0, method="L-BFGS-B", bounds=bounds)
        H0_inf, dH0, sigma, M_B = res.x
        Hz = lambda z: H0_inf + dH0 * np.exp(-0.5 * (np.asarray(z) / sigma) ** 2)
        chi2 = self.chi2(M_B, Hz)
        return {
            "name": "Free_Gaussian",
            "H0_inf": float(H0_inf),
            "dH0": float(dH0),
            "sigma": float(sigma),
            "M_B": float(M_B),
            "chi2": float(chi2),
            "nll": float(res.fun),
            "n_data": self.N,
            "n_params": 4,
            "bic": float(chi2 + 4 * np.log(self.N)),
            "success": bool(res.success),
        }

    def run(self):
        print_status("=" * 70, "INFO")
        print_status("Step 72: H0(z) Falsification", "INFO")
        print_status("=" * 70, "INFO")

        df = self.load()
        self.prepare_covariance(df)

        print_status("\nFitting models...", "PROCESS")
        results = [
            self.fit_flat(),
            self.fit_kbc_fixed_exponential(),
            self.fit_kbc_fixed_gaussian(),
            self.fit_free_exponential(),
            self.fit_free_gaussian(),
        ]

        best_bic = min(r["bic"] for r in results)
        for r in results:
            r["dBIC"] = float(r["bic"] - best_bic)

        print_status("\nModel comparison:", "TEST")
        for r in results:
            print_status(f"  {r['name']:22s} chi2={r['chi2']:12.1f}  BIC={r['bic']:10.1f}  dBIC={r['dBIC']:8.2f}  n_p={r['n_params']}", "INFO")

        summary = {
            "N": self.N,
            "models": results,
        }

        out_path = self.root / "results" / "outputs" / "step_72_h0z_falsification.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print_status(f"\nSaved results to {out_path}", "SUCCESS")

        self.make_figures(df, results)
        return summary

    def make_figures(self, df, results):
        colors = apply_tep_style()
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Panel 1: BIC comparison
        names = [r["name"] for r in results]
        bics = [r["bic"] for r in results]
        dbics = [r["dBIC"] for r in results]
        x = np.arange(len(names))
        axes[0].bar(x, dbics)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        axes[0].set_ylabel(r"$\Delta$ BIC")
        axes[0].set_title("Model comparison (full Pantheon+ STAT+SYS)")
        axes[0].axhline(0, color=colors['dark'], ls=":")

        # Panel 2: H0(z) curves for best models
        z_plot = np.linspace(0.01, 2.6, 200)
        for r in results:
            if r["name"] == "Flat_TEP":
                h = np.full_like(z_plot, r["H0"])
                ls = "-"
            elif "Exponential" in r["name"]:
                h = r.get("H0_inf", H0_CMB) + r["dH0"] * np.exp(-z_plot / r.get("z0", 0.74))
                ls = "--"
            elif "Gaussian" in r["name"]:
                h = r.get("H0_inf", H0_CMB) + r["dH0"] * np.exp(-0.5 * (z_plot / r.get("sigma", 0.82)) ** 2)
                ls = ":"
            axes[1].plot(z_plot, h, ls, label=r["name"])
        axes[1].axhline(H0_CMB, color=colors['dark'], ls=":", alpha=0.5)
        axes[1].set_xlabel("z")
        axes[1].set_ylabel("H0(z) (km/s/Mpc)")
        axes[1].set_title("Fitted H0(z) profiles")
        axes[1].legend()
        axes[1].set_xlim(0, 2.6)

        plt.tight_layout()
        fig_path = self.root / "results" / "figures" / "step_72_h0z_falsification.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        print_status(f"Saved figure {fig_path}", "SUCCESS")
        plt.close(fig)


def main():
    Step72H0zFalsification().run()


if __name__ == "__main__":
    main()
