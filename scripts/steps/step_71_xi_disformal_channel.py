#!/usr/bin/env python3
"""
Step 71: X_i Disformal Channel — Local TEP Signal in Pantheon+ SNe
==================================================================
Tests the inhomogeneous disformal transport channel: the SN light-curve
stretch x1 and Hubble residuals should correlate with the host potential
coordinate X_i, not with a global sky dipole.

The Mount Wilson Equivalence Theorem (step_70) shows that global temporal
shear is conformally degenerate with kinematic bulk flow. TEP becomes
observationally distinguishable only through local, non-integrable
inhomogeneous disformal transport inside galactic potential wells. This
step targets that channel.

Predictions:
  1. x1 (stretch) is modified by the local potential: x1_obs = x1_int * q_i.
     For the observed negative x1_int, a clock compression q_i < 1 in deeper
     potentials makes x1_obs less negative (larger). This produces a
     POSITIVE slope dx1/dX_i.
  2. The SALT3 standardization uses the biased x1, so the Hubble residual
     may retain little additional X_i information once x1 is included.
  3. The X_i effect should persist after removing the standard mass step,
     because X_i contains additional information about the gravitational
     potential beyond the binary logM=10.5 cut.

Models:
  - M0: x1 = a
  - Mz:  x1 = a + b*z + c*logM
  - MX:  x1 = a + b*z + c*logM + d*X_i
  - HR  = a
  - HR  = a + b*z + c*logM
  - HR  = a + b*z + c*logM + d*mass_step
  - HR  = a + b*z + c*logM + d*X_i
  - HR  = a + b*z + c*logM + d*mass_step + e*x1 + f*X_i

Outputs:
    results/outputs/step_71_xi_disformal_channel.json
    results/figures/step_71_xi_disformal_channel.png
    logs/step_71_xi_disformal_channel.log
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
from scripts.utils.screening import U_REF_SCREENED, compute_screening

warnings.filterwarnings("ignore", message="divide by zero encountered in matmul")
warnings.filterwarnings("ignore", message="overflow encountered in matmul")
warnings.filterwarnings("ignore", message="invalid value encountered in matmul")

C_KMS = 299792.458
H0_REF = 73.04
OMEGA_M = 0.334
V_STAR = 10000.0
MASS_STEP_CUT = 10.5


def mu_lcdm(z):
    """Flat LambdaCDM distance modulus at H0_REF, OMEGA_M."""
    z_fine = np.linspace(0, max(np.max(z) + 0.01, 2.7), 8000)
    E_fine = np.sqrt(OMEGA_M * (1 + z_fine) ** 3 + (1 - OMEGA_M))
    d_c_fine = cumulative_trapezoid(1.0 / E_fine, z_fine, initial=0)
    d_c = np.interp(z, z_fine, d_c_fine)
    d_l = (1 + z) * d_c * C_KMS / H0_REF
    return 5.0 * np.log10(d_l) + 25.0


def wls_fit(y, X, w):
    """Weighted least squares. Returns params, chi2, nll."""
    Xw = X * np.sqrt(w)[:, None]
    yw = y * np.sqrt(w)
    params, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    resid = y - X @ params
    chi2 = np.sum(w * resid ** 2)
    nll = 0.5 * chi2
    return params, chi2, nll


def bic(nll, n, k):
    return 2 * nll + k * np.log(n)


class Step71XiDisformalChannel:
    """Step 71: X_i disformal channel for Pantheon+ stretch and Hubble residuals."""

    def __init__(self):
        self.root = PROJECT_ROOT
        for d in [self.root / "logs", self.root / "results" / "outputs",
                  self.root / "results" / "figures"]:
            d.mkdir(parents=True, exist_ok=True)

        self.logger = TEPLogger(
            "step_71",
            log_file_path=self.root / "logs" / "step_71_xi_disformal_channel.log",
        )
        set_step_logger(self.logger)

    def load(self):
        print_status("Loading Pantheon+ and measured V_rot (deep) catalog...", "PROCESS")

        pan = pd.read_csv(
            self.root / "data" / "raw" / "Pantheon+SH0ES.dat", sep=r"\s+"
        )
        pan = pan.rename(columns={"zCMB": "zcmb", "RA": "ra", "DEC": "dec"})

        # Use the deep catalog with measured V_rot; compute X_i on the fly.
        vrot = pd.read_csv(self.root / "data" / "processed" / "pantheon_host_vrot_deep.csv")
        measured = vrot[vrot["v_rot_deep"].notna() & (vrot["v_rot_deep"] > 0)].copy()
        measured = measured.rename(columns={"v_rot_deep": "V_rot"})
        pgc = measured["pgc"].fillna(0).astype(int).values
        S_total = compute_screening(pgc, self.root)
        measured["S_total"] = S_total
        measured["U_i"] = (measured["V_rot"].values ** 2) / 2.0
        measured["X_i"] = (S_total * measured["U_i"].values - U_REF_SCREENED) / C_KMS ** 2

        # Merge
        measured = measured.rename(columns={"v_rot_source": "V_rot_source"})
        df = pan.merge(
            measured[["CID", "V_rot", "X_i", "V_rot_source"]],
            on="CID", how="left",
        )
        df = df[df["X_i"].notna() & np.isfinite(df["X_i"])].copy()
        df["mu_lcdm"] = mu_lcdm(df["zcmb"].values)
        df["HR"] = df["MU_SH0ES"] - df["mu_lcdm"]

        # CMB dipole direction for any residual global component
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        cmb = SkyCoord(l=264.021 * u.deg, b=48.253 * u.deg, frame="galactic").icrs
        coords = SkyCoord(ra=df["ra"].values * u.deg, dec=df["dec"].values * u.deg, frame="icrs")
        sep = coords.separation(cmb)
        df["cos_theta_cmb"] = np.cos(np.radians(sep.deg))

        # Restrict to Hubble flow
        df = df[df["IS_CALIBRATOR"] == 0].copy()
        df = df[df["HOST_LOGMASS"].notna() & (df["HOST_LOGMASS"] > 7)].copy()

        print_status(f"  {len(df)} SNe with measured V_rot and mass", "SUCCESS")
        print_status(f"  V_rot source: {df['V_rot_source'].value_counts().to_dict()}", "INFO")
        print_status(f"  X_i range: {df['X_i'].min():.2e} - {df['X_i'].max():.2e}", "INFO")

        return df

    def fit_x1_models(self, df):
        """Fit x1 as a function of X_i, redshift, mass, and global dipole."""
        y = df["x1"].values
        err = df["x1ERR"].values
        w = np.where(err > 0, 1.0 / err ** 2, 0.0)
        X = df["X_i"].values
        z = df["zcmb"].values
        logm = df["HOST_LOGMASS"].values
        V = z * C_KMS
        cos = df["cos_theta_cmb"].values

        results = {}
        models = {
            "M0": np.column_stack([np.ones(len(y))]),
            "Mz": np.column_stack([np.ones(len(y)), z, logm]),
            "MX": np.column_stack([np.ones(len(y)), z, logm, X]),
        }
        for name, design in models.items():
            params, chi2, nll = wls_fit(y, design, w)
            results[name] = {
                "chi2": float(chi2),
                "nll": float(nll),
                "bic": float(bic(nll, len(y), design.shape[1])),
                "n_data": int(len(y)),
                "n_params": int(design.shape[1]),
                "params": [float(p) for p in params],
            }
            if name == "MX":
                results[name]["d_x1_dX"] = float(params[3])
        best = min(results, key=lambda k: results[k]["bic"])
        for name in results:
            results[name]["dBIC"] = float(results[name]["bic"] - results[best]["bic"])
        return results

    def fit_hr_models(self, df):
        """Fit Hubble residual as a function of X_i, redshift, mass step, and x1."""
        y = df["HR"].values
        err = df["MU_SH0ES_ERR_DIAG"].values
        w = np.where(err > 0, 1.0 / err ** 2, 0.0)
        X = df["X_i"].values
        z = df["zcmb"].values
        logm = df["HOST_LOGMASS"].values
        mass_step = (logm >= MASS_STEP_CUT).astype(float)
        x1 = df["x1"].values

        results = {}
        models = {
            "M0": np.column_stack([np.ones(len(y))]),
            "Mzlogm": np.column_stack([np.ones(len(y)), z, logm]),
            "Mmass": np.column_stack([np.ones(len(y)), z, logm, mass_step]),
            "MX": np.column_stack([np.ones(len(y)), z, logm, X]),
            "MmassX": np.column_stack([np.ones(len(y)), z, logm, mass_step, X]),
            "MmassXx1": np.column_stack([np.ones(len(y)), z, logm, mass_step, x1, X]),
        }
        for name, design in models.items():
            params, chi2, nll = wls_fit(y, design, w)
            results[name] = {
                "chi2": float(chi2),
                "nll": float(nll),
                "bic": float(bic(nll, len(y), design.shape[1])),
                "n_data": int(len(y)),
                "n_params": int(design.shape[1]),
                "params": [float(p) for p in params],
            }
            if name in ("MX", "MmassX", "MmassXx1"):
                results[name]["slope_X"] = float(params[-1])
        best = min(results, key=lambda k: results[k]["bic"])
        for name in results:
            results[name]["dBIC"] = float(results[name]["bic"] - results[best]["bic"])
        return results

    def binned_visualisation(self, df):
        """Compute binned X_i-step for figure and robustness."""
        df = df.copy()
        df["X_bin"] = pd.qcut(df["X_i"], 5, duplicates="drop")
        out = []
        for bin_label, g in df.groupby("X_bin", observed=True):
            out.append({
                "X_low": float(bin_label.left),
                "X_high": float(bin_label.right),
                "N": int(len(g)),
                "x1_mean": float(g["x1"].mean()),
                "x1_err": float(g["x1"].std() / np.sqrt(len(g))),
                "hr_mean": float(g["HR"].mean()),
                "hr_err": float(g["HR"].std() / np.sqrt(len(g))),
                "X_med": float(g["X_i"].median()),
            })
        return out

    def run(self):
        print_status("=" * 70, "INFO")
        print_status("Step 71: X_i Disformal Channel", "INFO")
        print_status("=" * 70, "INFO")

        df = self.load()

        res_x1 = self.fit_x1_models(df)
        res_hr = self.fit_hr_models(df)
        binned = self.binned_visualisation(df)

        # Log summaries
        print_status("\nx1 models:", "TEST")
        for name in res_x1:
            r = res_x1[name]
            extra = f" d_x1/dX={r['d_x1_dX']:.2e}" if "d_x1_dX" in r else ""
            print_status(f"  {name:10s} BIC={r['bic']:.1f} dBIC={r['dBIC']:.2f}{extra}", "INFO")

        print_status("\nHubble residual models:", "TEST")
        for name in res_hr:
            r = res_hr[name]
            extra = f" dHR/dX={r['slope_X']:.2e}" if "slope_X" in r else ""
            print_status(f"  {name:10s} BIC={r['bic']:.1f} dBIC={r['dBIC']:.2f}{extra}", "INFO")

        summary = {
            "N": int(len(df)),
            "x1_models": res_x1,
            "hr_models": res_hr,
            "binned_steps": binned,
        }

        out_path = self.root / "results" / "outputs" / "step_71_xi_disformal_channel.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print_status(f"\nSaved results to {out_path}", "SUCCESS")

        self.make_figures(df, res_x1, res_hr, binned)
        return summary

    def make_figures(self, df, res_x1, res_hr, binned):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # x1 vs X_i: partial regression line from the best model
        ax = axes[0, 0]
        X = df["X_i"].values
        ax.scatter(X * 1e7, df["x1"].values, s=5, alpha=0.3)
        best_x1 = min(res_x1, key=lambda k: res_x1[k]["bic"])
        slope = res_x1[best_x1].get("d_x1_dX", 0.0)
        x_plot = np.linspace(X.min(), X.max(), 100)
        y_mean = df["x1"].mean()
        x_mean = X.mean()
        y_plot = y_mean + slope * (x_plot - x_mean)
        ax.plot(x_plot * 1e7, y_plot, "r--")
        ax.set_xlabel(r"$X_i \times 10^7$")
        ax.set_ylabel(r"$x_1$")
        ax.set_title(f"x1 vs $X_i$ (best: {best_x1})")

        # Hubble residual vs X_i: partial regression line from the best model
        ax = axes[0, 1]
        ax.scatter(X * 1e7, df["HR"].values, s=5, alpha=0.3)
        best_hr = min(res_hr, key=lambda k: res_hr[k]["bic"])
        slope = res_hr[best_hr].get("slope_X", 0.0)
        y_mean = df["HR"].mean()
        y_plot = y_mean + slope * (x_plot - x_mean)
        ax.plot(x_plot * 1e7, y_plot, "r--")
        ax.axhline(0, color="k", ls=":")
        ax.set_xlabel(r"$X_i \times 10^7$")
        ax.set_ylabel("Hubble residual (mag)")
        ax.set_title(f"HR vs $X_i$ (best: {best_hr})")

        # Binned x1
        ax = axes[1, 0]
        Xmed = [b["X_med"] * 1e7 for b in binned]
        x1m = [b["x1_mean"] for b in binned]
        x1e = [b["x1_err"] for b in binned]
        ax.errorbar(Xmed, x1m, yerr=x1e, fmt="o-")
        ax.set_xlabel(r"$X_i \times 10^7$")
        ax.set_ylabel(r"Mean $x_1$")
        ax.set_title("Binned x1 step")

        # Binned HR
        ax = axes[1, 1]
        hrm = [b["hr_mean"] for b in binned]
        hre = [b["hr_err"] for b in binned]
        ax.errorbar(Xmed, hrm, yerr=hre, fmt="o-", color="C1")
        ax.axhline(0, color="k", ls=":")
        ax.set_xlabel(r"$X_i \times 10^7$")
        ax.set_ylabel("Mean Hubble residual (mag)")
        ax.set_title("Binned HR step")

        plt.tight_layout()
        fig_path = self.root / "results" / "figures" / "step_71_xi_disformal_channel.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        print_status(f"Saved figure {fig_path}", "SUCCESS")
        plt.close(fig)


def main():
    Step71XiDisformalChannel().run()


if __name__ == "__main__":
    main()
