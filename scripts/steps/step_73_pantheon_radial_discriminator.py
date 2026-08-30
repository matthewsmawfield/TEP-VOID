#!/usr/bin/env python3
"""
Step 73: Pantheon+ Low-z Temporal-Kinematic Discriminator (Gate F v2 audit)
==========================================================================

This is a diagnostic implementation, not a final TEP-positive test.
It is restricted to z < 0.07, where the low-redshift kernels

    delta_mu_K ~ (5/ln 10) (B dot n) / V
    delta_mu_T ~ T (V / V_*) cos_theta

are approximately valid. It tests two redshift frames (zCMB, zHD) and
two dipole axes (CMB solar dipole and the Watkins et al. CF4 bulk-flow
direction) to audit the sensitivity of the nominal result to these
choices. Pantheon+ is also deduplicated to one row per CID.

The script does not attempt to reproduce the CF4 TF bulk-flow observable
or to derive the TEP forward model from first principles; those are
separate, necessary developments.

Outputs:
    results/outputs/step_73_pantheon_radial_discriminator.json
    results/figures/step_73_pantheon_radial_discriminator.png
    logs/step_73_pantheon_radial_discriminator.log
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
from astropy.coordinates import SkyCoord
import astropy.units as u

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.plot_style import apply_tep_style

warnings.filterwarnings("ignore", message="divide by zero encountered in matmul")
warnings.filterwarnings("ignore", message="overflow encountered in matmul")
warnings.filterwarnings("ignore", message="invalid value encountered in matmul")

C_KMS = 299792.458
CMB_DIPOLE_GAL_L = 264.021
CMB_DIPOLE_GAL_B = 48.253
CF4_BULK_FLOW_GAL_L = 298.0
CF4_BULK_FLOW_GAL_B = -7.0
V_STAR = 10000.0
H0_REF = 73.04
OMEGA_M = 0.334


def _gal_to_unit(l_deg, b_deg):
    c = SkyCoord(l=l_deg * u.deg, b=b_deg * u.deg, frame="galactic").icrs
    return np.array([
        np.cos(np.radians(c.dec.deg)) * np.cos(np.radians(c.ra.deg)),
        np.cos(np.radians(c.dec.deg)) * np.sin(np.radians(c.ra.deg)),
        np.sin(np.radians(c.dec.deg)),
    ])


CMB_DIPOLE_UNIT = _gal_to_unit(CMB_DIPOLE_GAL_L, CMB_DIPOLE_GAL_B)
CF4_BULK_FLOW_UNIT = _gal_to_unit(CF4_BULK_FLOW_GAL_L, CF4_BULK_FLOW_GAL_B)


def ra_dec_to_unit_vectors(ra_deg, dec_deg):
    ra_rad = np.radians(ra_deg)
    dec_rad = np.radians(dec_deg)
    return np.column_stack([
        np.cos(dec_rad) * np.cos(ra_rad),
        np.cos(dec_rad) * np.sin(ra_rad),
        np.sin(dec_rad),
    ])


def mu_lcdm(z):
    """Flat LambdaCDM distance modulus at H0_REF, OMEGA_M."""
    z_fine = np.linspace(0, max(np.max(z) + 0.01, 2.7), 8000)
    E_fine = np.sqrt(OMEGA_M * (1 + z_fine) ** 3 + (1 - OMEGA_M))
    d_c_fine = cumulative_trapezoid(1.0 / E_fine, z_fine, initial=0)
    d_c = np.interp(z, z_fine, d_c_fine)
    d_l = (1 + z) * d_c * C_KMS / H0_REF
    return 5.0 * np.log10(d_l) + 25.0


def nll(params, model_type, y, v, x, nx, ny, nz, cos_theta, w):
    alpha = params[0]
    if model_type == "M0":
        y_pred = alpha * np.ones_like(y)
    elif model_type == "MK":
        bx, by, bz = params[1:4]
        b_dot_n = bx * nx + by * ny + bz * nz
        y_pred = alpha + (5.0 / np.log(10.0)) * (b_dot_n / v)
    elif model_type == "MT":
        T_amp = params[1]
        y_pred = alpha + T_amp * (v / V_STAR) * cos_theta
    elif model_type == "MKT":
        T_amp = params[1]
        bx, by, bz = params[2:5]
        b_dot_n = bx * nx + by * ny + bz * nz
        y_pred = alpha + T_amp * (v / V_STAR) * cos_theta + (5.0 / np.log(10.0)) * (b_dot_n / v)
    elif model_type == "MX":
        gamma = params[1]
        y_pred = alpha + (5.0 / np.log(10.0)) * gamma * x / v
    elif model_type == "MKX":
        bx, by, bz = params[1:4]
        gamma = params[4]
        b_dot_n = bx * nx + by * ny + bz * nz
        y_pred = alpha + (5.0 / np.log(10.0)) * (b_dot_n + gamma * x) / v
    resid = y - y_pred
    return 0.5 * np.sum(w * resid * resid)


def fit_model(y, v, x, nx, ny, nz, cos_theta, w, model_type):
    alpha0 = np.median(y)
    if model_type == "M0":
        p0 = [alpha0]
    elif model_type == "MK":
        p0 = [alpha0, 0.0, 0.0, 0.0]
    elif model_type == "MT":
        p0 = [alpha0, 0.0]
    elif model_type == "MKT":
        p0 = [alpha0, 0.0, 0.0, 0.0, 0.0]
    elif model_type == "MX":
        p0 = [alpha0, 0.0]
    elif model_type == "MKX":
        p0 = [alpha0, 0.0, 0.0, 0.0, 0.0]
    res = minimize(nll, p0, args=(model_type, y, v, x, nx, ny, nz, cos_theta, w), method="BFGS")
    return res


def bic(nll_val, n_data, n_params):
    return 2 * nll_val + n_params * np.log(n_data)


def fit_global_models(y, v, x, nx, ny, nz, cos_theta, w):
    models = ["M0", "MK", "MT", "MKT", "MX", "MKX"]
    out = {}
    best_bic = np.inf
    for m in models:
        res = fit_model(y, v, x, nx, ny, nz, cos_theta, w, m)
        n_p = len(res.x)
        b = bic(res.fun, len(y), n_p)
        best_bic = min(best_bic, b)
        out[m] = {"nll": float(res.fun), "bic": float(b), "n_params": n_p,
                  "params": [float(p) for p in res.x]}
    for m in out:
        out[m]["dBIC"] = out[m]["bic"] - best_bic
    return out


def radial_binned_fit(df, z_col, v_col, cos_col, v_bins, z_max):
    """Fit MK in radial bins and compare to the TEP V^2 prediction."""
    out = []
    T_global = float(df["T_global"].iloc[0]) if "T_global" in df.columns else 0.0
    for i in range(len(v_bins) - 1):
        v_min, v_max = v_bins[i], v_bins[i + 1]
        mask = (df[v_col] >= v_min) & (df[v_col] < v_max) & (df[z_col] <= z_max)
        sub = df[mask]
        if len(sub) < 25:
            continue
        y = sub["hubble_resid"].values
        v = sub[v_col].values
        x = sub["x_local"].values
        w = 1.0 / (sub["MU_SH0ES_ERR_DIAG"].values ** 2)
        res = fit_model(y, v, x, sub["nx"].values, sub["ny"].values, sub["nz"].values,
                        sub[cos_col].values, w, "MK")
        bx, by, bz = res.x[1], res.x[2], res.x[3]
        b_vec = np.array([bx, by, bz])
        b_par = float(np.dot(b_vec, CMB_DIPOLE_UNIT if "cmb" in cos_col else CF4_BULK_FLOW_UNIT))
        v_med = float(np.median(v))
        b_app = float(T_global * (v_med ** 2 / V_STAR) * (np.log(10.0) / 5.0))
        out.append({
            "v_min": float(v_min), "v_max": float(v_max), "N": int(len(sub)),
            "v_med": v_med, "B_par": b_par, "B_mag": float(np.linalg.norm(b_vec)),
            "T_prediction": b_app,
        })
    return out


class Step73PantheonRadialDiscriminator:
    def __init__(self):
        self.root = PROJECT_ROOT
        for d in [self.root / "logs", self.root / "results" / "outputs",
                  self.root / "results" / "figures"]:
            d.mkdir(parents=True, exist_ok=True)
        self.logger = TEPLogger(
            "step_73",
            log_file_path=self.root / "logs" / "step_73_pantheon_radial_discriminator.log",
        )
        set_step_logger(self.logger)

    def load(self):
        print_status("Loading Pantheon+ for low-z Gate F v2 audit...", "PROCESS")
        path = self.root / "data" / "raw" / "Pantheon+SH0ES.dat"
        if not path.exists():
            print_status(f"Pantheon+ not found at {path}", "ERROR")
            return None
        df = pd.read_csv(path, sep=r"\s+")

        print_status(f"  Raw file: {len(df)} rows, {df['CID'].nunique()} unique CIDs", "INFO")

        # Deduplicate: one row per CID, prefer the row used in SH0ES Hubble flow
        df = df.sort_values("USED_IN_SH0ES_HF", ascending=False)
        df = df.drop_duplicates(subset="CID", keep="first")

        n_vecs = ra_dec_to_unit_vectors(df["RA"].values, df["DEC"].values)
        df["nx"] = n_vecs[:, 0]
        df["ny"] = n_vecs[:, 1]
        df["nz"] = n_vecs[:, 2]
        df["cos_theta_cmb"] = n_vecs @ CMB_DIPOLE_UNIT
        df["cos_theta_cf4"] = n_vecs @ CF4_BULK_FLOW_UNIT

        # Reference distance moduli for zCMB and zHD
        df["mu_lcdm_zcmb"] = mu_lcdm(df["zCMB"].values)
        df["mu_lcdm_zhd"] = mu_lcdm(df["zHD"].values)

        # Hubble residuals
        df["resid_zcmb"] = df["MU_SH0ES"].values - df["mu_lcdm_zcmb"].values
        df["resid_zhd"] = df["MU_SH0ES"].values - df["mu_lcdm_zhd"].values

        # Velocities
        df["v_cmb"] = df["zCMB"].values * C_KMS
        df["v_hd"] = df["zHD"].values * C_KMS

        # Local host-potential proxy for the topological TEP model
        m = df["HOST_LOGMASS"].values
        m_ref = np.nanmedian(m)
        df["x_local"] = m - m_ref

        print_status(f"  After CID dedup: {len(df)} rows", "SUCCESS")
        return df

    def run_config(self, df, z_col, v_col, cos_col, z_max, label):
        """Run one (redshift, direction, low-z) configuration."""
        sub = df[(df["IS_CALIBRATOR"] == 0) & (df[z_col] <= z_max)].copy()
        resid_col = "resid_zcmb" if z_col == "zCMB" else "resid_zhd"
        valid = (
            np.isfinite(sub[resid_col]) & np.isfinite(sub["MU_SH0ES"])
            & np.isfinite(sub["MU_SH0ES_ERR_DIAG"]) & (sub["MU_SH0ES_ERR_DIAG"] > 0)
            & np.isfinite(sub[v_col]) & (sub[v_col] > 0)
            & np.isfinite(sub[cos_col])
        )
        sub = sub[valid]
        if len(sub) < 100:
            print_status(f"  {label}: too few SNe ({len(sub)}); skipping", "WARNING")
            return None

        y = sub[resid_col].values
        v = sub[v_col].values
        x = sub["x_local"].values
        w = 1.0 / (sub["MU_SH0ES_ERR_DIAG"].values ** 2)

        global_res = fit_global_models(
            y, v, x, sub["nx"].values, sub["ny"].values, sub["nz"].values,
            sub[cos_col].values, w,
        )

        # carry the best MT T amplitude for the radial TEP prediction
        sub["hubble_resid"] = y
        sub["v"] = v
        sub["T_global"] = global_res["MT"]["params"][1]

        v_bins = [2000, 5000, 10000, 20000, 30000]
        bin_res = radial_binned_fit(sub, z_col, v_col, cos_col, v_bins, z_max)

        return {
            "label": label,
            "N": int(len(sub)),
            "z_max": float(z_max),
            "v_med_min": float(v.min()),
            "v_med_max": float(v.max()),
            "global": global_res,
            "radial_bins": bin_res,
        }

    def run(self):
        print_status("=" * 70, "INFO")
        print_status("Step 73: Pantheon+ Low-z Temporal-Kinematic Discriminator", "INFO")
        print_status("=" * 70, "INFO")

        df = self.load()
        if df is None:
            return

        configs = [
            ("zCMB", "v_cmb", "cos_theta_cmb", 0.07, "zCMB + CMB axis"),
            ("zCMB", "v_cmb", "cos_theta_cf4", 0.07, "zCMB + CF4 axis"),
            ("zHD", "v_hd", "cos_theta_cmb", 0.07, "zHD + CMB axis"),
            ("zHD", "v_hd", "cos_theta_cf4", 0.07, "zHD + CF4 axis"),
        ]

        summary = []
        for z_col, v_col, cos_col, z_max, label in configs:
            res = self.run_config(df, z_col, v_col, cos_col, z_max, label)
            if res is None:
                continue
            summary.append(res)
            print_status(f"\n=== {label} (z < {z_max}, N = {res['N']}) ===", "TEST")
            for m, v in res["global"].items():
                print_status(f"  {m:6s} BIC={v['bic']:10.1f} dBIC={v['dBIC']:7.2f} n_p={v['n_params']}", "INFO")
            print_status("  Radial bins:", "INFO")
            for b in res["radial_bins"]:
                print_status(
                    f"    [{b['v_min']:7.0f}-{b['v_max']:7.0f}] N={b['N']:4d} "
                    f"V_med={b['v_med']:8.0f} B_||={b['B_par']:8.1f} "
                    f"|B|={b['B_mag']:8.1f} TEP predicts {b['T_prediction']:7.1f}",
                    "INFO",
                )

        self.make_figure(summary)

        out_path = self.root / "results" / "outputs" / "step_73_pantheon_radial_discriminator.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print_status(f"\nSaved results to {out_path}", "SUCCESS")

        return summary

    def make_figure(self, summary):
        colors = apply_tep_style()
        n = len(summary)
        if n == 0:
            return
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        for ax, res in zip(axes, summary):
            if not res["radial_bins"]:
                continue
            x = np.array([b["v_med"] for b in res["radial_bins"]])
            y = np.array([b["B_par"] for b in res["radial_bins"]])
            t = np.array([b["T_prediction"] for b in res["radial_bins"]])
            ax.plot(x, y, "o-", label="Fitted kinematic $B_{||}$")
            ax.plot(x, t, "--", color=colors['red'], label="TEP $T \\cdot V^2 / V_*$")
            ax.axhline(0, color=colors['dark'], ls=":", alpha=0.5)
            ax.set_xlabel(r"$V$ (km s$^{-1}$)")
            ax.set_ylabel(r"$B_{||}$ (km s$^{-1}$)")
            ax.set_title(res["label"] + f"\n$N={res['N']}$, $z<{res['z_max']:.2f}")
            ax.legend()
            ax.grid(True)
        fig.suptitle("Pantheon+ low-z temporal-kinematic discriminator (diagnostic)", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(self.root / "results" / "figures" / "step_73_pantheon_radial_discriminator.png",
                    dpi=300, bbox_inches="tight")
        plt.close(fig)
        print_status("\nSaved figure to results/figures/step_73_pantheon_radial_discriminator.png", "SUCCESS")


if __name__ == "__main__":
    Step73PantheonRadialDiscriminator().run()
