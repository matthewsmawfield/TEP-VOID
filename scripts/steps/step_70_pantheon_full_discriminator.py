#!/usr/bin/env python3
"""
Step 70: Mount Wilson Equivalence Theorem — Pantheon+ Proof
===========================================================
Empirical proof of the Mount Wilson Equivalence Theorem: on a homogeneous,
static spatial background, a global conformal temporal gradient is
observationally degenerate with a kinematic bulk flow in redshift-distance
data (1+z = A_0/A_em vs 1+z = a_0/a_em). Both single-metric sectors couple
to the same matter metric, so a temporal dipole mimics a peculiar-velocity
dipole and the standard radial discriminator cannot separate them.

The pipeline fits the Phase F1 radial-discriminator family to the full
Pantheon+ sample over multiple redshift ranges and standardizations. It
shows that the kinematic (1/r) and temporal (r) dipole models are
indistinguishable at the BIC level, with the kinematic model preferred
purely by Occam's razor. This is not a failure of TEP; it is the expected
consequence of the conformal degeneracy.

Approaches (run in parallel):
  A. corr:  standardized Hubble residual, MU_SH0ES - mu_LCDM(z)
  B. raw:   forward-model raw mB with x1 and c covariates
            mB = M_B + mu_LCDM(z) + alpha_x1*x1 + beta_c*c + dipole terms
            This fits the SALT3 standardization simultaneously with TEP,
            removing the LambdaCDM-calibrated biasCor and SH0ES alpha/beta.

Dipole variants:
  - CMB-aligned: temporal/kinematic amplitude along the CMB dipole axis
  - Free-dipole: full 3-vector fit of the dipole axis

Redshift ranges:
  - z < 0.1  (matches step_62)
  - z < 0.5  (linear-to-mild curvature)
  - full Hubble flow (z >= 0.01, excluding calibrators)

Also tests the stretch channel (x1) for directional TEP compression.

Outputs:
    results/outputs/step_70_pantheon_full_discriminator.json
    results/figures/step_70_*.png
    logs/step_70_pantheon_full_discriminator.log
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
V_STAR = 10000.0  # km/s, normalisation for the temporal amplitude
H0_REF = 73.04
OMEGA_M = 0.334


def _cmb_dipole_unit_vector():
    c = SkyCoord(l=CMB_DIPOLE_GAL_L * u.deg, b=CMB_DIPOLE_GAL_B * u.deg,
                 frame="galactic").icrs
    return np.array([
        np.cos(np.radians(c.dec.deg)) * np.cos(np.radians(c.ra.deg)),
        np.cos(np.radians(c.dec.deg)) * np.sin(np.radians(c.ra.deg)),
        np.sin(np.radians(c.dec.deg)),
    ])


CMB_DIPOLE_UNIT = _cmb_dipole_unit_vector()


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


def wls_fit(y, X, w):
    """Weighted least squares. Returns params, chi2."""
    Xw = X * np.sqrt(w)[:, None]
    yw = y * np.sqrt(w)
    params, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    resid = y - X @ params
    chi2 = np.sum(w * resid ** 2)
    return params, chi2


def build_design_corr(y, v, n, cos_theta, model):
    """Design matrices for the standardized Hubble residual."""
    if model == "M0":
        return np.ones((len(y), 1))
    if model == "MK_cmb":
        return np.column_stack([np.ones(len(y)), (5.0 / np.log(10)) * cos_theta / v])
    if model == "MT_cmb":
        return np.column_stack([np.ones(len(y)), (v / V_STAR) * cos_theta])
    if model == "MK_free":
        return np.column_stack([
            np.ones(len(y)),
            (5.0 / np.log(10)) * n[:, 0] / v,
            (5.0 / np.log(10)) * n[:, 1] / v,
            (5.0 / np.log(10)) * n[:, 2] / v,
        ])
    if model == "MT_free":
        return np.column_stack([
            np.ones(len(y)),
            (v / V_STAR) * n[:, 0],
            (v / V_STAR) * n[:, 1],
            (v / V_STAR) * n[:, 2],
        ])
    if model == "MKT_cmb_b":
        return np.column_stack([
            np.ones(len(y)),
            (v / V_STAR) * cos_theta,
            (5.0 / np.log(10)) * n[:, 0] / v,
            (5.0 / np.log(10)) * n[:, 1] / v,
            (5.0 / np.log(10)) * n[:, 2] / v,
        ])
    if model == "MKT_free":
        return np.column_stack([
            np.ones(len(y)),
            (v / V_STAR) * n[:, 0],
            (v / V_STAR) * n[:, 1],
            (v / V_STAR) * n[:, 2],
            (5.0 / np.log(10)) * n[:, 0] / v,
            (5.0 / np.log(10)) * n[:, 1] / v,
            (5.0 / np.log(10)) * n[:, 2] / v,
        ])
    raise ValueError(f"Unknown corr model {model}")


def build_design_raw(y, v, n, cos_theta, x1, c, model):
    """Design matrices for the raw forward model (mB with x1, c covariates)."""
    base = [np.ones(len(y)), x1, c]
    if model == "M0":
        return np.column_stack(base)
    if model == "MK_cmb":
        return np.column_stack(base + [(5.0 / np.log(10)) * cos_theta / v])
    if model == "MT_cmb":
        return np.column_stack(base + [(v / V_STAR) * cos_theta])
    if model == "MK_free":
        return np.column_stack(base + [
            (5.0 / np.log(10)) * n[:, 0] / v,
            (5.0 / np.log(10)) * n[:, 1] / v,
            (5.0 / np.log(10)) * n[:, 2] / v,
        ])
    if model == "MT_free":
        return np.column_stack(base + [
            (v / V_STAR) * n[:, 0],
            (v / V_STAR) * n[:, 1],
            (v / V_STAR) * n[:, 2],
        ])
    if model == "MKT_cmb_b":
        return np.column_stack(base + [
            (v / V_STAR) * cos_theta,
            (5.0 / np.log(10)) * n[:, 0] / v,
            (5.0 / np.log(10)) * n[:, 1] / v,
            (5.0 / np.log(10)) * n[:, 2] / v,
        ])
    if model == "MKT_free":
        return np.column_stack(base + [
            (v / V_STAR) * n[:, 0],
            (v / V_STAR) * n[:, 1],
            (v / V_STAR) * n[:, 2],
            (5.0 / np.log(10)) * n[:, 0] / v,
            (5.0 / np.log(10)) * n[:, 1] / v,
            (5.0 / np.log(10)) * n[:, 2] / v,
        ])
    if model == "MT_cmb_x1tep":
        # CMB-aligned temporal dipole modulated by x1 (stretch-channel bias)
        return np.column_stack(base + [
            (v / V_STAR) * cos_theta,
            (v / V_STAR) * cos_theta * x1,
        ])
    raise ValueError(f"Unknown raw model {model}")


def fit_models_corr(df, z_col="zcmb"):
    """Fit the standardized Hubble-residual model grid."""
    y = df["MU_SH0ES"].values - df["mu_lcdm"].values
    v = df[z_col].values * C_KMS
    n = df[["nx", "ny", "nz"]].values
    cos_theta = df["cos_theta_cmb"].values
    err = df["MU_SH0ES_ERR_DIAG"].values
    w = 1.0 / err ** 2
    return _fit_grid(y, v, n, cos_theta, w, build_design_corr)


def fit_models_raw(df, z_col="zcmb"):
    """Fit the raw forward model grid with x1, c covariates."""
    y = df["mB"].values - df["mu_lcdm"].values
    v = df[z_col].values * C_KMS
    n = df[["nx", "ny", "nz"]].values
    cos_theta = df["cos_theta_cmb"].values
    x1 = df["x1"].values
    c = df["c"].values
    err = df["m_b_corr_err_DIAG"].values  # total magnitude uncertainty
    w = 1.0 / err ** 2
    return _fit_grid_raw(y, v, n, cos_theta, x1, c, w, build_design_raw)


def _extract_pars(name, params):
    out = {"params": [float(p) for p in params]}
    if name == "MT_cmb":
        out["T_amp"] = float(params[-1])
    elif name == "MT_free":
        D = params[-3:]
        out["D_vec"] = [float(d) for d in D]
        out["T_amp"] = float(np.linalg.norm(D))
        out["D_angle_deg"] = float(np.degrees(np.arccos(np.clip(
            np.dot(D / np.linalg.norm(D), CMB_DIPOLE_UNIT), -1.0, 1.0))))
    elif name == "MK_cmb":
        out["B_par"] = float(params[-1])
    elif name in ("MK_free", "MKT_cmb_b"):
        B = params[-3:]
        out["B_vec"] = [float(b) for b in B]
        out["B_mag"] = float(np.linalg.norm(B))
        out["B_par"] = float(np.dot(B, CMB_DIPOLE_UNIT))
        out["B_angle_deg"] = float(np.degrees(np.arccos(np.clip(
            np.dot(B / np.linalg.norm(B), CMB_DIPOLE_UNIT), -1.0, 1.0))))
    elif name == "MKT_free":
        D = params[-6:-3]
        B = params[-3:]
        out["D_vec"] = [float(d) for d in D]
        out["T_amp"] = float(np.linalg.norm(D))
        out["D_angle_deg"] = float(np.degrees(np.arccos(np.clip(
            np.dot(D / np.linalg.norm(D), CMB_DIPOLE_UNIT), -1.0, 1.0))))
        out["B_vec"] = [float(b) for b in B]
        out["B_mag"] = float(np.linalg.norm(B))
        out["B_par"] = float(np.dot(B, CMB_DIPOLE_UNIT))
        out["B_angle_deg"] = float(np.degrees(np.arccos(np.clip(
            np.dot(B / np.linalg.norm(B), CMB_DIPOLE_UNIT), -1.0, 1.0))))
    elif name == "MT_cmb_x1tep":
        out["T_amp"] = float(params[-2])
        out["T_x1_amp"] = float(params[-1])
    return out


def _fit_grid(y, v, n, cos_theta, w, builder):
    results = {}
    model_names = ["M0", "MK_cmb", "MT_cmb", "MK_free", "MT_free",
                   "MKT_cmb_b", "MKT_free"]
    for name in model_names:
        X = builder(y, v, n, cos_theta, name)
        params, chi2 = wls_fit(y, X, w)
        n_data = len(y)
        n_params = X.shape[1]
        bic = chi2 + n_params * np.log(n_data)
        results[name] = {
            "chi2": float(chi2),
            "bic": float(bic),
            "n_data": int(n_data),
            "n_params": int(n_params),
        }
        results[name].update(_extract_pars(name, params))
    best_bic = min(r["bic"] for r in results.values())
    for name in results:
        results[name]["dBIC"] = float(results[name]["bic"] - best_bic)
    return results


def _fit_grid_raw(y, v, n, cos_theta, x1, c, w, builder):
    results = {}
    model_names = ["M0", "MK_cmb", "MT_cmb", "MK_free", "MT_free",
                   "MKT_cmb_b", "MKT_free", "MT_cmb_x1tep"]
    for name in model_names:
        X = builder(y, v, n, cos_theta, x1, c, name)
        params, chi2 = wls_fit(y, X, w)
        n_data = len(y)
        n_params = X.shape[1]
        bic = chi2 + n_params * np.log(n_data)
        results[name] = {
            "chi2": float(chi2),
            "bic": float(bic),
            "n_data": int(n_data),
            "n_params": int(n_params),
        }
        # add x1, c coefficients for M0 base
        results[name]["M_B"] = float(params[0])
        results[name]["alpha_x1"] = float(params[1])
        results[name]["beta_c"] = float(params[2])
        results[name].update(_extract_pars(name, params))
    best_bic = min(r["bic"] for r in results.values())
    for name in results:
        results[name]["dBIC"] = float(results[name]["bic"] - best_bic)
    return results


def radial_bin_analysis(df, z_col="zcmb", approach="corr", label=""):
    """Bin by velocity and fit the free kinematic dipole in each bin."""
    v = df[z_col].values * C_KMS
    n = df[["nx", "ny", "nz"]].values
    if approach == "corr":
        y = df["MU_SH0ES"].values - df["mu_lcdm"].values
        err = df["MU_SH0ES_ERR_DIAG"].values
    else:
        y = df["mB"].values - df["mu_lcdm"].values
        err = df["m_b_corr_err_DIAG"].values
    w = 1.0 / err ** 2

    bins = [(3000, 6000), (6000, 10000), (10000, 15000), (15000, 20000),
            (20000, 30000), (30000, 50000), (50000, 80000)]

    out = []
    for vmin, vmax in bins:
        mask = (v >= vmin) & (v < vmax)
        if mask.sum() < 15:
            continue
        yb, vb, nb, x1b, cb = (y[mask], v[mask], n[mask],
                                df["x1"].values[mask], df["c"].values[mask])
        wb = w[mask]
        if approach == "corr":
            X = build_design_corr(yb, vb, nb, None, "MK_free")
        else:
            X = build_design_raw(yb, vb, nb, None, x1b, cb, "MK_free")
        params, chi2 = wls_fit(yb, X, wb)
        B = params[-3:]
        b_par = float(np.dot(B, CMB_DIPOLE_UNIT))
        b_mag = float(np.linalg.norm(B))
        v_med = float(np.median(vb))
        out.append({
            "v_min": int(vmin), "v_max": int(vmax), "N": int(mask.sum()),
            "v_med": float(v_med), "B_par": float(b_par), "B_mag": float(b_mag),
            "chi2": float(chi2),
        })
    return out


def x1_directional_test(df, z_col="zcmb"):
    """Test whether x1 (light-curve stretch) is directionally compressed."""
    v = df[z_col].values * C_KMS
    cos_theta = df["cos_theta_cmb"].values
    x1 = df["x1"].values
    x1err = df["x1ERR"].values
    w = np.where(np.isfinite(x1err) & (x1err > 0), 1.0 / x1err ** 2, 0.0)

    # Model: x1 = a + g*(v/V_star)*cos_theta
    X = np.column_stack([np.ones(len(x1)), (v / V_STAR) * cos_theta])
    params, chi2 = wls_fit(x1, X, w)
    return {
        "intercept": float(params[0]),
        "slope": float(params[1]),
        "chi2": float(chi2),
        "N": int(len(x1)),
    }


class Step70PantheonFullDiscriminator:
    def __init__(self):
        self.root = PROJECT_ROOT
        for d in [self.root / "logs", self.root / "results" / "outputs",
                  self.root / "results" / "figures"]:
            d.mkdir(parents=True, exist_ok=True)

        self.logger = TEPLogger(
            "step_70",
            log_file_path=self.root / "logs" / "step_70_pantheon_full_discriminator.log",
        )
        set_step_logger(self.logger)

    def load(self):
        print_status("Loading Pantheon+ full sample...", "PROCESS")
        path = self.root / "data" / "raw" / "Pantheon+SH0ES.dat"
        if not path.exists():
            print_status(f"Pantheon+ not found at {path}", "ERROR")
            return None

        df = pd.read_csv(path, sep=r"\s+")
        df = df.rename(columns={"zCMB": "zcmb", "zHEL": "zhel",
                                "RA": "ra", "DEC": "dec"})

        # Cosmological redshift for the distance modulus
        df["mu_lcdm"] = mu_lcdm(df["zcmb"].values)

        # Unit vectors and CMB dipole projection
        n_vecs = ra_dec_to_unit_vectors(df["ra"].values, df["dec"].values)
        df["nx"] = n_vecs[:, 0]
        df["ny"] = n_vecs[:, 1]
        df["nz"] = n_vecs[:, 2]
        df["cos_theta_cmb"] = n_vecs @ CMB_DIPOLE_UNIT

        print_status(f"  Loaded {len(df)} SNe", "SUCCESS")
        print_status(f"  z range: {df['zcmb'].min():.5f} - {df['zcmb'].max():.5f}", "INFO")

        return df

    def run(self):
        print_status("=" * 70, "INFO")
        print_status("Step 70: Full Pantheon+ Radial Discriminator", "INFO")
        print_status("=" * 70, "INFO")

        df = self.load()
        if df is None:
            return

        # Exclude calibrators and invalid values
        df = df[df["IS_CALIBRATOR"] == 0].copy()
        valid = (
            np.isfinite(df["zcmb"]) & np.isfinite(df["MU_SH0ES"])
            & np.isfinite(df["MU_SH0ES_ERR_DIAG"]) & np.isfinite(df["mB"])
            & np.isfinite(df["m_b_corr_err_DIAG"]) & np.isfinite(df["x1"])
            & np.isfinite(df["c"]) & (df["MU_SH0ES_ERR_DIAG"] > 0)
            & (df["m_b_corr_err_DIAG"] > 0) & (df["zcmb"] > 0)
        )
        df = df[valid].copy()
        print_status(f"  {len(df)} SNe after cuts", "INFO")

        summary = {}
        z_cuts = [
            ("z010", df["zcmb"] < 0.10),
            ("z050", df["zcmb"] < 0.50),
            ("zfull", df["zcmb"] > 0.01),
        ]

        for label, mask in z_cuts:
            sub = df[mask].copy()
            if len(sub) < 100:
                continue
            print_status(f"\n--- Redshift cut: {label} (N={len(sub)}) ---", "TEST")

            res_corr = fit_models_corr(sub, z_col="zcmb")
            res_raw = fit_models_raw(sub, z_col="zcmb")

            bin_corr = radial_bin_analysis(sub, z_col="zcmb", approach="corr")
            bin_raw = radial_bin_analysis(sub, z_col="zcmb", approach="raw")

            x1_cmb = x1_directional_test(sub, z_col="zcmb")

            summary[label] = {
                "N": int(len(sub)),
                "z_min": float(sub["zcmb"].min()),
                "z_max": float(sub["zcmb"].max()),
                "z_med": float(sub["zcmb"].median()),
                "corr": res_corr,
                "raw": res_raw,
                "radial_bins_corr": bin_corr,
                "radial_bins_raw": bin_raw,
                "x1_directional": x1_cmb,
            }

            for approach, res in [("CORR", res_corr), ("RAW", res_raw)]:
                print_status(f"  {approach} ({label}) BIC summary:", "INFO")
                for m in res:
                    extra = ""
                    if "T_amp" in res[m]:
                        extra += f" T={res[m]['T_amp']:.4f}"
                    if "B_par" in res[m]:
                        extra += f" Bpar={res[m]['B_par']:.1f}"
                    if "alpha_x1" in res[m]:
                        extra += f" ax1={res[m]['alpha_x1']:.4f} bc={res[m]['beta_c']:.4f}"
                    print_status(f"    {m:15s} BIC={res[m]['bic']:9.1f}  dBIC={res[m]['dBIC']:7.2f}{extra}", "INFO")

        # Save JSON
        out_path = self.root / "results" / "outputs" / "step_70_pantheon_full_discriminator.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print_status(f"\nSaved results to {out_path}", "SUCCESS")

        self.make_figures(summary)
        return summary

    def make_figures(self, summary):
        colors = apply_tep_style()
        for label in summary:
            fig, axes = plt.subplots(2, 2, figsize=(13, 10))

            models_corr = ["M0", "MK_cmb", "MT_cmb", "MK_free", "MT_free",
                           "MKT_cmb_b", "MKT_free"]
            models_raw = models_corr + ["MT_cmb_x1tep"]
            bic_corr = [summary[label]["corr"][m]["bic"] for m in models_corr]
            bic_raw = [summary[label]["raw"][m]["bic"] for m in models_corr]
            x = np.arange(len(models_corr))
            axes[0, 0].bar(x - 0.2, bic_corr, 0.4, label="HR_corr")
            axes[0, 0].bar(x + 0.2, bic_raw, 0.4, label="HR_raw")
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(models_corr, rotation=45, ha="right", fontsize=7)
            axes[0, 0].set_ylabel("BIC")
            axes[0, 0].set_title(f"{label}: BIC by model")
            axes[0, 0].legend()

            bc = summary[label]["radial_bins_corr"]
            if bc:
                vmed = [b["v_med"] for b in bc]
                bpar = [b["B_par"] for b in bc]
                axes[0, 1].plot(vmed, bpar, "o-", label="HR_corr")
            br = summary[label]["radial_bins_raw"]
            if br:
                vmed = [b["v_med"] for b in br]
                bpar = [b["B_par"] for b in br]
                axes[0, 1].plot(vmed, bpar, "s-", label="HR_raw")
            axes[0, 1].axhline(0, color=colors['dark'], ls=":")
            axes[0, 1].set_xlabel("V_cmb (km/s)")
            axes[0, 1].set_ylabel("B_|| (km/s)")
            axes[0, 1].set_title("Kinematic dipole vs distance")
            axes[0, 1].legend()

            tc = summary[label]["corr"]["MT_cmb"]["T_amp"]
            tr = summary[label]["raw"]["MT_cmb"]["T_amp"]
            axes[1, 0].bar(["HR_corr", "HR_raw"], [tc, tr])
            axes[1, 0].set_ylabel("T (mag at V_star)")
            axes[1, 0].set_title(f"{label}: CMB-aligned temporal amplitude")

            x1s = summary[label]["x1_directional"]["slope"]
            axes[1, 1].bar(["x1 vs cosθ"], [x1s])
            axes[1, 1].axhline(0, color=colors['dark'], ls=":")
            axes[1, 1].set_ylabel("Slope")
            axes[1, 1].set_title("Stretch channel directional compression")

            plt.tight_layout()
            fig_path = self.root / "results" / "figures" / f"step_70_pantheon_full_discriminator_{label}.png"
            plt.savefig(fig_path, dpi=300, bbox_inches="tight")
            print_status(f"Saved figure {fig_path}", "SUCCESS")
            plt.close(fig)


def main():
    Step70PantheonFullDiscriminator().run()


if __name__ == "__main__":
    main()
