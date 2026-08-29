#!/usr/bin/env python3
"""
Step 49: Band-Dependence Test — Optical vs NIR Cepheid Distance Differential
=============================================================================
Test the TEP disformal mechanism prediction that the Cepheid period bias
scales with the Leavitt law slope |b|, producing a band-dependent distance
differential that correlates with the gravitational potential coordinate X_i.

TEP Prediction:
    The optical V-band PL slope is b_V ~ -2.76 and the NIR H-band slope is
    b_H ~ -3.26.  The TEP disformal mechanism predicts that the Cepheid
    distance compression scales with |b|, so the inter-band differential

        Delta_mu_band = mu_NIR - mu_optical

    should correlate with X_i.  The predicted slope is

        (|b_H| - |b_V|) * kappa_P ~ 0.50 * kappa_P

    where kappa_P is the period-bias coupling constant.

Data Sources:
    1. Madore & Freedman (2023, arXiv:2309.10859) Table 2 — same-team
       comparison of W(V,VI) optical and W(H,VI) NIR Cepheid true distance
       moduli for 20 SH0ES SN Ia host galaxies.  This is the primary
       band-dependence sample because both bandpasses are analysed with
       identical methodology, photometry, and Cepheid samples, isolating
       the purely band-dependent effect.

    2. Freedman et al. (2001, ApJ 553 47) Key Project Table 3 — optical
       (V, I band) Cepheid true distance moduli for 31 galaxies, matched
       to SH0ES R22 NIR (H-band) distances for the overlap subsample.

    3. host_potential_catalog.csv — HyperLEDA rotation velocities for X_i.

Outputs:
    data/processed/band_dependence_matched.csv
    results/outputs/step_49_band_dependence.json
    results/figures/step_49_band_dependence.png
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status


# ── TEP constants ──────────────────────────────────────────────────────
SIGMA_REF = 87.165          # km/s  — anchor reference potential scale
U_REF = SIGMA_REF ** 2      # (km/s)^2
C_KMS = 299792.458           # km/s

# Leavitt law slopes (Madore & Freedman 2023, Riess et al. 2022)
B_V = -2.76                  # optical V-band PL slope
B_H = -3.26                  # NIR H-band PL slope
ABS_B_V = abs(B_V)
ABS_B_H = abs(B_H)
DELTA_B = ABS_B_H - ABS_B_V  # 0.50

# kappa_P candidates from TEP-H0 (Paper 11)
KAPPA_P_DEFAULT = 0.365e6    # mag
KAPPA_P_JOINT = 0.400e6      # mag


def normalize_galaxy_name(name: str) -> str:
    """Normalise a galaxy name to a common form for cross-catalog matching."""
    s = str(name).strip().upper()
    # M101 / M 101 / NGC 5457 → NGC 5457 handled by alias map
    s = re.sub(r"\s+", " ", s)
    # Remove leading zeros in NGC numbers: NGC 0691 → NGC 691
    s = re.sub(r"NGC 0*(\d+)", r"NGC \1", s)
    s = re.sub(r"UGC 0*(\d+)", r"UGC \1", s)
    s = re.sub(r"IC 0*(\d+)", r"IC \1", s)
    return s


# Alias map: alternative names → canonical NGC/UGC form
ALIAS = {
    "M 101": "NGC 5457",
    "M101": "NGC 5457",
    "M 31": "M 31",
    "M31": "M 31",
    "M 81": "NGC 3031",
    "M81": "NGC 3031",
    "M 82": "NGC 3034",
    "M82": "NGC 3034",
    "M 96": "NGC 3368",
    "M96": "NGC 3368",
    "M 66": "NGC 3627",
    "M66": "NGC 3627",
    "M 95": "NGC 3351",
    "M95": "NGC 3351",
    "M 100": "NGC 4321",
    "M100": "NGC 4321",
    "M 106": "NGC 4258",
    "M106": "NGC 4258",
    "MRK 1337": "MRK 1337",
    "M1337": "MRK 1337",
}


def canonical_name(name: str) -> str:
    """Return the canonical galaxy name used for cross-catalog matching."""
    norm = normalize_galaxy_name(name)
    return ALIAS.get(norm, norm)


# R22 source_id → canonical name
R22_ALIAS = {
    "M101": "NGC 5457",
    "M1337": "MRK 1337",
    "N0691": "NGC 691",
    "N1015": "NGC 1015",
    "N105A": "NGC 105",
    "N1309": "NGC 1309",
    "N1365": "NGC 1365",
    "N1448": "NGC 1448",
    "N1559": "NGC 1559",
    "N2442": "NGC 2442",
    "N2525": "NGC 2525",
    "N2608": "NGC 2608",
    "N3021": "NGC 3021",
    "N3147": "NGC 3147",
    "N3254": "NGC 3254",
    "N3370": "NGC 3370",
    "N3447": "NGC 3447",
    "N3583": "NGC 3583",
    "N3972": "NGC 3972",
    "N3982": "NGC 3982",
    "N4038": "NGC 4038",
    "N4424": "NGC 4424",
    "N4536": "NGC 4536",
    "N4639": "NGC 4639",
    "N4680": "NGC 4680",
    "N5468": "NGC 5468",
    "N5584": "NGC 5584",
    "N5643": "NGC 5643",
    "N5728": "NGC 5728",
    "N5861": "NGC 5861",
    "N5917": "NGC 5917",
    "N7250": "NGC 7250",
    "N7329": "NGC 7329",
    "N7541": "NGC 7541",
    "N7678": "NGC 7678",
    "N976A": "NGC 976",
    "U9391": "UGC 9391",
    "N4258": "NGC 4258",
    "LMC": "LMC",
    "LMC_GRND": "LMC",
    "LMC_HST": "LMC",
    "M31": "M 31",
    "SMC": "SMC",
}


class Step49BandDependence:
    """Step 49: TEP band-dependence test — optical vs NIR Cepheid distances."""

    def __init__(self):
        self.root = PROJECT_ROOT
        self.data_raw = self.root / "data" / "raw" / "external"
        self.data_proc = self.root / "data" / "processed"
        self.results = self.root / "results" / "outputs"
        self.figures = self.root / "results" / "figures"
        self.logs = self.root / "logs"

        for d in [self.data_proc, self.results, self.figures, self.logs]:
            d.mkdir(parents=True, exist_ok=True)

        self.logger = TEPLogger(
            "step_49",
            log_file_path=self.logs / "step_49_band_dependence.log",
        )
        set_step_logger(self.logger)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def load_madore_freedman2023(self):
        """Load Madore & Freedman (2023) Table 2 — VIH and VI distances."""
        print_status("Loading Madore & Freedman (2023) Table 2...", "PROCESS")
        path = self.data_raw / "madore_freedman2023_vih_vi.csv"
        if not path.exists():
            print_status(f"File not found: {path}", "ERROR")
            return pd.DataFrame()
        df = pd.read_csv(path)
        df["canonical"] = df["galaxy"].apply(canonical_name)
        df["delta_mu_band"] = df["mu_vih"] - df["mu_vi"]
        df["delta_mu_band_err"] = np.sqrt(
            df["mu_vih_err"] ** 2 + df["mu_vi_err"] ** 2
        )
        print_status(
            f"  {len(df)} galaxies with VIH (NIR) and VI (optical) distances",
            "SUCCESS",
        )
        return df

    def load_key_project(self):
        """Load Freedman et al. (2001) Key Project optical Cepheid distances."""
        print_status("Loading Key Project (Freedman et al. 2001) distances...",
                     "PROCESS")
        path = self.data_raw / "key_project_cepheid_distances.csv"
        if not path.exists():
            print_status(f"File not found: {path}", "ERROR")
            return pd.DataFrame()
        df = pd.read_csv(path)
        df["canonical"] = df["galaxy"].apply(canonical_name)
        print_status(f"  {len(df)} Key Project galaxies loaded", "SUCCESS")
        return df

    def load_r22(self):
        """Load SH0ES R22 NIR Cepheid distances."""
        print_status("Loading SH0ES R22 NIR Cepheid distances...", "PROCESS")
        path = self.data_raw / "r22_cepheid_distances.csv"
        if not path.exists():
            print_status(f"File not found: {path}", "ERROR")
            return pd.DataFrame()
        df = pd.read_csv(path)
        df["canonical"] = df["source_id"].map(R22_ALIAS)
        df = df.rename(columns={"value": "mu_nir", "error": "mu_nir_err"})
        print_status(f"  {len(df)} R22 galaxies loaded", "SUCCESS")
        return df

    def load_host_potential(self):
        """Load host potential catalog with V_rot from HyperLEDA."""
        print_status("Loading host potential catalog...", "PROCESS")
        path = self.data_proc / "host_potential_catalog.csv"
        if not path.exists():
            print_status(f"File not found: {path}", "ERROR")
            return pd.DataFrame()
        df = pd.read_csv(path)
        df["canonical"] = df["galaxy"].apply(canonical_name)
        # Compute X_i = (V_rot^2/2 - U_ref) / c^2
        df["U_i"] = df["sigma_kms"] ** 2 / 2.0
        df["X_i"] = (df["U_i"] - U_REF) / C_KMS ** 2
        df["X_i_err"] = (
            df["sigma_kms"] * df["error_kms"] / C_KMS ** 2
        )
        print_status(f"  {len(df)} galaxies with V_rot and X_i", "SUCCESS")
        return df

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------
    def merge_primary(self, mf_df, host_df):
        """Merge Madore & Freedman (2023) Table 2 with host potential catalog."""
        print_status(
            "\nMerging Madore & Freedman (2023) with host potential catalog...",
            "PROCESS",
        )
        merged = mf_df.merge(
            host_df[["canonical", "galaxy", "sigma_kms", "error_kms",
                      "U_i", "X_i", "X_i_err"]],
            on="canonical",
            how="inner",
            suffixes=("", "_host"),
        )
        merged = merged.rename(columns={
            "galaxy_host": "galaxy_name",
            "sigma_kms": "V_rot",
            "error_kms": "V_rot_err",
        })
        print_status(f"  {len(merged)} galaxies matched (primary sample)",
                     "SUCCESS")
        return merged

    def merge_secondary(self, kp_df, r22_df, host_df):
        """Merge Key Project optical with R22 NIR and host potential catalog."""
        print_status(
            "\nMerging Key Project + R22 + host potential catalog...",
            "PROCESS",
        )
        # Match Key Project with R22
        kp_r22 = kp_df.merge(
            r22_df[["canonical", "mu_nir", "mu_nir_err"]],
            on="canonical",
            how="inner",
        )
        print_status(f"  {len(kp_r22)} Key Project ↔ R22 matches", "TEST")

        # Match with host potential catalog
        merged = kp_r22.merge(
            host_df[["canonical", "galaxy", "sigma_kms", "error_kms",
                      "U_i", "X_i", "X_i_err"]],
            on="canonical",
            how="inner",
        )
        merged = merged.rename(columns={
            "galaxy": "galaxy_name",
            "mu_true": "mu_opt",
            "mu_true_err": "mu_opt_err",
            "sigma_kms": "V_rot",
            "error_kms": "V_rot_err",
        })
        merged["delta_mu_band"] = merged["mu_nir"] - merged["mu_opt"]
        merged["delta_mu_band_err"] = np.sqrt(
            merged["mu_nir_err"] ** 2 + merged["mu_opt_err"] ** 2
        )
        print_status(f"  {len(merged)} galaxies matched (secondary sample)",
                     "SUCCESS")
        return merged

    # ------------------------------------------------------------------
    # Regression
    # ------------------------------------------------------------------
    def weighted_regression(self, x, y, yerr):
        """Weighted linear regression y = slope * x + intercept."""
        n = len(x)
        if n < 3:
            return None
        w = 1.0 / yerr ** 2
        S = np.sum(w)
        Sx = np.sum(w * x)
        Sy = np.sum(w * y)
        Sxx = np.sum(w * x * x)
        Sxy = np.sum(w * x * y)
        denom = S * Sxx - Sx * Sx
        if abs(denom) < 1e-30:
            return None
        slope = (S * Sxy - Sx * Sy) / denom
        intercept = (Sxx * Sy - Sx * Sxy) / denom
        slope_err = np.sqrt(S / denom)
        intercept_err = np.sqrt(Sxx / denom)
        slope_sigma = abs(slope / slope_err) if slope_err > 0 else 0.0

        residuals = y - (slope * x + intercept)
        chi2 = float(np.sum((residuals / yerr) ** 2))
        dof = max(n - 2, 1)
        chi2_red = chi2 / dof

        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        pearson_r, pearson_p = sp_stats.pearsonr(x, y)
        spearman_rho, spearman_p = sp_stats.spearmanr(x, y)

        return {
            "n": n,
            "slope": float(slope),
            "slope_err": float(slope_err),
            "slope_significance_sigma": float(slope_sigma),
            "intercept": float(intercept),
            "intercept_err": float(intercept_err),
            "r_squared": float(r2),
            "chi2": chi2,
            "chi2_reduced": float(chi2_red),
            "dof": dof,
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "spearman_rho": float(spearman_rho),
            "spearman_p": float(spearman_p),
        }

    def run_regression(self, df, label, x_col="X_i", y_col="delta_mu_band",
                       err_col="delta_mu_band_err"):
        """Run weighted regression and print diagnostics."""
        print_status(f"\nRegression: {label}", "PROCESS")
        x = df[x_col].values
        y = df[y_col].values
        yerr = df[err_col].values
        n = len(df)

        res = self.weighted_regression(x, y, yerr)
        if res is None:
            print_status(f"  N={n} — too few for regression", "WARNING")
            return None

        # TEP predicted slope
        tep_slope_default = DELTA_B * KAPPA_P_DEFAULT
        tep_slope_joint = DELTA_B * KAPPA_P_JOINT

        print_status(
            f"  N={n}, slope = {res['slope']:.4e} ± {res['slope_err']:.4e} "
            f"({res['slope_significance_sigma']:.2f}σ)",
            "TEST",
        )
        print_status(
            f"  intercept = {res['intercept']:+.4f} ± "
            f"{res['intercept_err']:.4f}",
            "TEST",
        )
        print_status(
            f"  Pearson r = {res['pearson_r']:+.4f} "
            f"(p = {res['pearson_p']:.4f})",
            "TEST",
        )
        print_status(
            f"  Spearman ρ = {res['spearman_rho']:+.4f} "
            f"(p = {res['spearman_p']:.4f})",
            "TEST",
        )
        print_status(
            f"  R² = {res['r_squared']:.4f}, "
            f"χ²/dof = {res['chi2_reduced']:.2f}",
            "TEST",
        )
        print_status(
            f"  TEP predicted slope: "
            f"{tep_slope_default:.4e} (default κ_P), "
            f"{tep_slope_joint:.4e} (joint κ_P)",
            "TEST",
        )

        # Consistency with TEP prediction
        for kname, kval in [("default", KAPPA_P_DEFAULT),
                            ("joint", KAPPA_P_JOINT)]:
            pred = DELTA_B * kval
            diff = abs(res["slope"] - pred)
            cons = diff / res["slope_err"] if res["slope_err"] > 0 else 99
            print_status(
                f"  vs TEP ({kname}): |slope - pred|/σ = {cons:.2f}σ",
                "TEST",
            )
            res[f"tep_consistency_{kname}"] = float(cons)

        res["tep_predicted_slope_default"] = float(tep_slope_default)
        res["tep_predicted_slope_joint"] = float(tep_slope_joint)
        res["slope_sign_positive"] = bool(res["slope"] > 0)
        return res

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    def plot_band_dependence(self, df_primary, df_secondary, res_primary,
                             res_secondary):
        """Generate the band-dependence regression figure."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # --- Panel 1: Primary (Madore & Freedman 2023 VIH vs VI) ---
        ax = axes[0]
        x = df_primary["X_i"].values
        y = df_primary["delta_mu_band"].values
        yerr = df_primary["delta_mu_band_err"].values

        ax.errorbar(x, y, yerr=yerr, fmt="o", color="steelblue", ms=5,
                    capsize=3, alpha=0.8, zorder=3)

        if res_primary:
            slope = res_primary["slope"]
            intercept = res_primary["intercept"]
            slope_err = res_primary["slope_err"]
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, slope * x_line + intercept, "k--", lw=1.5,
                    label=f"Fit: slope = {slope:.2e} ± {slope_err:.2e}")

            # TEP predicted line
            tep_slope = DELTA_B * KAPPA_P_DEFAULT
            ax.plot(x_line, tep_slope * x_line, "g-", lw=1.5, alpha=0.6,
                    label=f"TEP pred (κ_P = {KAPPA_P_DEFAULT:.3g})")

        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        ax.set_xlabel(
            r"$X_i = (V_{\rm rot}^2/2 - U_{\rm ref}) / c^2$", fontsize=12
        )
        ax.set_ylabel(
            r"$\Delta\mu_{\rm band} = \mu_{\rm NIR} - \mu_{\rm opt}$ (mag)",
            fontsize=12,
        )
        ax.set_title(
            f"Same-team VIH vs VI  (N={len(df_primary)})\n"
            f"Madore & Freedman (2023)",
            fontsize=12,
        )
        ax.legend(fontsize=8, loc="best")

        # --- Panel 2: Secondary (Key Project vs R22) ---
        ax = axes[1]
        if df_secondary is not None and len(df_secondary) > 0:
            x2 = df_secondary["X_i"].values
            y2 = df_secondary["delta_mu_band"].values
            yerr2 = df_secondary["delta_mu_band_err"].values

            ax.errorbar(x2, y2, yerr=yerr2, fmt="s", color="crimson", ms=5,
                        capsize=3, alpha=0.8, zorder=3)

            if res_secondary:
                slope2 = res_secondary["slope"]
                intercept2 = res_secondary["intercept"]
                slope_err2 = res_secondary["slope_err"]
                x_line = np.linspace(x2.min(), x2.max(), 100)
                ax.plot(x_line, slope2 * x_line + intercept2, "k--", lw=1.5,
                        label=f"Fit: slope = {slope2:.2e} ± {slope_err2:.2e}")

                tep_slope = DELTA_B * KAPPA_P_DEFAULT
                ax.plot(x_line, tep_slope * x_line, "g-", lw=1.5, alpha=0.6,
                        label=f"TEP pred (κ_P = {KAPPA_P_DEFAULT:.3g})")

            ax.set_title(
                f"Key Project vs R22  (N={len(df_secondary)})\n"
                f"Freedman (2001) vs Riess (2022)",
                fontsize=12,
            )
        else:
            ax.set_title("Key Project vs R22  (no matches)")

        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        ax.set_xlabel(
            r"$X_i = (V_{\rm rot}^2/2 - U_{\rm ref}) / c^2$", fontsize=12
        )
        ax.set_ylabel(
            r"$\Delta\mu_{\rm band} = \mu_{\rm NIR} - \mu_{\rm opt}$ (mag)",
            fontsize=12,
        )
        ax.legend(fontsize=8, loc="best")

        fig.suptitle(
            "TEP Band-Dependence: Optical vs NIR Cepheid Distance "
            "Differential vs Gravitational Potential",
            fontsize=13, y=1.02,
        )
        fig.tight_layout()
        fig_path = self.figures / "step_49_band_dependence.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print_status(f"Figure saved to {fig_path}", "SUCCESS")
        return str(fig_path)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def save_matched_catalog(self, df_primary, df_secondary):
        """Save the matched band-dependence catalog."""
        # Build a unified output DataFrame
        rows = []
        for _, r in df_primary.iterrows():
            rows.append({
                "galaxy": r.get("galaxy_name", r["canonical"]),
                "mu_opt": r["mu_vi"],
                "mu_opt_err": r["mu_vi_err"],
                "mu_nir": r["mu_vih"],
                "mu_nir_err": r["mu_vih_err"],
                "delta_mu_band": r["delta_mu_band"],
                "delta_mu_band_err": r["delta_mu_band_err"],
                "X_i": r["X_i"],
                "V_rot": r["V_rot"],
                "sample": "primary_MF2023",
            })
        if df_secondary is not None and len(df_secondary) > 0:
            for _, r in df_secondary.iterrows():
                rows.append({
                    "galaxy": r.get("galaxy_name", r["canonical"]),
                    "mu_opt": r["mu_opt"],
                    "mu_opt_err": r["mu_opt_err"],
                    "mu_nir": r["mu_nir"],
                    "mu_nir_err": r["mu_nir_err"],
                    "delta_mu_band": r["delta_mu_band"],
                    "delta_mu_band_err": r["delta_mu_band_err"],
                    "X_i": r["X_i"],
                    "V_rot": r["V_rot"],
                    "sample": "secondary_KP_R22",
                })
        out = pd.DataFrame(rows)
        path = self.data_proc / "band_dependence_matched.csv"
        out.to_csv(path, index=False)
        print_status(f"Matched catalog saved to {path}", "SUCCESS")
        return out

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------
    def run(self):
        print_status("Step 49: Band-Dependence Test", "TITLE")
        print_status(
            "Test the TEP prediction that the optical-to-NIR Cepheid distance "
            "differential scales with the gravitational potential coordinate "
            "X_i, with a predicted slope of (|b_H| - |b_V|) * kappa_P "
            f"= {DELTA_B:.2f} * kappa_P.",
            "PROCESS",
        )
        print_status(
            f"Leavitt law slopes: b_V = {B_V}, b_H = {B_H}, "
            f"|b_H|/|b_V| = {ABS_B_H/ABS_B_V:.3f} "
            f"({(ABS_B_H/ABS_B_V - 1)*100:.1f}% larger in NIR)",
            "PROCESS",
        )

        # Load data
        mf_df = self.load_madore_freedman2023()
        kp_df = self.load_key_project()
        r22_df = self.load_r22()
        host_df = self.load_host_potential()

        # Merge
        df_primary = self.merge_primary(mf_df, host_df)
        df_secondary = self.merge_secondary(kp_df, r22_df, host_df)

        # Regressions
        res_primary = None
        res_secondary = None
        if len(df_primary) >= 3:
            res_primary = self.run_regression(
                df_primary, "Primary: MF2023 VIH vs VI"
            )
        if len(df_secondary) >= 3:
            res_secondary = self.run_regression(
                df_secondary, "Secondary: Key Project vs R22"
            )

        # Save matched catalog
        matched = self.save_matched_catalog(df_primary, df_secondary)

        # Figure
        fig_path = self.plot_band_dependence(
            df_primary, df_secondary, res_primary, res_secondary
        )

        # Summary
        summary = {
            "step": "49_band_dependence",
            "description": (
                "Test the TEP disformal mechanism prediction that the "
                "optical-to-NIR Cepheid distance differential scales with "
                "the gravitational potential coordinate X_i."
            ),
            "tep_prediction": {
                "b_V": B_V,
                "b_H": B_H,
                "abs_b_V": ABS_B_V,
                "abs_b_H": ABS_B_H,
                "delta_b": DELTA_B,
                "nir_excess_percent": (ABS_B_H / ABS_B_V - 1) * 100,
                "predicted_slope_default": DELTA_B * KAPPA_P_DEFAULT,
                "predicted_slope_joint": DELTA_B * KAPPA_P_JOINT,
                "formula": "Delta_mu_band = (|b_H| - |b_V|) * kappa_P * X_i",
            },
            "primary_analysis": {
                "label": "Madore & Freedman (2023) VIH vs VI — same-team",
                "n_matched": len(df_primary),
                "data_source": (
                    "Madore & Freedman 2023 (arXiv:2309.10859) Table 2: "
                    "W(H,VI) NIR and W(V,VI) optical Cepheid true distance "
                    "moduli for 20 SH0ES SN Ia host galaxies, analysed with "
                    "identical methodology."
                ),
                "regression": res_primary,
            },
            "secondary_analysis": {
                "label": "Key Project (Freedman 2001) vs R22 (Riess 2022)",
                "n_matched": len(df_secondary) if df_secondary is not None else 0,
                "data_source": (
                    "Freedman et al. (2001) Key Project optical (V,I band) "
                    "Cepheid distances matched to Riess et al. (2022) SH0ES "
                    "NIR (H-band) Cepheid distances for the overlap "
                    "subsample."
                ),
                "regression": res_secondary,
            },
            "constants": {
                "sigma_ref_kms": SIGMA_REF,
                "u_ref_kms2": U_REF,
                "c_kms": C_KMS,
                "kappa_p_default_mag": KAPPA_P_DEFAULT,
                "kappa_p_joint_mag": KAPPA_P_JOINT,
            },
            "matched_catalog": {
                "n_total": len(matched),
                "n_primary": len(df_primary),
                "n_secondary": len(df_secondary) if df_secondary is not None else 0,
                "file": str(self.data_proc / "band_dependence_matched.csv"),
            },
            "output_files": [
                str(self.data_proc / "band_dependence_matched.csv"),
                str(self.results / "step_49_band_dependence.json"),
                fig_path,
            ],
            "methodology": (
                "Weighted linear regression Delta_mu_band = slope * X_i + "
                "intercept, where Delta_mu_band = mu_NIR - mu_optical and "
                "X_i = (V_rot^2/2 - U_ref) / c^2. Two samples are analysed: "
                "(1) Primary — Madore & Freedman (2023) Table 2 with "
                "same-team VIH (NIR) and VI (optical) Cepheid distances for "
                "20 galaxies; (2) Secondary — Key Project (Freedman et al. "
                "2001) optical distances matched to SH0ES R22 (Riess et al. "
                "2022) NIR distances for the overlap subsample. The TEP "
                "predicted slope is (|b_H| - |b_V|) * kappa_P = 0.50 * "
                "kappa_P, positive if NIR distances are more compressed in "
                "deeper potentials."
            ),
            "provenance": {
                "data_sources": [
                    "Madore & Freedman 2023 (arXiv:2309.10859) Table 2",
                    "Freedman et al. 2001 (ApJ 553 47) Key Project Table 3",
                    "Riess et al. 2022 (ApJ 934 L7) SH0ES R22 NIR distances",
                    "HyperLEDA (Makarov et al. 2014) rotation velocities",
                ],
                "pipeline_block": "standalone",
            },
            "scientific_context": (
                "The TEP disformal mechanism predicts that the Cepheid "
                "period bias scales with the Leavitt law slope |b|. Because "
                "the NIR H-band slope (|b_H| = 3.26) is steeper than the "
                "optical V-band slope (|b_V| = 2.76), the TEP distance "
                "compression is ~18% larger in NIR than in optical at the "
                "same potential depth. The inter-band differential "
                "Delta_mu_band = mu_NIR - mu_optical should therefore "
                "correlate with X_i, with a predicted slope of "
                "(|b_H| - |b_V|) * kappa_P ~ 0.50 * kappa_P."
            ),
            "tep_prediction_text": (
                "Delta_mu_band = (|b_H| - |b_V|) * kappa_P * X_i, with a "
                "positive slope indicating that NIR Cepheid distances are "
                "more compressed in deeper gravitational potentials."
            ),
            "void_prediction": (
                "No physical mechanism in the void model predicts a "
                "band-dependent scaling of Cepheid distance differences "
                "with gravitational potential. The regression slope should "
                "be consistent with zero."
            ),
            "downstream_consumers": [],
        }

        # Key finding
        if res_primary:
            s = res_primary["slope"]
            s_err = res_primary["slope_err"]
            sig = res_primary["slope_significance_sigma"]
            sign = "positive" if s > 0 else "negative"
            summary["key_finding"] = (
                f"Primary sample (N={res_primary['n']}): slope = "
                f"{s:.4e} ± {s_err:.4e} ({sig:.2f}σ, {sign}). "
                f"TEP predicted slope = {DELTA_B * KAPPA_P_DEFAULT:.4e}."
            )
        else:
            summary["key_finding"] = "Insufficient data for regression."

        path = self.results / "step_49_band_dependence.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        print_status(f"Summary saved to {path}", "SUCCESS")
        print_status("Step 49 complete", "SUCCESS")


if __name__ == "__main__":
    step = Step49BandDependence()
    step.run()
