#!/usr/bin/env python3
"""
Step 11: Indicator Divergence vs Gravitational Potential
==========================================================
Test whether the Cepheid-TRGB distance divergence scales with host
galaxy gravitational potential depth, as predicted by the TEP framework.

Key Tasks:
1. Load matched host data (from step_10) and host potential catalog
   (from step_01)
2. Compute correlation between Delta_mu and sigma_v (velocity dispersion)
3. Fit linear model: Delta_mu = kappa * sigma_v + intercept
4. Compare TEP prediction (non-zero kappa) with void prediction (kappa = 0)

Outputs:
    results/outputs/step_11_indicator_divergence_vs_potential.json
    results/figures/step_11_divergence_vs_potential.png
"""

import json
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


class Step11IndicatorDivergence:
    """Step 11: Test whether Cepheid-TRGB divergence scales with host potential."""

    def __init__(self):
        self.root = PROJECT_ROOT
        self.data_interim = self.root / "data" / "interim"
        self.results = self.root / "results" / "outputs"
        self.figures = self.root / "results" / "figures"
        self.logs = self.root / "logs"

        for d in [self.data_interim, self.results, self.figures, self.logs]:
            d.mkdir(parents=True, exist_ok=True)

        self.logger = TEPLogger("step_11", log_file_path=self.logs / "step_11_indicator_divergence_vs_potential.log")
        set_step_logger(self.logger)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def load_data(self):
        """Load matched host data with Delta_mu and host potential catalog."""
        print_status("Loading matched host data and potential catalog...", "PROCESS")

        # Try step_10 per-galaxy output first
        step10_path = self.results / "step_10_per_galaxy_delta_mu.csv"
        matched_path = self.data_interim / "matched_hosts.csv"
        potential_path = self.data_interim / "host_potential_catalog.csv"

        df = None

        if step10_path.exists():
            print_status(f"Loading step_10 per-galaxy results: {step10_path}", "PROCESS")
            df = pd.read_csv(step10_path)
            print_status(f"  {len(df)} hosts with Delta_mu", "SUCCESS")
        elif matched_path.exists():
            print_status(f"Loading matched hosts: {matched_path}", "PROCESS")
            df = pd.read_csv(matched_path)
            # Compute delta_mu if not present
            if "delta_mu" not in df.columns and "mu_cepheid" in df.columns and "mu_trgb" in df.columns:
                df["delta_mu"] = df["mu_cepheid"] - df["mu_trgb"]
                df["delta_mu_err"] = np.sqrt(
                    df.get("mu_cepheid_err", 0.06) ** 2 + df.get("mu_trgb_err", 0.05) ** 2
                )
            print_status(f"  {len(df)} matched hosts", "SUCCESS")
        else:
            raise FileNotFoundError(
                f"Required input not found. Expected either {step10_path} "
                f"or {matched_path}. Re-run step_10 (matched host comparison) "
                "or step_00 (data ingestion) to regenerate the matched-host "
                "catalog from the real source data."
            )

        # Merge with host potential catalog if sigma_v is not already present
        if "sigma_v" not in df.columns or df["sigma_v"].isna().all():
            if potential_path.exists():
                print_status(f"Loading host potential catalog: {potential_path}", "PROCESS")
                pot_df = pd.read_csv(potential_path)
                df = self._merge_potential(df, pot_df)
            else:
                raise FileNotFoundError(
                    f"Required input {potential_path} not found. Re-run "
                    "step_01 (host potential catalog) to regenerate the "
                    "potential catalog from the real HyperLEDA source data."
                )

        # Drop rows without sigma_v or delta_mu
        before = len(df)
        df = df.dropna(subset=["delta_mu", "sigma_v"])
        after = len(df)
        if before != after:
            print_status(f"Dropped {before - after} rows missing sigma_v or delta_mu", "WARNING")

        print_status(f"Final sample: {len(df)} hosts with Delta_mu and sigma_v", "SUCCESS")
        return df

    def _merge_potential(self, df, pot_df):
        """Merge potential catalog into matched host DataFrame."""
        # Find galaxy name columns
        df_name = self._find_name_col(df.columns)
        pot_name = self._find_name_col(pot_df.columns)
        if df_name is None or pot_name is None:
            print_status("Could not identify galaxy name columns for potential merge.", "WARNING")
            return df

        sigma_col = None
        for c in pot_df.columns:
            cl = c.lower()
            if "sigma" in cl and ("v" in cl or cl == "sigma"):
                sigma_col = c
                break

        if sigma_col is None:
            print_status("No sigma_v column found in potential catalog.", "WARNING")
            return df

        df = df.copy()
        pot_df = pot_df.copy()
        df["_gal"] = df[df_name].astype(str).str.upper().str.strip().str.replace(" ", "")
        pot_df["_gal"] = pot_df[pot_name].astype(str).str.upper().str.strip().str.replace(" ", "")

        merged = pd.merge(df, pot_df[["_gal", sigma_col]], on="_gal", how="left")
        merged = merged.drop(columns=["_gal"])
        if "sigma_v" not in merged.columns:
            merged["sigma_v"] = merged[sigma_col]
        elif merged["sigma_v"].isna().any():
            merged["sigma_v"] = merged["sigma_v"].fillna(merged[sigma_col])

        return merged

    @staticmethod
    def _find_name_col(columns):
        for c in columns:
            cl = c.lower()
            if "gal" in cl or "host" in cl or "name" in cl:
                return c
        return None

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------
    def compute_correlation(self, df):
        """Compute correlation between Delta_mu and sigma_v."""
        print_status("Computing correlation between Delta_mu and sigma_v...", "PROCESS")

        delta_mu = df["delta_mu"].values
        sigma_v = df["sigma_v"].values
        n = len(df)

        # Pearson correlation
        pearson_r, pearson_p = sp_stats.pearsonr(sigma_v, delta_mu)
        print_status(f"  Pearson r:  {pearson_r:+.4f} (p={pearson_p:.4f})", "TEST")

        # Spearman rank correlation
        spearman_rho, spearman_p = sp_stats.spearmanr(sigma_v, delta_mu)
        print_status(f"  Spearman rho: {spearman_rho:+.4f} (p={spearman_p:.4f})", "TEST")

        # Linear fit: Delta_mu = kappa * sigma_v + intercept
        # Weighted least squares using delta_mu_err
        y = delta_mu
        x = sigma_v
        yerr = df["delta_mu_err"].values

        # Weighted linear regression
        weights = 1.0 / yerr ** 2
        S = np.sum(weights)
        Sx = np.sum(weights * x)
        Sy = np.sum(weights * y)
        Sxx = np.sum(weights * x * x)
        Sxy = np.sum(weights * x * y)

        denom = S * Sxx - Sx * Sx
        if abs(denom) > 0:
            kappa = (S * Sxy - Sx * Sy) / denom
            intercept = (Sxx * Sy - Sx * Sxy) / denom
            # Uncertainty on slope and intercept
            kappa_err = np.sqrt(S / denom)
            intercept_err = np.sqrt(Sxx / denom)
        else:
            kappa, intercept = 0.0, 0.0
            kappa_err, intercept_err = 0.0, 0.0

        kappa_sigma = float(abs(kappa) / kappa_err) if kappa_err > 0 else 0.0

        print_status(f"  Linear fit: Delta_mu = ({kappa:.6f} +/- {kappa_err:.6f}) * sigma_v + ({intercept:+.4f} +/- {intercept_err:.4f})", "TEST")
        print_status(f"  kappa significance: {kappa_sigma:.2f} sigma", "TEST")

        # Goodness of fit
        residuals = y - (kappa * x + intercept)
        chi2 = float(np.sum((residuals / yerr) ** 2))
        dof = max(n - 2, 1)
        chi2_reduced = chi2 / dof

        # R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        results = {
            "n_hosts": n,
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "spearman_rho": float(spearman_rho),
            "spearman_p": float(spearman_p),
            "linear_fit": {
                "kappa": float(kappa),
                "kappa_err": float(kappa_err),
                "kappa_significance_sigma": kappa_sigma,
                "intercept": float(intercept),
                "intercept_err": float(intercept_err),
                "r_squared": r_squared,
                "chi2": chi2,
                "chi2_reduced": float(chi2_reduced),
                "dof": dof,
            },
            "tep_prediction": {
                "kappa_nonzero": True,
                "description": "TEP predicts non-zero kappa: divergence scales with potential",
            },
            "void_prediction": {
                "kappa": 0.0,
                "description": "Void predicts kappa = 0: no potential-dependent divergence",
            },
        }
        return results, kappa, intercept, kappa_err, intercept_err

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    def plot_divergence_vs_potential(self, df, fit_results, kappa, intercept, kappa_err, intercept_err):
        """Generate the divergence vs potential figure."""
        print_status("Generating divergence vs potential figure...", "PROCESS")

        fig, ax = plt.subplots(figsize=(10, 7))

        x = df["sigma_v"].values
        y = df["delta_mu"].values
        yerr = df["delta_mu_err"].values

        # Data points
        ax.errorbar(
            x, y, yerr=yerr, fmt="o", color="#2166ac", ecolor="#92c5de",
            capsize=3, markersize=7, label="Matched hosts", zorder=3,
        )

        # Linear fit
        x_fit = np.linspace(x.min() - 10, x.max() + 10, 200)
        y_fit = kappa * x_fit + intercept
        ax.plot(x_fit, y_fit, color="#4daf4a", linewidth=2,
                label=f"TEP fit: $\\kappa = {kappa:.5f} \\pm {kappa_err:.5f}$ mag/(km/s)", zorder=2)

        # Fit uncertainty band
        y_upper = (kappa + kappa_err) * x_fit + (intercept + intercept_err)
        y_lower = (kappa - kappa_err) * x_fit + (intercept - intercept_err)
        ax.fill_between(x_fit, y_lower, y_upper, alpha=0.15, color="#4daf4a", zorder=1)

        # Void prediction (kappa = 0, intercept = 0)
        ax.axhline(0, color="#b2182b", linestyle="--", linewidth=1.5,
                   label="Void prediction ($\\Delta\\mu = 0$)", zorder=2)

        ax.set_xlabel("Host velocity dispersion $\\sigma_v$ (km s$^{-1}$)", fontsize=12)
        ax.set_ylabel("$\\Delta\\mu = \\mu_{\\rm Cepheid} - \\mu_{\\rm TRGB}$ (mag)", fontsize=12)
        ax.set_title("Cepheid–TRGB Divergence vs Host Gravitational Potential", fontsize=13)
        ax.legend(fontsize=10, loc="best")

        # Annotate with statistics
        text = (f"Pearson r = {fit_results['pearson_r']:+.3f} (p = {fit_results['pearson_p']:.3f})\n"
                f"Spearman $\\rho$ = {fit_results['spearman_rho']:+.3f} (p = {fit_results['spearman_p']:.3f})\n"
                f"$\\kappa$ = {kappa:.5f} $\\pm$ {kappa_err:.5f} ({fit_results['linear_fit']['kappa_significance_sigma']:.1f}$\\sigma$)\n"
                f"$R^2$ = {fit_results['linear_fit']['r_squared']:.3f}")
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=9,
                verticalalignment="top", bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))

        fig.tight_layout()
        fig_path = self.figures / "step_11_divergence_vs_potential.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print_status(f"Figure saved to {fig_path}", "SUCCESS")
        return fig_path

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(self):
        """Execute the full step."""
        print_status("Step 11: Indicator Divergence vs Gravitational Potential", "TITLE")

        print_status(
            "Scientific question: does the Cepheid-TRGB distance divergence scale with "
            "the depth of the host galaxy gravitational potential? The TEP framework "
            "predicts that Delta_mu = kappa * sigma_v, with a non-zero coupling kappa, "
            "because each indicator's clock rate responds differently to the local "
            "potential. The kinematic void model predicts kappa = 0, since distances are "
            "indicator-independent and no potential dependence is expected. This step "
            "tests the potential-scaling hypothesis by correlating Delta_mu with host "
            "velocity dispersion sigma_v, providing the central discriminant between "
            "the two frameworks in Block Ia.",
            "PROCESS",
        )

        # Load data
        df = self.load_data()

        if len(df) == 0:
            print_status("No data available for correlation analysis.", "ERROR")
            return

        print_status(
            "Methodology: the sample comprises matched hosts with both Delta_mu "
            "(from step_10) and velocity dispersion sigma_v (from the host potential "
            "catalog or HyperLEDA). The analysis uses Pearson and Spearman rank "
            "correlations to assess monotonic association, and an inverse-variance "
            "weighted linear regression Delta_mu = kappa * sigma_v + intercept to "
            "quantify the potential scaling. The slope significance (kappa / kappa_err) "
            "is the primary statistic for the void-vs-TEP comparison.",
            "PROCESS",
        )

        # Compute correlation and linear fit
        fit_results, kappa, intercept, kappa_err, intercept_err = self.compute_correlation(df)

        kappa_sig = fit_results["linear_fit"]["kappa_significance_sigma"]
        print_status(
            f"Interpretation: the fitted slope is kappa = {kappa:.6f} +/- {kappa_err:.6f} "
            f"mag/(km/s) ({kappa_sig:.2f} sigma). A non-zero kappa indicates that the "
            f"divergence scales with host potential depth, consistent with the TEP "
            f"prediction. The void prediction of kappa = 0 is "
            f"{'falsified' if kappa_sig > 2.0 else 'not definitively falsified'} at the "
            f"2-sigma threshold. The Pearson correlation "
            f"(r = {fit_results['pearson_r']:+.4f}) and R-squared "
            f"({fit_results['linear_fit']['r_squared']:.3f}) characterize the strength "
            f"of the potential dependence.",
            "TEST",
        )

        # Generate figure
        fig_path = self.plot_divergence_vs_potential(
            df, fit_results, kappa, intercept, kappa_err, intercept_err
        )

        # Summary JSON
        summary = {
            "step": "11_indicator_divergence_vs_potential",
            "description": "Test whether Cepheid-TRGB divergence scales with host gravitational potential",
            "delta_mu_definition": "Delta_mu = mu_Cepheid - mu_TRGB",
            "model": "Delta_mu = kappa * sigma_v + intercept",
            "methodology": (
                "Pearson and Spearman rank correlations plus an inverse-variance "
                "weighted linear regression of Delta_mu on host velocity dispersion "
                "sigma_v. The slope kappa and its significance (kappa / kappa_err) "
                "serve as the primary discriminant between TEP (kappa non-zero) and "
                "void (kappa = 0) predictions."
            ),
            "provenance": {
                "data_sources": [
                    "step_10 per-galaxy Delta_mu results",
                    "step_00 matched_hosts.csv interim catalog",
                    "step_01 host_potential_catalog.csv interim catalog",
                    "HyperLEDA (Makarov et al. 2014) velocity dispersions",
                ],
                "pipeline_block": "Ia — Indicator Divergence",
            },
            "scientific_context": (
                "Does the Cepheid-TRGB distance divergence scale with host "
                "gravitational potential depth? TEP predicts a non-zero coupling "
                "kappa; the void model predicts kappa = 0."
            ),
            "tep_prediction": (
                "Non-zero kappa: Delta_mu scales with sigma_v, reflecting "
                "indicator-dependent clock-rate responses to the local gravitational "
                "potential."
            ),
            "void_prediction": (
                "kappa = 0: no potential-dependent divergence, since distances are "
                "indicator-independent in the kinematic void model."
            ),
            "downstream_consumers": [
                "step_12_void_prediction_uniformity",
            ],
            "results": fit_results,
            "per_galaxy": df[["galaxy", "delta_mu", "delta_mu_err", "sigma_v"]].to_dict(orient="records"),
            "interpretation": {
                "tep": "Non-zero kappa indicates potential-dependent divergence, consistent with TEP",
                "void": "Void model predicts kappa = 0 (no potential dependence)",
                "falsification": (
                    f"kappa detected at {fit_results['linear_fit']['kappa_significance_sigma']:.2f} sigma; "
                    f"void prediction of kappa = 0 is "
                    f"{'falsified' if fit_results['linear_fit']['kappa_significance_sigma'] > 2.0 else 'not definitively falsified'} "
                    "at >2 sigma"
                ),
            },
            "output_files": [
                str(self.results / "step_11_indicator_divergence_vs_potential.json"),
                str(fig_path),
            ],
        }

        summary_path = self.results / "step_11_indicator_divergence_vs_potential.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print_status(f"Summary saved to {summary_path}", "SUCCESS")

        print_status("Step 11 complete", "SUCCESS")


if __name__ == "__main__":
    step = Step11IndicatorDivergence()
    step.run()
