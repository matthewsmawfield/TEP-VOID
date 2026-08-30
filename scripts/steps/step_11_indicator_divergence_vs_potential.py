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

    # TEP constants (matching step_36 and TEP-H0)
    SIGMA_REF = 87.165  # km/s — unscreened anchor reference potential scale
    U_REF = SIGMA_REF ** 2  # (km/s)^2 — unscreened reference potential proxy
    # Screened anchor reference (from TEP-H0 tep_correction.compute_anchor_sigma_ref(screened=True))
    # U_ref_screened = sum(w_a * S_a * sigma_a^2) / sum(w_a) = 30.507^2
    SIGMA_REF_SCREENED = 30.507  # km/s — screened anchor reference
    U_REF_SCREENED = SIGMA_REF_SCREENED ** 2  # ≈ 930.7 (km/s)^2
    C_KMS = 299792.458  # km/s
    KAPPA_CEP_DEFAULT = 0.365e6  # mag (TEP-H0 closure)
    KAPPA_CEP_JOINT = 0.400e6  # mag (joint multi-block)
    KAPPA_CEP_WLS = 0.452e6  # mag (redshift-only WLS — manuscript primary)

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

        # Compute X_i = (sigma_v^2 - U_ref) / c^2 (dimensionless potential)
        # This is the correct TEP potential coordinate, not sigma_v itself.
        # TEP predicts delta_mu = -kappa_Cep * X_i, which is quadratic in sigma_v.
        df["X_i"] = (df["sigma_v"].values ** 2 - self.U_REF) / self.C_KMS ** 2
        print_status(f"  Computed X_i = (sigma_v^2 - U_ref) / c^2", "INFO")
        print_status(f"  X_i range: {df['X_i'].min()*1e7:.4f} to {df['X_i'].max()*1e7:.4f} x 1e-7", "INFO")

        # Load TEP screening factors from TEP-H0 Step 03
        # Match by canonical galaxy name to avoid collisions when multiple
        # hosts share the same sigma_inferred value.
        tep_h0_root = self.root.parent / "TEP-H0"
        strat_path = tep_h0_root / "results" / "outputs" / "step_03_stratified_h0.csv"
        hosts_path = tep_h0_root / "data" / "processed" / "hosts_processed.csv"
        if strat_path.exists() and hosts_path.exists():
            strat = pd.read_csv(strat_path)
            hosts = pd.read_csv(hosts_path)

            # Build canonical-name -> pgc mapping
            def _canonical(name):
                return str(name).upper().strip().replace(" ", "").replace("NGC", "N").replace("IC", "I")

            hosts["_canon"] = hosts["normalized_name"].apply(_canonical)
            df["_canon"] = df["galaxy"].apply(_canonical)
            host_pgc = dict(zip(hosts["_canon"], hosts["pgc"]))
            host_source = dict(zip(hosts["pgc"], hosts["source_id"]))
            strat_screening = dict(zip(strat["source_id"], strat["shear_suppression"]))

            df["pgc"] = df["_canon"].map(host_pgc)
            df["source_id"] = df["pgc"].map(host_source)
            df["S_total"] = df["source_id"].map(strat_screening).fillna(1.0)
            # Screened X_i uses the SCREENED anchor reference (U_ref_screened),
            # not the unscreened one. The TEP endpoint form is:
            #   X_i = (S_total * sigma_v^2 - U_ref_screened) / c^2
            # where U_ref_screened = sum(w_a * S_a * sigma_a^2) / sum(w_a).
            # Using the unscreened U_ref with screened host potentials is
            # inconsistent and destroys the signal (see TEP-H0 tep_correction.py).
            df["X_i_screened"] = (df["S_total"].values * df["sigma_v"].values ** 2 - self.U_REF_SCREENED) / self.C_KMS ** 2
            n_matched = df["pgc"].notna().sum()
            print_status(f"  Loaded TEP screening factors from TEP-H0 Step 03 ({n_matched}/{len(df)} matched)", "INFO")
            df = df.drop(columns=["_canon"])
        else:
            df["S_total"] = 1.0
            df["X_i_screened"] = df["X_i"]
            print_status(f"  TEP-H0 screening not available, using S_total = 1", "WARNING")

        print_status(f"Final sample: {len(df)} hosts with Delta_mu and X_i", "SUCCESS")
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
        """Compute correlation between Delta_mu and the screened TEP potential coordinate.

        TEP predicts delta_mu = -kappa_Cep * X_i, where the canonical
        screened coordinate is X_i = (S_total * sigma_v^2 - U_ref_screened) / c^2.
        The screened fit is primary because kappa_Cep from TEP-H0 is defined
        on this scale.  An unscreened diagnostic fit is also reported.
        """
        print_status("Computing correlation between Delta_mu and X_i...", "PROCESS")

        delta_mu = df["delta_mu"].values
        sigma_v = df["sigma_v"].values
        x_i = df["X_i"].values
        x_i_screened = df["X_i_screened"].values
        n = len(df)

        # --- Correlations with sigma_v (legacy, for comparison) ---
        pearson_r_sv, pearson_p_sv = sp_stats.pearsonr(sigma_v, delta_mu)
        spearman_rho_sv, spearman_p_sv = sp_stats.spearmanr(sigma_v, delta_mu)
        print_status(f"  [sigma_v]  Pearson r:  {pearson_r_sv:+.4f} (p={pearson_p_sv:.4f})", "TEST")
        print_status(f"  [sigma_v]  Spearman rho: {spearman_rho_sv:+.4f} (p={spearman_p_sv:.4f})", "TEST")

        # --- Correlations with X_i (correct TEP coordinate) ---
        pearson_r_xi, pearson_p_xi = sp_stats.pearsonr(x_i, delta_mu)
        spearman_rho_xi, spearman_p_xi = sp_stats.spearmanr(x_i, delta_mu)
        print_status(f"  [X_i uns]  Pearson r:  {pearson_r_xi:+.4f} (p={pearson_p_xi:.4f})", "TEST")
        print_status(f"  [X_i uns]  Spearman rho: {spearman_rho_xi:+.4f} (p={spearman_p_xi:.4f})", "TEST")

        # --- Correlations with screened X_i (primary) ---
        pearson_r_xi_scr, pearson_p_xi_scr = sp_stats.pearsonr(x_i_screened, delta_mu)
        spearman_rho_xi_scr, spearman_p_xi_scr = sp_stats.spearmanr(x_i_screened, delta_mu)
        print_status(f"  [X_i scr]  Pearson r:  {pearson_r_xi_scr:+.4f} (p={pearson_p_xi_scr:.4f})", "TEST")
        print_status(f"  [X_i scr]  Spearman rho: {spearman_rho_xi_scr:+.4f} (p={spearman_p_xi_scr:.4f})", "TEST")

        y = delta_mu
        yerr = df["delta_mu_err"].values

        # --- Legacy fit: Delta_mu = kappa_sv * sigma_v + intercept ---
        x_sv = sigma_v
        w = 1.0 / yerr ** 2
        S = np.sum(w); Sx = np.sum(w * x_sv); Sy = np.sum(w * y)
        Sxx = np.sum(w * x_sv ** 2); Sxy = np.sum(w * x_sv * y)
        denom = S * Sxx - Sx * Sx
        if abs(denom) > 0:
            kappa_sv = (S * Sxy - Sx * Sy) / denom
            intercept_sv = (Sxx * Sy - Sx * Sxy) / denom
            kappa_sv_err = np.sqrt(S / denom)
            intercept_sv_err = np.sqrt(Sxx / denom)
        else:
            kappa_sv, intercept_sv = 0.0, 0.0
            kappa_sv_err, intercept_sv_err = 0.0, 0.0
        kappa_sv_sigma = float(abs(kappa_sv) / kappa_sv_err) if kappa_sv_err > 0 else 0.0
        print_status(f"  [sigma_v]  Linear fit: kappa_sv = {kappa_sv:.6f} +/- {kappa_sv_err:.6f} ({kappa_sv_sigma:.2f}sigma)", "TEST")

        # --- Primary fit: Delta_mu = -kappa * X_i_screened + intercept ---
        # The screened coordinate X_i = (S_total * sigma_v^2 - U_ref_screened) / c^2
        # is the canonical TEP-H0 environmental coordinate.  kappa_Cep from
        # TEP-H0 is defined on this scale, so the screened fit is the physically
        # consistent one for comparing fitted and predicted slopes.
        x = x_i_screened
        w = 1.0 / yerr ** 2
        S = np.sum(w); Sx = np.sum(w * x); Sy = np.sum(w * y)
        Sxx = np.sum(w * x ** 2); Sxy = np.sum(w * x * y)
        denom = S * Sxx - Sx * Sx
        if abs(denom) > 0:
            kappa = (S * Sxy - Sx * Sy) / denom
            intercept = (Sxx * Sy - Sx * Sxy) / denom
            kappa_err = np.sqrt(S / denom)
            intercept_err = np.sqrt(Sxx / denom)
        else:
            kappa, intercept = 0.0, 0.0
            kappa_err, intercept_err = 0.0, 0.0
        kappa_sigma = float(abs(kappa) / kappa_err) if kappa_err > 0 else 0.0
        print_status(f"  [X_i scr]  Linear fit: kappa = {kappa:.4e} +/- {kappa_err:.4e} ({kappa_sigma:.2f}sigma)", "TEST")
        print_status(f"  [X_i scr]  TEP predicted kappa = -{self.KAPPA_CEP_DEFAULT:.0f} (default), -{self.KAPPA_CEP_WLS:.0f} (WLS)", "TEST")
        print_status(f"  [X_i scr]  Consistency: |kappa - pred|/err = {abs(kappa - (-self.KAPPA_CEP_DEFAULT))/kappa_err:.2f}sigma (default), {abs(kappa - (-self.KAPPA_CEP_WLS))/kappa_err:.2f}sigma (WLS)", "TEST")

        residuals = y - (kappa * x + intercept)
        chi2 = float(np.sum((residuals / yerr) ** 2))
        dof = max(n - 2, 1)
        chi2_reduced = chi2 / dof
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        # --- Unscreened fit (diagnostic / screening sensitivity) ---
        x_un = x_i
        w = 1.0 / yerr ** 2
        S = np.sum(w); Sx = np.sum(w * x_un); Sy = np.sum(w * y)
        Sxx = np.sum(w * x_un ** 2); Sxy = np.sum(w * x_un * y)
        denom = S * Sxx - Sx * Sx
        if abs(denom) > 0:
            kappa_un = (S * Sxy - Sx * Sy) / denom
            intercept_un = (Sxx * Sy - Sx * Sxy) / denom
            kappa_un_err = np.sqrt(S / denom)
            intercept_un_err = np.sqrt(Sxx / denom)
        else:
            kappa_un, intercept_un = 0.0, 0.0
            kappa_un_err, intercept_un_err = 0.0, 0.0
        kappa_un_sigma = float(abs(kappa_un) / kappa_un_err) if kappa_un_err > 0 else 0.0
        print_status(f"  [X_i uns]  Linear fit: kappa = {kappa_un:.4e} +/- {kappa_un_err:.4e} ({kappa_un_sigma:.2f}sigma)", "TEST")

        results = {
            "n_hosts": n,
            "pearson_r": float(pearson_r_xi_scr),
            "pearson_p": float(pearson_p_xi_scr),
            "spearman_rho": float(spearman_rho_xi_scr),
            "spearman_p": float(spearman_p_xi_scr),
            "unscreened_pearson_r": float(pearson_r_xi),
            "unscreened_pearson_p": float(pearson_p_xi),
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
                "coordinate": "screened",
                "description": "Primary fit on screened X_i = (S_total * sigma_v^2 - U_ref_screened) / c^2",
            },
            "unscreened_fit": {
                "kappa": float(kappa_un),
                "kappa_err": float(kappa_un_err),
                "kappa_significance_sigma": kappa_un_sigma,
                "intercept": float(intercept_un),
                "intercept_err": float(intercept_un_err),
                "description": "Diagnostic fit on unscreened X_i = (sigma_v^2 - U_ref_unscreened) / c^2",
            },
            "legacy_sigma_v_fit": {
                "kappa": float(kappa_sv),
                "kappa_err": float(kappa_sv_err),
                "kappa_significance_sigma": kappa_sv_sigma,
                "intercept": float(intercept_sv),
                "intercept_err": float(intercept_sv_err),
                "note": "Legacy fit on sigma_v (linear), not the correct TEP coordinate X_i (quadratic in sigma_v). Kept for comparison only.",
            },
            "tep_prediction": {
                "kappa_nonzero": True,
                "kappa_default": -self.KAPPA_CEP_DEFAULT,
                "kappa_wls": -self.KAPPA_CEP_WLS,
                "description": "TEP predicts delta_mu = -kappa_Cep * X_i with negative slope",
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

        x = df["X_i_screened"].values * 1e7  # plot in units of 1e-7
        y = df["delta_mu"].values
        yerr = df["delta_mu_err"].values

        # Data points
        ax.errorbar(
            x, y, yerr=yerr, fmt="o", color="#2166ac", ecolor="#92c5de",
            capsize=3, markersize=7, label="Matched hosts", zorder=3,
        )

        # Linear fit (in screened X_i space)
        x_fit = np.linspace(x.min() - 0.1, x.max() + 0.1, 200)
        # kappa is in mag per X_i (dimensionless), so scale by 1e-7 for plotting
        y_fit = (kappa * 1e7) * x_fit + intercept
        ax.plot(x_fit, y_fit, color="#4daf4a", linewidth=2,
                label=f"TEP fit: $\\kappa = {kappa:.2e} \\pm {kappa_err:.2e}$ mag", zorder=2)

        # TEP predicted slope
        x_pred = np.array([x.min() - 0.1, x.max() + 0.1])
        y_pred_default = (-self.KAPPA_CEP_DEFAULT * 1e7) * x_pred
        ax.plot(x_pred, y_pred_default, color="#ff7f00", linestyle=":", linewidth=1.5,
                label=f"TEP prediction ($\\kappa = -{self.KAPPA_CEP_DEFAULT/1e6:.2f} \\times 10^6$)", zorder=2)

        # Fit uncertainty band
        y_upper = ((kappa + kappa_err) * 1e7) * x_fit + (intercept + intercept_err)
        y_lower = ((kappa - kappa_err) * 1e7) * x_fit + (intercept - intercept_err)
        ax.fill_between(x_fit, y_lower, y_upper, alpha=0.15, color="#4daf4a", zorder=1)

        # Void prediction (kappa = 0, intercept = 0)
        ax.axhline(0, color="#b2182b", linestyle="--", linewidth=1.5,
                   label="Void prediction ($\\Delta\\mu = 0$)", zorder=2)

        ax.set_xlabel("$X_i = (S_{\\rm tot}\\,\\sigma_v^2 - U_{\\rm ref}^{\\rm scr}) / c^2$  ($\\times 10^{-7}$)", fontsize=12)
        ax.set_ylabel("$\\Delta\\mu = \\mu_{\\rm Cepheid} - \\mu_{\\rm TRGB}$ (mag)", fontsize=12)
        ax.set_title("Cepheid–TRGB Divergence vs TEP Potential Coordinate", fontsize=13)
        ax.legend(fontsize=9, loc="best")

        # Annotate with statistics
        text = (f"Pearson r = {fit_results['pearson_r']:+.3f} (p = {fit_results['pearson_p']:.3f})\n"
                f"Spearman $\\rho$ = {fit_results['spearman_rho']:+.3f} (p = {fit_results['spearman_p']:.3f})\n"
                f"$\\kappa$ = {kappa:.2e} $\\pm$ {kappa_err:.2e} ({fit_results['linear_fit']['kappa_significance_sigma']:.1f}$\\sigma$)\n"
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
            "predicts that Delta_mu = -kappa_Cep * X_i, where X_i = (sigma_v^2 - U_ref) / c^2 "
            "is the dimensionless TEP potential coordinate. A non-zero negative slope kappa "
            "is expected because each indicator's clock rate responds differently to the local "
            "potential. The kinematic void model predicts kappa = 0, since distances are "
            "indicator-independent and no potential dependence is expected.",
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
            "catalog or HyperLEDA). The TEP potential coordinate X_i = (sigma_v^2 - U_ref) / c^2 "
            "is computed for each host, with optional TEP screening S_total from TEP-H0 Step 03. "
            "The analysis uses Pearson and Spearman rank correlations and an inverse-variance "
            "weighted linear regression Delta_mu = -kappa * X_i + intercept to quantify the "
            "potential scaling. The slope significance (kappa / kappa_err) is the primary "
            "statistic for the void-vs-TEP comparison.",
            "PROCESS",
        )

        # Compute correlation and linear fit
        fit_results, kappa, intercept, kappa_err, intercept_err = self.compute_correlation(df)

        kappa_sig = fit_results["linear_fit"]["kappa_significance_sigma"]
        print_status(
            f"Interpretation: the fitted slope is kappa = {kappa:.4e} +/- {kappa_err:.4e} "
            f"mag ({kappa_sig:.2f} sigma). The TEP predicted slope is "
            f"-{self.KAPPA_CEP_DEFAULT:.0f} (default) or -{self.KAPPA_CEP_WLS:.0f} (WLS). "
            f"The observed slope is within "
            f"{abs(kappa - (-self.KAPPA_CEP_DEFAULT))/kappa_err:.2f} sigma of the default "
            f"prediction. The void prediction of kappa = 0 is "
            f"{'falsified' if kappa_sig > 2.0 else 'not definitively falsified'} at the "
            f"2-sigma threshold.",
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
            "model": "Delta_mu = -kappa * X_i + intercept, where X_i = (sigma_v^2 - U_ref) / c^2",
            "methodology": (
                "Pearson and Spearman rank correlations plus an inverse-variance "
                "weighted linear regression of Delta_mu on the TEP potential coordinate "
                "X_i = (sigma_v^2 - U_ref) / c^2. The slope kappa and its significance "
                "(kappa / kappa_err) serve as the primary discriminant between TEP "
                "(kappa non-zero, negative) and void (kappa = 0) predictions. "
                "A legacy fit on sigma_v (linear) is retained for comparison."
            ),
            "provenance": {
                "data_sources": [
                    "step_10 per-galaxy Delta_mu results",
                    "step_00 matched_hosts.csv interim catalog",
                    "step_01 host_potential_catalog.csv interim catalog",
                    "HyperLEDA (Makarov et al. 2014) velocity dispersions",
                    "TEP-H0 Step 03 screening factors (Tully 2015 group catalog)",
                ],
                "pipeline_block": "Ia — Indicator Divergence",
            },
            "scientific_context": (
                "Does the Cepheid-TRGB distance divergence scale with host "
                "gravitational potential depth? TEP predicts a non-zero negative "
                "coupling kappa (Delta_mu = -kappa_Cep * X_i); the void model predicts "
                "kappa = 0."
            ),
            "tep_prediction": (
                "Non-zero negative kappa: Delta_mu scales with X_i, reflecting "
                "indicator-dependent clock-rate responses to the local gravitational "
                "potential. Deeper potentials (higher X_i) yield more negative Delta_mu."
            ),
            "void_prediction": (
                "kappa = 0: no potential-dependent divergence, since distances are "
                "indicator-independent in the kinematic void model."
            ),
            "downstream_consumers": [
                "step_12_void_prediction_uniformity",
            ],
            "results": fit_results,
            "per_galaxy": df[["galaxy", "delta_mu", "delta_mu_err", "sigma_v", "X_i", "X_i_screened", "S_total"]].to_dict(orient="records"),
            "interpretation": {
                "tep": "Non-zero negative kappa indicates potential-dependent divergence, consistent with TEP",
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
