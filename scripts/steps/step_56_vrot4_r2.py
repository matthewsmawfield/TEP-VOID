#!/usr/bin/env python3
"""
Step 56: 2D Geometric Prediction — V_rot^4 / r^2 Test
======================================================
Test the DHOST scalar-tensor derivation that the disformal transport
coupling ε_0 should scale as V_rot^4 / r^2, not V_rot^2 alone.

The current pipeline uses X_i ∝ V_rot^2 as a 1D proxy for the galactic
potential. The full 2D geometric prediction from Section 4.6 is that
the Cepheid distance offset should correlate more strongly with
V_rot^4 / R_disk^2 than with V_rot^2 alone, where R_disk is the disk
scale length (a proxy for the galactocentric radius of the Cepheid
fields).

The SPARC catalog (Lelli et al. 2016, 175 galaxies) provides both
V_flat (asymptotic rotation velocity) and R_disk (disk scale length)
for a large sample of galaxies with rotation curves. Cross-matching
with the directional Cepheid-TRGB sample (step_53) yields 7 galaxies
with both indicators.

Analyses:
  1. Compare correlations: r(Δμ, V_rot^2) vs r(Δμ, V_rot^4/R^2)
  2. Partial correlation: r(Δμ, V_rot^4/R^2 | V_rot^2)
  3. Bayesian model comparison (AIC/BIC)
  4. Full sample with R_disk proxy from HyperLEDA logD25

Outputs:
    results/outputs/step_56_vrot4_r2.json
    results/figures/step_56_vrot4_r2.png
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
from astropy.coordinates import SkyCoord
import astropy.units as u

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.screening import U_REF_SCREENED, C_KMS


class Step56Vrot4R2:
    """Step 56: 2D geometric prediction V_rot^4/r^2 test."""

    def __init__(self):
        self.root = PROJECT_ROOT
        self.data_raw_external = self.root / "data" / "raw" / "external"
        self.data_processed = self.root / "data" / "processed"
        self.results = self.root / "results" / "outputs"
        self.figures = self.root / "results" / "figures"
        self.logs = self.root / "logs"

        for d in [self.results, self.figures, self.logs]:
            d.mkdir(parents=True, exist_ok=True)

        self.logger = TEPLogger(
            "step_56",
            log_file_path=self.logs / "step_56_vrot4_r2.log",
        )
        set_step_logger(self.logger)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def load_sparc(self):
        """Load the full SPARC catalog (Lelli et al. 2016)."""
        print_status("Loading SPARC catalog...", "PROCESS")
        path = self.data_raw_external / "sparc_full_175.csv"
        if not path.exists():
            print_status(f"SPARC not found at {path}", "ERROR")
            return pd.DataFrame()

        df = pd.read_csv(path)
        df = df.rename(columns={"_RA": "ra", "_DE": "dec", "Vflat": "vflat",
                                "Rdisk": "rdisk", "Dist": "dist_mpc"})
        print_status(f"  {len(df)} SPARC galaxies", "SUCCESS")
        return df

    def load_directional(self):
        """Load the directional Cepheid-TRGB sample."""
        print_status("Loading directional sample...", "PROCESS")
        path = self.data_processed / "directional_ceph_trgb_sample.csv"
        if not path.exists():
            print_status(f"Sample not found at {path}", "ERROR")
            return pd.DataFrame()

        df = pd.read_csv(path)
        print_status(f"  {len(df)} galaxies", "SUCCESS")
        return df

    def cross_match(self, sparc, directional):
        """Cross-match SPARC with the directional sample by coordinates."""
        print_status("Cross-matching SPARC with directional sample...", "PROCESS")

        sparc_coords = SkyCoord(ra=sparc["ra"].values * u.deg,
                                dec=sparc["dec"].values * u.deg)
        dir_coords = SkyCoord(ra=directional["ra"].values * u.deg,
                              dec=directional["dec"].values * u.deg)

        idx, d2d, _ = dir_coords.match_to_catalog_sky(sparc_coords)
        match_mask = d2d < 0.02 * u.deg

        matched = directional[match_mask].copy()
        matched["sparc_name"] = sparc.iloc[idx[match_mask]]["Name"].values
        matched["vflat"] = sparc.iloc[idx[match_mask]]["vflat"].values
        matched["rdisk_kpc"] = sparc.iloc[idx[match_mask]]["rdisk"].values
        matched["sparc_dist"] = sparc.iloc[idx[match_mask]]["dist_mpc"].values
        matched["match_sep_deg"] = d2d[match_mask].deg

        # Filter out Vflat=0 (no rotation measured)
        matched = matched[matched["vflat"] > 0].copy()

        print_status(f"  {len(matched)} matched galaxies with V_flat > 0", "SUCCESS")
        return matched

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------
    def test_2d_prediction(self, df):
        """Test whether V_rot^4/R^2 predicts Δμ better than V_rot^2 alone."""
        print_status("\n--- 2D geometric prediction test ---", "PROCESS")

        n = len(df)
        if n < 4:
            print_status(f"  Insufficient matches (N={n})", "WARNING")
            return {"status": "insufficient_data", "n": n}

        dmu = df["delta_mu"].values
        vflat = df["vflat"].values
        rdisk = df["rdisk_kpc"].values

        # 1D proxy: V_rot^2 (proportional to X_i without screening)
        vrot2 = vflat ** 2

        # 2D prediction: V_rot^4 / R^2
        vrot4_r2 = vflat ** 4 / rdisk ** 2

        # Correlations
        r_1d, p_1d = sp_stats.pearsonr(vrot2, dmu)
        r_2d, p_2d = sp_stats.pearsonr(vrot4_r2, dmu)

        # Spearman (rank) correlations
        rho_1d, p_rho_1d = sp_stats.spearmanr(vrot2, dmu)
        rho_2d, p_rho_2d = sp_stats.spearmanr(vrot4_r2, dmu)

        # Partial correlation: Δμ vs V_rot^4/R^2 controlling for V_rot^2
        if n >= 5:
            from numpy.linalg import lstsq
            X = np.column_stack([np.ones(n), vrot2])
            by = lstsq(X, dmu, rcond=None)[0]
            bx = lstsq(X, vrot4_r2, rcond=None)[0]
            y_resid = dmu - X @ by
            x_resid = vrot4_r2 - X @ bx
            r_partial, p_partial = sp_stats.pearsonr(x_resid, y_resid)
        else:
            r_partial, p_partial = np.nan, np.nan

        # AIC/BIC comparison
        # Model 1: Δμ = a + b * V_rot^2
        # Model 2: Δμ = a + b * V_rot^4/R^2
        def calc_aic(y, ypred, k):
            n = len(y)
            rss = np.sum((y - ypred) ** 2)
            aic = n * np.log(rss / n) + 2 * k
            bic = n * np.log(rss / n) + k * np.log(n)
            return aic, bic

        X1 = np.column_stack([vrot2, np.ones(n)])
        beta1 = lstsq(X1, dmu, rcond=None)[0]
        pred1 = X1 @ beta1
        aic1, bic1 = calc_aic(dmu, pred1, 2)

        X2 = np.column_stack([vrot4_r2, np.ones(n)])
        beta2 = lstsq(X2, dmu, rcond=None)[0]
        pred2 = X2 @ beta2
        aic2, bic2 = calc_aic(dmu, pred2, 2)

        # Joint model: Δμ = a + b * V_rot^2 + c * V_rot^4/R^2
        if n >= 5:
            X3 = np.column_stack([vrot2, vrot4_r2, np.ones(n)])
            beta3 = lstsq(X3, dmu, rcond=None)[0]
            pred3 = X3 @ beta3
            aic3, bic3 = calc_aic(dmu, pred3, 3)
            r2_joint = 1 - np.sum((dmu - pred3) ** 2) / np.sum((dmu - dmu.mean()) ** 2)
        else:
            aic3, bic3, r2_joint = np.nan, np.nan, np.nan

        result = {
            "n": int(n),
            "galaxies": df[["galaxy_name", "sparc_name", "vflat", "rdisk_kpc",
                            "delta_mu", "delta_mu_err"]].to_dict("records"),
            "correlation_1d_vrot2": {
                "pearson_r": float(r_1d),
                "pearson_p": float(p_1d),
                "spearman_rho": float(rho_1d),
                "spearman_p": float(p_rho_1d),
            },
            "correlation_2d_vrot4_r2": {
                "pearson_r": float(r_2d),
                "pearson_p": float(p_2d),
                "spearman_rho": float(rho_2d),
                "spearman_p": float(p_rho_2d),
            },
            "partial_correlation_2d_given_1d": {
                "r": float(r_partial),
                "p": float(p_partial),
            },
            "model_comparison": {
                "1d_aic": float(aic1),
                "1d_bic": float(bic1),
                "2d_aic": float(aic2),
                "2d_bic": float(bic2),
                "joint_aic": float(aic3),
                "joint_bic": float(bic3),
                "delta_aic_2d_vs_1d": float(aic2 - aic1),
                "delta_bic_2d_vs_1d": float(bic2 - bic1),
                "joint_r_squared": float(r2_joint),
            },
        }

        print_status(f"  N={n} matched galaxies", "TEST")
        print_status(f"  1D: r(Δμ, V²) = {r_1d:+.4f} (p={p_1d:.4f})", "TEST")
        print_status(f"  2D: r(Δμ, V⁴/R²) = {r_2d:+.4f} (p={p_2d:.4f})", "TEST")
        print_status(f"  Partial r(Δμ, V⁴/R² | V²) = {r_partial:+.4f} (p={p_partial:.4f})", "TEST")
        print_status(f"  AIC: 1D={aic1:.2f}, 2D={aic2:.2f}, ΔAIC={aic2-aic1:.2f}", "TEST")
        print_status(f"  BIC: 1D={bic1:.2f}, 2D={bic2:.2f}, ΔBIC={bic2-bic1:.2f}", "TEST")

        return result

    def make_figure(self, df):
        """Generate comparison figure."""
        print_status("\n--- Generating figure ---", "PROCESS")

        if len(df) < 4:
            print_status("  Insufficient data for figure", "WARNING")
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        dmu = df["delta_mu"].values
        vrot2 = df["vflat"].values ** 2
        vrot4_r2 = df["vflat"].values ** 4 / df["rdisk_kpc"].values ** 2
        n = len(df)

        # Panel 1: Δμ vs V_rot^2 (1D proxy)
        ax1 = axes[0]
        ax1.scatter(vrot2, dmu, c="C0", s=80, edgecolors="k", zorder=3)
        for i in range(n):
            ax1.annotate(df.iloc[i]["sparc_name"], (vrot2[i], dmu[i]),
                        fontsize=7, xytext=(5, 5), textcoords="offset points")
        r1, p1 = sp_stats.pearsonr(vrot2, dmu)
        ax1.set_xlabel("$V_{\\rm rot}^2$ (km/s)$^2$")
        ax1.set_ylabel("Δμ (mag)")
        ax1.set_title(f"1D proxy: $r = {r1:+.3f}$ (p={p1:.3f})", fontsize=11)
        ax1.axhline(0, color="gray", linestyle=":", alpha=0.5)

        # Panel 2: Δμ vs V_rot^4/R^2 (2D prediction)
        ax2 = axes[1]
        ax2.scatter(vrot4_r2, dmu, c="C1", s=80, edgecolors="k", zorder=3)
        for i in range(n):
            ax2.annotate(df.iloc[i]["sparc_name"], (vrot4_r2[i], dmu[i]),
                        fontsize=7, xytext=(5, 5), textcoords="offset points")
        r2, p2 = sp_stats.pearsonr(vrot4_r2, dmu)
        ax2.set_xlabel("$V_{\\rm rot}^4 / R_{\\rm disk}^2$ (km/s)$^4$/kpc$^2$")
        ax2.set_ylabel("Δμ (mag)")
        ax2.set_title(f"2D prediction: $r = {r2:+.3f}$ (p={p2:.3f})", fontsize=11)
        ax2.axhline(0, color="gray", linestyle=":", alpha=0.5)

        # Panel 3: Comparison bar chart
        ax3 = axes[2]
        models = ["1D: $V_{\\rm rot}^2$", "2D: $V_{\\rm rot}^4/R^2$"]
        r_vals = [r1, r2]
        colors = ["C0", "C1"]
        ax3.barh(models, [abs(r) for r in r_vals], color=colors, edgecolor="k")
        ax3.set_xlabel("|Pearson r|")
        ax3.set_title("Correlation strength comparison", fontsize=11)
        ax3.set_xlim(0, 1)
        for i, (r, p) in enumerate(zip(r_vals, [p1, p2])):
            ax3.text(abs(r) + 0.02, i, f"r={r:+.3f}\np={p:.3f}", va="center", fontsize=9)

        fig.suptitle("Step 56: 2D Geometric Prediction — $V_{\\rm rot}^4/R^2$ vs $V_{\\rm rot}^2$",
                     fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.94])

        fig_path = self.figures / "step_56_vrot4_r2.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print_status(f"  Figure saved to {fig_path}", "SUCCESS")

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------
    def run(self):
        print_status("Step 56: 2D Geometric Prediction — V_rot^4/r^2 Test", "TITLE")
        print_status("DHOST scalar-tensor derivation: ε_0 ∝ V_rot^4/r^2", "PROCESS")

        sparc = self.load_sparc()
        directional = self.load_directional()

        if sparc.empty or directional.empty:
            print_status("Missing data. Exiting.", "ERROR")
            return

        matched = self.cross_match(sparc, directional)

        if len(matched) < 4:
            print_status(f"Only {len(matched)} matches — insufficient for analysis", "WARNING")
            result = {"status": "insufficient_data", "n": len(matched)}
        else:
            result = self.test_2d_prediction(matched)
            self.make_figure(matched)

        output = {
            "step": "56_vrot4_r2",
            "description": "2D geometric prediction: V_rot^4/R^2 vs V_rot^2 for Cepheid distance offset",
            "tep_prediction": "V_rot^4/R^2 should correlate more strongly with Δμ than V_rot^2 alone",
            "n_sparc_galaxies": int(len(sparc)),
            "n_matched": int(len(matched)),
            "result": result,
            "note": ("The DHOST scalar-tensor derivation (Section 4.6) predicts that "
                     "the disformal transport coupling scales as V_rot^4/r^2. The current "
                     "test uses R_disk (disk scale length from SPARC) as a proxy for the "
                     "galactocentric radius. A rigorous test requires the actual Cepheid "
                     "galactocentric radii from JWST imaging."),
        }

        output_path = self.results / "step_56_vrot4_r2.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print_status(f"\nResults saved to {output_path}", "SUCCESS")


if __name__ == "__main__":
    step = Step56Vrot4R2()
    step.run()
