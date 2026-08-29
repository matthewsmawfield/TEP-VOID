#!/usr/bin/env python3
"""
Step 10: Matched Host Comparison — Cepheid vs TRGB Distance Moduli
====================================================================
Compare Cepheid-derived and TRGB-derived distance moduli for matched
host galaxies to quantify the indicator-specific distance divergence.

Key Tasks:
1. Load matched host catalog (from step_00), SH0ES Cepheid hosts, and
   CCHP TRGB hosts
2. Compute Delta_mu = mu_Cepheid - mu_TRGB for each matched host
3. Compute statistics: mean, median, std, significance
4. Generate divergence figure and JSON summary

Outputs:
    results/outputs/step_10_matched_host_comparison.json
    results/figures/step_10_cepheid_trgb_divergence.png
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status


# NOTE: No hardcoded fallback catalog. If data/interim/matched_hosts.csv is
# missing, the step fails loudly rather than emitting plausible-looking
# results from typed-in values. Re-run step_00 (data ingestion) to
# regenerate the matched-host catalog from the real source data.


class Step10MatchedHostComparison:
    """Step 10: Compare Cepheid and TRGB distance moduli for matched hosts."""

    def __init__(self):
        self.root = PROJECT_ROOT
        self.data_interim = self.root / "data" / "interim"
        self.data_raw = self.root / "data" / "raw"
        self.data_external = self.data_raw / "external"
        self.results = self.root / "results" / "outputs"
        self.figures = self.root / "results" / "figures"
        self.logs = self.root / "logs"

        for d in [self.data_interim, self.results, self.figures, self.logs]:
            d.mkdir(parents=True, exist_ok=True)

        self.logger = TEPLogger("step_10", log_file_path=self.logs / "step_10_matched_host_comparison.log")
        set_step_logger(self.logger)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def load_matched_hosts(self):
        """Load matched host catalog from step_00 interim files.

        The matched_hosts.csv produced by step_00 contains the full
        cross-match of R22 Cepheid, Freedman 2025 TRGB, and HyperLEDA
        velocity dispersions, with columns: galaxy, mu_cepheid,
        mu_cepheid_err, trgb_mu, trgb_mu_err, sigma_kms, delta_mu.
        """
        matched_path = self.data_interim / "matched_hosts.csv"

        print_status("Loading matched host catalog...", "PROCESS")

        # Try the matched_hosts.csv from step_00 first
        if matched_path.exists():
            matched_df = pd.read_csv(matched_path)
            if len(matched_df) > 0 and "mu_cepheid" in matched_df.columns:
                print_status(f"Loaded {len(matched_df)} matched hosts from {matched_path}", "SUCCESS")

                # Normalize column names
                result = pd.DataFrame()
                result["galaxy"] = matched_df.get("galaxy", matched_df.get("galaxy_y", ""))
                result["mu_cepheid"] = pd.to_numeric(matched_df["mu_cepheid"], errors="coerce")
                result["mu_cepheid_err"] = pd.to_numeric(
                    matched_df.get("mu_cepheid_err", 0.06), errors="coerce"
                )
                result["mu_trgb"] = pd.to_numeric(
                    matched_df.get("trgb_mu", matched_df.get("mu_trgb", np.nan)), errors="coerce"
                )
                result["mu_trgb_err"] = pd.to_numeric(
                    matched_df.get("trgb_mu_err", matched_df.get("mu_trgb_err", 0.05)), errors="coerce"
                )
                result["sigma_v"] = pd.to_numeric(
                    matched_df.get("sigma_kms", matched_df.get("sigma_v", np.nan)), errors="coerce"
                )
                result["delta_mu"] = pd.to_numeric(
                    matched_df.get("delta_mu", result["mu_cepheid"] - result["mu_trgb"]), errors="coerce"
                )

                result = result.dropna(subset=["mu_cepheid", "mu_trgb"])
                print_status(f"  {len(result)} hosts with both Cepheid and TRGB moduli", "SUCCESS")
                if result["sigma_v"].notna().sum() > 0:
                    print_status(f"  {int(result['sigma_v'].notna().sum())} hosts with velocity dispersions", "SUCCESS")
                return result

        # No fallback: fail loudly if the real matched-host catalog is missing.
        raise FileNotFoundError(
            f"Required input {matched_path} not found or not usable. "
            "Re-run step_00 (data ingestion) to regenerate the matched-host "
            "catalog from the real SH0ES Cepheid, CCHP TRGB, and HyperLEDA sources."
        )

    def _merge_ceph_trgb(self, cepheid_df, trgb_df):
        """Merge Cepheid and TRGB DataFrames on galaxy name."""
        # Identify galaxy name columns
        ceph_name_col = self._find_name_col(cepheid_df.columns)
        trgb_name_col = self._find_name_col(trgb_df.columns)

        if ceph_name_col is None or trgb_name_col is None:
            print_status("Could not identify galaxy name columns for merge.", "WARNING")
            return None

        # Identify modulus columns
        ceph_mu_col = self._find_mu_col(cepheid_df.columns, "cepheid")
        ceph_err_col = self._find_err_col(cepheid_df.columns, "cepheid")
        trgb_mu_col = self._find_mu_col(trgb_df.columns, "trgb")
        trgb_err_col = self._find_err_col(trgb_df.columns, "trgb")

        if ceph_mu_col is None or trgb_mu_col is None:
            print_status("Could not identify distance modulus columns for merge.", "WARNING")
            return None

        cepheid_df = cepheid_df.copy()
        trgb_df = trgb_df.copy()
        cepheid_df["_gal"] = cepheid_df[ceph_name_col].astype(str).str.upper().str.strip().str.replace(" ", "")
        trgb_df["_gal"] = trgb_df[trgb_name_col].astype(str).str.upper().str.strip().str.replace(" ", "")

        merged = pd.merge(
            cepheid_df, trgb_df, on="_gal", suffixes=("_cep", "_trgb")
        )

        # Rename to standard columns
        result = pd.DataFrame()
        result["galaxy"] = merged[ceph_name_col]
        result["mu_cepheid"] = merged[ceph_mu_col]
        result["mu_cepheid_err"] = merged[ceph_err_col] if ceph_err_col else 0.06
        result["mu_trgb"] = merged[trgb_mu_col]
        result["mu_trgb_err"] = merged[trgb_err_col] if trgb_err_col else 0.05

        # Try to find sigma_v
        sigma_col = self._find_sigma_col(merged.columns)
        result["sigma_v"] = merged[sigma_col] if sigma_col else np.nan

        return result.dropna(subset=["mu_cepheid", "mu_trgb"])

    @staticmethod
    def _find_name_col(columns):
        for c in columns:
            cl = c.lower()
            if "gal" in cl or "host" in cl or "name" in cl:
                return c
        return None

    @staticmethod
    def _find_mu_col(columns, indicator):
        candidates = []
        for c in columns:
            cl = c.lower()
            if "mu" in cl and ("err" not in cl) and ("h0" not in cl):
                candidates.append(c)
            if indicator in cl and "mu" in cl and "err" not in cl:
                return c
        return candidates[0] if candidates else None

    @staticmethod
    def _find_err_col(columns, indicator):
        for c in columns:
            cl = c.lower()
            if "mu" in cl and "err" in cl:
                return c
        return None

    @staticmethod
    def _find_sigma_col(columns):
        for c in columns:
            cl = c.lower()
            if "sigma" in cl and "v" in cl:
                return c
            if cl == "sigma":
                return c
        return None

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------
    def compute_divergence(self, df):
        """Compute Delta_mu = mu_Cepheid - mu_TRGB and associated statistics."""
        print_status("Computing Delta_mu = mu_Cepheid - mu_TRGB...", "PROCESS")

        df = df.copy()
        df["delta_mu"] = df["mu_cepheid"] - df["mu_trgb"]
        df["delta_mu_err"] = np.sqrt(df["mu_cepheid_err"] ** 2 + df["mu_trgb_err"] ** 2)

        n = len(df)
        mean_delta = float(np.mean(df["delta_mu"]))
        median_delta = float(np.median(df["delta_mu"]))
        std_delta = float(np.std(df["delta_mu"], ddof=1)) if n > 1 else 0.0
        sem_delta = std_delta / np.sqrt(n) if n > 0 else 0.0

        # Weighted mean (inverse-variance)
        weights = 1.0 / df["delta_mu_err"] ** 2
        weighted_mean = float(np.sum(weights * df["delta_mu"]) / np.sum(weights))
        weighted_err = float(1.0 / np.sqrt(np.sum(weights)))

        # Significance of the divergence (weighted)
        significance_sigma = float(abs(weighted_mean) / weighted_err) if weighted_err > 0 else 0.0

        # t-test against zero
        if n > 1 and sem_delta > 0:
            t_stat = mean_delta / sem_delta
        else:
            t_stat = 0.0

        print_status(f"  N matched hosts: {n}", "INFO")
        print_status(f"  Mean Delta_mu:   {mean_delta:+.4f} mag", "INFO")
        print_status(f"  Median Delta_mu: {median_delta:+.4f} mag", "INFO")
        print_status(f"  Std Delta_mu:    {std_delta:.4f} mag", "INFO")
        print_status(f"  Weighted mean:   {weighted_mean:+.4f} +/- {weighted_err:.4f} mag", "INFO")
        print_status(f"  Significance:    {significance_sigma:.2f} sigma", "TEST")

        stats = {
            "n_hosts": n,
            "mean_delta_mu": mean_delta,
            "median_delta_mu": median_delta,
            "std_delta_mu": std_delta,
            "sem_delta_mu": float(sem_delta),
            "weighted_mean_delta_mu": weighted_mean,
            "weighted_err_delta_mu": weighted_err,
            "significance_sigma": significance_sigma,
            "t_statistic": float(t_stat),
            "definition": "Delta_mu = mu_Cepheid - mu_TRGB",
        }
        return df, stats

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    def plot_divergence(self, df, stats):
        """Generate the Cepheid-TRGB divergence figure."""
        print_status("Generating divergence figure...", "PROCESS")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left panel: per-galaxy Delta_mu with error bars
        ax = axes[0]
        galaxies = df["galaxy"].tolist()
        x = np.arange(len(galaxies))
        ax.errorbar(
            x, df["delta_mu"], yerr=df["delta_mu_err"],
            fmt="o", color="#2166ac", ecolor="#92c5de", capsize=3, markersize=6,
        )
        ax.axhline(0, color="#b2182b", linestyle="--", linewidth=1.5, label="Void prediction ($\\Delta\\mu = 0$)")
        ax.axhline(
            stats["weighted_mean_delta_mu"],
            color="#4daf4a", linestyle="-", linewidth=1.5,
            label=f"Weighted mean: {stats['weighted_mean_delta_mu']:+.3f} mag",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(galaxies, rotation=75, ha="right", fontsize=8)
        ax.set_ylabel("$\\Delta\\mu = \\mu_{\\rm Cepheid} - \\mu_{\\rm TRGB}$ (mag)")
        ax.set_title("Per-Galaxy Cepheid–TRGB Distance Modulus Divergence")
        ax.legend(fontsize=9, loc="best")
        ax.set_ylim(-0.35, 0.35)

        # Right panel: histogram of Delta_mu
        ax = axes[1]
        ax.hist(
            df["delta_mu"], bins=min(10, max(5, len(df) // 2)),
            color="#2166ac", alpha=0.7, edgecolor="white",
        )
        ax.axvline(0, color="#b2182b", linestyle="--", linewidth=1.5, label="Void prediction ($\\Delta\\mu = 0$)")
        ax.axvline(
            stats["weighted_mean_delta_mu"],
            color="#4daf4a", linestyle="-", linewidth=1.5,
            label=f"Weighted mean: {stats['weighted_mean_delta_mu']:+.3f} mag",
        )
        ax.set_xlabel("$\\Delta\\mu$ (mag)")
        ax.set_ylabel("Count")
        ax.set_title(f"Distribution of $\\Delta\\mu$ (N={stats['n_hosts']})")
        ax.legend(fontsize=9, loc="best")

        fig.tight_layout()
        fig_path = self.figures / "step_10_cepheid_trgb_divergence.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print_status(f"Figure saved to {fig_path}", "SUCCESS")
        return fig_path

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(self):
        """Execute the full step."""
        print_status("Step 10: Matched Host Comparison — Cepheid vs TRGB", "TITLE")

        print_status(
            "Scientific question: do Cepheid and TRGB distance moduli agree for galaxies "
            "observed by both indicators? The kinematic void model predicts that all "
            "distance indicators yield identical moduli (Delta_mu = 0), because distances "
            "are purely geometric and indicator-independent. The TEP framework predicts a "
            "non-zero, potential-dependent divergence, since each indicator's clock rate "
            "responds differently to the local gravitational potential. This step "
            "quantifies the per-galaxy divergence Delta_mu = mu_Cepheid - mu_TRGB and "
            "establishes the observational baseline for the void-vs-TEP comparison in "
            "Block Ia.",
            "PROCESS",
        )

        # Load data
        df = self.load_matched_hosts()

        print_status(
            "Methodology: the matched-host sample is restricted to galaxies with both "
            "Cepheid (Riess et al. 2022) and TRGB (Freedman et al. 2025) distance moduli. "
            "For each host, Delta_mu is computed as the direct difference of the two "
            "moduli, with Gaussian error propagation from the published per-indicator "
            "uncertainties. Sample statistics include unweighted and inverse-variance "
            "weighted means, standard error, and a t-statistic against the null "
            "hypothesis Delta_mu = 0.",
            "PROCESS",
        )

        # Compute divergence
        df, stats = self.compute_divergence(df)

        print_status(
            f"Interpretation: the weighted mean divergence is "
            f"{stats['weighted_mean_delta_mu']:+.4f} +/- {stats['weighted_err_delta_mu']:.4f} mag "
            f"({stats['significance_sigma']:.2f} sigma). A non-zero weighted mean is "
            f"inconsistent with the void prediction of Delta_mu = 0 and is consistent "
            f"with the TEP prediction of indicator-dependent distance moduli. The "
            f"per-galaxy scatter ({stats['std_delta_mu']:.4f} mag) provides the input "
            f"for the potential-scaling test in step_11.",
            "TEST",
        )

        # Generate figure
        fig_path = self.plot_divergence(df, stats)

        # Save per-galaxy results
        per_galaxy_path = self.results / "step_10_per_galaxy_delta_mu.csv"
        df.to_csv(per_galaxy_path, index=False)
        print_status(f"Per-galaxy results saved to {per_galaxy_path}", "SUCCESS")

        # Summary JSON
        summary = {
            "step": "10_matched_host_comparison",
            "description": "Cepheid vs TRGB distance modulus divergence for matched hosts",
            "delta_mu_definition": "Delta_mu = mu_Cepheid - mu_TRGB",
            "methodology": (
                "Direct difference of Cepheid (Riess et al. 2022) and TRGB "
                "(Freedman et al. 2025) distance moduli for matched hosts, with "
                "Gaussian error propagation. Statistics include unweighted and "
                "inverse-variance weighted means, standard error, and a t-statistic "
                "against Delta_mu = 0."
            ),
            "provenance": {
                "data_sources": [
                    "Riess et al. 2022 (SH0ES Cepheid distance moduli)",
                    "Freedman et al. 2025 (CCHP TRGB distance moduli)",
                    "HyperLEDA (Makarov et al. 2014) velocity dispersions",
                    "step_00 matched_hosts.csv interim catalog",
                ],
                "pipeline_block": "Ia — Indicator Divergence",
            },
            "scientific_context": (
                "Do Cepheid and TRGB distance moduli agree for galaxies observed by "
                "both indicators? The void model predicts Delta_mu = 0 for all hosts; "
                "TEP predicts a non-zero, potential-dependent divergence."
            ),
            "tep_prediction": (
                "Non-zero Delta_mu that scales with host gravitational potential, "
                "arising from indicator-dependent clock-rate responses to the local "
                "potential."
            ),
            "void_prediction": (
                "Delta_mu = 0 for all hosts: all distance indicators yield identical "
                "moduli because distances are purely geometric and indicator-independent."
            ),
            "downstream_consumers": [
                "step_11_indicator_divergence_vs_potential",
                "step_12_void_prediction_uniformity",
            ],
            "statistics": stats,
            "per_galaxy": df[["galaxy", "mu_cepheid", "mu_cepheid_err", "mu_trgb", "mu_trgb_err",
                              "delta_mu", "delta_mu_err"]].to_dict(orient="records"),
            "void_prediction": {
                "delta_mu": 0.0,
                "description": "Kinematic void predicts identical distances for all indicators",
            },
            "tep_prediction": {
                "delta_mu_nonzero": True,
                "scales_with_potential": True,
                "description": "TEP predicts non-zero, potential-dependent divergence",
            },
            "output_files": [
                str(self.results / "step_10_matched_host_comparison.json"),
                str(per_galaxy_path),
                str(fig_path),
            ],
        }

        summary_path = self.results / "step_10_matched_host_comparison.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print_status(f"Summary saved to {summary_path}", "SUCCESS")

        print_status("Step 10 complete", "SUCCESS")


if __name__ == "__main__":
    step = Step10MatchedHostComparison()
    step.run()
