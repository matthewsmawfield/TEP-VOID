#!/usr/bin/env python3
"""
Step 12: Void Prediction Uniformity Test
==========================================
Quantify the kinematic void model prediction that all distance indicators
yield identical H0 (Delta_mu = 0 for all hosts) and test it against the
observed Cepheid-TRGB divergence.

Key Tasks:
1. Load the observed Delta_mu distribution (from step_10)
2. Formulate the void null hypothesis: Delta_mu = 0 for all hosts
3. Compute chi-squared and p-value for the void null hypothesis
4. Compare with the TEP prediction of non-zero, potential-dependent divergence

Outputs:
    results/outputs/step_12_void_prediction_uniformity.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status


class Step12VoidUniformity:
    """Step 12: Quantify and test the void model prediction of indicator uniformity."""

    def __init__(self):
        self.root = PROJECT_ROOT
        self.data_interim = self.root / "data" / "interim"
        self.results = self.root / "results" / "outputs"
        self.figures = self.root / "results" / "figures"
        self.logs = self.root / "logs"

        for d in [self.data_interim, self.results, self.figures, self.logs]:
            d.mkdir(parents=True, exist_ok=True)

        self.logger = TEPLogger("step_12", log_file_path=self.logs / "step_12_void_prediction_uniformity.log")
        set_step_logger(self.logger)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def load_data(self):
        """Load the observed Delta_mu distribution from step_10."""
        print_status("Loading observed Delta_mu distribution...", "PROCESS")

        step10_path = self.results / "step_10_per_galaxy_delta_mu.csv"
        step10_json = self.results / "step_10_matched_host_comparison.json"

        if step10_path.exists():
            print_status(f"Loading step_10 per-galaxy results: {step10_path}", "PROCESS")
            df = pd.read_csv(step10_path)
            print_status(f"  {len(df)} hosts with Delta_mu", "SUCCESS")
            return df

        # Try to load from step_10 JSON summary
        if step10_json.exists():
            print_status(f"Loading step_10 JSON summary: {step10_json}", "PROCESS")
            with open(step10_json) as f:
                step10_data = json.load(f)
            per_galaxy = step10_data.get("per_galaxy", [])
            if per_galaxy:
                df = pd.DataFrame(per_galaxy)
                print_status(f"  {len(df)} hosts extracted from JSON", "SUCCESS")
                return df

        # No fallback: fail loudly if step_10 output is missing.
        raise FileNotFoundError(
            f"Required input not found. Expected either {step10_path} "
            f"or {step10_json}. Re-run step_10 (matched host comparison) "
            "to regenerate the per-galaxy Delta_mu results from the real "
            "matched-host catalog."
        )

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------
    def test_void_hypothesis(self, df):
        """Test the void null hypothesis: Delta_mu = 0 for all hosts.

        Computes:
        - Chi-squared statistic against the void prediction (Delta_mu = 0)
        - p-value from the chi-squared distribution
        - Significance in sigma equivalents
        - One-sample t-test against zero
        - Wilcoxon signed-rank test against zero
        """
        print_status("Testing void null hypothesis: Delta_mu = 0 for all hosts...", "PROCESS")

        delta_mu = df["delta_mu"].values
        delta_mu_err = df["delta_mu_err"].values
        n = len(df)

        # --- Chi-squared test ---
        # Under the void null hypothesis, Delta_mu_i = 0 for all i.
        # Chi-squared = sum((Delta_mu_i - 0)^2 / sigma_i^2)
        chi2 = float(np.sum((delta_mu / delta_mu_err) ** 2))
        dof = n  # all parameters are fixed under the null (no free parameters)
        chi2_reduced = chi2 / dof

        # p-value: probability of observing chi2 or larger under the null
        p_value_chi2 = float(sp_stats.chi2.sf(chi2, dof))

        # Significance in sigma (one-sided)
        significance_sigma = float(sp_stats.norm.ppf(1.0 - p_value_chi2 / 2.0))

        print_status(f"  Chi-squared:     {chi2:.2f} (dof={dof})", "TEST")
        print_status(f"  Reduced chi-sq:  {chi2_reduced:.3f}", "TEST")
        print_status(f"  p-value (chi2):  {p_value_chi2:.6e}", "TEST")
        print_status(f"  Significance:    {significance_sigma:.2f} sigma", "TEST")

        # --- One-sample t-test against zero ---
        t_stat, t_p = sp_stats.ttest_1samp(delta_mu, 0.0)
        print_status(f"  t-test:          t={t_stat:.3f}, p={t_p:.4f}", "TEST")

        # --- Wilcoxon signed-rank test ---
        try:
            wilcoxon_stat, wilcoxon_p = sp_stats.wilcoxon(delta_mu)
            print_status(f"  Wilcoxon:        W={wilcoxon_stat:.1f}, p={wilcoxon_p:.4f}", "TEST")
        except Exception:
            wilcoxon_stat, wilcoxon_p = float("nan"), float("nan")

        # --- Weighted mean and its significance ---
        weights = 1.0 / delta_mu_err ** 2
        weighted_mean = float(np.sum(weights * delta_mu) / np.sum(weights))
        weighted_err = float(1.0 / np.sqrt(np.sum(weights)))
        weighted_sigma = float(abs(weighted_mean) / weighted_err) if weighted_err > 0 else 0.0

        print_status(f"  Weighted mean:   {weighted_mean:+.4f} +/- {weighted_err:.4f} mag", "TEST")
        print_status(f"  Weighted signif: {weighted_sigma:.2f} sigma", "TEST")

        # --- TEP model comparison ---
        # Under TEP, Delta_mu = kappa * sigma_v.  The TEP model has 1 free
        # parameter (kappa), so the chi-squared under TEP would be lower.
        # We report the void chi-squared and the improvement that any
        # non-zero model would provide.

        results = {
            "n_hosts": n,
            "void_null_hypothesis": {
                "prediction": "Delta_mu = 0 for all hosts",
                "description": "Kinematic void requires all indicators yield identical distances",
            },
            "chi_squared_test": {
                "chi2": chi2,
                "dof": dof,
                "chi2_reduced": chi2_reduced,
                "p_value": p_value_chi2,
                "significance_sigma": significance_sigma,
            },
            "t_test": {
                "t_statistic": float(t_stat),
                "p_value": float(t_p),
            },
            "wilcoxon_test": {
                "statistic": float(wilcoxon_stat),
                "p_value": float(wilcoxon_p),
            },
            "weighted_mean": {
                "value": weighted_mean,
                "error": weighted_err,
                "significance_sigma": weighted_sigma,
            },
            "tep_prediction": {
                "prediction": "Delta_mu = kappa * sigma_v (non-zero, potential-dependent)",
                "description": "TEP predicts a non-zero, potential-dependent divergence",
                "free_parameters": 1,
            },
            "falsification_verdict": {
                "void_falsified": significance_sigma > 2.0,
                "threshold_sigma": 2.0,
                "verdict": (
                    f"Void null hypothesis (Delta_mu = 0) is "
                    f"{'FALSIFIED' if significance_sigma > 2.0 else 'not definitively falsified'} "
                    f"at {significance_sigma:.2f} sigma"
                ),
            },
        }
        return results

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(self):
        """Execute the full step."""
        print_status("Step 12: Void Prediction Uniformity Test", "TITLE")

        print_status(
            "Scientific question: can the kinematic void model's prediction of "
            "indicator-uniform distances (Delta_mu = 0 for all hosts) survive a "
            "rigorous statistical test against the observed Cepheid-TRGB divergence? "
            "The void model requires all distance indicators to yield identical moduli, "
            "so any significant departure from Delta_mu = 0 falsifies the void null "
            "hypothesis. The TEP framework predicts a non-zero, potential-dependent "
            "divergence and is therefore not constrained by this null. This step "
            "formally tests the void null hypothesis using multiple independent "
            "statistics and renders a falsification verdict, closing Block Ia.",
            "PROCESS",
        )

        # Load data
        df = self.load_data()

        if len(df) == 0:
            print_status("No data available for void uniformity test.", "ERROR")
            return

        print_status(
            "Methodology: the observed per-galaxy Delta_mu distribution (from step_10) "
            "is tested against the void null hypothesis Delta_mu = 0 for all hosts. "
            "Four independent statistics are computed: a chi-squared statistic "
            "(sum of Delta_mu_i^2 / sigma_i^2 with N degrees of freedom), a one-sample "
            "t-test against zero, a Wilcoxon signed-rank test, and the inverse-variance "
            "weighted mean with its significance. The chi-squared significance in sigma "
            "equivalents is the primary falsification criterion, using a 2-sigma "
            "threshold.",
            "PROCESS",
        )

        # Test void hypothesis
        results = self.test_void_hypothesis(df)

        sig = results["chi_squared_test"]["significance_sigma"]
        verdict = results["falsification_verdict"]["verdict"]
        print_status(
            f"Interpretation: the void null hypothesis (Delta_mu = 0) yields a "
            f"chi-squared of {results['chi_squared_test']['chi2']:.2f} for "
            f"{results['chi_squared_test']['dof']} degrees of freedom "
            f"(reduced chi-squared = {results['chi_squared_test']['chi2_reduced']:.3f}), "
            f"corresponding to {sig:.2f} sigma. {verdict}. The weighted mean divergence "
            f"is {results['weighted_mean']['value']:+.4f} +/- "
            f"{results['weighted_mean']['error']:.4f} mag "
            f"({results['weighted_mean']['significance_sigma']:.2f} sigma). A "
            f"significant rejection of the void null is consistent with the TEP "
            f"prediction of non-zero, potential-dependent divergence.",
            "TEST",
        )

        # Summary JSON
        summary = {
            "step": "12_void_prediction_uniformity",
            "description": "Quantify and test the void model prediction of indicator uniformity",
            "delta_mu_definition": "Delta_mu = mu_Cepheid - mu_TRGB",
            "methodology": (
                "Formal test of the void null hypothesis Delta_mu = 0 for all hosts "
                "using four independent statistics: chi-squared (N dof), one-sample "
                "t-test, Wilcoxon signed-rank test, and inverse-variance weighted "
                "mean significance. The chi-squared significance in sigma equivalents "
                "is the primary falsification criterion at a 2-sigma threshold."
            ),
            "provenance": {
                "data_sources": [
                    "step_10 per-galaxy Delta_mu results",
                    "step_10 matched_host_comparison.json summary",
                ],
                "pipeline_block": "Ia — Indicator Divergence",
            },
            "scientific_context": (
                "Can the void model's prediction of indicator-uniform distances "
                "(Delta_mu = 0 for all hosts) survive formal statistical testing "
                "against the observed Cepheid-TRGB divergence? This step renders the "
                "falsification verdict for the void null hypothesis."
            ),
            "tep_prediction": (
                "Non-zero, potential-dependent divergence (Delta_mu = kappa * sigma_v) "
                "is expected; the void null hypothesis is not a prediction of TEP and "
                "its rejection is consistent with the TEP framework."
            ),
            "void_prediction": (
                "Delta_mu = 0 for all hosts: all distance indicators yield identical "
                "moduli. Any significant departure falsifies the void null hypothesis."
            ),
            "downstream_consumers": [],
            "results": results,
            "per_galaxy": df[["galaxy", "delta_mu", "delta_mu_err"]].to_dict(orient="records"),
            "output_files": [
                str(self.results / "step_12_void_prediction_uniformity.json"),
            ],
        }

        summary_path = self.results / "step_12_void_prediction_uniformity.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print_status(f"Summary saved to {summary_path}", "SUCCESS")

        print_status("Step 12 complete", "SUCCESS")


if __name__ == "__main__":
    step = Step12VoidUniformity()
    step.run()
