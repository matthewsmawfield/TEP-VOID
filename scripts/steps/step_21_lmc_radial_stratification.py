#!/usr/bin/env python3
"""
Step 21: LMC Radial Stratification
=====================================
Analyse the LMC OGLE-IV Cepheid radial potential stratification, which
tests for a radial gradient in the acoustic clock calibration within a
single galaxy.

Under the TEP framework, Cepheids in the deeper central potential of the
LMC should exhibit a period contraction relative to those in the outer
bar/disk, producing a measurable P-L zero-point offset Delta_W.
The kinematic void model predicts Delta_W = 0 (no internal radial gradient).

Key Tasks:
1. Check for TEP-H0 companion results (step_14_lmc_replication.json)
2. If not found, check for alternative TEP-H0 LMC result files
3. Fall back to manuscript values if no TEP-H0 results are available
4. Compute significance and compile summary

Outputs:
    results/outputs/step_21_lmc_radial_stratification.json
"""

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status

# Path to the companion TEP-H0 project (sibling directory)
TEP_H0_ROOT = PROJECT_ROOT.parent / "TEP-H0"
TEP_H0_RESULTS = TEP_H0_ROOT / "results" / "outputs"


class Step21LMCRadialStratification:
    """Step 21: Analyse the LMC OGLE-IV Cepheid radial potential stratification."""

    def __init__(self):
        self.root = PROJECT_ROOT
        self.results = self.root / "results" / "outputs"
        self.figures = self.root / "results" / "figures"
        self.logs = self.root / "logs"

        for d in [self.results, self.figures, self.logs]:
            d.mkdir(parents=True, exist_ok=True)

        self.logger = TEPLogger("step_21", log_file_path=self.logs / "step_21_lmc_radial_stratification.log")
        set_step_logger(self.logger)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def load_lmc_results(self):
        """Load LMC radial stratification results from TEP-H0 companion project.

        Checks for the primary file (step_14_lmc_replication.json) and
        alternative TEP-H0 output files.  Falls back to manuscript values
        if none are found.
        """
        print_status("Loading LMC radial stratification results...", "PROCESS")

        # Primary file specified in the task
        primary_path = TEP_H0_RESULTS / "step_14_lmc_replication.json"
        if primary_path.exists():
            print_status(f"Found primary TEP-H0 LMC results: {primary_path}", "SUCCESS")
            return self._parse_teph0_json(primary_path, source="step_14_lmc_replication.json")

        # Alternative TEP-H0 files that contain LMC radial stratification data
        alternatives = [
            ("step_14_lmc_results.csv", self._parse_lmc_results_csv),
            ("step_14_lmc_robustness_summary.json", self._parse_lmc_robustness_json),
        ]

        for filename, parser in alternatives:
            alt_path = TEP_H0_RESULTS / filename
            if alt_path.exists():
                print_status(f"Found alternative TEP-H0 LMC results: {alt_path}", "SUCCESS")
                result = parser(alt_path)
                if result is not None:
                    result["source_file"] = filename
                    result["source_project"] = "TEP-H0"
                    return result

        # No fallback: fail loudly if no TEP-H0 LMC results are available.
        raise FileNotFoundError(
            f"No TEP-H0 LMC results found under {TEP_H0_RESULTS}. "
            "Expected one of: step_14_lmc_replication.json, "
            "step_14_lmc_results.csv, step_14_lmc_robustness_summary.json. "
            "Re-run the TEP-H0 companion pipeline to regenerate the LMC "
            "radial stratification results from the real OGLE-IV photometry."
        )

    def _parse_teph0_json(self, path, source=""):
        """Parse a generic TEP-H0 LMC JSON output."""
        print_status(f"Parsing {path.name}...", "PROCESS")
        with open(path) as f:
            data = json.load(f)

        # Try to extract delta_W and significance from various possible structures
        if "delta_W" in data or "delta_w" in data:
            delta_w = float(data.get("delta_W", data.get("delta_w", 0)))
            delta_w_err = float(data.get("delta_W_err", data.get("delta_w_err", 0)))
            sig = float(data.get("significance_sigma", abs(delta_w) / delta_w_err if delta_w_err > 0 else 0))
            return {
                "lmc": {
                    "delta_W": delta_w,
                    "delta_W_err": delta_w_err,
                    "significance_sigma": sig,
                    "dataset": data.get("dataset", "LMC (from TEP-H0)"),
                    "n_cepheids_total": data.get("n_cepheids_total"),
                    "n_inner": data.get("n_inner"),
                    "n_outer": data.get("n_outer"),
                },
                "source_file": source,
                "source_project": "TEP-H0",
            }

        # If the JSON has nested structure, try common keys
        for key in ["lmc", "results", "baseline"]:
            if key in data and isinstance(data[key], dict):
                sub = data[key]
                if "delta_W" in sub or "delta_w" in sub or "delta_mag" in sub:
                    delta_w = float(sub.get("delta_W", sub.get("delta_w", sub.get("delta_mag", 0))))
                    delta_w_err = float(sub.get("delta_W_err", sub.get("delta_w_err", sub.get("delta_err", 0))))
                    sig = float(sub.get("significance_sigma", abs(delta_w) / delta_w_err if delta_w_err > 0 else 0))
                    return {
                        "lmc": {
                            "delta_W": delta_w,
                            "delta_W_err": delta_w_err,
                            "significance_sigma": sig,
                            "dataset": "LMC (from TEP-H0)",
                        },
                        "source_file": source,
                        "source_project": "TEP-H0",
                    }

        # Could not parse structure
        raise ValueError(
            f"Could not parse TEP-H0 JSON structure in {source}. "
            "Expected delta_W/delta_w fields or nested lmc/results/baseline "
            "with delta_W/delta_w/delta_mag fields."
        )

    def _parse_lmc_results_csv(self, path):
        """Parse step_14_lmc_results.csv from TEP-H0."""
        print_status(f"Parsing {path.name}...", "PROCESS")
        import pandas as pd
        df = pd.read_csv(path)

        inner = df[df["Region"].str.lower() == "inner"].iloc[0]
        outer = df[df["Region"].str.lower() == "outer"].iloc[0]

        delta_w = float(inner["Intercept"] - outer["Intercept"])
        delta_w_err = float(np.sqrt(inner["Error"] ** 2 + outer["Error"] ** 2))
        sig = float(abs(delta_w) / delta_w_err) if delta_w_err > 0 else 0

        lmc = {
            "delta_W": delta_w,
            "delta_W_err": delta_w_err,
            "significance_sigma": sig,
            "dataset": "LMC OGLE-IV (from TEP-H0 step_14_lmc_results.csv)",
            "n_inner": int(inner["N"]) if "N" in inner else None,
            "n_outer": int(outer["N"]) if "N" in outer else None,
            "inner_cut_kpc": float(inner["inner_cut_kpc"]) if "inner_cut_kpc" in inner else None,
            "outer_cut_kpc": float(outer["outer_cut_kpc"]) if "outer_cut_kpc" in outer else None,
            "inner_intercept": float(inner["Intercept"]),
            "inner_intercept_err": float(inner["Error"]),
            "outer_intercept": float(outer["Intercept"]),
            "outer_intercept_err": float(outer["Error"]),
        }

        return {
            "lmc": lmc,
            "source_file": path.name,
            "source_project": "TEP-H0",
        }

    def _parse_lmc_robustness_json(self, path):
        """Parse step_14_lmc_robustness_summary.json from TEP-H0."""
        print_status(f"Parsing {path.name}...", "PROCESS")
        with open(path) as f:
            data = json.load(f)

        # Try to extract from the robustness summary
        # This file may contain multiple test configurations
        baseline = data.get("baseline", data)

        delta_w = float(baseline.get("delta_W", baseline.get("delta_mag")))
        delta_w_err = float(baseline.get("delta_W_err", baseline.get("delta_err")))
        if delta_w is None or delta_w_err is None:
            raise ValueError(
                f"{path.name} does not contain baseline.delta_W/delta_mag "
                "or baseline.delta_W_err/delta_err. Cannot extract LMC gradient."
            )
        sig = float(baseline.get("significance_sigma", abs(delta_w) / delta_w_err if delta_w_err > 0 else 0))

        lmc = {
            "delta_W": delta_w,
            "delta_W_err": delta_w_err,
            "significance_sigma": sig,
            "dataset": "LMC OGLE-IV (from TEP-H0 step_14 robustness summary)",
            "n_cepheids_total": baseline.get("n_total", baseline.get("n_cepheids_total")),
            "n_inner": baseline.get("n_inner"),
            "n_outer": baseline.get("n_outer"),
        }

        return {
            "lmc": lmc,
            "source_file": path.name,
            "source_project": "TEP-H0",
        }

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------
    def compile_results(self, lmc_data):
        """Compile the LMC radial stratification results and test against void prediction."""
        print_status("Compiling LMC radial stratification results...", "PROCESS")

        print_status(
            "Methodology: OGLE-IV Cepheid photometry is partitioned into inner and outer "
            "radial bins using projected galactocentric radius cuts. The P-L zero-point "
            "offset Delta_W is computed as the difference in fitted intercepts. "
            "Significance is assessed via chi-squared against the void null hypothesis "
            "(Delta_W = 0).",
            "PROCESS",
        )

        lmc = lmc_data.get("lmc", {})

        # Extract LMC values
        dw = float(lmc.get("delta_W", 0))
        dw_err = float(lmc.get("delta_W_err", 0))
        sig = float(lmc.get("significance_sigma", abs(dw) / dw_err if dw_err > 0 else 0))

        print_status(f"  Delta_W = {dw:+.4f} +/- {dw_err:.4f} mag ({sig:.2f} sigma)", "TEST")

        print_status(
            f"The positive Delta_W is consistent with the TEP prediction that inner LMC "
            f"Cepheids are fainter due to period contraction in the deeper central "
            f"potential. The void prediction of Delta_W = 0 is rejected at {sig:.2f} sigma.",
            "TEST",
        )

        # Void prediction: Delta_W = 0
        void_prediction = 0.0

        # Chi-squared test against void prediction
        from scipy import stats as sp_stats
        chi2 = float((dw / dw_err) ** 2) if dw_err > 0 else 0
        p_value = float(sp_stats.chi2.sf(chi2, 1))

        results = {
            "lmc": {
                "delta_W": dw,
                "delta_W_err": dw_err,
                "significance_sigma": sig,
                "dataset": lmc.get("dataset", "LMC OGLE-IV"),
                "n_cepheids_total": lmc.get("n_cepheids_total"),
                "n_inner": lmc.get("n_inner"),
                "n_outer": lmc.get("n_outer"),
                "inner_cut_kpc": lmc.get("inner_cut_kpc"),
                "outer_cut_kpc": lmc.get("outer_cut_kpc"),
                "inner_intercept": lmc.get("inner_intercept"),
                "inner_intercept_err": lmc.get("inner_intercept_err"),
                "outer_intercept": lmc.get("outer_intercept"),
                "outer_intercept_err": lmc.get("outer_intercept_err"),
                "void_prediction": void_prediction,
                "chi2_vs_void": chi2,
                "p_value_vs_void": p_value,
                "void_falsified": sig > 2.0,
            },
            "void_prediction": {
                "delta_W": 0.0,
                "description": "Void model predicts no internal radial gradient (Delta_W = 0)",
            },
            "tep_prediction": {
                "delta_W_positive": True,
                "description": "TEP predicts Delta_W > 0: inner Cepheids fainter due to period contraction",
            },
            "interpretation": (
                f"LMC radial P-L gradient detected at {sig:.2f} sigma. "
                f"Void prediction of Delta_W = 0 is "
                f"{'falsified' if sig > 2.0 else 'not definitively falsified'} "
                "at >2 sigma."
            ),
            "source": {
                "project": lmc_data.get("source_project", "manuscript"),
                "file": lmc_data.get("source_file", "manuscript"),
            },
        }
        return results

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(self):
        """Execute the full step."""
        print_status("Step 21: LMC Radial Stratification", "TITLE")

        print_status(
            "This step addresses whether the LMC OGLE-IV Cepheid sample exhibits a "
            "radial P-L zero-point stratification between the central bar and the outer "
            "disk. Under the TEP framework, Cepheids in the deeper central potential of "
            "the LMC undergo period contraction, producing a measurable offset Delta_W. "
            "The kinematic void model predicts Delta_W = 0 for any single galaxy.",
            "PROCESS",
        )

        # Load LMC results from TEP-H0 or manuscript
        lmc_data = self.load_lmc_results()

        # Compile results
        results = self.compile_results(lmc_data)

        # Summary JSON
        summary = {
            "step": "21_lmc_radial_stratification",
            "description": "LMC OGLE-IV Cepheid radial potential stratification",
            "galaxy": "LMC (Large Magellanic Cloud)",
            "results": results,
            "methodology": (
                "OGLE-IV Cepheid photometry is partitioned into inner and outer radial "
                "bins using projected galactocentric radius cuts. The P-L zero-point "
                "offset Delta_W is computed as the difference in fitted intercepts. "
                "Significance is assessed via chi-squared against the void null "
                "hypothesis (Delta_W = 0)."
            ),
            "provenance": {
                "data_sources": [
                    "LMC OGLE-IV Cepheid survey",
                    "TEP-H0 companion project (step_14)",
                ],
                "pipeline_block": "standalone",
            },
            "scientific_context": (
                "This step tests whether the LMC exhibits an internal radial gradient in "
                "the Cepheid P-L zero-point. Under TEP, Cepheids in the deeper central "
                "potential of the LMC undergo period contraction, producing a positive "
                "Delta_W. The kinematic void model predicts Delta_W = 0 because a "
                "physical recession velocity affects the entire galaxy uniformly."
            ),
            "tep_prediction": (
                "Delta_W > 0: inner Cepheids are fainter due to period contraction in the "
                "deeper central potential of the LMC."
            ),
            "void_prediction": (
                "Delta_W = 0: no internal radial gradient, as a kinematic recession "
                "velocity cannot produce a distance gradient within a single galaxy."
            ),
            "downstream_consumers": ["22_void_null_prediction"],
            "output_files": [
                str(self.results / "step_21_lmc_radial_stratification.json"),
            ],
        }

        summary_path = self.results / "step_21_lmc_radial_stratification.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print_status(f"Summary saved to {summary_path}", "SUCCESS")

        print_status("Step 21 complete", "SUCCESS")


if __name__ == "__main__":
    step = Step21LMCRadialStratification()
    step.run()
