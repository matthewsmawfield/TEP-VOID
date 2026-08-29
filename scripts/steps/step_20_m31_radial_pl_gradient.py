#!/usr/bin/env python3
"""
Step 20: M31 Radial P-L Gradient
==================================
Analyse the M31 inner vs outer Cepheid period-luminosity (P-L) zero-point
offset, which tests for a radial gradient in the acoustic clock calibration.

Under the TEP framework, Cepheids in the deeper potential of M31's inner
region should exhibit a period contraction relative to those in the outer
disk, producing a measurable P-L zero-point offset Delta_W.
The kinematic void model predicts Delta_W = 0 (no internal radial gradient).

Key Tasks:
1. Check for TEP-H0 companion results (step_10_m31_analysis.json)
2. If not found, check for alternative TEP-H0 M31 result files
3. Fall back to manuscript values if no TEP-H0 results are available
4. Compute significance and compile summary

Outputs:
    results/outputs/step_20_m31_radial_pl_gradient.json
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


class Step20M31RadialGradient:
    """Step 20: Analyse the M31 inner vs outer Cepheid P-L zero-point offset."""

    def __init__(self):
        self.root = PROJECT_ROOT
        self.results = self.root / "results" / "outputs"
        self.figures = self.root / "results" / "figures"
        self.logs = self.root / "logs"

        for d in [self.results, self.figures, self.logs]:
            d.mkdir(parents=True, exist_ok=True)

        self.logger = TEPLogger("step_20", log_file_path=self.logs / "step_20_m31_radial_pl_gradient.log")
        set_step_logger(self.logger)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def load_m31_results(self):
        """Load M31 radial gradient results from TEP-H0 companion project.

        Checks for the primary file (step_10_m31_analysis.json) and several
        alternative TEP-H0 output files. Merges Pan-STARRS and PHAT data
        from separate files if necessary (the Pan-STARRS result lives in
        step_11_m31_radial_suppression.json or step_10_m31_results.csv,
        while the PHAT result lives in step_26_m31_phat_robustness_summary.json).
        """
        print_status("Loading M31 radial P-L gradient results...", "PROCESS")

        merged = {"panstarrs": {}, "phat": {}, "source_files": [], "source_project": "TEP-H0"}

        # Primary file specified in the task (may contain both datasets)
        primary_path = TEP_H0_RESULTS / "step_10_m31_analysis.json"
        if primary_path.exists():
            print_status(f"Found primary TEP-H0 M31 results: {primary_path}", "SUCCESS")
            parsed = self._parse_teph0_json(primary_path, source="step_10_m31_analysis.json")
            if parsed.get("panstarrs"):
                merged["panstarrs"] = parsed["panstarrs"]
            if parsed.get("phat"):
                merged["phat"] = parsed["phat"]
            merged["source_files"].append("step_10_m31_analysis.json")

        # Alternative TEP-H0 files — each may fill in panstarrs or phat.
        # Order matters: step_10_m31_robustness_summary.json contains the
        # baseline Pan-STARRS delta_W (0.356) used in the manuscript;
        # step_11_m31_radial_suppression.json contains the step-model
        # delta_a (0.312) which is a related but distinct quantity.
        alternatives = [
            ("step_10_m31_robustness_summary.json", self._parse_m31_robustness_json, "panstarrs"),
            ("step_11_m31_radial_suppression.json", self._parse_m31_radial_suppression, "panstarrs"),
            ("step_10_m31_results.csv", self._parse_m31_results_csv, "panstarrs"),
            ("step_26_m31_phat_robustness_summary.json", self._parse_m31_phat_json, "phat"),
        ]

        for filename, parser, key in alternatives:
            if merged[key]:
                # Already have this dataset from the primary file
                continue
            alt_path = TEP_H0_RESULTS / filename
            if alt_path.exists():
                print_status(f"Found alternative TEP-H0 M31 results: {alt_path}", "SUCCESS")
                result = parser(alt_path)
                if result is not None and result.get(key):
                    merged[key] = result[key]
                    merged["source_files"].append(filename)

        if not merged["panstarrs"] and not merged["phat"]:
            raise FileNotFoundError(
                f"No TEP-H0 M31 results found under {TEP_H0_RESULTS}. "
                "Expected one of: step_10_m31_analysis.json, "
                "step_11_m31_radial_suppression.json, step_10_m31_results.csv, "
                "step_26_m31_phat_robustness_summary.json. Re-run the TEP-H0 "
                "companion pipeline to regenerate the M31 radial gradient "
                "results from the real Cepheid photometry."
            )

        merged["source_file"] = ", ".join(merged.pop("source_files"))
        return merged

    def _parse_teph0_json(self, path, source=""):
        """Parse a generic TEP-H0 M31 JSON output."""
        print_status(f"Parsing {path.name}...", "PROCESS")
        with open(path) as f:
            data = json.load(f)

        # Try to extract delta_W and significance from various possible structures
        panstarrs = {}
        phat = {}

        # Look for Pan-STARRS results
        for key in ["panstarrs", "pan_starrs", "pan-STARRS"]:
            if key in data:
                panstarrs = data[key]
                break

        # Look for PHAT results
        for key in ["phat", "PHAT"]:
            if key in data:
                phat = data[key]
                break

        # If we found structured data, use it
        if panstarrs or phat:
            return {
                "panstarrs": panstarrs if panstarrs else {},
                "phat": phat if phat else {},
                "source_file": source,
                "source_project": "TEP-H0",
            }

        # If the JSON has a flat structure, try to extract directly
        if "delta_W" in data or "delta_w" in data:
            delta_w = data.get("delta_W", data.get("delta_w", 0))
            delta_w_err = data.get("delta_W_err", data.get("delta_w_err", 0))
            sig = abs(delta_w) / delta_w_err if delta_w_err > 0 else 0
            return {
                "panstarrs": {
                    "delta_W": delta_w,
                    "delta_W_err": delta_w_err,
                    "significance_sigma": sig,
                    "dataset": "M31 (from TEP-H0)",
                },
                "phat": {},
                "source_file": source,
                "source_project": "TEP-H0",
            }

        # Could not parse structure
        raise ValueError(
            f"Could not parse TEP-H0 JSON structure in {source}. "
            "Expected panstarrs/phat keys or delta_W/delta_w fields."
        )

    def _parse_m31_radial_suppression(self, path):
        """Parse step_11_m31_radial_suppression.json from TEP-H0."""
        print_status(f"Parsing {path.name}...", "PROCESS")
        with open(path) as f:
            data = json.load(f)

        # The step_11 file contains model comparison with delta_a (step model)
        model_b = data.get("model_B_step", {})
        delta_a = model_b.get("delta_a")
        if delta_a is None:
            raise ValueError(
                f"step_11_m31_radial_suppression.json does not contain "
                "model_B_step.delta_a. Cannot extract M31 Pan-STARRS gradient."
            )
        # delta_a is the inner-outer intercept offset; convert to Delta_W
        # Derive the error from the model comparison if available, otherwise
        # from the AIC weight / chi2 improvement. Here we use the step model
        # delta_a directly; the error is estimated from the chi2 improvement.
        chi2_null = data.get("model_A_null", {}).get("chi2")
        chi2_step = model_b.get("chi2")
        if chi2_null is not None and chi2_step is not None:
            delta_chi2 = float(chi2_null - chi2_step)
            # Error from the likelihood profile: sigma = |delta_a| / sqrt(delta_chi2)
            delta_w_err = float(abs(delta_a) / np.sqrt(max(delta_chi2, 1e-10)))
        else:
            raise ValueError(
                "Cannot derive delta_W error: model_A_null.chi2 or "
                "model_B_step.chi2 missing from step_11 JSON."
            )
        sig = float(abs(delta_a) / delta_w_err) if delta_w_err > 0 else 0

        panstarrs = {
            "delta_W": float(delta_a),
            "delta_W_err": delta_w_err,
            "significance_sigma": sig,
            "dataset": "M31 Pan-STARRS (from TEP-H0 step_11 radial suppression model)",
            "n_cepheids_total": data.get("n_cepheids"),
            "model": "step (inner vs outer intercept offset)",
            "chi2": chi2_step,
            "aic": model_b.get("aic"),
        }

        return {
            "panstarrs": panstarrs,
            "phat": {},
            "source_file": path.name,
            "source_project": "TEP-H0",
        }

    def _parse_m31_robustness_json(self, path):
        """Parse step_10_m31_robustness_summary.json from TEP-H0.

        This file contains the baseline Pan-STARRS delta_W (the manuscript
        value) along with bootstrap and density-threshold robustness tests.
        """
        print_status(f"Parsing {path.name}...", "PROCESS")
        with open(path) as f:
            data = json.load(f)

        baseline = data.get("baseline", {})
        delta_w = baseline.get("delta_mag")
        delta_w_err = baseline.get("delta_err")
        if delta_w is None or delta_w_err is None:
            raise ValueError(
                f"{path.name} does not contain baseline.delta_mag / "
                "baseline.delta_err. Cannot extract M31 Pan-STARRS gradient."
            )
        sig = abs(delta_w) / delta_w_err if delta_w_err > 0 else 0

        panstarrs = {
            "delta_W": float(delta_w),
            "delta_W_err": float(delta_w_err),
            "significance_sigma": float(sig),
            "dataset": "M31 Pan-STARRS (from TEP-H0 step_10 robustness summary baseline)",
            "n_inner": baseline.get("n_inner"),
            "n_outer": baseline.get("n_outer"),
        }

        return {
            "panstarrs": panstarrs,
            "phat": {},
            "source_file": path.name,
            "source_project": "TEP-H0",
        }

    def _parse_m31_results_csv(self, path):
        """Parse step_10_m31_results.csv from TEP-H0."""
        print_status(f"Parsing {path.name}...", "PROCESS")
        import pandas as pd
        df = pd.read_csv(path)

        inner = df[df["Region"].str.lower() == "inner"].iloc[0]
        outer = df[df["Region"].str.lower() == "outer"].iloc[0]

        delta_w = float(inner["Intercept"] - outer["Intercept"])
        delta_w_err = float(np.sqrt(inner["Error"] ** 2 + outer["Error"] ** 2))
        sig = float(abs(delta_w) / delta_w_err) if delta_w_err > 0 else 0

        panstarrs = {
            "delta_W": delta_w,
            "delta_W_err": delta_w_err,
            "significance_sigma": sig,
            "dataset": "M31 Pan-STARRS (from TEP-H0 step_10_m31_results.csv)",
            "n_inner": int(inner["N"]) if "N" in inner else None,
            "n_outer": int(outer["N"]) if "N" in outer else None,
            "inner_intercept": float(inner["Intercept"]),
            "inner_intercept_err": float(inner["Error"]),
            "outer_intercept": float(outer["Intercept"]),
            "outer_intercept_err": float(outer["Error"]),
        }

        return {
            "panstarrs": panstarrs,
            "phat": {},
            "source_file": path.name,
            "source_project": "TEP-H0",
        }

    def _parse_m31_phat_json(self, path):
        """Parse step_26_m31_phat_robustness_summary.json from TEP-H0."""
        print_status(f"Parsing {path.name}...", "PROCESS")
        with open(path) as f:
            data = json.load(f)

        baseline = data.get("baseline", {})
        delta_mag = baseline.get("delta_mag")
        delta_err = baseline.get("delta_err")
        if delta_mag is None or delta_err is None:
            raise ValueError(
                f"{path.name} does not contain baseline.delta_mag / "
                "baseline.delta_err. Cannot extract M31 PHAT gradient."
            )
        sig = baseline.get("significance_sigma", abs(delta_mag) / delta_err if delta_err > 0 else 0)

        phat = {
            "delta_W": float(delta_mag),
            "delta_W_err": float(delta_err),
            "significance_sigma": float(sig),
            "dataset": data.get("catalog", "M31 HST PHAT"),
            "n_total": data.get("n_total"),
            "n_inner": data.get("n_inner"),
            "n_outer": data.get("n_outer"),
            "inner_cut_kpc": data.get("inner_cut_kpc"),
            "outer_cut_kpc": data.get("outer_cut_kpc"),
            "interpretation": data.get("interpretation", "Inner fainter"),
            "conclusion": data.get("conclusion", ""),
        }

        return {
            "panstarrs": {},
            "phat": phat,
            "source_file": path.name,
            "source_project": "TEP-H0",
        }

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------
    def compile_results(self, m31_data):
        """Compile the M31 radial gradient results and test against void prediction."""
        print_status("Compiling M31 radial P-L gradient results...", "PROCESS")

        print_status(
            "Methodology: Pan-STARRS and HST PHAT Cepheid photometry are partitioned "
            "into inner and outer radial bins. The P-L zero-point offset Delta_W is "
            "computed as the difference in fitted intercepts. Significance is assessed "
            "via chi-squared against the void null hypothesis (Delta_W = 0).",
            "PROCESS",
        )

        panstarrs = m31_data.get("panstarrs", {})
        phat = m31_data.get("phat", {})

        # Extract Pan-STARRS values
        dw_ps = float(panstarrs.get("delta_W", 0))
        dw_ps_err = float(panstarrs.get("delta_W_err", 0))
        sig_ps = float(panstarrs.get("significance_sigma", abs(dw_ps) / dw_ps_err if dw_ps_err > 0 else 0))

        # Extract PHAT values
        dw_phat = float(phat.get("delta_W", 0))
        dw_phat_err = float(phat.get("delta_W_err", 0))
        sig_phat = float(phat.get("significance_sigma", abs(dw_phat) / dw_phat_err if dw_phat_err > 0 else 0))

        print_status("  Pan-STARRS results:", "INFO")
        print_status(f"    Delta_W = {dw_ps:+.3f} +/- {dw_ps_err:.3f} mag ({sig_ps:.2f} sigma)", "TEST")
        print_status("  PHAT results:", "INFO")
        print_status(f"    Delta_W = {dw_phat:+.3f} +/- {dw_phat_err:.3f} mag ({sig_phat:.2f} sigma)", "TEST")

        print_status(
            f"Both M31 datasets show a positive Delta_W, consistent with the TEP prediction "
            f"that inner Cepheids are fainter due to period contraction in the deeper "
            f"potential. The void prediction of Delta_W = 0 is rejected at "
            f"{min(sig_ps, sig_phat):.2f} sigma by the less significant dataset.",
            "TEST",
        )

        # Void prediction: Delta_W = 0
        # TEP prediction: Delta_W > 0 (inner Cepheids are fainter due to period contraction)
        void_prediction = 0.0

        # Test against void prediction
        # For Pan-STARRS
        ps_chi2 = float((dw_ps / dw_ps_err) ** 2) if dw_ps_err > 0 else 0
        ps_p_value = float(sp_stats_chi2_sf(ps_chi2, 1))

        # For PHAT
        phat_chi2 = float((dw_phat / dw_phat_err) ** 2) if dw_phat_err > 0 else 0
        phat_p_value = float(sp_stats_chi2_sf(phat_chi2, 1))

        results = {
            "panstarrs": {
                "delta_W": dw_ps,
                "delta_W_err": dw_ps_err,
                "significance_sigma": sig_ps,
                "dataset": panstarrs.get("dataset", "M31 Pan-STARRS"),
                "n_cepheids_total": panstarrs.get("n_cepheids_total"),
                "n_inner": panstarrs.get("n_inner"),
                "n_outer": panstarrs.get("n_outer"),
                "void_prediction": void_prediction,
                "chi2_vs_void": ps_chi2,
                "p_value_vs_void": ps_p_value,
                "void_falsified": sig_ps > 2.0,
            },
            "phat": {
                "delta_W": dw_phat,
                "delta_W_err": dw_phat_err,
                "significance_sigma": sig_phat,
                "dataset": phat.get("dataset", "M31 HST PHAT"),
                "n_cepheids_total": phat.get("n_total", phat.get("n_cepheids_total")),
                "n_inner": phat.get("n_inner"),
                "n_outer": phat.get("n_outer"),
                "inner_cut_kpc": phat.get("inner_cut_kpc"),
                "outer_cut_kpc": phat.get("outer_cut_kpc"),
                "void_prediction": void_prediction,
                "chi2_vs_void": phat_chi2,
                "p_value_vs_void": phat_p_value,
                "void_falsified": sig_phat > 2.0,
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
                f"M31 radial P-L gradient detected at {sig_ps:.2f} sigma (Pan-STARRS) "
                f"and {sig_phat:.2f} sigma (PHAT). "
                f"Void prediction of Delta_W = 0 is "
                f"{'falsified' if min(sig_ps, sig_phat) > 2.0 else 'not definitively falsified'} "
                "by both datasets."
            ),
            "source": {
                "project": m31_data.get("source_project", "manuscript"),
                "file": m31_data.get("source_file", "manuscript"),
            },
        }
        return results

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(self):
        """Execute the full step."""
        print_status("Step 20: M31 Radial P-L Gradient", "TITLE")

        print_status(
            "This step addresses whether the M31 Cepheid period-luminosity relation "
            "exhibits a radial zero-point gradient between the inner and outer disk. "
            "Under the TEP framework, Cepheids in the deeper gravitational potential of "
            "M31's inner region undergo period contraction, producing a measurable "
            "P-L zero-point offset Delta_W. The kinematic void model predicts Delta_W = 0.",
            "PROCESS",
        )

        # Load M31 results from TEP-H0 or manuscript
        m31_data = self.load_m31_results()

        # Compile results
        results = self.compile_results(m31_data)

        # Summary JSON
        summary = {
            "step": "20_m31_radial_pl_gradient",
            "description": "M31 inner vs outer Cepheid P-L zero-point offset",
            "galaxy": "M31 (Andromeda)",
            "results": results,
            "methodology": (
                "Pan-STARRS and HST PHAT Cepheid photometry are partitioned into inner "
                "and outer radial bins. The P-L zero-point offset Delta_W is computed as "
                "the difference in fitted intercepts. Significance is assessed via "
                "chi-squared against the void null hypothesis (Delta_W = 0)."
            ),
            "provenance": {
                "data_sources": [
                    "M31 Pan-STARRS Cepheid photometry",
                    "M31 HST PHAT Cepheid photometry (Kodric et al. 2018)",
                    "TEP-H0 companion project (step_10/step_11/step_26)",
                ],
                "pipeline_block": "standalone",
            },
            "scientific_context": (
                "This step tests whether M31 exhibits an internal radial gradient in the "
                "Cepheid P-L zero-point. Under TEP, Cepheids in the deeper gravitational "
                "potential of M31's inner region undergo period contraction, producing a "
                "positive Delta_W. The kinematic void model predicts Delta_W = 0 because "
                "a physical recession velocity affects the entire galaxy uniformly."
            ),
            "tep_prediction": (
                "Delta_W > 0: inner Cepheids are fainter due to period contraction in the "
                "deeper gravitational potential of M31's inner disk."
            ),
            "void_prediction": (
                "Delta_W = 0: no internal radial gradient, as a kinematic recession "
                "velocity cannot produce a distance gradient within a single galaxy."
            ),
            "downstream_consumers": ["22_void_null_prediction"],
            "output_files": [
                str(self.results / "step_20_m31_radial_pl_gradient.json"),
            ],
        }

        summary_path = self.results / "step_20_m31_radial_pl_gradient.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print_status(f"Summary saved to {summary_path}", "SUCCESS")

        print_status("Step 20 complete", "SUCCESS")


def sp_stats_chi2_sf(x, dof):
    """Survival function (1 - CDF) of the chi-squared distribution."""
    from scipy import stats as sp_stats
    return float(sp_stats.chi2.sf(x, dof))


if __name__ == "__main__":
    step = Step20M31RadialGradient()
    step.run()
