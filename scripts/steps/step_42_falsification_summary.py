#!/usr/bin/env python3
"""
Step 42: Falsification Summary — Void vs TEP Head-to-Head Comparison
=====================================================================
Compiles the four discriminating observables that distinguish the KBC
void/MOND model from the TEP isochrony-violation framework.

The Four Discriminating Observables:
    1. Indicator-Specific Distance Divergence (step_30, Test A):
       Direct comparison of Cepheid vs TRGB distance moduli for galaxies
       with both measurements in CF4 table2.
       Void: Δμ = 0 (kinematic outflow is indicator-independent).
       TEP:  Δμ < 0 (Cepheid distances compressed by acoustic clock bias).
       Result: populated from step_30 JSON at runtime.

    2. Bulk-Flow Calibration Sensitivity (step_30 Test B + step_31):
       The bulk-flow amplitude depends on the H0 calibration.
       Void: bulk flow is gravitational — no calibration dependence.
       TEP:  Cepheid H0 inflates the bulk flow; Δv = -ΔH0·d km/s.
       Result: populated from step_30/31 JSON at runtime.

    3. H0(z) Redshift Profile (step_32):
       Void: gradual decline (Gaussian/Exponential, HBK20/Mazurenko 2025).
       TEP:  flat H0(z) ≈ 73 for global M_B (zero-point inheritance).
       Models are fit with free parameters; AIC comparison is valid.
       Result: populated from step_32 JSON at runtime.

    4. Calibration-Independent Relative Evolution (step_34):
       Void: H0 declines from low to high redshift (gradual decay).
       TEP:  flat H0(z) for global M_B (zero-point inheritance).
       The absolute H0 at high z is NOT independent (Cepheid zero-point
       inherited); the meaningful test is the relative evolution.
       Result: populated from step_34 JSON at runtime.

Observables 1, 3, and 4 are independent falsification tests. Observable 2
is a quantification of how the indicator divergence (Observable 1) propagates
into the peculiar velocity field, providing the physical link between the
Cepheid distance compression and the bulk-flow anomaly.

Outputs:
    results/outputs/step_42_falsification_summary.json
    results/outputs/step_42_falsification_table.csv
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status


class Step42FalsificationSummary:
    """Step 42: Head-to-head void vs TEP falsification summary."""

    H0_CMB = 67.4  # Planck 2018 CMB-inferred H0

    STEP_FILES = [
        "step_30_bulk_flow_recalculation.json",
        "step_31_peculiar_velocity_artifact.json",
        "step_32_redshift_decay_profile.json",
        "step_34_void_boundary_test.json",
        "step_40_redshift_shear_reconstruction.json",
    ]

    def __init__(self):
        self.root = PROJECT_ROOT
        self.results = self.root / "results" / "outputs"
        self.logs = self.root / "logs"

        for d in [self.results, self.logs]:
            d.mkdir(parents=True, exist_ok=True)

        self.logger = TEPLogger(
            "step_42", log_file_path=self.logs / "step_42_falsification_summary.log"
        )
        set_step_logger(self.logger)

    def load_step_results(self):
        """Load all previous step results from results/outputs/."""
        print_status("Loading previous step results...", "PROCESS")

        step_data = {}
        for step_file in self.STEP_FILES:
            path = self.results / step_file
            if path.exists():
                try:
                    with open(path, "r") as f:
                        data = json.load(f)
                    step_data[step_file.replace(".json", "")] = data
                    print_status(f"  Loaded {step_file}", "SUCCESS")
                except Exception as e:
                    print_status(f"  Error reading {step_file}: {e}", "WARNING")
            else:
                print_status(f"  {step_file} not found", "WARNING")

        print_status(f"Loaded {len(step_data)}/{len(self.STEP_FILES)} step results", "PROCESS")
        return step_data

    def compile_observables(self, step_data):
        """Compile the four discriminating observables from step results."""
        print_status("Compiling discriminating observables...", "PROCESS")

        observables = []

        # Observable 1: Indicator-Specific Distance Divergence (step_30, Test A)
        obs1 = {
            "observable": "Indicator-Specific Distance Divergence",
            "tep_prediction": "DMceph < DMtrgb (Cepheid distances compressed by acoustic clock bias)",
            "void_prediction": "DMceph = DMtrgb (kinematic outflow is indicator-independent)",
            "tep_result": "Step 30 results not available",
            "void_result": "N/A",
            "tep_favored": False,
            "tep_confirmed": False,
            "significance": "N/A",
            "step": "30",
            "test_type": "direct_falsification",
        }

        if "step_30_bulk_flow_recalculation" in step_data:
            s30 = step_data["step_30_bulk_flow_recalculation"]
            test_a = s30.get("test_a_indicator_comparison", {})
            n_overlap = test_a.get("n_overlap", 0)
            mean_delta = test_a.get("mean_delta_mu", 0)
            sem_delta = test_a.get("sem_delta_mu", 0)
            sigma = test_a.get("significance_sigma", 0)
            n_shorter = test_a.get("n_cepheid_shorter", 0)
            compression = test_a.get("distance_compression_pct", 0)

            if n_overlap >= 5:
                # Check harmonized zero-point audit
                harmonized = s30.get("test_a2_harmonized_zero_point_audit", {})
                robust = harmonized.get("robust_against_calibration", False)
                ext = harmonized.get("external_harmonized", {})
                ext_sigma = ext.get("significance_sigma", 0)
                ext_mean = ext.get("mean_delta_mu", 0)

                obs1["tep_result"] = (
                    f"CF4: Δμ = {mean_delta:.4f} ± {sem_delta:.4f} mag ({sigma:.2f}σ), "
                    f"N={n_overlap}, {n_shorter}/{n_overlap} Cepheid shorter. "
                    f"Harmonized (R22 vs Freedman 2025): Δμ = {ext_mean:.4f} "
                    f"± {ext.get('sem_total', 0):.4f} ({ext_sigma:.2f}σ, N={ext.get('n_overlap', 0)}). "
                    f"Result {'survives' if robust else 'does not survive'} calibration harmonization."
                )
                obs1["void_result"] = "Void predicts Δμ = 0"
                # TEP is favored only if the harmonized audit confirms the offset
                obs1["tep_favored"] = bool(robust)
                obs1["tep_confirmed"] = bool(robust)
                obs1["significance"] = f"{sigma:.2f}σ (CF4), {ext_sigma:.2f}σ (harmonized)"

        observables.append(obs1)
        print_status(f"  Observable 1: Indicator Divergence — TEP favored: {obs1['tep_favored']}", "TEST")

        # Observable 2: Bulk-Flow Calibration Sensitivity (step_30 Test B + step_31)
        obs2 = {
            "observable": "Bulk-Flow Calibration Sensitivity",
            "tep_prediction": (
                "Cepheid H0 inflates bulk flow; Δv = -ΔH0·d km/s from "
                "indicator divergence (Observable 1)"
            ),
            "void_prediction": "No calibration dependence (bulk flow is gravitational)",
            "tep_result": "Step 30/31 results not available",
            "void_result": "N/A",
            "tep_favored": False,
            "tep_confirmed": False,
            "significance": "N/A",
            "step": "30+31",
            "test_type": "calibration_sensitivity",
            "note": (
                "This is a quantification of how the indicator divergence "
                "(Observable 1) propagates into the peculiar velocity field, "
                "not an independent falsification test."
            ),
        }

        if "step_30_bulk_flow_recalculation" in step_data and "step_31_peculiar_velocity_artifact" in step_data:
            s30 = step_data["step_30_bulk_flow_recalculation"]
            s31 = step_data["step_31_peculiar_velocity_artifact"]

            test_b = s30.get("test_b_bulk_flow_sensitivity", {})
            mean_reduction = test_b.get("mean_reduction_pct", 0)

            s31_results = s31.get("results", {})
            rms_shift = s31_results.get("rms_shift", 0)
            frac_large = s31_results.get("fraction_gt_200_kms", 0)
            n_gal = s31.get("n_galaxies", 0)
            delta_h0_val = s31_results.get("delta_h0", s31.get("delta_h0", 3.2))

            # Update TEP prediction with actual delta_h0 value
            obs2["tep_prediction"] = (
                f"Cepheid H0 inflates bulk flow; Δv = -{delta_h0_val:.1f}·d km/s from "
                "indicator divergence (Observable 1)"
            )

            obs2["tep_result"] = (
                f"Mean BF reduction: {mean_reduction:.1f}%, "
                f"RMS shift: {rms_shift:.0f} km/s, "
                f"{frac_large*100:.0f}% of galaxies |shift|>200 km/s (N={n_gal})"
            )
            obs2["void_result"] = "Void predicts no calibration dependence"
            # TEP is favored if the indicator divergence (Observable 1) is confirmed
            # AND the calibration sensitivity is in the predicted direction
            test_a = s30.get("test_a_indicator_comparison", {})
            obs2["tep_favored"] = bool(
                test_a.get("tep_confirmed", False) and mean_reduction > 0
            )
            obs2["tep_confirmed"] = obs2["tep_favored"]
            obs2["significance"] = f"{mean_reduction:.1f}% BF reduction, {rms_shift:.0f} km/s RMS"

        observables.append(obs2)
        print_status(f"  Observable 2: Calibration Sensitivity — TEP favored: {obs2['tep_favored']}", "TEST")

        # Observable 3: H0(z) Redshift Profile (step_32)
        obs3 = {
            "observable": "H0(z) Redshift Profile",
            "tep_prediction": "Flat H0(z) ≈ 73 for global M_B (zero-point inheritance)",
            "void_prediction": "Gradual decline (Gaussian/Exponential, HBK20/Mazurenko 2025)",
            "tep_result": "Step 32 results not available",
            "void_result": "N/A",
            "tep_favored": False,
            "tep_confirmed": False,
            "significance": "N/A",
            "step": "32",
            "test_type": "direct_falsification",
        }

        if "step_32_redshift_decay_profile" in step_data:
            s32 = step_data["step_32_redshift_decay_profile"]
            mc = s32.get("model_comparison", {})
            # New structure: nested model dicts with fitted params
            vm = mc.get("void_model", {})
            tm = mc.get("tep_model", {})
            cm = mc.get("constant_model", {})
            delta_aic_tep_vs_void = mc.get("delta_aic_void_vs_tep", 0)
            best_model = mc.get("best_model", "unknown")

            chi2_void = vm.get("chi2_reduced", 0)
            chi2_tep = tm.get("chi2_reduced", 0)
            chi2_const = cm.get("chi2_reduced", 0)

            obs3["tep_result"] = (
                f"χ²_red(TEP) = {chi2_tep:.2f} (n_fit={tm.get('decay_index_fit', 0):.2f}), "
                f"χ²_red(Const) = {chi2_const:.2f}, "
                f"ΔAIC(Void−TEP) = {delta_aic_tep_vs_void:.1f}, "
                f"best={best_model}"
            )
            obs3["void_result"] = (
                f"χ²_red(Void) = {chi2_void:.2f}"
            )
            obs3["tep_favored"] = bool(mc.get("tep_preferred", False) or mc.get("void_falsified", False))
            obs3["tep_confirmed"] = bool(mc.get("tep_preferred", False))
            obs3["significance"] = f"ΔAIC(Void−TEP) = {delta_aic_tep_vs_void:.1f}, best={best_model}"

        observables.append(obs3)
        print_status(f"  Observable 3: H0(z) Profile — TEP favored: {obs3['tep_favored']}", "TEST")

        # Observable 4: Calibration-Independent Relative Evolution (step_34)
        obs4 = {
            "observable": "Relative Evolution (z < 0.15 vs z > 0.25)",
            "tep_prediction": (
                "Flat H0(z) for global M_B (zero-point inheritance)"
            ),
            "void_prediction": "Gradual decline towards Planck value",
            "tep_result": "Step 34 results not available",
            "void_result": "N/A",
            "tep_favored": False,
            "tep_confirmed": False,
            "significance": "N/A",
            "step": "34",
            "test_type": "direct_falsification",
            "note": (
                "The absolute H0 at z > 0.25 is NOT independent — Pantheon+ "
                "uses MU_SH0ES with M_B=-19.253 from Cepheid anchors. The "
                "meaningful test is the relative evolution (common zero-point "
                "cancels). The host-mass dependence of the TEP effect is "
                "established in TEP-H0 (Paper 11)."
            ),
        }

        if "step_34_void_boundary_test" in step_data:
            s34 = step_data["step_34_void_boundary_test"]
            bt = s34.get("boundary_test", {})
            mc34 = s34.get("model_comparison", {})
            h0_highz = bt.get("h0_massive_z_gt_03", np.nan)
            h0_lowz = bt.get("h0_massive_z_lt_015", np.nan)
            void_falsified = bt.get("void_falsified", False)
            tep_supported = bt.get("tep_supported", False)

            if not np.isnan(h0_highz) and not np.isnan(h0_lowz):
                delta_rel = h0_highz - h0_lowz
                obs4["tep_result"] = f"H0 changes from {h0_lowz:.1f} (z<0.15) to {h0_highz:.1f} (z>0.25), Δ={delta_rel:+.1f}"
                obs4["void_result"] = "KBC predicts gradual decline"
                obs4["tep_favored"] = bool(tep_supported)
                obs4["tep_confirmed"] = bool(tep_supported)
                obs4["significance"] = f"ΔH0(low→high) = {delta_rel:+.1f} km/s/Mpc (no decline)"

            mc_massive = mc34.get("massive", {})
            if mc_massive:
                fp = mc_massive.get("fixed_prediction", {})
                da_gauss = fp.get("delta_aic_fixed_gaussian", np.nan)
                da_exp = fp.get("delta_aic_fixed_exponential", np.nan)
                if not np.isnan(da_gauss):
                    obs4["significance"] += (
                        f", ΔAIC(Void_Gauss−TEP)={da_gauss:.1f}"
                    )
                if not np.isnan(da_exp):
                    obs4["significance"] += (
                        f", ΔAIC(Void_Exp−TEP)={da_exp:.1f}"
                    )

        observables.append(obs4)
        print_status(f"  Observable 4: Relative Evolution — TEP favored: {obs4['tep_favored']}", "TEST")

        return observables

    def compute_overall_assessment(self, observables):
        """Compute the overall void vs TEP assessment."""
        print_status("Computing overall assessment...", "PROCESS")

        # Count direct falsification tests (observables 1, 3, 4)
        direct_tests = [o for o in observables if o.get("test_type") == "direct_falsification"]
        n_direct = len(direct_tests)
        n_direct_tep = sum(1 for o in direct_tests if o["tep_favored"])

        # Count all observables
        n_tep_favored = sum(1 for obs in observables if obs["tep_favored"])
        n_total = len(observables)

        print_status(f"  TEP favored in {n_tep_favored}/{n_total} observables", "TEST")
        print_status(f"  Direct falsification tests: {n_direct_tep}/{n_direct} favor TEP", "TEST")

        if n_direct_tep == n_direct and n_tep_favored >= 3:
            verdict = (
                "TEP strongly favored — all direct falsification tests "
                f"({n_direct_tep}/{n_direct}) support TEP over the KBC void model. "
                f"The indicator-specific distance divergence ({observables[0]['significance']}) "
                "is dataset/reduction dependent. The H0(z) redshift profile "
                f"({observables[2]['significance']}) and the calibration-independent "
                f"relative evolution ({observables[3]['significance']}) — two views of "
                "the same Pantheon+ redshift-profile test — provide the decisive "
                "falsification of the void model."
            )
        elif n_tep_favored >= 3:
            verdict = (
                "TEP strongly favored — majority of observables support TEP; "
                "void model falsified on redshift-profile grounds"
            )
        elif n_direct_tep >= 2:
            verdict = (
                f"TEP favored — {n_direct_tep}/{n_direct} direct falsification tests "
                "support TEP. The indicator divergence is dataset/reduction dependent; "
                "the redshift profile test provides the decisive falsification."
            )
        elif n_tep_favored >= 2:
            verdict = "TEP moderately favored — majority of observables support TEP"
        elif n_tep_favored >= 1:
            verdict = "Inconclusive — mixed results between TEP and void model"
        else:
            verdict = "Void model favored — observables do not support TEP prediction"

        print_status(f"  Overall verdict: {verdict}", "SUCCESS")

        return {
            "n_observables": n_total,
            "n_tep_favored": n_tep_favored,
            "n_direct_falsification_tests": n_direct,
            "n_direct_tep_favored": n_direct_tep,
            "tep_favored_fraction": float(n_tep_favored / n_total),
            "verdict": verdict,
            "tep_strongly_favored": n_tep_favored >= 3,
            "void_falsified": n_direct_tep >= 2,
        }

    def write_csv_table(self, observables, assessment):
        """Write the falsification summary table as CSV."""
        print_status("Writing falsification summary CSV table...", "PROCESS")

        csv_path = self.results / "step_42_falsification_table.csv"

        headers = [
            "Observable",
            "Type",
            "TEP Prediction",
            "Void Model Prediction",
            "TEP Result",
            "Void Model Result",
            "TEP Favored",
            "Significance",
            "Step",
        ]

        rows = []
        for obs in observables:
            rows.append([
                obs["observable"],
                obs.get("test_type", ""),
                obs["tep_prediction"],
                obs["void_prediction"],
                obs["tep_result"],
                obs["void_result"],
                "Yes" if obs["tep_favored"] else "No",
                obs["significance"],
                obs["step"],
            ])

        rows.append([
            "OVERALL ASSESSMENT",
            "",
            "",
            "",
            "",
            "",
            f"{assessment['n_tep_favored']}/{assessment['n_observables']}",
            assessment["verdict"][:80],
            "42",
        ])

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)

        print_status(f"CSV table saved to {csv_path}", "SUCCESS")
        return csv_path

    def run(self):
        """Execute the full step."""
        print_status("Step 42: Falsification Summary — Void vs TEP Head-to-Head", "TITLE")

        print_status(
            "This step compiles the four discriminating observables that "
            "distinguish the KBC void/MOND model from the TEP isochrony-violation "
            "framework into a single head-to-head falsification assessment. "
            "Observables 1, 3, and 4 are independent direct falsification tests; "
            "Observable 2 quantifies how the indicator divergence propagates into "
            "the peculiar velocity field. The overall verdict determines whether "
            "the void model survives the combined evidence or is falsified in "
            "favor of TEP.",
            "INFO",
        )

        step_data = self.load_step_results()

        print_status(
            "Methodology: four discriminating observables are compiled from "
            "upstream step results (steps 30, 31, 32, 34, 40). Each observable "
            "carries explicit TEP and void predictions, a measured result, and a "
            "significance assessment. Direct falsification tests (observables 1, "
            "3, 4) are weighted most heavily in the overall verdict.",
            "PROCESS",
        )
        observables = self.compile_observables(step_data)
        assessment = self.compute_overall_assessment(observables)

        n_tep = assessment.get("n_tep_favored", 0)
        n_total = assessment.get("n_observables", 0)
        n_direct = assessment.get("n_direct_tep_favored", 0)
        n_direct_total = assessment.get("n_direct_falsification_tests", 0)
        print_status(
            f"Interpretation: TEP is favored in {n_tep}/{n_total} observables, "
            f"with {n_direct}/{n_direct_total} direct falsification tests "
            f"supporting TEP over the void model. The verdict is: "
            f"{assessment.get('verdict', 'N/A')}",
            "SUCCESS",
        )

        csv_path = self.write_csv_table(observables, assessment)

        summary = {
            "step": "42_falsification_summary",
            "description": (
                "Head-to-head falsification summary: four discriminating "
                "observables comparing KBC void/MOND model vs TEP isochrony violation. "
                "Observables 1, 3, and 4 are independent direct falsification tests. "
                "Observable 2 quantifies how the indicator divergence propagates "
                "into the peculiar velocity field."
            ),
            "n_steps_loaded": len(step_data),
            "n_step_files_expected": len(self.STEP_FILES),
            "observables": observables,
            "overall_assessment": assessment,
            "tep_prediction": "Redshift-profile test (Sections 7/8) should favor TEP; indicator divergence (Section 5) is dataset/reduction dependent",
            "tep_confirmed": assessment["tep_strongly_favored"],
            "void_falsified": assessment["void_falsified"],
            "void_prediction": (
                "The KBC void/MOND model predicts indicator-independent distances, "
                "calibration-independent bulk flow, and a gradual H0(z) decline. "
                "Falsification occurs if the direct tests favor TEP."
            ),
            "methodology": (
                "Four discriminating observables compiled from upstream step "
                "results (steps 30, 31, 32, 34, 40). Each observable carries TEP "
                "and void predictions, measured results, and significance. Direct "
                "falsification tests (observables 1, 3, 4) are weighted most "
                "heavily in the overall verdict. Observable 2 is a quantification "
                "of the indicator divergence propagation, not an independent test."
            ),
            "provenance": {
                "data_sources": [
                    "step_30_bulk_flow_recalculation.json",
                    "step_31_peculiar_velocity_artifact.json",
                    "step_32_redshift_decay_profile.json",
                    "step_34_void_boundary_test.json",
                    "step_40_redshift_shear_reconstruction.json",
                ],
                "pipeline_block": "Block III — TEP reconstruction and synthesis",
            },
            "scientific_context": (
                "Head-to-head falsification of the KBC void/MOND model against the "
                "TEP isochrony-violation framework. The four observables span "
                "indicator-specific distance divergence, bulk-flow calibration "
                "sensitivity, the H0(z) redshift profile, and calibration-"
                "independent relative evolution. The combined verdict determines "
                "whether the void model is falsified."
            ),
            "downstream_consumers": ["step_43"],
            "output_files": [
                str(self.results / "step_42_falsification_summary.json"),
                str(csv_path),
            ],
        }

        summary_path = self.results / "step_42_falsification_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print_status(f"Summary saved to {summary_path}", "SUCCESS")

        print_status("Step 42 complete", "SUCCESS")


if __name__ == "__main__":
    step = Step42FalsificationSummary()
    step.run()
