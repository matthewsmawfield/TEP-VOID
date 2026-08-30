#!/usr/bin/env python3
"""
TEP-VOID Analysis Pipeline Master Script
========================================
Orchestrates the analysis pipeline for Paper 31: "Cosmological Voids vs
Temporal Shear: An Empirical Falsification of Kinematic Hubble Tension
Solutions".

This paper's unique contribution is a head-to-head falsification of the
KBC void/MOND model against the TEP temporal-shear framework.  The
pipeline tests four discriminating observables where the two models make
mutually exclusive predictions, plus a forward-looking survey design.
The primary falsification is the H0(z) redshift-profile test against
Pantheon+; the auxiliary channels (indicator divergence, host-potential
scaling, single-galaxy radial gradients) are supporting diagnostics.

Pipeline Blocks:
  Block 0 (Steps 00-03): Data Ingestion
    - SH0ES + CCHP host samples
    - Host galaxy gravitational potential catalog
    - CosmicFlows-4 peculiar velocity catalog
    - Pantheon+ supernova catalog

  Block I (Steps 10-12, 30-33, 36): Indicator Divergence & Void Falsification
    - Matched host comparison: Cepheid vs TRGB distance moduli
    - Indicator divergence vs gravitational potential
    - Void prediction uniformity test
    - Bulk-flow recalculation (Cepheid vs TRGB calibrated)
    - Peculiar velocity calibration sensitivity (H0 bias propagation)
    - H0(z) decay profile: void sharp boundary vs TEP smooth decay
    - Digitization sensitivity (Table 4)
    - Omega_m sensitivity (robustness of void rejection)
    - Jia et al. (2023) full replication and validation
    - Host-mass z > 0.25 survey design
    - Xi regression: Δμ vs potential coordinate (Tables 2-3)

  Block II (Steps 34-35): Void Falsification — Boundary Test
    - Does the Hubble tension persist at z > 0.25 (well beyond void wall)?
    - Void predicts: tension vanishes beyond the void wall
    - TEP predicts: tension persists wherever deep-potential hosts are used
    - Floating M_B analysis: SALT2 absorption quantification

  Block III (Steps 40, 42, 43): TEP Reconstruction & Synthesis
    - Pantheon+ TEP correction removes the apparent H0(z) decay
    - Head-to-head falsification summary: void vs TEP
    - Manuscript figure generation

  Block IV (Steps 32c, 44-52): Auxiliary TEP Supporting Tests
    - Free-parameter void fits, H0 vs potential, X_i-step, band dependence
    - JWST matched sample, Bayesian band analysis, eta_P derivation

  Block V (Steps 53-58, 70-73): Bulk-Flow Estimator Audit & Radial Discriminators
    - Directional Δμ sample and analysis (CMB correlation is a confound)
    - Gate D: Cartesian dipole rebuild with Freedman-Lane permutations
    - Gate F: dual-calibration TF — H0-invariant estimator gives ΔB = 0.0
    - Mount Wilson Equivalence Theorem (step 70)
    - X_i disformal channel (step 71)
    - H0(z) falsification with full Pantheon+ covariance (step 72)
    - Low-z radial discriminator audit (step 73)

Note: Single-galaxy radial gradients (M31, LMC) and the full distance-ladder
H0 unification are published in the companion paper TEP-H0 (Paper 11,
DOI: 10.5281/zenodo.18209702) and are not duplicated here.

Usage:
    python3 scripts/run_pipeline.py
    python3 scripts/run_pipeline.py --block I
    python3 scripts/run_pipeline.py --step 30

Author: Matthew Lukin Smawfield
Date: August 2026
"""

import sys
import time
import argparse
import traceback
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status

# Pipeline definition: (block, step_num, module_name, class_name, description)
PIPELINE = [
    # Block 0: Data Ingestion
    ("0", 0, "scripts.steps.step_00_data_ingestion", "Step00DataIngestion",
     "Data ingestion: SH0ES Cepheid + CCHP TRGB host samples"),
    ("0", "00b", "scripts.steps.step_00b_external_data", "Step00bExternalData",
     "External data download: all datasets with SHA-256 checksums + provenance manifest"),
    ("0", "00c", "scripts.steps.step_00c_vrot_deep_catalogs", "Step00cDeepVrot",
     "Deep V_rot catalogs cross-match (SPARC, ALFALFA)"),
    ("0", 1, "scripts.steps.step_01_host_potential_catalog", "Step01HostPotentialCatalog",
     "Host galaxy gravitational potential catalog"),
    ("0", 2, "scripts.steps.step_02_cosmicflows_ingestion", "Step02CosmicflowsIngestion",
     "CosmicFlows-4 peculiar velocity catalog"),
    ("0", 3, "scripts.steps.step_03_pantheon_ingestion", "Step03PantheonIngestion",
     "Pantheon+ supernova distance-redshift catalog"),

    # Block I: Indicator Divergence & Void Falsification
    ("I", 10, "scripts.steps.step_10_matched_host_comparison", "Step10MatchedHostComparison",
     "Matched host comparison: Cepheid vs TRGB distance moduli for overlapping galaxies"),
    ("I", 11, "scripts.steps.step_11_indicator_divergence_vs_potential", "Step11IndicatorDivergence",
     "Indicator divergence vs gravitational potential: test TEP potential-scaling prediction"),
    ("I", 12, "scripts.steps.step_12_void_prediction_uniformity", "Step12VoidUniformity",
     "Void prediction uniformity test: does Δμ scatter match the void's zero-prediction?"),
    ("I", 30, "scripts.steps.step_30_bulk_flow_recalculation", "Step30BulkFlowRecalculation",
     "Bulk-flow recalculation: Cepheid-calibrated vs TRGB-calibrated"),
    ("I", 31, "scripts.steps.step_31_peculiar_velocity_artifact", "Step31PeculiarVelocityArtifact",
     "Peculiar velocity artifact: Cepheid distance compression inflates v_pec"),
    ("I", 32, "scripts.steps.step_32_redshift_decay_profile", "Step32RedshiftDecayProfile",
     "H0(z) decay profile: void sharp step vs TEP smooth decay (fitted models, valid AIC)"),
    ("I", "32d", "scripts.steps.step_32_digitization_sensitivity", "Step32DigitizationSensitivity",
     "Digitization sensitivity: ΔAIC under ±0.5 and ±1.0 km/s/Mpc curve shifts (Table 4)"),
    ("I", "32m", "scripts.steps.step_32_omega_m_sensitivity", "Step32OmegaMSensitivity",
     "Omega_m sensitivity: ΔAIC and R_H as functions of Omega_m (robustness of void rejection)"),
    ("I", "32r", "scripts.steps.step_32b_jia_replication", "Step32bJiaReplication",
     "Jia et al. (2023) full replication: six-bin H_0(z) reconstruction from Pantheon+"),
    ("I", "32p", "scripts.steps.step_32b_jia_proper_replication", "Step32bJiaProperReplication",
     "Proper replication: Jia method parameter extraction"),
    ("I", "32b", "scripts.steps.step_32b_jia_validation", "Step32BJiaValidation",
     "VALIDATION: reproduce Jia et al. H_0(z) reconstruction, confirm direct-μ likelihood differs"),
    ("I", 33, "scripts.steps.step_33_host_mass_z03_survey", "Step33HostMassZ03Survey",
     "Host-mass survey design for SNe Ia at z > 0.3"),
    ("I", 36, "scripts.steps.step_36_xi_regression", "Step36XiRegression",
     "Xi regression: Δμ vs gravitational potential coordinate across 22-galaxy CF4 sample (Tables 2–3)"),

    # Block II: Void Falsification — Boundary Test
    ("II", 34, "scripts.steps.step_34_void_boundary_test", "Step34VoidBoundaryTest",
     "Void boundary test: does H0 tension persist at z > 0.25 (well beyond void wall)?"),
    ("II", 35, "scripts.steps.step_35_float_mb_analysis", "Step35FloatMBAnalysis",
     "Floating M_B analysis: release global M_B constraint, quantify SALT2 absorption"),

    # Block III: TEP Reconstruction & Synthesis
    ("III", 40, "scripts.steps.step_40_redshift_shear_reconstruction", "Step40RedshiftShearReconstruction",
     "TEP correction removes the apparent H0(z) decay in Pantheon+"),
    ("III", 42, "scripts.steps.step_42_falsification_summary", "Step42FalsificationSummary",
     "Head-to-head falsification summary: void vs TEP"),
    ("III", 43, "scripts.steps.step_43_manuscript_figures", "Step43ManuscriptFigures",
     "Manuscript figure generation from all results"),

    # Block IV: Auxiliary TEP Supporting Tests (steps 32c, 44-52)
    # These steps are cited in the manuscript (Sections 5, 9, 10) and are
    # registered here so the full pipeline reproduces every cited number.
    ("IV", "32c", "scripts.steps.step_32c_free_param_native", "Step32cFreeParamNative",
     "Free-parameter void family fits in native mu-space (Table 5): void collapses to flat"),
    ("IV", 44, "scripts.steps.step_44_h0_vs_potential", "Step44H0VsPotential",
     "H0 vs calibrator-population potential depth (JAGB/TRGB/Cepheids/SBF ordering)"),
    ("IV", 45, "scripts.steps.step_45_xi_step", "Step45XiStep",
     "X_i-step in Pantheon+ Hubble residuals (TF + measured V_rot, screened)"),
    ("IV", 46, "scripts.steps.step_46_anchor_sensitivity", "Step46AnchorSensitivity",
     "Anchor sensitivity: Cepheid channel bound under anchor variations"),
    ("IV", 47, "scripts.steps.step_47_measured_vrot_analysis", "MeasuredVrotAnalysis",
     "Measured V_rot analysis: Vizier/HyperLEDA rotation velocities for Pantheon+ hosts"),
    ("IV", 48, "scripts.steps.step_48_xi_step_measured_vrot", "MeasuredVrotXiStep",
     "X_i-step with measured V_rot from HyperLEDA (full sample + measured subsample)"),
    ("IV", 49, "scripts.steps.step_49_band_dependence", "Step49BandDependence",
     "Band-dependence: optical vs NIR Cepheid offset vs X_i (MF2023 + KP/R22 cross-team)"),
    ("IV", 50, "scripts.steps.step_50_jwst_matched", "Step50JWSTMatched",
     "JWST matched Cepheid/TRGB sample (GO-1995, GO-1685, GO-2875)"),
    ("IV", 51, "scripts.steps.step_51_band_bayesian", "Step51BandBayesian",
     "Bayesian hierarchical analysis of band-dependence (MCMC + bootstrap)"),
    ("IV", 52, "scripts.steps.step_52_eta_p_derivation", "Step52EtaP",
     "Derivation of stellar-pulsation response coefficient eta_P"),

    # Block V: Bulk-Flow Estimator Audit & Radial Discriminators
    ("V", 53, "scripts.steps.step_53_directional_sample", "Step53DirectionalSample",
     "Directional Cepheid-TRGB sample compilation (download + cross-match + X_i + cmb_dot)"),
    ("V", 54, "scripts.steps.step_54_directional_dmu", "Step54DirectionalDmu",
     "Directional Δμ analysis: CMB correlation absorbed by X_i and R22 provenance"),
    ("V", 55, "scripts.steps.step_55_directional_pantheon", "Step55DirectionalPantheon",
     "Directional Pantheon+ hemisphere split: CMB-dipole-aligned Hubble residuals"),
    ("V", 56, "scripts.steps.step_56_vrot4_r2", "Step56Vrot4R2",
     "2D geometric prediction: V_rot^4/R^2 vs V_rot^2 for Cepheid distance offset"),
    ("V", 57, "scripts.steps.step_57_differential_dipole", None,
     "Gate D: Cartesian dipole rebuild with Freedman-Lane permutations (directional signal not robust)"),
    ("V", 58, "scripts.steps.step_58_dual_calibration_tf", None,
     "Gate F: dual-calibration TF experiment — H0-invariant log-distance estimator gives ΔB = 0.0 km/s"),
    ("V", 63, "scripts.steps.step_63_raw_sn_temporal_audit", None,
     "Gate G: raw SN temporal audit — pre-standardization magnitude residuals vs SALT3-standardized residuals"),
    ("V", 64, "scripts.steps.step_64_mechanism_resolved_audit", None,
     "Mechanism-resolved audit — simultaneous decomposition of temporal and kinematic dipole channels"),
    ("V", 65, "scripts.steps.step_65_finite_coherence_audit", None,
     "Finite-coherence kernel audit — grid search over L_T on Pantheon+ raw magnitude residuals"),
    ("V", 66, "scripts.steps.step_66_cross_dataset_coherence_audit", None,
     "Cross-dataset coherence audit — continuous L_T optimization on Pantheon+ and zero-parameter CF4 cross-prediction"),
    ("V", 70, "scripts.steps.step_70_pantheon_full_discriminator", "Step70PantheonFullDiscriminator",
     "Mount Wilson Equivalence Theorem: global temporal dipole is indistinguishable from kinematic bulk flow in SNe"),
    ("V", 71, "scripts.steps.step_71_xi_disformal_channel", "Step71XiDisformalChannel",
     "X_i disformal channel: local TEP signal in Pantheon+ SN stretch and Hubble residuals"),
    ("V", 72, "scripts.steps.step_72_h0z_falsification", "Step72H0zFalsification",
     "H0(z) falsification: KBC/MOND gradual decay vs TEP flat profile with full Pantheon+ likelihood"),
    ("V", 73, "scripts.steps.step_73_pantheon_radial_discriminator", "Step73PantheonRadialDiscriminator",
     "Pantheon+ low-z radial discriminator audit: zCMB/zHD × CMB/CF4 axis sensitivity test"),
]


def run_step(block, step_num, module_name, class_name, description, master_logger):
    """Execute a single pipeline step with proper logging and error handling."""

    if isinstance(step_num, int):
        step_id = f"step_{step_num:02d}"
    else:
        step_id = f"step_{step_num}"
    # Clear any previous step logger so runner messages go to console only;
    # each step creates its own descriptive logger and calls set_step_logger().
    set_step_logger(None)
    print_status(f"Starting {step_id}: {description}", "TITLE")

    start_time = time.time()

    try:
        # Dynamically import the step module
        import importlib
        module = importlib.import_module(module_name)
        if class_name and hasattr(module, class_name):
            step_class = getattr(module, class_name)
            step_instance = step_class()
            step_instance.run()
        elif hasattr(module, "run"):
            module.run()
        else:
            raise AttributeError(f"Module {module_name} has neither class '{class_name}' nor function 'run'")

        elapsed = time.time() - start_time
        print_status(f"{step_id} completed in {elapsed:.1f}s", "SUCCESS")
        master_logger.info(f"  {step_id}: OK ({elapsed:.1f}s)")
        return True

    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = f"{step_id} FAILED after {elapsed:.1f}s: {e}"
        print_status(error_msg, "ERROR")
        master_logger.error(f"  {step_id}: FAILED ({elapsed:.1f}s) — {e}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="TEP-VOID Analysis Pipeline (Paper 31)"
    )
    parser.add_argument(
        "--block",
        choices=["0", "I", "II", "III", "IV", "V"],
        help="Run only a specific block",
    )
    parser.add_argument(
        "--step",
        type=str,
        help="Run only a specific step (by number, e.g. 30, or by ID, e.g. 32d)",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running subsequent steps even if one fails",
    )
    args = parser.parse_args()

    # Create output directories
    (PROJECT_ROOT / "results" / "figures").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "results" / "outputs").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "logs").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "data" / "raw" / "external").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "data" / "interim").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)

    # Master logger
    master_log = PROJECT_ROOT / "logs" / "pipeline_master.log"
    master_logger = TEPLogger("pipeline_master", log_file_path=master_log, reset_log=True)
    set_step_logger(master_logger)

    print_status("TEP-VOID Analysis Pipeline (Paper 31)", "TITLE")
    print_status("Cosmological Voids vs Temporal Shear", "PROCESS")
    print_status(f"Project root: {PROJECT_ROOT}", "INFO")
    print_status(f"Results: {PROJECT_ROOT / 'results'}", "INFO")
    print_status(f"Logs: {PROJECT_ROOT / 'logs'}", "INFO")

    # Filter steps
    steps_to_run = PIPELINE
    if args.block:
        steps_to_run = [s for s in PIPELINE if s[0] == args.block]
    elif args.step is not None:
        # Try int match first (e.g. "30"), then string match (e.g. "32d")
        try:
            step_val = int(args.step)
        except ValueError:
            step_val = args.step
        steps_to_run = [s for s in PIPELINE if s[1] == step_val]

    print_status(f"Running {len(steps_to_run)} step(s)", "PROCESS")

    succeeded = 0
    failed = 0
    total_start = time.time()

    for block, step_num, module_name, class_name, description in steps_to_run:
        ok = run_step(block, step_num, module_name, class_name, description, master_logger)
        if ok:
            succeeded += 1
        else:
            failed += 1
            if not args.continue_on_error:
                print_status(f"Stopping pipeline (use --continue-on-error to skip failures)", "WARNING")
                break

    total_elapsed = time.time() - total_start
    summary = f"Pipeline complete: {succeeded} succeeded, {failed} failed, {total_elapsed:.1f}s total"
    print_status(summary, "TITLE")
    master_logger.info("")
    master_logger.info("=" * 80)
    master_logger.info(f"   {summary}")
    master_logger.info("=" * 80)
    master_logger.info("")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
