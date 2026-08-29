#!/usr/bin/env python3
"""
TEP-VOID Analysis Pipeline Master Script
========================================
Orchestrates the analysis pipeline for Paper 31: "Cosmological Voids vs
Isochrony Violation: Differentiating Kinematic Outflows from Proper-Time
Bias in the Local Distance Ladder".

This paper's unique contribution is a head-to-head falsification of the
KBC void/MOND model against the TEP isochrony-violation framework.  The
pipeline tests four discriminating observables where the two models make
mutually exclusive predictions, plus a forward-looking survey design.
Three of the four are independent direct falsification tests; the fourth
quantifies how the indicator divergence propagates into the peculiar
velocity field.

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
]


def run_step(block, step_num, module_name, class_name, description, master_logger):
    """Execute a single pipeline step with proper logging and error handling."""

    if isinstance(step_num, int):
        step_id = f"step_{step_num:02d}"
    else:
        step_id = f"step_{step_num}"
    print_status(f"Starting {step_id}: {description}", "TITLE")

    # Create step-specific logger
    log_file = PROJECT_ROOT / "logs" / f"{step_id}.log"
    step_logger = TEPLogger(step_id, log_file_path=log_file)
    set_step_logger(step_logger)

    start_time = time.time()

    try:
        # Dynamically import the step module
        import importlib
        module = importlib.import_module(module_name)
        step_class = getattr(module, class_name)
        step_instance = step_class()
        step_instance.run()

        elapsed = time.time() - start_time
        print_status(f"{step_id} completed in {elapsed:.1f}s", "SUCCESS")
        master_logger.info(f"  {step_id}: OK ({elapsed:.1f}s)")
        # Clean up empty step_NN.log files (actual content is in descriptively-named logs)
        if log_file.exists() and log_file.stat().st_size == 0:
            log_file.unlink()
        return True

    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = f"{step_id} FAILED after {elapsed:.1f}s: {e}"
        print_status(error_msg, "ERROR")
        master_logger.error(f"  {step_id}: FAILED ({elapsed:.1f}s) — {e}")
        traceback.print_exc()
        # Clean up empty step_NN.log files
        if log_file.exists() and log_file.stat().st_size == 0:
            log_file.unlink()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="TEP-VOID Analysis Pipeline (Paper 31)"
    )
    parser.add_argument(
        "--block",
        choices=["0", "I", "II", "III"],
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
    print_status("Cosmological Voids vs Isochrony Violation", "PROCESS")
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
    print_status(f"Pipeline complete: {succeeded} succeeded, {failed} failed, {total_elapsed:.1f}s total", "TITLE")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
