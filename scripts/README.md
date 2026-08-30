# TEP-VOID Scripts

## Analysis Pipeline

```bash
cd "/Users/matthewsmawfield/www/Temporal Equivalence Principle/TEP-VOID"
python3 scripts/run_pipeline.py
```

Options:
```bash
python3 scripts/run_pipeline.py --block I          # Run only Block I (steps 30-31)
python3 scripts/run_pipeline.py --step 32           # Run only step 32
python3 scripts/run_pipeline.py --continue-on-error # Don't stop on failures
```

## Pipeline Structure

The pipeline consists of 48 registered steps in 6 blocks:

### Block 0: Data Ingestion (Steps 00-03)
- `step_00_data_ingestion.py` — SH0ES Cepheid + CCHP TRGB host samples
- `step_01_host_potential_catalog.py` — Host galaxy gravitational potential catalog
- `step_02_cosmicflows_ingestion.py` — CosmicFlows-4 peculiar velocity catalog
- `step_03_pantheon_ingestion.py` — Pantheon+ supernova distance-redshift catalog (deduplicated by CID: 1,701 raw → 1,543 unique SNe)

### Block I: Indicator Divergence & Void Falsification (Steps 10-12, 30-33, 36)
- `step_10_matched_host_comparison.py` — Cepheid vs TRGB distance moduli for matched hosts; Δμ statistics and per-galaxy table
- `step_11_indicator_divergence_vs_potential.py` — Indicator divergence vs gravitational potential; test TEP potential-scaling prediction
- `step_12_void_prediction_uniformity.py` — Void prediction uniformity test; does Δμ scatter match the void's zero-prediction?
- `step_30_bulk_flow_recalculation.py` — Cepheid vs TRGB calibrated bulk flows; self-consistent H0_TRGB derivation from Δμ
- `step_31_peculiar_velocity_artifact.py` — Peculiar velocity artifact: ΔH0 from step_30 results (not hardcoded)
- `step_32_redshift_decay_profile.py` — H0(z) decay: void vs TEP vs constant models, all fit with free parameters; valid AIC comparison; per-SN error propagation; host-mass null result reported honestly
- `step_32_digitization_sensitivity.py` — Digitization sensitivity: ΔAIC under ±0.5 and ±1.0 km/s/Mpc curve shifts (Table 4)
- `step_32_omega_m_sensitivity.py` — Omega_m sensitivity: ΔAIC and R_H as functions of Omega_m (robustness of void rejection)
- `step_32b_jia_replication.py` — Full replication of Jia et al. (2023) six-bin H_0(z) piecewise H_th method with Pantheon+ SN-only data
- `step_32b_jia_validation.py` — VALIDATION: reproduce Jia et al. (2023) H_0(z) reconstruction; Level A (curve comparison) and Level B (binned inversion); confirms direct-μ likelihood differs from parameter reconstruction
- `step_33_host_mass_z03_survey.py` — Host-mass survey design for z > 0.25
- `step_36_xi_regression.py` — Xi regression: Δμ vs gravitational potential coordinate across 22-galaxy CF4 sample (Tables 2-3); leave-one-out analysis; hierarchical pipeline-intercept regression

### Block II: Void Boundary Test (Steps 34-35)
- `step_34_void_boundary_test.py` — Void boundary test at z > 0.25; consistent threshold; free-parameter model fits; valid AIC
- `step_35_float_mb_analysis.py` — Floating M_B analysis: releases global M_B constraint, quantifies SALT2 absorption of TEP host-mass signal, forward-models TEP-predicted per-host M_B

### Block III: Synthesis & Figures (Steps 40-43)
- `step_40_redshift_shear_reconstruction.py` — Pantheon+ TEP temporal shear correction; κ_Cep imported from TEP-H0 (Paper 11); -9.0 mass placeholders excluded
- `step_42_falsification_summary.py` — Falsification pathway summary; all results (including nulls) reported honestly
- `step_43_manuscript_figures.py` — Manuscript figure generation

Note: Single-galaxy radial gradients (M31, LMC) and the full distance-ladder H0 unification are published in the companion paper TEP-H0 (Paper 11, DOI: 10.5281/zenodo.18209702) and are not duplicated here. Steps 20-22 and step 41 exist on disk as bridge scripts with fallback to manuscript values but are intentionally excluded from the active pipeline.

### Block IV: Auxiliary TEP Supporting Tests (Steps 32c, 44-52)
- `step_32c_free_param_native.py` — Free-parameter void family fits in native mu-space with full STAT+SYS covariance (Table 5): void collapses to flat; TEP fixed-n=0.3 with dH0>=0 also collapses to flat
- `step_44_h0_vs_potential.py` — H0 vs calibrator-population potential depth (JAGB < TRGB < Cepheids < SBF ordering)
- `step_45_xi_step.py` — X_i-step in Pantheon+ Hubble residuals (TF + measured V_rot, screened); underpowered due to screening compression
- `step_46_anchor_sensitivity.py` — Anchor sensitivity: NGC 4258 sigma audit; Cepheid channel bound preserved under anchor variations
- `step_47_measured_vrot_analysis.py` — Measured V_rot from Vizier/HyperLEDA; tracer-type classification; band-dependence by indicator
- `step_48_xi_step_measured_vrot.py` — X_i-step with measured V_rot (full sample N=1470 + measured subsample N=122)
- `step_49_band_dependence.py` — Band-dependence: optical vs NIR Cepheid offset vs X_i (MF2023 same-team + KP/R22 cross-team)
- `step_50_jwst_matched.py` — JWST matched Cepheid/TRGB sample (GO-1995, GO-1685, GO-2875); filter-corrected analysis
- `step_51_band_bayesian.py` — Bayesian hierarchical MCMC analysis of band-dependence (intrinsic scatter, leave-one-out)
- `step_52_eta_p_derivation.py` — Derivation of stellar-pulsation response coefficient eta_P (geometric fiducial + radiative range)

### Block V: Bulk-Flow Estimator Audit & Radial Discriminators (Steps 53-58, 70-73)
- `step_53_directional_sample.py` — Directional Cepheid-TRGB sample compilation (download + cross-match + X_i + cmb_dot)
- `step_54_directional_dmu.py` — Directional Δμ analysis: CMB correlation absorbed by X_i and R22 provenance
- `step_55_directional_pantheon.py` — Directional Pantheon+ hemisphere split: CMB-dipole-aligned Hubble residuals
- `step_56_vrot4_r2.py` — 2D geometric prediction: V_rot^4/R^2 vs V_rot^2 for Cepheid distance offset
- `step_57_differential_dipole.py` — Gate D: Cartesian dipole rebuild with Freedman-Lane permutations; directional signal not robust (bootstrap p = 0.262)
- `step_58_dual_calibration_tf.py` — Gate F: dual-calibration TF experiment; H0-invariant log-distance estimator gives ΔB = 0.0 km/s between Cepheid and TRGB calibrations
- `step_63_raw_sn_temporal_audit.py` — Gate G: raw SN temporal audit; pre-standardization magnitude residuals vs SALT3-standardized residuals
- `step_64_mechanism_resolved_audit.py` — Mechanism-resolved audit; simultaneous decomposition of temporal and kinematic dipole channels
- `step_65_finite_coherence_audit.py` — Finite-coherence kernel audit; grid search over L_T on Pantheon+ raw magnitude residuals
- `step_66_cross_dataset_coherence_audit.py` — Cross-dataset coherence audit; continuous L_T optimization on Pantheon+ and zero-parameter CF4 cross-prediction
- `step_70_pantheon_full_discriminator.py` — Mount Wilson Equivalence Theorem: global temporal dipole indistinguishable from kinematic bulk flow in SNe
- `step_71_xi_disformal_channel.py` — X_i disformal channel: local TEP signal in Pantheon+ SN stretch and Hubble residuals
- `step_72_h0z_falsification.py` — H0(z) falsification: KBC/MOND gradual decay vs TEP flat profile with full Pantheon+ likelihood
- `step_73_pantheon_radial_discriminator.py` — Pantheon+ low-z radial discriminator audit: zCMB/zHD × CMB/CF4 axis sensitivity test

Note: Steps 59-62 (CF4 registration attack, radial/heliocentric/Pantheon discriminators) exist on disk as earlier iterations superseded by step_73. They are not registered in the pipeline and not cited in the manuscript. Step_58b (M101 forensics) is also unregistered; it is a validation pass for the V_rot^4/R^2 predictor analysis in step_56.

## Output Structure

- `results/outputs/` — Step-number-prefixed JSON summaries and CSV tables
- `results/figures/` — Step-number-prefixed PNG figures
- `logs/` — Step-number-prefixed log files + `pipeline_master.log`
- `data/raw/` — Downloaded raw data (never modified)
- `data/interim/` — Intermediate processing outputs
- `data/processed/` — Final processed catalogs

## Data Sources

All data is downloaded automatically by the pipeline. See `data/DATA_PROVENANCE.md`
for complete provenance information.

Key data:
- Pantheon+SH0ES.dat (1,701 raw entries → 1,543 unique SNe after CID deduplication) — Riess et al. 2022 / Scolnic et al. 2022
- TRGB distances — Freedman et al. 2024 CCHP
- Velocity dispersions — Curated from literature (Ho+2009, Campbell+2014, etc.)
- CosmicFlows-4 bulk flows — Watkins et al. 2023 (published values)
- M31/LMC Cepheid gradients — From TEP-H0 companion paper

## Key Methodological Notes

- **H0 inference**: Uses the proper LCDM luminosity distance relation (valid at all z), not the linear cz/d approximation.
- **Error propagation**: Per-SN μ_err from Pantheon+ propagated to H0 via σ_H0 = H0·ln(10)/5·σ_μ.
- **Model fitting**: All models (void, TEP, constant) fit with free parameters via Nelder-Mead. AIC comparison is valid (not zero-parameter).
- **Host-mass split**: Excludes -9.0 placeholder values; split at log10(M*/M☉) = 10.0. Not discriminating at Pantheon+ level (global M_B calibration).
- **κ_Cep = 0.040**: Imported from TEP-H0 (Paper 11), not derived within this pipeline.
- **z > 0.25**: Consistent redshift threshold used across steps 32, 33, 34.

## PDF Generation

```bash
python3 scripts/generate_site_pdf.py
python3 scripts/generate_site_pdf.py --quality high --wait-time 10
```

Generates `31-TEP-VOID-v0.2-Valencia.pdf` from the built static site.
Requires the site to be built first (`cd site && npm run build`).

## Utilities

- `utils/logger.py` — TEPLogger: color-coded console + file logging
- `utils/plot_style.py` — Matplotlib style for consistent figures
- `utils/html_to_pdf.py` — HTML to PDF converter
- `utils/compress_pdf.py` — PDF compression
- `utils/process_pdf.py` — PDF metadata embedding
- `utils/setup_pdf_converter.sh` — Setup PDF converter dependencies
