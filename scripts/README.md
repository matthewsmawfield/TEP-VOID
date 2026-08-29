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

The pipeline consists of 21 active steps in 4 blocks:

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

Generates `31-TEP-VOID-v0.1-Valencia.pdf` from the built static site.
Requires the site to be built first (`cd site && npm run build`).

## Utilities

- `utils/logger.py` — TEPLogger: color-coded console + file logging
- `utils/plot_style.py` — Matplotlib style for consistent figures
- `utils/html_to_pdf.py` — HTML to PDF converter
- `utils/compress_pdf.py` — PDF compression
- `utils/process_pdf.py` — PDF metadata embedding
- `utils/setup_pdf_converter.sh` — Setup PDF converter dependencies
