# TEP-VOID Data Provenance

This document provides complete provenance information for all external data
used in the TEP-VOID analysis pipeline.

## 1. SH0ES Cepheid Distance Ladder Data

**Source:** Riess et al. (2022) - SH0ES Team
**Reference:** Riess, A. G., et al. 2022, ApJ, 934, L7
**arXiv:** 2112.04510
**Data Repository:** https://github.com/PantheonPlusSH0ES/DataRelease

**Files Used:**
- `Pantheon+SH0ES.dat` — Cepheid + SN Ia distance ladder data

**Ingestion:** Automated download via `steps/step_00_data_ingestion.py`

---

## 2. CCHP TRGB Distances

**Source:** Freedman et al. (2024) - Chicago-Carnegie Hubble Program
**Reference:** Freedman, W. L., et al. 2024, ApJ, 961, 121
**Data:** TRGB distance moduli for SN Ia host galaxies

**Ingestion:** Curated CSV at `data/raw/external/trgb_distances_freedman2024.csv`

---

## 3. Kinematic Potential Proxy (σ_v)

**Source:** HyperLEDA galaxy database (Makarov et al. 2014)
**Reference:** Makarov, D., Prugniel, P., Terekhova, N., et al. 2014, A&A, 570, A13
**Primary File:** `data/raw/external/velocity_dispersions_literature.csv`

The `sigma_kms` column is NOT a measured central stellar velocity
dispersion. It is the declared potential-equivalent scale
`u_phi = V_rot / sqrt(2)`, computed from HyperLEDA's
inclination-corrected maximum rotation velocity. The source `V_rot`
and catalog uncertainty are preserved in adjacent columns
(`vrot_kms`, `vrot_error_kms`). The filename is retained for
historical pipeline compatibility. Rows 1–37 are the complete R22
SN-host set; final rows are ladder anchors.

**Ingestion:** Read by `steps/step_01_host_potential_catalog.py`

---

## 4. CosmicFlows-4 Catalog

**Source:** Tully et al. (2023) - CosmicFlows-4
**Reference:** Tully, R. B., et al. 2023, ApJ, 944, 94
**arXiv:** 2209.11238
**CDS Reference:** J/ApJ/944/94
**CDS URL:** http://cdsarc.u-strasbg.fr/ftp/J/ApJ/944/94/

**Files Used:**
- `table2.dat` — 55,877 individual galaxy distances (all methodologies)
  - Columns: PGC, Vcmb, DM, e_DM, DMtrgb, DMceph, DMsnIa, RAdeg, DEdeg, etc.
- `table4.dat` — 38,053 galaxy groups with peculiar velocities
  - Columns: PGC1, DMzp, Dist, V3k, Vpec, Hi, RAdeg, DEdeg, SGX, SGY, SGZ, etc.
- `ReadMe` — Column format documentation

**Download Method:** Playwright headless browser bypasses the CDS Anubis
proof-of-work bot protection. The download is automated in
`steps/step_02_cosmicflows_ingestion.py`.

**Ingestion:** Automated download and parsing via `steps/step_02_cosmicflows_ingestion.py`

---

## 5. Pantheon+ SN Ia Catalog

**Source:** Scolnic et al. (2022) - Pantheon+ Collaboration
**Reference:** Scolnic, D., et al. 2022, ApJ, 938, 113
**arXiv:** 2112.03863
**Data Repository:** https://github.com/PantheonPlusSH0ES/DataRelease

**Ingestion:** Automated download via `steps/step_03_pantheon_ingestion.py`

---

## 6. Companion Paper Data (TEP-H0, Paper 11)

**Source:** Smawfield, M. L. 2026, Zenodo DOI: 10.5281/zenodo.18209702
**Repository:** https://github.com/matthewsmawfield/TEP-H0

The following TEP-H0 results are referenced (not re-downloaded):
- 37-host endpoint likelihood analysis
- M31 inner vs outer Cepheid P-L offset (+0.356 ± 0.136 mag, 2.6σ)
- M31 PHAT-restricted offset (+0.630 ± 0.195 mag, 3.24σ)
- LMC OGLE-IV radial stratification (+0.0284 ± 0.0086 mag, 3.3σ)
- Unified H₀ = 66.65 ± 1.58 km s⁻¹ Mpc⁻¹

---

## 7. Mazurenko et al. (2025) Void H(z) Curves (Digitized)

**Source:** Mazurenko, Banik & Kroupa 2025, MNRAS 536, 3232–3241, Figure 3
**Reference:** "Method-3" full-GR H₀(z) curves for the KBC void
**Primary Directory:** `data/raw/external/mazurenko_curves/`

The directory contains digitized (z, H₀) points for the Gaussian and
Exponential void-density-profile H₀(z) curves from Figure 3 of
Mazurenko et al. (2025), using the HBK20 best-fitting void parameters.
Files:

- `mazurenko_curves_all.csv` — combined (curve, z, H0) table (388 rows)
- `gaussian_method3.json` — raw Gaussian Method-3 digitized points
- `exponential_method3.json` — raw Exponential Method-3 digitized points
- `gaussian_method3_cleaned.json` — Gaussian curve after artifact removal
- `gaussian_method3_fitted.json` — parametric Gaussian-decline denoised fit
- `maxwell_boltzmann_method3.json` — Maxwell-Boltzmann profile curve
- `cleaning_log.json` — digitization-noise mitigation log

**Digitization noise mitigation:** The Gaussian Method-3 curve exhibits
interleaved upper/lower envelope tracing (30 sign changes in 35
intervals at z < 0.5). A parametric Gaussian-decline fit
H(z) = H_inf + A·exp(-z²/(2σ²)) is applied to denoise (reduced χ² ≈ 0.93,
σ_digitize = 0.5 km/s/Mpc). The Exponential curve is already clean
(0 sign changes) and requires no denoising.

**Ingestion:** Read by `steps/step_32_redshift_decay_profile.py` and
related step_32 variants; also consumed by `step_34_void_boundary_test.py`
and `step_32b_jia_validation.py`.

---

## 8. Jia et al. (2023) H(z) Data

**Source:** Jia et al. 2023, A&A 674, A45
**Reference:** "Cosmological constraints from the Hubble diagram of
Type Ia supernovae and the H(z) data"
**Data Repository:** https://github.com/JoJo20221003/Hz-Code

**Files Used:**
- `data/raw/external/jia_hz_data.txt` — 60 H(z) measurements (cosmic
  chronographs + BAO-derived) used in Jia et al.'s piecewise H_th
  analysis. Copied from Jia's public repository for reproducibility.

**Ingestion:** Read by `steps/step_32b_jia_proper_replication.py` as
the secondary data source (after `/tmp/Hz-Code/Data/Hz data.txt`) for
the H(z)+SNe Ia joint fit replication.

---

## Reproducibility

All data is downloaded automatically by the pipeline. To regenerate from
scratch:

```bash
cd "/Users/matthewsmawfield/www/Temporal Equivalence Principle/TEP-VOID"
python3 scripts/run_pipeline.py
```

The pipeline will download all required data, process it through 12 analysis
steps, and populate `results/outputs/`, `results/figures/`, and `logs/` with
step-number-prefixed files.
