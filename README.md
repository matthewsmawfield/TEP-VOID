# Cosmological Voids vs Temporal Shear: An Empirical Falsification of Kinematic Hubble Tension Solutions

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22150139.svg)](https://doi.org/10.5281/zenodo.22150139)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
![TEP-VOID: Cosmological Voids vs Temporal Shear](site/public/image.webp)

**Author:** Matthew Lukin Smawfield  
**Version:** v0.4 (Valencia)  
**First published:** 29 August 2026 · **Last updated:** 17 October 2026  
**Status:** Preprint  
**DOI:** [10.5281/zenodo.22150139](https://doi.org/10.5281/zenodo.22150139)  
**Website:** [https://mlsmawfield.com/tep/void/](https://mlsmawfield.com/tep/void/)  
**Paper Series:** TEP Series · Group G (VII. Domain-Boundary & Observational Anomaly Case Studies) · Paper 31 (`G31`)

## Abstract

<!-- Generated from site/components/1_abstract.html; update the manuscript source, then run tools/sync_current_corpus_readmes.py. -->

The Hubble tension admits fundamentally different interpretations: a physical expansion rate inflated by local kinematics (e.g., a Gpc-scale underdensity), or an operational redshift-distance calibration artifact. This paper tests the kinematic void hypothesis against the Temporal Equivalence Principle (TEP), which models cosmological redshift as accumulated temporal shear on a static spatial manifold. Using the matched $z \ge 0.05$ submatrix of the native Pantheon+ STAT+SYS covariance, the published KBC/MOND gradual-decay profiles are strongly rejected ($\Delta\chi² > 100$). A calibration-independent ratio $R_H = 1.009 \pm 0.006$ excludes the predicted declines — $4.3\%$ (Gaussian profile) and $5.1\%$ (Exponential) — at $8.4\sigma$ and $9.7\sigma$ respectively, falsifying the published KBC/MOND gradual-decay void solution on its own native coordinates. While this competitor exclusion carries no direct evidential weight for TEP by itself, the residual directional and environmental data are consistent with the independent clock-transport predictions of TEP. An exploratory fit to pre-standardization Pantheon+ residuals has an interior optimum at $L_T=55.6$ Mpc along the pre-specified CF4 axis, surviving a 300-draw permutation look-elsewhere calibration with zero null exceedances ($-2\Delta\ln L = 24.0$ versus a null maximum of $10.8$, setting a Monte Carlo resolution-limited bound $p = 1/301 \approx 0.0033$, or $\gtrsim 2.7\sigma$), though CF4 cannot independently distinguish the kernel from its $1/r$ asymptote. However, the identified coherence scale places the characteristic structure in the tens-of-Mpc local-environment regime rather than the Gpc scale required by the KBC void. Because the standard rest-frame time mapping used by SALT3 implicitly assumes that the spectroscopic redshift fully accounts for the transformation between observed and emitted light-curve timescales, aggregate SALT3 standardization attenuates the pre-standardization directional signal by $70 \pm 16\%$, consistent with partial absorption of a temporal contribution. The underlying temporal shear instead manifests locally: host-galaxy systemic redshifts are weighted toward deeper central potentials than disk standard candles, artificially contracting rest-frame timescales. Consequently, Cepheid distances are systematically shorter than TRGB distances within the same galaxies ($2.39\sigma$ directional preference), and single-galaxy Period-Luminosity gradients within the LMC ($3.30\sigma$, robust to confounder controls) and M31 ($3.65\sigma$ baseline in the HST PHAT catalogue, $2.18\sigma$ period–colour matched, and $6.17\sigma$ under flux-independent neighbour-density crowding control; Paper 11) are directionally consistent with the predicted clock-rate differential; potential-scaling tests across independent samples are presently leverage-limited rather than excluded. The results are consistent with a clock-transport contribution across the distance-ladder measurement chain driven by localized temporal topology, and motivate the independent environmental and spectroscopic tests specified below.

Keywords: Hubble tension, Temporal Equivalence Principle, isochrony violation, Cepheid variables, TRGB, KBC void, MOND, bulk flow, CosmicFlows-4, JWST, Pantheon+, scalar-tensor theory, distance ladder, cosmology

Current manuscript source: [1_abstract.html](site/components/1_abstract.html).

## Key Findings

The current findings, numerical results and comparison assumptions are reported in [12. Conclusions](site/components/13_conclusion.html). The abstract above is synchronized from the current manuscript.

## The TEP Research Program

| Paper | Repository | Title | DOI |
|-------|-----------|-------|-----|
| **Paper 0** | [TEP](https://github.com/matthewsmawfield/TEP) | Temporal Equivalence Principle: Dynamic Time & Emergent Light Speed | [10.5281/zenodo.16921911](https://doi.org/10.5281/zenodo.16921911) |
| **Paper 9** | [TEP-EXP](https://github.com/matthewsmawfield/TEP-EXP) | What Do Precision Tests of General Relativity Actually Measure? | [10.5281/zenodo.18109760](https://doi.org/10.5281/zenodo.18109760) |
| **Paper 11** | [TEP-H0](https://github.com/matthewsmawfield/TEP-H0) | The Cepheid Bias: Resolving the Hubble Tension | [10.5281/zenodo.18209702](https://doi.org/10.5281/zenodo.18209702) |
| **Paper 12** | [TEP-JWST](https://github.com/matthewsmawfield/TEP-JWST) | A Unified Resolution to the JWST High-Redshift Anomalies | [10.5281/zenodo.19000827](https://doi.org/10.5281/zenodo.19000827) |
| **Paper 31** | **TEP-VOID** (This repo) | Cosmological Voids vs Temporal Shear | [10.5281/zenodo.22150139](https://doi.org/10.5281/zenodo.22150139) |

## Pipeline Structure

The pipeline (48 registered steps) falsifies the void model and tests the two-channel
TEP calibration framework. The full step register is documented in `scripts/README.md`;
the summary below highlights the main manuscript blocks.

```
Block 0 (Steps 00-03): Data Ingestion
  - SH0ES + CCHP host samples, host potential catalog, CosmicFlows-4, Pantheon+

Block I (Steps 10-12, 30-33, 36): Indicator Divergence & Void Falsification
  - Matched-host comparison, indicator divergence, void prediction uniformity
  - Indicator-specific distance divergence (Cepheid vs TRGB, CF4 tab ≤ 2, 2.39σ sign test)
  - Peculiar velocity calibration sensitivity (H0 bias propagation)
  - H0(z) redshift profile: published KBC gradual decay curves vs TEP flat prediction
  - Digitization sensitivity, Ω_m sensitivity, Jia et al. replication
  - Host-mass z > 0.25 survey design, Xi regression

Block II (Steps 34-35): Void Falsification — Boundary Test & Float M_B
  - Does H0 decline from low to high redshift (KBC prediction)?
  - Floating M_B analysis: SALT2 absorption of host-mass dependence

Block III (Steps 40, 42, 43): TEP Reconstruction & Synthesis
  - TEP correction: Cepheid calibrator → ΔM_B → Pantheon+ (+0.025 ± 0.021 mag)
  - Falsification summary: void vs TEP
  - Manuscript figure generation

Standalone (not in pipeline):
  - step_32c: Free-parameter void family fits in native mu-space
  - step_32b_jia_proper_replication: MCMC replication of Jia et al. with emcee
  - step_44: H0 vs calibrator-population potential depth
  - step_45: Xi-step test in Pantheon+ Hubble residuals
  - step_46: Anchor sensitivity analysis (NGC 4258 σ_local)
```

## Repository Structure

```
TEP-VOID/
├── site/                           # Academic manuscript site
│   ├── components/                 # HTML section files (edit these)
│   ├── public/                     # Static assets
│   └── dist/                       # Built site (generated)
├── scripts/steps/                  # Reproducible analysis pipeline (48 registered steps)
├── core/                           # Shared TEP framework modules
├── scripts/                        # Utility scripts (PDF generation, etc.)
├── G31-TEP-VOID-v0.4-Valencia.md   # Generated manuscript (built from site/components)
└── VERSION.json                    # Version metadata
```

## Building the Site

```bash
cd site
npm install
npm run build
```

The built site will be in `site/dist/`. The build also regenerates `G31-TEP-VOID-v0.4-Valencia.md` at the repository root.

## Manuscript Editing

Edit `site/components/*.html` files only. The markdown and `site/dist/` files are auto-generated by `npm run build`. Do not edit generated files directly.

## Citation

```bibtex
@misc{smawfield2026void,
  title        = {Cosmological Voids vs Temporal Shear: An Empirical Falsification of Kinematic Hubble Tension Solutions},
  author       = {Smawfield, Matthew Lukin},
  year         = {2026},
  doi          = {10.5281/zenodo.22150139},
  url          = {https://doi.org/10.5281/zenodo.22150139},
  note         = {Preprint, Version v0.4 (Valencia)}
}
```

---

## Open Science Statement

These are working preprints shared in the spirit of open science—all manuscripts, analysis code, and data products are openly available under Creative Commons and MIT licenses to encourage and facilitate replication. Feedback and collaboration are warmly invited and welcome.

---
  
**Contact:** matthew@mlsmawfield.com  
**ORCID:** [0009-0003-8219-3159](https://orcid.org/0009-0003-8219-3159)
