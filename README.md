# Cosmological Voids vs Temporal Shear: An Empirical Falsification of Kinematic Hubble Tension Solutions

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22150139.svg)](https://doi.org/10.5281/zenodo.22150139)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

![TEP-VOID: Cosmological Voids vs Temporal Shear](site/public/image.webp)

**Author:** Matthew Lukin Smawfield  
**Version:** v0.2 (Valencia)  
**Date:** First published: 29 August 2026 · Last updated: 30 August 2026  
**Status:** Preprint (Draft)  
**DOI:** [10.5281/zenodo.22150139](https://doi.org/10.5281/zenodo.22150139)  
**Website:** [https://mlsmawfield.com/tep/void/](https://mlsmawfield.com/tep/void/)  
**Paper Series:** TEP Series: Paper 31 (Cosmological Observations)

## Abstract

The Hubble tension admits fundamentally different interpretations. In an expanding-space interpretation, it may arise from local kinematics such as a Gpc-scale underdensity. In the Temporal Equivalence Principle (TEP) framework, the premise is different: the spatial universe is static, non-expanding and eternal, cosmological redshift is accumulated temporal shear, and the locally inferred $H_0$ is an operational redshift–distance calibration rather than a physical expansion rate. Under this framework, the local tension is resolved by environment-dependent clock biases affecting specific rungs of the distance ladder. These frameworks make opposing predictions for the redshift profile of the inferred $H_0$ and for the consistency of distance indicators within a single galaxy. This paper tests both.

Against the Pantheon+ distance-modulus vector using the matched $z \ge 0.05$ submatrix of the native $1701 \times 1701$ STAT+SYS covariance, the published KBC/MOND profiles are rejected by $\Delta\chi^2 = 101.5$ (Gaussian) and $117.1$ (Exponential) over their stated $z \ge 0.05$ validity domain — and when amplitude and scale are free to vary, maximum likelihood occurs at zero void amplitude ($\Delta H_0^{\rm ML} = 0$): the void family collapses to its flat limit. A calibration-independent ratio $R_H = 1.0066 \pm 0.0064$ excludes the predicted 5% decline at $7.7\sigma$ (Gaussian) and $8.9\sigma$ (Exponential). The published Gpc-scale KBC/MOND kinematic-void solution cannot explain the tension.

The remaining evidence is consistent with the Temporal Equivalence Principle (TEP), which promotes proper time from a static geometric coordinate to a dynamical physical field coupled to matter. Under TEP, cosmic redshift is not physical metric expansion, but the accumulation of dynamical temporal shear ($1+z = A_0/A_{\rm em}$) across cosmic time. Locally, the Hubble tension emerges as a dynamical clock artifact: because a host galaxy's systemic redshift can be weighted toward deeper central regions than the disk standard candles, de-redshifting operations contract rest-frame timescales and artificially inflate the inferred $H_0$. Single-galaxy differential tests provide supporting evidence: internal radial Period--Luminosity gradients within M31 ($\Delta W = +0.630 \pm 0.195$ mag, $3.24\sigma$ under HST PHAT spatial matching, exhibiting the TEP-predicted sign robust to metallicity controls) and the LMC ($\Delta W = +0.0284 \pm 0.0086$ mag, $3.30\sigma$ from OGLE-IV) independently detect the predicted clock-gradient signal. At the cosmological level, a redshift-only regression across 33 Hubble-flow hosts yields $\kappa_{\rm Cep} = (0.452 \pm 0.220) \times 10^6$ mag ($2.05\sigma$), with joint multi-block likelihoods providing consistent, though statistically diluted, support ($1.58\sigma$). The conformal sector can satisfy existing solar-system bounds; complete disformal constraint matching requires the remaining natural-unit mass-scale identification. These auxiliary tests are supporting diagnostics, not the primary result; the void falsification stands independently of them at $\Delta\chi^2 > 100$. Full quantitative development of these channels and the definitive radial-gradient tests appear in the companion analysis, TEP-H0.

## Key Findings

1. **Primary falsification — $H_0(z)$ native $\mu$-space likelihood**: The unbinned native $\mu$-space likelihood with the full $1701 \times 1701$ Pantheon+ STAT+SYS covariance rejects the published KBC Method-3 curves over $z \ge 0.05$ with $\Delta{\rm AIC} = +101.5$ (Gaussian) and $+117.1$ (Exponential). Even when amplitude and scale are free to vary, the void family's best fit is flat ($\Delta H_0 = 0$, $\Delta{\rm AIC} = +4.0$).

2. **Calibration-independent relative evolution**: $R_H = 1.0093 \pm 0.0062$, $8.4\sigma$ from the KBC Gaussian prediction ($R_H = 0.9569$) and $9.7\sigma$ from the Exponential, with KBC predictions evaluated through the matched GLS estimator. The common zero-point cancels; the test is independent of the Cepheid calibration.

3. **Indicator-specific distance divergence**: 17 of 22 galaxies with both Cepheid and TRGB distances in CF4 show Cepheid distances shorter than TRGB ($2.39\sigma$ one-sided). $\kappa_{\rm Cep} = (0.452 \pm 0.220) \times 10^6$ mag from the redshift-only WLS regression ($2.05\sigma$), with joint multi-block likelihoods providing consistent, though statistically diluted, support ($1.58\sigma$).

4. **Two-channel amplitude ledger**: Cepheid channel bounded at $\sim 0.09$ mag; SN stretch channel bounded at $\sim 0.10$ mag; combined budget $\sim 0.06$–$0.19$ mag spans the $0.174$ mag tension. The SN channel provides a mechanism for the $\sim$15-year-unexplained host-mass step.

5. **Preliminary $X_i$-step test**: $+22.3 \pm 15.1$ mmag ($1.47\sigma$) in the TEP-predicted direction on 1,470 Pantheon+ Hubble-flow SNe (step_48); joint OLS $X_i$ coefficient after mass correction $+7.92 \times 10^4 \pm 3.75 \times 10^4$ ($2.11\sigma$, TEP-predicted); the Tully-Fisher $V_{\rm rot}$ proxy subsample (step_45, $N=1353$) is underpowered ($+7.5 \pm 50.2$ mmag, $0.15\sigma$) due to screening compression.

## The TEP Research Program

| Paper | Repository | Title | DOI |
|-------|-----------|-------|-----|
| **Paper 0** | [TEP](https://github.com/matthewsmawfield/TEP) | Temporal Equivalence Principle: Dynamic Time & Emergent Light Speed | [10.5281/zenodo.16921911](https://doi.org/10.5281/zenodo.16921911) |
| **Paper 9** | [TEP-EXP](https://github.com/matthewsmawfield/TEP-EXP) | What Do Precision Tests of General Relativity Actually Measure? | [10.5281/zenodo.18109760](https://doi.org/10.5281/zenodo.18109760) |
| **Paper 11** | [TEP-H0](https://github.com/matthewsmawfield/TEP-H0) | The Cepheid Bias: Resolving the Hubble Tension | [10.5281/zenodo.18209702](https://doi.org/10.5281/zenodo.18209702) |
| **Paper 12** | [TEP-JWST](https://github.com/matthewsmawfield/TEP-JWST) | A Unified Resolution to the JWST High-Redshift Anomalies | [10.5281/zenodo.19000827](https://doi.org/10.5281/zenodo.19000827) |
| **Paper 31** | **TEP-VOID** (This repo) | Cosmological Voids vs Temporal Shear | [10.5281/zenodo.22150139](https://doi.org/10.5281/zenodo.22150139) |

## Pipeline Structure

The pipeline (46 registered steps) falsifies the void model and tests the two-channel
TEP calibration framework. The full step register is documented in `scripts/README.md`;
the summary below highlights the main manuscript blocks.

```
Block 0 (Steps 00-03): Data Ingestion
  - SH0ES + CCHP host samples, host potential catalog, CosmicFlows-4, Pantheon+

Block I (Steps 10-12, 30-33, 36): Indicator Divergence & Void Falsification
  - Matched-host comparison, indicator divergence, void prediction uniformity
  - Indicator-specific distance divergence (Cepheid vs TRGB, CF4 table2, 2.39σ sign test)
  - Peculiar velocity calibration sensitivity (H0 bias propagation)
  - H0(z) redshift profile: published KBC gradual decay curves vs TEP flat prediction
  - Digitization sensitivity, Omega_m sensitivity, Jia et al. replication
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
  - step_46: Anchor sensitivity analysis (NGC 4258 sigma_local)
```

## Repository Structure

```
TEP-VOID/
├── site/                           # Academic manuscript site
│   ├── components/                 # HTML section files (edit these)
│   ├── public/                     # Static assets
│   └── dist/                       # Built site (generated)
├── scripts/steps/                  # Reproducible analysis pipeline (12 steps)
├── core/                           # Shared TEP framework modules
├── scripts/                        # Utility scripts (PDF generation, etc.)
├── 31-TEP-VOID-v0.2-Valencia.md   # Generated manuscript (built from site/components)
└── VERSION.json                    # Version metadata
```

## Building the Site

```bash
cd site
npm install
npm run build
```

The built site will be in `site/dist/`. The build also regenerates `31-TEP-VOID-v0.2-Valencia.md` at the repository root.

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
  note         = {Preprint, Version v0.2 (Valencia)}
}
```

---

## Open Science Statement

These are working preprints shared in the spirit of open science—all manuscripts, analysis code, and data products are openly available under Creative Commons and MIT licenses to encourage and facilitate replication. Feedback and collaboration are warmly invited and welcome.

---
  
**Contact:** matthew@mlsmawfield.com  
**ORCID:** [0009-0003-8219-3159](https://orcid.org/0009-0003-8219-3159)
