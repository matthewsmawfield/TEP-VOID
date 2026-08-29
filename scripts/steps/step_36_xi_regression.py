#!/usr/bin/env python3
"""
Step 36: Xi Regression — Δμ vs Gravitational Potential Coordinate
===================================================================
Test whether the Cepheid-TRGB distance divergence scales with the TEP
gravitational potential coordinate X_i across the full 22-galaxy CF4
matched sample, rather than splitting by provenance (R22 vs non-R22).

TEP predicts Δμ = κ_μ · X_i, where X_i = (U_i - U_ref) / c² and
U_i = u_phi² = vrot² / 2. The slope κ_μ should be non-zero with the
predicted sign (deeper potential → shorter Cepheid distance → Δμ < 0).

A pipeline-offset systematic would produce a constant offset independent
of X_i (slope ≈ 0). The regression therefore discriminates between TEP
(slope ≠ 0) and reduction artifact (slope = 0).

Additional analyses:
  - Mass/potential confound check: compare X_i distributions for R22 vs non-R22
  - Leave-one-out on the weighted 6.13σ result
  - Power calculation for the N=6 R22-matched test

Outputs:
    results/outputs/step_36_xi_regression.json
    results/figures/step_36_xi_regression.png
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status


class Step36XiRegression:
    """Step 36: Xi regression of Δμ vs gravitational potential coordinate."""

    # TEP constants (from TEP-H0, Paper 11, and step_40)
    SIGMA_REF = 87.165  # km/s — anchor reference potential scale
    U_REF = SIGMA_REF ** 2  # (km/s)^2 — reference potential proxy
    C_KMS = 299792.458  # km/s
    KAPPA_CEP_DEFAULT = 0.365e6  # mag (TEP-H0 closure)
    KAPPA_CEP_JOINT = 0.400e6  # mag (joint multi-block)
    KAPPA_CEP_CANONICAL = 0.960e6  # mag (canonical reference)

    def __init__(self):
        self.root = PROJECT_ROOT
        self.data_raw_external = self.root / "data" / "raw" / "external"
        self.data_interim = self.root / "data" / "interim"
        self.results = self.root / "results" / "outputs"
        self.figures = self.root / "results" / "figures"
        self.logs = self.root / "logs"

        for d in [self.data_interim, self.results, self.figures, self.logs]:
            d.mkdir(parents=True, exist_ok=True)

        self.logger = TEPLogger(
            "step_36",
            log_file_path=self.logs / "step_36_xi_regression.log",
        )
        set_step_logger(self.logger)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def load_cf4_galaxies(self):
        """Load the 22 CF4 galaxies with both Cepheid and TRGB distances."""
        print_status("Loading CF4 matched galaxy sample...", "PROCESS")

        cf4_path = self.data_raw_external / "cf4_table2.dat"
        if not cf4_path.exists():
            print_status(f"CF4 table2 not found at {cf4_path}", "ERROR")
            return pd.DataFrame()

        # Parse fixed-width format (column positions from cf4_readme.txt)
        colspecs = [
            (0, 7),       # PGC
            (102, 107),   # DMtrgb
            (108, 112),   # e_DMtrgb
            (113, 119),   # DMceph
            (120, 125),   # e_DMceph
        ]
        names = ["PGC", "DMtrgb", "e_DMtrgb", "DMceph", "e_DMceph"]
        df = pd.read_fwf(cf4_path, colspecs=colspecs, names=names, header=None)

        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["PGC"] = df["PGC"].astype("Int64")

        # Filter to galaxies with both measurements
        both = df[df["DMceph"].notna() & df["DMtrgb"].notna()].copy()
        both["delta_mu"] = both["DMceph"] - both["DMtrgb"]
        both["delta_mu_err"] = np.sqrt(both["e_DMceph"] ** 2 + both["e_DMtrgb"] ** 2)

        print_status(f"  {len(both)} galaxies with both Cepheid and TRGB", "SUCCESS")
        return both

    def load_teph0_data(self):
        """Load the TEP-H0 (Paper 11) raw SH0ES Cepheid + EDD/CCHP TRGB data.

        This is the dataset used in TEP-H0 Step 44 to calibrate kappa_Cep.
        It uses raw SH0ES Cepheid distances (not CF4 registered) and
        EDD/CCHP TRGB distances. Comparing the Xi regression on this dataset
        vs the CF4 registered dataset reveals whether CF4 registration
        preserves or distorts the TEP signal.
        """
        print_status("Loading TEP-H0 raw data for comparison...", "PROCESS")

        tep_h0_root = self.root.parent / "TEP-H0"
        if not tep_h0_root.exists():
            print_status(f"TEP-H0 directory not found at {tep_h0_root}", "WARNING")
            return pd.DataFrame()

        # Load R22 Cepheid distances
        r22_path = tep_h0_root / "data" / "interim" / "r22_distances.csv"
        hosts_path = tep_h0_root / "data" / "processed" / "hosts_processed.csv"
        trgb_path = tep_h0_root / "results" / "outputs" / "step_15_trgb_hosts_data.csv"

        if not all(p.exists() for p in [r22_path, hosts_path, trgb_path]):
            print_status("One or more TEP-H0 data files not found", "WARNING")
            return pd.DataFrame()

        r22 = pd.read_csv(r22_path)
        r22["host"] = r22["parameter"].str.replace("mu_", "")
        r22 = r22.rename(columns={"value": "mu_cep", "error": "mu_cep_err"})

        hosts = pd.read_csv(hosts_path)
        host_pgc = dict(zip(hosts["source_id"], hosts["pgc"]))
        host_sigma = dict(zip(hosts["source_id"], hosts["sigma_inferred"]))
        r22["pgc"] = r22["host"].map(host_pgc)
        r22["u_phi"] = r22["host"].map(host_sigma)

        # Load screening factors from TEP-H0 Step 03
        strat_path = tep_h0_root / "results" / "outputs" / "step_03_stratified_h0.csv"
        if strat_path.exists():
            strat = pd.read_csv(strat_path)
            strat_screening = dict(zip(strat["source_id"], strat["shear_suppression"]))
        else:
            strat_screening = {}

        trgb = pd.read_csv(trgb_path)
        trgb_pgc = trgb.merge(
            hosts[["source_id", "pgc", "normalized_name"]],
            left_on="galaxy",
            right_on="normalized_name",
            how="left",
        )

        tep_h0 = trgb_pgc.merge(
            r22[["pgc", "mu_cep", "mu_cep_err", "u_phi"]], on="pgc", how="inner"
        )
        tep_h0["delta_mu"] = tep_h0["mu_cep"] - tep_h0["mu_trgb"]
        tep_h0["delta_mu_err"] = np.sqrt(
            tep_h0["mu_cep_err"] ** 2 + tep_h0["mu_trgb_err"] ** 2
        )

        # Screening factor from Step 03
        tep_h0["S_total"] = tep_h0["source_id"].map(strat_screening).fillna(1.0)

        # Compute X_i with screening (matching TEP-H0 Step 20 methodology)
        # X_i = (S_total * u_phi^2 - U_ref) / c^2
        tep_h0["U_i_screened"] = tep_h0["S_total"] * tep_h0["u_phi"] ** 2
        tep_h0["X_i"] = (tep_h0["U_i_screened"] - self.U_REF) / self.C_KMS ** 2
        tep_h0["X_i_unscreened"] = (tep_h0["u_phi"] ** 2 - self.U_REF) / self.C_KMS ** 2

        # Exclude geometric anchors (NGC 4258, LMC) to match TEP-H0 Step 44
        anchor_pgcs = [39600, 17223]
        tep_h0 = tep_h0[~tep_h0["pgc"].isin(anchor_pgcs)].copy()

        print_status(
            f"  {len(tep_h0)} TEP-H0 galaxies (excluding anchors)", "SUCCESS"
        )
        print_status(
            f"  Screening factors loaded from TEP-H0 Step 03", "INFO"
        )
        return tep_h0

    def load_vrot_catalog(self):
        """Load the rotation velocity catalog for the 22 galaxies."""
        print_status("Loading rotation velocity catalog...", "PROCESS")

        vrot_path = self.data_raw_external / "cf4_matched_galaxies_vrot.csv"
        if not vrot_path.exists():
            print_status(f"Vrot catalog not found at {vrot_path}", "ERROR")
            return pd.DataFrame()

        vrot = pd.read_csv(vrot_path, comment="#")
        vrot["PGC"] = vrot["pgc"]
        print_status(f"  {len(vrot)} galaxies with rotation velocities", "SUCCESS")
        return vrot

    def merge_data(self, cf4_df, vrot_df):
        """Merge CF4 distances with rotation velocities and compute screening."""
        print_status("Merging CF4 distances with rotation velocities...", "PROCESS")

        merged = cf4_df.merge(vrot_df, on="PGC", how="inner")
        print_status(f"  {len(merged)} galaxies matched", "SUCCESS")

        # Compute potential coordinate
        # u_phi = vrot / sqrt(2)
        # U_i = u_phi^2 = vrot^2 / 2
        # X_i = (S_total * U_i - U_ref) / c^2  (screened, matching TEP-H0)
        merged["u_phi"] = merged["vrot_kms"] / np.sqrt(2)
        merged["U_i"] = merged["u_phi"] ** 2

        # Compute screening factors from Tully 2015 group catalog
        merged["S_total"] = self._compute_screening(merged)

        merged["X_i"] = (merged["S_total"] * merged["U_i"] - self.U_REF) / self.C_KMS ** 2
        merged["X_i_unscreened"] = (merged["U_i"] - self.U_REF) / self.C_KMS ** 2

        # Propagate uncertainty on X_i
        # dU_i = vrot * dvrot (since U = vrot^2/2, dU = vrot * dvrot)
        merged["U_i_err"] = merged["vrot_kms"] * merged["vrot_error_kms"]
        merged["X_i_err"] = merged["U_i_err"] / self.C_KMS ** 2

        return merged

    def _compute_screening(self, df):
        """Compute TEP screening factors for each galaxy.

        Uses the Tully 2015 2MRS group catalog (Nmb = group richness) to
        compute the group screening factor S_group = 1/(1 + (Nmb/Ncrit)^gamma),
        matching the TEP-H0 methodology (tep_correction.py).

        For galaxies in the TEP-H0 Step 03 stratified sample, the full
        screening factor (S_local * S_group) is loaded directly.

        For galaxies not in any catalog (isolated dwarfs), S = 1.0.
        """
        # TEP-H0 screening parameters
        N_CRIT = 10.0
        GAMMA = 1.2
        # Special Nmb values for anchors (from TEP-H0 tep_correction.py)
        ANCHOR_NMB = {2557: 11, 39600: 65, 17223: 2}  # M31, NGC4258, LMC

        # Try to load TEP-H0 Step 03 screening for known hosts
        tep_h0_root = self.root.parent / "TEP-H0"
        strat_path = tep_h0_root / "results" / "outputs" / "step_03_stratified_h0.csv"
        known_screening = {}
        if strat_path.exists():
            strat = pd.read_csv(strat_path)
            # Map by PGC
            for _, row in strat.iterrows():
                pgc = row.get("pgc", None)
                S = row.get("shear_suppression", None)
                if pd.notna(pgc) and pd.notna(S):
                    known_screening[int(pgc)] = float(S)

        # Try to load Tully catalog
        tully_path = tep_h0_root / "data" / "raw" / "external" / "tully2015_2mrs_groups_table5.csv"
        tully_nmb = {}
        if tully_path.exists():
            tully = pd.read_csv(tully_path)
            for _, row in tully.iterrows():
                tully_nmb[int(row["PGC"])] = float(row["Nmb"])

        screening = []
        for _, row in df.iterrows():
            pgc = int(row["PGC"])

            # Check TEP-H0 known screening first
            if pgc in known_screening:
                screening.append(known_screening[pgc])
                continue

            # Check Tully catalog
            if pgc in tully_nmb:
                nmb = tully_nmb[pgc]
                S_group = 1.0 / (1.0 + (nmb / N_CRIT) ** GAMMA)
                screening.append(S_group)
                continue

            # Check anchor special values
            if pgc in ANCHOR_NMB:
                nmb = ANCHOR_NMB[pgc]
                S_group = 1.0 / (1.0 + (nmb / N_CRIT) ** GAMMA)
                screening.append(S_group)
                continue

            # Isolated galaxy — no group screening
            screening.append(1.0)

        return np.array(screening)

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------
    def xi_regression(self, df):
        """
        Weighted linear regression: Δμ = slope · X_i + intercept.

        TEP predicts a non-zero slope. A pipeline offset predicts slope ≈ 0.
        """
        print_status("Running Xi regression...", "PROCESS")

        x = df["X_i"].values
        y = df["delta_mu"].values
        yerr = df["delta_mu_err"].values
        n = len(df)

        # Weighted linear regression
        weights = 1.0 / yerr ** 2
        S = np.sum(weights)
        Sx = np.sum(weights * x)
        Sy = np.sum(weights * y)
        Sxx = np.sum(weights * x * x)
        Sxy = np.sum(weights * x * y)

        denom = S * Sxx - Sx * Sx
        if abs(denom) > 0:
            slope = (S * Sxy - Sx * Sy) / denom
            intercept = (Sxx * Sy - Sx * Sxy) / denom
            slope_err = np.sqrt(S / denom)
            intercept_err = np.sqrt(Sxx / denom)
        else:
            slope, intercept = 0.0, 0.0
            slope_err, intercept_err = 0.0, 0.0

        slope_sigma = float(abs(slope) / slope_err) if slope_err > 0 else 0.0

        # Goodness of fit
        residuals = y - (slope * x + intercept)
        chi2 = float(np.sum((residuals / yerr) ** 2))
        dof = max(n - 2, 1)
        chi2_reduced = chi2 / dof

        # R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Pearson and Spearman correlations
        pearson_r, pearson_p = sp_stats.pearsonr(x, y)
        spearman_rho, spearman_p = sp_stats.spearmanr(x, y)

        # TEP predicted slope
        # Δμ = κ_μ · X_i, where X_i = (U_i - U_ref) / c² is dimensionless.
        # The slope of Δμ vs X_i IS κ_μ (in mag), NOT κ_μ/c².
        # Sign: deeper potential → shorter Cepheid distance → Δμ < 0.
        # X_i > 0 for deep potentials, so slope should be NEGATIVE.
        # The TEP-H0 values (0.365e6, 0.400e6, 0.960e6) are magnitudes;
        # the physical prediction is slope = -κ_μ.
        tep_slope_default = -self.KAPPA_CEP_DEFAULT
        tep_slope_joint = -self.KAPPA_CEP_JOINT
        tep_slope_canonical = -self.KAPPA_CEP_CANONICAL

        print_status(
            f"  Weighted fit: Δμ = ({slope:.4e} ± {slope_err:.4e}) · X_i + ({intercept:+.4f} ± {intercept_err:.4f})",
            "TEST",
        )
        print_status(f"  Slope significance: {slope_sigma:.2f}σ", "TEST")
        print_status(f"  Pearson r: {pearson_r:+.4f} (p={pearson_p:.4f})", "TEST")
        print_status(f"  Spearman ρ: {spearman_rho:+.4f} (p={spearman_p:.4f})", "TEST")
        print_status(f"  R² = {r_squared:.4f}, χ²/dof = {chi2_reduced:.2f}", "TEST")
        print_status(
            f"  TEP predicted slope: {tep_slope_default:.4e} (default), "
            f"{tep_slope_joint:.4e} (joint), {tep_slope_canonical:.4e} (canonical)",
            "TEST",
        )

        # Test: is the observed slope consistent with the TEP prediction?
        # (within 2σ of the predicted value)
        for label, tep_slope in [
            ("default", tep_slope_default),
            ("joint", tep_slope_joint),
            ("canonical", tep_slope_canonical),
        ]:
            diff = abs(slope - tep_slope)
            consistency_sigma = diff / slope_err if slope_err > 0 else 99
            print_status(
                f"  vs TEP ({label}): |slope - pred| / σ = {consistency_sigma:.2f}σ",
                "TEST",
            )

        results = {
            "n_galaxies": n,
            "slope": float(slope),
            "slope_err": float(slope_err),
            "slope_significance_sigma": slope_sigma,
            "intercept": float(intercept),
            "intercept_err": float(intercept_err),
            "r_squared": r_squared,
            "chi2": chi2,
            "chi2_reduced": float(chi2_reduced),
            "dof": dof,
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "spearman_rho": float(spearman_rho),
            "spearman_p": float(spearman_p),
            "tep_predicted_slope": {
                "default": float(tep_slope_default),
                "joint": float(tep_slope_joint),
                "canonical": float(tep_slope_canonical),
            },
            "tep_consistency_sigma": {
                "default": float(abs(slope - tep_slope_default) / slope_err) if slope_err > 0 else 99,
                "joint": float(abs(slope - tep_slope_joint) / slope_err) if slope_err > 0 else 99,
                "canonical": float(abs(slope - tep_slope_canonical) / slope_err) if slope_err > 0 else 99,
            },
        }

        # Unscreened variant (no S_total) for screening sensitivity
        if "X_i_unscreened" in df.columns:
            x_un = df["X_i_unscreened"].values
            S_u = np.sum(weights)
            Sx_u = np.sum(weights * x_un)
            Sxx_u = np.sum(weights * x_un * x_un)
            Sxy_u = np.sum(weights * x_un * y)
            denom_u = S_u * Sxx_u - Sx_u * Sx_u
            if abs(denom_u) > 0:
                slope_u = (S_u * Sxy_u - Sx_u * Sy) / denom_u
                intercept_u = (Sxx_u * Sy - Sx_u * Sxy_u) / denom_u
                slope_err_u = np.sqrt(S_u / denom_u)
            else:
                slope_u, intercept_u, slope_err_u = 0.0, 0.0, 0.0
            slope_sigma_u = float(abs(slope_u) / slope_err_u) if slope_err_u > 0 else 0.0
            print_status(
                f"  Unscreened fit: slope = {slope_u:.4e} ± {slope_err_u:.4e} "
                f"({slope_sigma_u:.2f}σ)",
                "TEST",
            )
            results["unscreened"] = {
                "slope": float(slope_u),
                "slope_err": float(slope_err_u),
                "slope_significance_sigma": slope_sigma_u,
                "intercept": float(intercept_u),
            }

        return results, slope, intercept, slope_err

    def regression_by_subset(self, df):
        """Run the Xi regression separately for R22-matched and non-R22 subsets.

        Both screened (S_total · U_i) and unscreened (U_i) potentials
        are fit so that the manuscript can report the screening
        sensitivity of each subset.
        """
        print_status("\nRunning Xi regression by subset...", "PROCESS")

        subset_results = {}
        for label, mask in [("r22_matched", df["r22_matched"] == True),
                            ("non_r22", df["r22_matched"] == False)]:
            subset = df[mask]
            if len(subset) < 3:
                print_status(f"  {label}: N={len(subset)} — too few for regression", "WARNING")
                subset_results[label] = {"n": len(subset), "slope": None, "slope_err": None}
                continue

            y = subset["delta_mu"].values
            yerr = subset["delta_mu_err"].values
            n = len(subset)

            subset_entry = {"n": n}
            for screen_label, xcol in [("screened", "X_i"),
                                       ("unscreened", "X_i_unscreened")]:
                x = subset[xcol].values

                weights = 1.0 / yerr ** 2
                S = np.sum(weights)
                Sx = np.sum(weights * x)
                Sy = np.sum(weights * y)
                Sxx = np.sum(weights * x * x)
                Sxy = np.sum(weights * x * y)

                denom = S * Sxx - Sx * Sx
                if abs(denom) > 0:
                    slope = (S * Sxy - Sx * Sy) / denom
                    intercept = (Sxx * Sy - Sx * Sxy) / denom
                    slope_err = np.sqrt(S / denom)
                else:
                    slope, intercept, slope_err = 0.0, 0.0, 0.0

                slope_sigma = float(abs(slope) / slope_err) if slope_err > 0 else 0.0
                pearson_r, pearson_p = sp_stats.pearsonr(x, y) if n >= 3 else (0, 1)

                print_status(
                    f"  {label} ({screen_label}, N={n}): "
                    f"slope = {slope:.4e} ± {slope_err:.4e} ({slope_sigma:.2f}σ), "
                    f"r = {pearson_r:+.4f}",
                    "TEST",
                )

                subset_entry[screen_label] = {
                    "slope": float(slope),
                    "slope_err": float(slope_err),
                    "slope_significance_sigma": slope_sigma,
                    "intercept": float(intercept),
                    "pearson_r": float(pearson_r),
                    "pearson_p": float(pearson_p),
                }

            subset_results[label] = subset_entry

        return subset_results

    def mass_potential_confound(self, df):
        """Check whether R22 and non-R22 subsamples have different X_i distributions."""
        print_status("\nChecking mass/potential confound...", "PROCESS")

        r22 = df[df["r22_matched"] == True]
        non_r22 = df[df["r22_matched"] == False]

        r22_xi = r22["X_i"].values
        non_r22_xi = non_r22["X_i"].values

        print_status(f"  R22-matched (N={len(r22)}): mean X_i = {np.mean(r22_xi):.4e}, "
                     f"median vrot = {r22['vrot_kms'].median():.1f} km/s", "TEST")
        print_status(f"  Non-R22 (N={len(non_r22)}): mean X_i = {np.mean(non_r22_xi):.4e}, "
                     f"median vrot = {non_r22['vrot_kms'].median():.1f} km/s", "TEST")

        # Welch's t-test on X_i distributions
        if len(r22_xi) >= 2 and len(non_r22_xi) >= 2:
            t_stat, t_p = sp_stats.ttest_ind(r22_xi, non_r22_xi, equal_var=False)
            print_status(f"  Welch's t-test on X_i: t = {t_stat:.2f}, p = {t_p:.4f}", "TEST")
        else:
            t_stat, t_p = 0, 1

        # Also compare mean Δμ
        r22_dmu = r22["delta_mu"].values
        non_r22_dmu = non_r22["delta_mu"].values
        print_status(f"  R22-matched: mean Δμ = {np.mean(r22_dmu):+.4f} mag", "TEST")
        print_status(f"  Non-R22: mean Δμ = {np.mean(non_r22_dmu):+.4f} mag", "TEST")

        # Key question: is the non-R22 sample actually LOWER potential than R22?
        # If so, TEP predicts smaller |Δμ| for non-R22, but we observe larger |Δμ|.
        # This would be a tension. If non-R22 has comparable or higher potential,
        # the pattern is consistent.
        r22_mean_vrot = r22["vrot_kms"].mean()
        non_r22_mean_vrot = non_r22["vrot_kms"].mean()
        print_status(f"  R22 mean vrot = {r22_mean_vrot:.1f} km/s", "TEST")
        print_status(f"  Non-R22 mean vrot = {non_r22_mean_vrot:.1f} km/s", "TEST")

        confound = {
            "r22_mean_xi": float(np.mean(r22_xi)),
            "non_r22_mean_xi": float(np.mean(non_r22_xi)),
            "r22_mean_vrot": float(r22_mean_vrot),
            "non_r22_mean_vrot": float(non_r22_mean_vrot),
            "r22_median_vrot": float(r22["vrot_kms"].median()),
            "non_r22_median_vrot": float(non_r22["vrot_kms"].median()),
            "r22_mean_delta_mu": float(np.mean(r22_dmu)),
            "non_r22_mean_delta_mu": float(np.mean(non_r22_dmu)),
            "welch_t_xi": float(t_stat),
            "welch_p_xi": float(t_p),
            "interpretation": (
                "If non-R22 has comparable or higher mean vrot than R22, "
                "the larger |Δμ| in non-R22 is consistent with TEP. "
                "If non-R22 has lower mean vrot, the pattern is in tension "
                "with the naive mass-scaling prediction."
            ),
        }
        return confound

    def xi_regression_teph0(self, df):
        """Xi regression on TEP-H0 raw data with screening.

        Uses raw SH0ES Cepheid distances and EDD/CCHP TRGB distances (not CF4
        registered), with the full TEP screening factor S_total from TEP-H0
        Step 03. This matches the methodology of TEP-H0 Step 20 (joint
        indicator model) and Step 44 (joint multi-block likelihood).

        TEP predicts: Δμ = μ_Cep - μ_TRGB = δm - κ_Cep * X_i
        where X_i = (S_total * u_phi^2 - U_ref) / c^2.
        The slope should be negative (= -κ_Cep).

        The TEP-H0 Step 44 joint multi-block likelihood combines:
          1. Redshift-distance block (N=31 Hubble-flow hosts)
          2. TRGB differential block (N=16 non-anchor calibrators)
          3. Geometric anchor block (N=2 independent anchors)
        to achieve κ_Cep = (0.400 ± 0.270) × 10^6 mag (1.48σ, correct sign).
        This simple regression alone is underpowered but provides a direct
        cross-check of the sign.
        """
        print_status("\nRunning Xi regression on TEP-H0 raw data (screened)...",
                     "PROCESS")

        y = df["delta_mu"].values
        yerr = df["delta_mu_err"].values
        n = len(df)

        tep_slope_default = -self.KAPPA_CEP_DEFAULT
        tep_slope_joint = -self.KAPPA_CEP_JOINT

        results = {}
        for label, xcol in [("screened", "X_i"), ("unscreened", "X_i_unscreened")]:
            x = df[xcol].values

            weights = 1.0 / yerr ** 2
            S = np.sum(weights)
            Sx = np.sum(weights * x)
            Sy = np.sum(weights * y)
            Sxx = np.sum(weights * x * x)
            Sxy = np.sum(weights * x * y)

            denom = S * Sxx - Sx * Sx
            if abs(denom) > 0:
                slope = (S * Sxy - Sx * Sy) / denom
                intercept = (Sxx * Sy - Sx * Sxy) / denom
                slope_err = np.sqrt(S / denom)
                intercept_err = np.sqrt(Sxx / denom)
            else:
                slope, intercept = 0.0, 0.0
                slope_err, intercept_err = 0.0, 0.0

            slope_sigma = float(abs(slope) / slope_err) if slope_err > 0 else 0.0
            pearson_r, pearson_p = sp_stats.pearsonr(x, y)

            sign_correct = bool(slope < 0)
            print_status(
                f"  TEP-H0 ({label}): slope = {slope:.4e} ± {slope_err:.4e} "
                f"({slope_sigma:.2f}σ), r = {pearson_r:+.4f}",
                "TEST",
            )
            print_status(
                f"    Sign: {'NEGATIVE (correct for TEP)' if slope < 0 else 'POSITIVE (wrong for TEP)'}",
                "TEST",
            )

            results[label] = {
                "slope": float(slope),
                "slope_err": float(slope_err),
                "slope_significance_sigma": slope_sigma,
                "intercept": float(intercept),
                "intercept_err": float(intercept_err),
                "pearson_r": float(pearson_r),
                "pearson_p": float(pearson_p),
                "sign_correct": sign_correct,
            }

        print_status(
            f"  TEP predicted slope: {tep_slope_default:.4e} (default), "
            f"{tep_slope_joint:.4e} (joint)",
            "TEST",
        )

        return {
            "n_galaxies": n,
            "screened": results["screened"],
            "unscreened": results["unscreened"],
            "tep_predicted_slope_default": float(tep_slope_default),
            "tep_predicted_slope_joint": float(tep_slope_joint),
            "data_source": (
                "Raw SH0ES Cepheid (R22) + EDD/CCHP TRGB (TEP-H0 Paper 11), "
                "with full TEP screening S_total from Step 03"
            ),
            "note": (
                "This regression uses the same data as TEP-H0 Step 20 and "
                "Step 44. The simple regression alone is underpowered "
                "(N=18, R^2<0.01); the joint multi-block likelihood (Step 44) "
                "combines redshift-distance and anchor constraints to achieve "
                "kappa_Cep = (0.400 +/- 0.270) x 10^6 mag (1.48sigma, correct sign)."
            ),
        }, results["screened"]["slope"], results["screened"]["intercept"], results["screened"]["slope_err"]

    def leave_one_out_weighted(self, df):
        """Leave-one-out analysis on the weighted mean Δμ."""
        print_status("\nRunning leave-one-out on weighted mean Δμ...", "PROCESS")

        y = df["delta_mu"].values
        yerr = df["delta_mu_err"].values
        n = len(df)

        # Full weighted mean
        w = 1.0 / yerr ** 2
        full_mean = np.sum(y * w) / np.sum(w)
        full_sem = np.sqrt(1.0 / np.sum(w))
        full_sigma = abs(full_mean / full_sem)

        print_status(f"  Full weighted mean: {full_mean:.4f} ± {full_sem:.4f} ({full_sigma:.2f}σ)", "TEST")

        # Leave-one-out
        loo_means = []
        loo_sigmas = []
        pgcs = df["PGC"].values
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            w_loo = 1.0 / yerr[mask] ** 2
            mean_loo = np.sum(y[mask] * w_loo) / np.sum(w_loo)
            sem_loo = np.sqrt(1.0 / np.sum(w_loo))
            sigma_loo = abs(mean_loo / sem_loo)
            loo_means.append(float(mean_loo))
            loo_sigmas.append(float(sigma_loo))

        # Find the most influential galaxy
        max_change_idx = np.argmax(np.abs(np.array(loo_means) - full_mean))
        print_status(
            f"  Most influential: PGC {pgcs[max_change_idx]} — "
            f"Δμ changes from {full_mean:.4f} to {loo_means[max_change_idx]:.4f} "
            f"({full_sigma:.2f}σ → {loo_sigmas[max_change_idx]:.2f}σ)",
            "TEST",
        )

        # Range of LOO sigmas
        print_status(
            f"  LOO sigma range: {min(loo_sigmas):.2f}σ to {max(loo_sigmas):.2f}σ",
            "TEST",
        )

        return {
            "full_weighted_mean": float(full_mean),
            "full_weighted_sem": float(full_sem),
            "full_weighted_sigma": float(full_sigma),
            "loo_means": loo_means,
            "loo_sigmas": loo_sigmas,
            "loo_pgcs": [int(p) for p in pgcs],
            "most_influential_pgc": int(pgcs[max_change_idx]),
            "min_loo_sigma": float(min(loo_sigmas)),
            "max_loo_sigma": float(max(loo_sigmas)),
        }

    def leave_one_out_regression(self, df):
        """Leave-one-out analysis on the Xi regression slope.

        This quantifies how much each individual galaxy influences the
        full-sample regression slope, making the manuscript claim
        "excluding M101 alone drops the slope from +5.36e5 to +1.04e5"
        fully reproducible.
        """
        print_status("\nRunning leave-one-out on Xi regression slope...", "PROCESS")

        x = df["X_i"].values
        y = df["delta_mu"].values
        yerr = df["delta_mu_err"].values
        n = len(df)
        pgcs = df["PGC"].values

        # Full sample regression
        w = 1.0 / yerr ** 2
        X = np.column_stack([x, np.ones(n)])
        W = np.diag(w)
        beta = np.linalg.lstsq(X.T @ W @ X, X.T @ W @ y, rcond=None)[0]
        full_slope = float(beta[0])

        # Leave-one-out
        loo_slopes = []
        loo_slope_errs = []
        loo_sigmas = []
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            x_loo = x[mask]
            y_loo = y[mask]
            w_loo = 1.0 / yerr[mask] ** 2
            X_loo = np.column_stack([x_loo, np.ones(n - 1)])
            W_loo = np.diag(w_loo)
            beta_loo = np.linalg.lstsq(
                X_loo.T @ W_loo @ X_loo, X_loo.T @ W_loo @ y_loo, rcond=None
            )[0]
            # Slope error from covariance
            cov = np.linalg.inv(X_loo.T @ W_loo @ X_loo)
            slope_err_loo = float(np.sqrt(cov[0, 0]))
            slope_loo = float(beta_loo[0])
            sigma_loo = abs(slope_loo / slope_err_loo) if slope_err_loo > 0 else 0
            loo_slopes.append(slope_loo)
            loo_slope_errs.append(slope_err_loo)
            loo_sigmas.append(float(sigma_loo))

        # Find most influential on slope
        slope_changes = [abs(s - full_slope) for s in loo_slopes]
        max_change_idx = int(np.argmax(slope_changes))

        print_status(
            f"  Full slope: {full_slope:.4e}", "TEST"
        )
        print_status(
            f"  Most influential on slope: PGC {pgcs[max_change_idx]} — "
            f"slope {full_slope:.4e} → {loo_slopes[max_change_idx]:.4e} "
            f"({loo_sigmas[max_change_idx]:.2f}σ)",
            "TEST",
        )

        # Print M101 specifically
        for i, pgc in enumerate(pgcs):
            if int(pgc) == 50063:
                print_status(
                    f"  M101 (PGC 50063) excluded: slope={loo_slopes[i]:.4e} "
                    f"± {loo_slope_errs[i]:.4e} ({loo_sigmas[i]:.2f}σ)",
                    "TEST",
                )

        return {
            "full_slope": full_slope,
            "loo_slopes": loo_slopes,
            "loo_slope_errs": loo_slope_errs,
            "loo_sigmas": loo_sigmas,
            "loo_pgcs": [int(p) for p in pgcs],
            "most_influential_pgc": int(pgcs[max_change_idx]),
            "m101_excluded_slope": next(
                (loo_slopes[i] for i, p in enumerate(pgcs) if int(p) == 50063), None
            ),
            "m101_excluded_slope_err": next(
                (loo_slope_errs[i] for i, p in enumerate(pgcs) if int(p) == 50063), None
            ),
            "m101_excluded_sigma": next(
                (loo_sigmas[i] for i, p in enumerate(pgcs) if int(p) == 50063), None
            ),
        }

    def power_calculation(self, df):
        """Power calculation for the R22-matched N=6 test."""
        print_status("\nRunning power calculation for N=6 R22-matched test...", "PROCESS")

        # Use the per-galaxy scatter from the full sample
        y = df["delta_mu"].values
        yerr = df["delta_mu_err"].values
        w = 1.0 / yerr ** 2
        weighted_mean = np.sum(y * w) / np.sum(w)
        weighted_sem = np.sqrt(1.0 / np.sum(w))

        # Per-galaxy typical error
        typical_err = np.median(yerr)

        # For a weighted mean with N galaxies each with error σ:
        # SEM = σ / sqrt(N)
        # Minimum detectable effect at 80% power, 5% significance (two-sided):
        # δ_min = (z_{1-α/2} + z_{1-β}) * SEM = (1.96 + 0.84) * σ / sqrt(N)
        # = 2.80 * σ / sqrt(N)
        from scipy.stats import norm
        z_alpha = norm.ppf(0.975)  # two-sided 5%
        z_beta = norm.ppf(0.80)  # 80% power
        factor = z_alpha + z_beta

        # For N=6 with typical_err
        n_r22 = 6
        sem_n6 = typical_err / np.sqrt(n_r22)
        mde_n6 = factor * sem_n6

        # What N is needed to detect 0.02, 0.03, 0.05 mag?
        results = {}
        for target_effect in [0.02, 0.03, 0.05]:
            # N needed: target_effect = factor * typical_err / sqrt(N)
            # sqrt(N) = factor * typical_err / target_effect
            # N = (factor * typical_err / target_effect)^2
            n_needed = int(np.ceil((factor * typical_err / target_effect) ** 2))
            print_status(
                f"  To detect {target_effect:.2f} mag at 80% power: N ≈ {n_needed}",
                "TEST",
            )
            results[f"n_needed_for_{target_effect:.2f}mag"] = n_needed

        print_status(
            f"  N=6 minimum detectable effect (80% power): {mde_n6:.3f} mag",
            "TEST",
        )
        print_status(f"  Typical per-galaxy error: {typical_err:.3f} mag", "TEST")

        return {
            "typical_per_galaxy_error_mag": float(typical_err),
            "n_r22_matched": n_r22,
            "sem_n6": float(sem_n6),
            "mde_n6_80pct_power": float(mde_n6),
            "factor_80pct_5pct": float(factor),
            **results,
        }

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    def plot_regression(self, df, slope, intercept, slope_err):
        """Generate the Xi vs Δμ regression figure."""
        fig, ax = plt.subplots(figsize=(8, 6))

        x = df["X_i"].values
        y = df["delta_mu"].values
        yerr = df["delta_mu_err"].values
        r22 = df["r22_matched"].values

        # Plot non-R22
        ax.errorbar(
            x[~r22], y[~r22], yerr=yerr[~r22],
            fmt="o", color="steelblue", ms=6, capsize=3,
            label=f"Non-R22 (N={sum(~r22)})", alpha=0.8,
        )
        # Plot R22
        ax.errorbar(
            x[r22], y[r22], yerr=yerr[r22],
            fmt="s", color="crimson", ms=6, capsize=3,
            label=f"R22-matched (N={sum(r22)})", alpha=0.8,
        )

        # Regression line
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, "k--", lw=1.5,
                label=f"Fit: slope = {slope:.2e} ± {slope_err:.2e}")

        # TEP predicted line (default κ_μ)
        tep_slope = -self.KAPPA_CEP_DEFAULT
        tep_intercept = 0  # TEP predicts zero intercept at X_i = 0
        y_tep = tep_slope * x_line + tep_intercept
        ax.plot(x_line, y_tep, "g-", lw=1.5, alpha=0.6,
                label=f"TEP prediction (κ = {self.KAPPA_CEP_DEFAULT:.3g} mag)")

        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        ax.set_xlabel("$X_i = (U_i - U_{\\rm ref}) / c^2$", fontsize=12)
        ax.set_ylabel("$\\Delta\\mu = \\mu_{\\rm Cep} - \\mu_{\\rm TRGB}$ (mag)", fontsize=12)
        ax.set_title("Cepheid–TRGB Distance Divergence vs Gravitational Potential", fontsize=13)
        ax.legend(fontsize=9, loc="upper right")

        fig.tight_layout()
        fig_path = self.figures / "step_36_xi_regression.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print_status(f"Figure saved to {fig_path}", "SUCCESS")

    def sensitivity_analysis(self, df):
        """Exclusion and hierarchical sensitivity tests for Table 3b.

        Computes:
        - Simple regression with M101 (PGC 51969) and NGC 5643 excluded
        - Hierarchical regression with a pipeline-offset indicator
          (β·X_i + γ·I_R22 + α), both Gaussian and Student-t (ν=4)
        """
        print_status("\nRunning sensitivity analysis (Table 3b)...", "PROCESS")

        results = {}

        # --- Identify M101 and NGC 5643 by PGC ---
        # The vrot catalog labels PGC 50063 as "M 101" and PGC 51969 as
        # "NGC 5643". These are the two highest-X_i R22-matched galaxies
        # and are the most influential in the leave-one-out analysis.
        exclude_pgcs = set()
        if "PGC" in df.columns:
            for pgc_id in [50063, 51969]:
                if pgc_id in df["PGC"].values:
                    exclude_pgcs.add(int(pgc_id))

        # --- Simple exclusion regression ---
        if exclude_pgcs:
            df_excl = df[~df["PGC"].isin(exclude_pgcs)]
        else:
            df_excl = df

        x_excl = df_excl["X_i"].values
        y_excl = df_excl["delta_mu"].values
        yerr_excl = df_excl["delta_mu_err"].values
        n_excl = len(df_excl)

        if n_excl >= 3:
            w = 1.0 / yerr_excl ** 2
            S = np.sum(w)
            Sx = np.sum(w * x_excl)
            Sy = np.sum(w * y_excl)
            Sxx = np.sum(w * x_excl * x_excl)
            Sxy = np.sum(w * x_excl * y_excl)
            denom = S * Sxx - Sx * Sx
            if abs(denom) > 0:
                slope_excl = (S * Sxy - Sx * Sy) / denom
                slope_err_excl = np.sqrt(S / denom)
            else:
                slope_excl, slope_err_excl = 0.0, 0.0
            sigma_excl = float(abs(slope_excl) / slope_err_excl) if slope_err_excl > 0 else 0.0
            print_status(
                f"  Excl. M101+NGC5643 (N={n_excl}): "
                f"slope = {slope_excl:.4e} ± {slope_err_excl:.4e} ({sigma_excl:.2f}σ)",
                "TEST",
            )
            results["exclusion_m101_ngc5643"] = {
                "n": n_excl,
                "excluded_pgcs": sorted(exclude_pgcs),
                "slope": float(slope_excl),
                "slope_err": float(slope_err_excl),
                "slope_significance_sigma": sigma_excl,
            }

        # --- Hierarchical regression: Δμ = β·X_i + γ·I_R22 + α ---
        # I_R22 = 1 for R22-matched galaxies, 0 otherwise
        if "r22_matched" in df.columns and len(df) >= 5:
            x = df["X_i"].values
            y = df["delta_mu"].values
            yerr = df["delta_mu_err"].values
            r22_ind = df["r22_matched"].astype(float).values
            n = len(df)

            # Weighted 3-parameter linear regression: y = β·x + γ·r22 + α
            # Design matrix: [x, r22, 1]
            A = np.column_stack([x, r22_ind, np.ones(n)])
            W = np.diag(1.0 / yerr ** 2)
            try:
                AtWA = A.T @ W @ A
                AtWy = A.T @ W @ y
                params = np.linalg.solve(AtWA, AtWy)
                cov = np.linalg.inv(AtWA)
                errs = np.sqrt(np.diag(cov))
                beta, gamma, alpha = params
                beta_err, gamma_err, alpha_err = errs
                beta_sigma = float(abs(beta) / beta_err) if beta_err > 0 else 0.0
                gamma_sigma = float(abs(gamma) / gamma_err) if gamma_err > 0 else 0.0

                print_status(
                    f"  Hierarchical (full, N={n}): "
                    f"β={beta:.4e}±{beta_err:.4e} ({beta_sigma:.2f}σ), "
                    f"γ={gamma:+.4f}±{gamma_err:.4f} ({gamma_sigma:.2f}σ)",
                    "TEST",
                )

                results["hierarchical_full"] = {
                    "n": n,
                    "slope": float(beta),
                    "slope_err": float(beta_err),
                    "slope_significance_sigma": beta_sigma,
                    "pipeline_offset": float(gamma),
                    "pipeline_offset_err": float(gamma_err),
                    "pipeline_offset_sigma": gamma_sigma,
                }

                # Hierarchical with exclusion
                if exclude_pgcs:
                    df_h_excl = df[~df["PGC"].isin(exclude_pgcs)]
                    xh = df_h_excl["X_i"].values
                    yh = df_h_excl["delta_mu"].values
                    yerrh = df_h_excl["delta_mu_err"].values
                    r22h = df_h_excl["r22_matched"].astype(float).values
                    nh = len(df_h_excl)
                    Ah = np.column_stack([xh, r22h, np.ones(nh)])
                    Wh = np.diag(1.0 / yerrh ** 2)
                    try:
                        params_h = np.linalg.solve(Ah.T @ Wh @ Ah, Ah.T @ Wh @ yh)
                        cov_h = np.linalg.inv(Ah.T @ Wh @ Ah)
                        errs_h = np.sqrt(np.diag(cov_h))
                        beta_h, gamma_h, alpha_h = params_h
                        beta_h_err = errs_h[0]
                        beta_h_sigma = float(abs(beta_h) / beta_h_err) if beta_h_err > 0 else 0.0
                        print_status(
                            f"  Hierarchical (excl, N={nh}): "
                            f"β={beta_h:.4e}±{beta_h_err:.4e} ({beta_h_sigma:.2f}σ)",
                            "TEST",
                        )
                        results["hierarchical_exclusion"] = {
                            "n": nh,
                            "slope": float(beta_h),
                            "slope_err": float(beta_h_err),
                            "slope_significance_sigma": beta_h_sigma,
                            "pipeline_offset": float(gamma_h),
                            "pipeline_offset_err": float(errs_h[1]),
                        }
                    except np.linalg.LinAlgError:
                        pass

                # Student-t (ν=4) robust regression on full sample
                # Iteratively reweighted least squares (IRWLS) for Student-t
                nu = 4
                beta_t, gamma_t, alpha_t = beta, gamma, alpha
                for iteration in range(50):
                    resid = y - (beta_t * x + gamma_t * r22_ind + alpha_t)
                    # Student-t weights
                    w_t = (nu + 1) / (nu + (resid / yerr) ** 2)
                    W_t = np.diag(w_t / yerr ** 2)
                    try:
                        AtWA_t = A.T @ W_t @ A
                        AtWy_t = A.T @ W_t @ y
                        params_t = np.linalg.solve(AtWA_t, AtWy_t)
                        cov_t = np.linalg.inv(AtWA_t)
                        errs_t = np.sqrt(np.diag(cov_t))
                        beta_t, gamma_t, alpha_t = params_t
                    except np.linalg.LinAlgError:
                        break
                beta_t_err = errs_t[0]
                beta_t_sigma = float(abs(beta_t) / beta_t_err) if beta_t_err > 0 else 0.0
                gamma_t_err = errs_t[1]
                gamma_t_sigma = float(abs(gamma_t) / gamma_t_err) if gamma_t_err > 0 else 0.0
                print_status(
                    f"  Student-t (ν=4, full, N={n}): "
                    f"β={beta_t:.4e}±{beta_t_err:.4e} ({beta_t_sigma:.2f}σ), "
                    f"γ={gamma_t:+.4f}±{gamma_t_err:.4f} ({gamma_t_sigma:.2f}σ)",
                    "TEST",
                )
                results["hierarchical_studentt_full"] = {
                    "n": n,
                    "nu": nu,
                    "slope": float(beta_t),
                    "slope_err": float(beta_t_err),
                    "slope_significance_sigma": beta_t_sigma,
                    "pipeline_offset": float(gamma_t),
                    "pipeline_offset_err": float(gamma_t_err),
                    "pipeline_offset_sigma": gamma_t_sigma,
                }
            except np.linalg.LinAlgError:
                print_status("  Hierarchical regression: singular matrix", "WARNING")

        return results

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------
    def run(self):
        print_status("=" * 60, "TITLE")
        print_status("Step 36: Xi Regression — Δμ vs Potential Coordinate", "TITLE")
        print_status("=" * 60, "TITLE")

        print_status(
            "This step tests whether the Cepheid-TRGB distance divergence "
            "scales with the TEP gravitational potential coordinate X_i "
            "across the 22-galaxy CF4 matched sample, rather than splitting "
            "by provenance (R22 vs non-R22). TEP predicts Delta_mu = "
            "kappa_mu * X_i with a non-zero negative slope, while a "
            "pipeline-offset systematic would produce a constant offset "
            "independent of X_i (slope approximately zero). The key "
            "discriminating observable is the weighted regression slope of "
            "Delta_mu versus X_i, with TEP requiring a statistically "
            "significant non-zero slope of the predicted sign.",
            "INFO",
        )

        # Load data
        cf4_df = self.load_cf4_galaxies()
        vrot_df = self.load_vrot_catalog()

        if cf4_df.empty or vrot_df.empty:
            print_status("Cannot proceed without data.", "ERROR")
            return

        # Merge
        df = self.merge_data(cf4_df, vrot_df)

        if len(df) < 5:
            print_status(f"Only {len(df)} galaxies matched — too few for analysis.", "ERROR")
            return

        print_status(f"\nFinal sample: {len(df)} galaxies with Δμ, vrot, and X_i", "SUCCESS")

        # Print summary table
        print_status("\n--- Galaxy Summary ---", "TEST")
        for _, row in df.sort_values("X_i").iterrows():
            r22_flag = "R22" if row["r22_matched"] else "   "
            print_status(
                f"  PGC {int(row['PGC']):>6} [{r22_flag}] {row['galaxy_name']:<25} "
                f"vrot={row['vrot_kms']:6.1f}  X_i={row['X_i']:+.4e}  "
                f"Δμ={row['delta_mu']:+.4f} ± {row['delta_mu_err']:.4f}",
                "TEST",
            )

        # 1. Xi regression (full sample)
        print_status(
            "Methodology: Weighted linear regression Delta_mu = slope * X_i "
            "+ intercept is performed on the full CF4 matched sample, with "
            "weights = 1/delta_mu_err^2. The TEP-predicted slope is "
            "-kappa_Cep (negative, as deeper potentials yield shorter "
            "Cepheid distances). Both screened (S_total * U_i) and "
            "unscreened (U_i) potential coordinates are evaluated to "
            "assess screening sensitivity.",
            "PROCESS",
        )
        reg_results, slope, intercept, slope_err = self.xi_regression(df)

        print_status(
            f"Interpretation: The full-sample regression slope of "
            f"{slope:.4e} ± {slope_err:.4e} "
            f"({reg_results['slope_significance_sigma']:.2f}σ) is compared "
            f"against the TEP-predicted slope of "
            f"{reg_results['tep_predicted_slope']['default']:.4e} (default). "
            f"A non-zero slope with the correct (negative) sign supports "
            f"the TEP interpretation, while a slope consistent with zero "
            f"supports a pipeline-offset systematic.",
            "TEST",
        )

        # 2. Regression by subset
        subset_results = self.regression_by_subset(df)

        # 3. Mass/potential confound check
        confound_results = self.mass_potential_confound(df)

        # 4. Leave-one-out on weighted mean
        loo_results = self.leave_one_out_weighted(df)

        # 4a. Leave-one-out on regression slope
        loo_regression_results = self.leave_one_out_regression(df)

        # 4b. Sensitivity analysis (Table 3b: exclusion + hierarchical)
        print_status(
            "Methodology: Sensitivity analysis comprises (1) simple "
            "regression with M101 and NGC 5643 excluded, (2) hierarchical "
            "regression Delta_mu = beta * X_i + gamma * I_R22 + alpha with "
            "a pipeline-offset indicator, and (3) Student-t (nu=4) robust "
            "regression via iteratively reweighted least squares to "
            "assess outlier influence on the slope.",
            "PROCESS",
        )
        sensitivity_results = self.sensitivity_analysis(df)

        # 5. Power calculation
        power_results = self.power_calculation(df)

        # 6. TEP-H0 comparison: run Xi regression on raw SH0ES + EDD/CCHP data
        teph0_df = self.load_teph0_data()
        teph0_results = None
        if not teph0_df.empty and len(teph0_df) >= 5:
            teph0_reg, teph0_slope, teph0_intercept, teph0_slope_err = \
                self.xi_regression_teph0(teph0_df)
            teph0_results = teph0_reg

        # Plot
        print_status(
            f"Interpretation: The Xi regression on {len(df)} galaxies "
            f"yields a slope of {slope:.4e} ± {slope_err:.4e}. The "
            f"leave-one-out analysis identifies PGC "
            f"{loo_regression_results['most_influential_pgc']} as the most "
            f"influential galaxy for the slope. The mass/potential "
            f"confound check (Welch p = {confound_results['welch_p_xi']:.4f}) "
            f"assesses whether R22 and non-R22 subsamples have comparable "
            f"X_i distributions. Under TEP, the slope should be non-zero "
            f"with negative sign; under a pipeline-offset systematic, the "
            f"slope should be consistent with zero.",
            "SUCCESS",
        )
        self.plot_regression(df, slope, intercept, slope_err)

        # Save JSON output
        output = {
            "step": "36_xi_regression",
            "description": (
                "Test whether Cepheid-TRGB distance divergence scales with "
                "the TEP gravitational potential coordinate X_i across the "
                "full 22-galaxy CF4 matched sample."
            ),
            "n_galaxies": len(df),
            "xi_regression": reg_results,
            "subset_regression": subset_results,
            "mass_potential_confound": confound_results,
            "leave_one_out": loo_results,
            "leave_one_out_regression": loo_regression_results,
            "sensitivity_analysis": sensitivity_results,
            "power_calculation": power_results,
            "tep_h0_comparison": teph0_results,
            "galaxy_table": [
                {
                    "PGC": int(row["PGC"]),
                    "galaxy_name": str(row.get("galaxy_name", "")),
                    "r22_matched": bool(row.get("r22_matched", False)),
                    "vrot_kms": float(row.get("vrot_kms", 0)),
                    "X_i": float(row["X_i"]),
                    "X_i_unscreened": float(row.get("X_i_unscreened", 0)),
                    "delta_mu": float(row["delta_mu"]),
                    "delta_mu_err": float(row.get("delta_mu_err", 0)),
                }
                for _, row in df.sort_values("X_i").iterrows()
            ],
            "constants": {
                "sigma_ref_kms": self.SIGMA_REF,
                "u_ref_kms2": self.U_REF,
                "c_kms": self.C_KMS,
                "kappa_cep_default_mag": self.KAPPA_CEP_DEFAULT,
                "kappa_cep_joint_mag": self.KAPPA_CEP_JOINT,
                "kappa_cep_canonical_mag": self.KAPPA_CEP_CANONICAL,
            },
            "data_source": {
                "cf4_table": "data/raw/external/cf4_table2.dat",
                "vrot_catalog": "data/raw/external/cf4_matched_galaxies_vrot.csv",
                "vrot_source": "HyperLEDA (Makarov et al. 2014, A&A 570, A13)",
                "vrot_query_date": "2026-08-28",
            },
            "output_files": [
                "results/outputs/step_36_xi_regression.json",
                "results/outputs/step_36_22galaxy_table.csv",
                "results/figures/step_36_xi_regression.png",
            ],
            "methodology": (
                "Weighted linear regression Delta_mu = slope * X_i + "
                "intercept on the 22-galaxy CF4 matched sample, with "
                "weights = 1/delta_mu_err^2. X_i = (S_total * U_i - U_ref) "
                "/ c^2 with screening from Tully 2015 group catalog. "
                "Subsets (R22 vs non-R22), leave-one-out, hierarchical "
                "regression with pipeline-offset indicator, Student-t "
                "(nu=4) robust regression, and power calculation for N=6."
            ),
            "provenance": {
                "data_sources": [
                    "CF4 Table 2 (cf4_table2.dat) — Cepheid and TRGB distance moduli",
                    "HyperLEDA rotation velocity catalog (cf4_matched_galaxies_vrot.csv)",
                    "Tully 2015 2MRS group catalog — screening factors",
                    "TEP-H0 Step 03 stratified screening factors",
                    "TEP-H0 raw SH0ES Cepheid + EDD/CCHP TRGB data for comparison",
                ],
                "pipeline_block": "Host mass and Xi regression",
            },
            "scientific_context": (
                "This step tests whether the Cepheid-TRGB distance "
                "divergence scales with the TEP gravitational potential "
                "coordinate X_i across the full 22-galaxy CF4 matched "
                "sample. TEP predicts Delta_mu = kappa_mu * X_i with a "
                "non-zero negative slope, while a pipeline-offset "
                "systematic would produce a constant offset independent "
                "of X_i (slope approximately zero). The regression "
                "discriminates between TEP (slope != 0) and reduction "
                "artifact (slope = 0)."
            ),
            "tep_prediction": (
                "Delta_mu = kappa_mu * X_i with a non-zero negative slope "
                "(deeper potential yields shorter Cepheid distance). The "
                "predicted slope magnitude is kappa_Cep ~ 0.365e6 mag "
                "(default), 0.400e6 mag (joint), or 0.960e6 mag (canonical)."
            ),
            "void_prediction": (
                "A pipeline-offset systematic produces a constant offset "
                "independent of X_i, yielding a regression slope "
                "consistent with zero. No physical scaling between "
                "Delta_mu and gravitational potential is expected."
            ),
            "downstream_consumers": [],
        }

        output_path = self.results / "step_36_xi_regression.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print_status(f"\nSummary saved to {output_path}", "SUCCESS")

        # Export the 22-galaxy table as a machine-readable CSV (referenced in
        # the manuscript as a released data product)
        csv_columns = [
            "PGC", "galaxy_name", "vrot_kms", "vrot_err_kms", "S_total",
            "X_i", "X_i_unscreened", "DM_cep", "DM_cep_err",
            "DM_trgb", "DM_trgb_err", "delta_mu", "delta_mu_err",
            "r22_matched",
        ]
        available_cols = [c for c in csv_columns if c in df.columns]
        csv_df = df[available_cols].sort_values("X_i").copy()
        csv_path = self.results / "step_36_22galaxy_table.csv"
        csv_df.to_csv(csv_path, index=False)
        print_status(f"Galaxy table CSV saved to {csv_path}", "SUCCESS")

        print_status("Step 36 complete", "SUCCESS")


if __name__ == "__main__":
    step = Step36XiRegression()
    step.run()
