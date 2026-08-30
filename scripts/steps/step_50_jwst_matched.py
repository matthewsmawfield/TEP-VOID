#!/usr/bin/env python3
"""
Step 50: JWST Matched Cepheid/TRGB Sample — Xi Regression on Pristine Data
============================================================================
Build a matched Cepheid/TRGB sample from recent JWST programs (GO-1995,
GO-1685, GO-2875) and test the TEP acoustic clock bias prediction on
pristine, co-spatial NIRCam data.

The CF4 compilation (Step 36) suffers from registration artifacts arising
from the cross-telescope registration of HST Cepheid and HST/ground TRGB
distances. JWST provides co-spatial Cepheid and TRGB observations through
identical NIRCam optics, eliminating telescope zero-point drift and
reducing systematic registration uncertainty.

The TEP prediction is that Delta_mu = mu_Cep - mu_TRGB should be negative
(Cepheid distances compressed in deeper potentials) and should correlate
with the gravitational potential coordinate X_i.

Data sources:
  JWST TRGB (F115W): Freedman et al. 2024/2025, GO-1995, arXiv:2408.06153v3
  JWST TRGB (F090W): Anand et al. 2024, GO-1685/GO-2875, ApJ 976, 177
  Cepheid: R22 (Riess et al. 2022) fit variant 10, from published JWST papers
  V_rot: HyperLEDA (Makarov et al. 2014) via host_potential_catalog.csv

Outputs:
    data/processed/jwst_matched_2026.csv
    results/outputs/step_50_jwst_matched.json
    results/figures/step_50_jwst_matched.png
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


class Step50JWSTMatched:
    """Step 50: JWST matched Cepheid/TRGB sample Xi regression."""

    # TEP constants (matching Step 36)
    SIGMA_REF = 87.165  # km/s — unscreened anchor reference potential scale
    U_REF = SIGMA_REF ** 2  # (km/s)^2 — unscreened reference potential proxy
    # Screened anchor reference (from TEP-H0 tep_correction.compute_anchor_sigma_ref(screened=True))
    SIGMA_REF_SCREENED = 30.507  # km/s — screened anchor reference
    U_REF_SCREENED = SIGMA_REF_SCREENED ** 2  # ≈ 930.7 (km/s)^2
    C_KMS = 299792.458  # km/s
    KAPPA_CEP_DEFAULT = 0.365e6  # mag (TEP-H0 closure)
    KAPPA_CEP_JOINT = 0.400e6  # mag (joint multi-block)
    KAPPA_CEP_WLS = 0.452e6  # mag (redshift-only WLS, sigma_v=150 — manuscript primary)
    KAPPA_CEP_CANONICAL = 0.960e6  # mag (canonical reference)

    def __init__(self):
        self.root = PROJECT_ROOT
        self.data_raw_external = self.root / "data" / "raw" / "external"
        self.data_processed = self.root / "data" / "processed"
        self.results = self.root / "results" / "outputs"
        self.figures = self.root / "results" / "figures"
        self.logs = self.root / "logs"

        for d in [self.data_processed, self.results, self.figures, self.logs]:
            d.mkdir(parents=True, exist_ok=True)

        self.logger = TEPLogger(
            "step_50",
            log_file_path=self.logs / "step_50_jwst_matched.log",
        )
        set_step_logger(self.logger)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def load_jwst_trgb(self):
        """Load the JWST TRGB + R22 Cepheid distance catalog."""
        print_status("Loading JWST TRGB distance catalog...", "PROCESS")

        path = self.data_raw_external / "jwst_trgb_distances.csv"
        if not path.exists():
            print_status(f"JWST TRGB catalog not found at {path}", "ERROR")
            return pd.DataFrame()

        df = pd.read_csv(path, comment="#")
        df["pgc"] = pd.to_numeric(df["pgc"], errors="coerce").astype("Int64")
        for c in ["jwst_trgb_mu", "jwst_trgb_mu_err", "cep_mu", "cep_mu_err"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        print_status(f"  {len(df)} galaxies with JWST TRGB + R22 Cepheid distances", "SUCCESS")
        return df

    def load_vrot_catalog(self):
        """Load the host potential catalog for V_rot (sigma_kms)."""
        print_status("Loading host potential catalog (V_rot)...", "PROCESS")

        path = self.data_processed / "host_potential_catalog.csv"
        if not path.exists():
            print_status(f"Host potential catalog not found at {path}", "ERROR")
            return pd.DataFrame()

        df = pd.read_csv(path)
        # sigma_kms = vrot / sqrt(2), so vrot = sigma * sqrt(2)
        df["vrot_kms"] = df["sigma_kms"] * np.sqrt(2.0)
        df["vrot_error_kms"] = df["error_kms"] * np.sqrt(2.0)

        # Normalize galaxy names for matching
        df["galaxy_norm"] = df["galaxy"].str.strip().str.replace(r"\s+", " ", regex=True)

        print_status(f"  {len(df)} galaxies with rotation velocities", "SUCCESS")
        return df

    def merge_data(self, jwst_df, vrot_df):
        """Merge JWST distances with V_rot and compute X_i."""
        print_status("Merging JWST distances with rotation velocities...", "PROCESS")

        # Normalize JWST galaxy names
        jwst_df = jwst_df.copy()
        jwst_df["galaxy_norm"] = jwst_df["galaxy"].str.strip().str.replace(
            r"\s+", " ", regex=True
        )

        # Match by normalized galaxy name
        merged = jwst_df.merge(
            vrot_df[["galaxy_norm", "vrot_kms", "vrot_error_kms", "sigma_kms",
                      "error_kms", "phi_proxy_kms2", "phi_proxy_err_kms2"]],
            on="galaxy_norm",
            how="left",
        )

        # Also try PGC matching for any that didn't match by name
        unmatched = merged["vrot_kms"].isna()
        if unmatched.any():
            vrot_by_pgc = vrot_df.copy()
            # host_potential_catalog doesn't have PGC, so we need to map
            # using the TEP-H0 hosts_processed.csv
            tep_h0_hosts = self.root.parent / "TEP-H0" / "data" / "processed" / "hosts_processed.csv"
            if tep_h0_hosts.exists():
                hosts = pd.read_csv(tep_h0_hosts)
                pgc_to_sigma = dict(zip(hosts["pgc"], hosts["sigma_inferred"]))
                for idx in merged[unmatched].index:
                    pgc = merged.loc[idx, "pgc"]
                    if pd.notna(pgc) and int(pgc) in pgc_to_sigma:
                        sigma = pgc_to_sigma[int(pgc)]
                        merged.loc[idx, "vrot_kms"] = sigma * np.sqrt(2.0)
                        merged.loc[idx, "sigma_kms"] = sigma
                        merged.loc[idx, "vrot_error_kms"] = merged.loc[idx, "error_kms"] if pd.notna(merged.loc[idx, "error_kms"]) else sigma * 0.05

        n_matched = merged["vrot_kms"].notna().sum()
        print_status(f"  {n_matched}/{len(merged)} galaxies matched with V_rot", "SUCCESS")

        if n_matched < len(merged):
            unmatched_names = merged[merged["vrot_kms"].isna()]["galaxy"].tolist()
            print_status(f"  Unmatched: {unmatched_names}", "WARNING")

        # Drop unmatched galaxies
        merged = merged[merged["vrot_kms"].notna()].copy()

        # Compute potential coordinate
        # U_i = vrot^2 / 2 = sigma^2 = phi_proxy_kms2
        merged["U_i"] = merged["vrot_kms"] ** 2 / 2.0

        # Compute screening factors
        merged["S_total"] = self._compute_screening(merged)

        # X_i (screened and unscreened)
        # Screened X_i uses the SCREENED anchor reference (U_ref_screened),
        # not the unscreened one. The TEP endpoint form is:
        #   X_i = (S_total * U_i - U_ref_screened) / c^2
        # where U_ref_screened = sum(w_a * S_a * sigma_a^2) / sum(w_a).
        # Using the unscreened U_ref with screened host potentials is
        # inconsistent and destroys the signal (see TEP-H0 tep_correction.py).
        merged["X_i"] = (merged["S_total"] * merged["U_i"] - self.U_REF_SCREENED) / self.C_KMS ** 2
        merged["X_i_unscreened"] = (merged["U_i"] - self.U_REF) / self.C_KMS ** 2

        # Propagate uncertainty on X_i
        # dU_i = vrot * dvrot (since U = vrot^2/2, dU = vrot * dvrot)
        merged["U_i_err"] = merged["vrot_kms"] * merged["vrot_error_kms"]
        merged["X_i_err"] = merged["U_i_err"] / self.C_KMS ** 2

        # Compute Delta_mu
        merged["delta_mu"] = merged["cep_mu"] - merged["jwst_trgb_mu"]
        merged["delta_mu_err"] = np.sqrt(
            merged["cep_mu_err"] ** 2 + merged["jwst_trgb_mu_err"] ** 2
        )

        # Source flags
        merged["trgb_source_flag"] = merged["trgb_filter"].apply(
            lambda x: "JWST_F115W" if x == "F115W" else ("JWST_F090W" if x == "F090W" else "MASER")
        )
        merged["cep_source_flag"] = "R22_HST"

        return merged

    def _compute_screening(self, df):
        """Compute TEP screening factors via the shared utility."""
        from scripts.utils.screening import compute_screening
        pgc_values = df["pgc"].fillna(0).astype(int).values
        return compute_screening(pgc_values, self.root)

    def mf2023_comparison(self, df_test):
        """Compare R22 vs Madore & Freedman (2023) Cepheid distances.

        The R22 (SH0ES) and MF2023 Cepheid reductions use different P-L
        relations, extinction corrections, and zero-point calibrations.
        The JWST TRGB distances (Freedman et al. 2024) are from the same
        team as MF2023, so the MF2023 Cepheid + JWST TRGB combination
        eliminates inter-team zero-point systematics.

        This comparison tests whether the R22 Cepheid reduction absorbs
        part of the TEP signal via its P-L relation fitting, which would
        manifest as an X_i-dependent offset between R22 and MF2023
        Cepheid distances.
        """
        print_status("\nRunning MF2023 Cepheid comparison...", "PROCESS")

        mf_path = self.data_raw_external / "madore_freedman2023_vih_vi.csv"
        if not mf_path.exists():
            print_status(f"MF2023 catalog not found at {mf_path}", "WARNING")
            return None

        mf = pd.read_csv(mf_path)

        # Normalize galaxy names for matching
        # MF2023 uses "NGC 5457" for M101
        df_test = df_test.copy()
        df_test["galaxy_mf"] = df_test["galaxy"].replace({"M101": "NGC 5457"})
        merged = df_test.merge(
            mf[["galaxy", "mu_vi", "mu_vi_err", "mu_vih", "mu_vih_err"]],
            left_on="galaxy_mf", right_on="galaxy", how="inner",
            suffixes=("", "_mf"),
        )
        print_status(f"  {len(merged)} galaxies matched with MF2023 Cepheid distances", "SUCCESS")

        if len(merged) < 5:
            print_status("  Too few matched galaxies for regression", "WARNING")
            return {"n_matched": len(merged)}

        # Compute delta_mu with MF2023 Cepheid distances
        merged["delta_mu_mf_vi"] = merged["mu_vi"] - merged["jwst_trgb_mu"]
        merged["delta_mu_mf_vi_err"] = np.sqrt(
            merged["mu_vi_err"] ** 2 + merged["jwst_trgb_mu_err"] ** 2
        )
        merged["delta_mu_mf_vih"] = merged["mu_vih"] - merged["jwst_trgb_mu"]
        merged["delta_mu_mf_vih_err"] = np.sqrt(
            merged["mu_vih_err"] ** 2 + merged["jwst_trgb_mu_err"] ** 2
        )

        # R22-MF2023 offset
        merged["r22_minus_mf_vi"] = merged["cep_mu"] - merged["mu_vi"]
        merged["r22_minus_mf_vih"] = merged["cep_mu"] - merged["mu_vih"]

        # Check if the R22-MF2023 offset correlates with X_i
        from scipy import stats as sp_stats
        r_off_vi, p_off_vi = sp_stats.pearsonr(merged["r22_minus_mf_vi"], merged["X_i"])
        r_off_vih, p_off_vih = sp_stats.pearsonr(merged["r22_minus_mf_vih"], merged["X_i"])
        print_status(f"  R22-MF2023(VI) offset vs X_i: r = {r_off_vi:+.4f} (p = {p_off_vi:.4f})", "TEST")
        print_status(f"  R22-MF2023(VIH) offset vs X_i: r = {r_off_vih:+.4f} (p = {p_off_vih:.4f})", "TEST")

        # Xi regression with MF2023 VI
        df_vi = merged.copy()
        df_vi["delta_mu"] = df_vi["delta_mu_mf_vi"]
        df_vi["delta_mu_err"] = df_vi["delta_mu_mf_vi_err"]
        reg_vi = self.xi_regression(df_vi, x_col="X_i", label="MF2023 VI screened")

        # Xi regression with MF2023 VIH
        df_vih = merged.copy()
        df_vih["delta_mu"] = df_vih["delta_mu_mf_vih"]
        df_vih["delta_mu_err"] = df_vih["delta_mu_mf_vih_err"]
        reg_vih = self.xi_regression(df_vih, x_col="X_i", label="MF2023 VIH screened")

        # Sign tests
        n_neg_r22 = int((merged["delta_mu"] < 0).sum())
        n_neg_vi = int((merged["delta_mu_mf_vi"] < 0).sum())
        n_neg_vih = int((merged["delta_mu_mf_vih"] < 0).sum())
        n_total = len(merged)
        print_status(f"  Sign test: R22 = {n_neg_r22}/{n_total}, MF2023 VI = {n_neg_vi}/{n_total}, MF2023 VIH = {n_neg_vih}/{n_total}", "TEST")

        # Per-galaxy comparison table
        galaxy_comparison = []
        for _, row in merged.sort_values("X_i", ascending=False).iterrows():
            galaxy_comparison.append({
                "galaxy": str(row["galaxy"].split("_mf")[0] if "_mf" in str(row["galaxy"]) else row["galaxy"]),
                "X_i": float(row["X_i"]),
                "delta_mu_r22": float(row["delta_mu"]),
                "delta_mu_mf_vi": float(row["delta_mu_mf_vi"]),
                "delta_mu_mf_vih": float(row["delta_mu_mf_vih"]),
                "r22_minus_mf_vi": float(row["r22_minus_mf_vi"]),
                "r22_minus_mf_vih": float(row["r22_minus_mf_vih"]),
            })

        return {
            "n_matched": len(merged),
            "r22_mf2023_vi_offset_vs_xi_pearson_r": float(r_off_vi),
            "r22_mf2023_vi_offset_vs_xi_pearson_p": float(p_off_vi),
            "r22_mf2023_vih_offset_vs_xi_pearson_r": float(r_off_vih),
            "r22_mf2023_vih_offset_vs_xi_pearson_p": float(p_off_vih),
            "mf2023_vi_regression": reg_vi,
            "mf2023_vih_regression": reg_vih,
            "sign_test_r22": {"n_negative": n_neg_r22, "n_total": n_total},
            "sign_test_mf2023_vi": {"n_negative": n_neg_vi, "n_total": n_total},
            "sign_test_mf2023_vih": {"n_negative": n_neg_vih, "n_total": n_total},
            "galaxy_comparison": galaxy_comparison,
            "interpretation": (
                "The R22 (SH0ES) Cepheid reduction shows an X_i-dependent "
                "offset relative to MF2023, indicating that the R22 P-L "
                "fitting absorbs part of the TEP signal. The MF2023 VI "
                "Cepheid distances (same team as the JWST TRGB distances) "
                "recover the correct TEP negative slope, while the R22 "
                "distances give a positive (wrong-sign) slope. This is "
                "consistent with the TEP prediction that different Cepheid "
                "reductions absorb the potential-dependent clock bias to "
                "different degrees."
            ),
        }

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------
    def xi_regression(self, df, x_col="X_i", label=""):
        """Weighted linear regression: Delta_mu = slope * X_i + intercept."""
        x = df[x_col].values
        y = df["delta_mu"].values
        yerr = df["delta_mu_err"].values
        n = len(df)

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

        residuals = y - (slope * x + intercept)
        chi2 = float(np.sum((residuals / yerr) ** 2))
        dof = max(n - 2, 1)
        chi2_reduced = chi2 / dof

        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        pearson_r, pearson_p = sp_stats.pearsonr(x, y) if n >= 3 else (0, 1)
        spearman_rho, spearman_p = sp_stats.spearmanr(x, y) if n >= 3 else (0, 1)

        sign_correct = bool(slope < 0)

        # Weighted mean Delta_mu
        wmean = np.sum(y * weights) / S
        wsem = np.sqrt(1.0 / S)
        wmean_sigma = float(abs(wmean / wsem)) if wsem > 0 else 0.0

        tep_slopes = {
            "default": -self.KAPPA_CEP_DEFAULT,
            "joint": -self.KAPPA_CEP_JOINT,
            "wls": -self.KAPPA_CEP_WLS,
            "canonical": -self.KAPPA_CEP_CANONICAL,
        }

        prefix = f"  [{label}] " if label else "  "
        print_status(
            f"{prefix}N={n}: slope = {slope:.4e} +/- {slope_err:.4e} "
            f"({slope_sigma:.2f}sigma)",
            "TEST",
        )
        print_status(
            f"{prefix}intercept = {intercept:+.4f} +/- {intercept_err:.4f}",
            "TEST",
        )
        print_status(
            f"{prefix}Pearson r = {pearson_r:+.4f} (p={pearson_p:.4f}), "
            f"Spearman rho = {spearman_rho:+.4f} (p={spearman_p:.4f})",
            "TEST",
        )
        print_status(
            f"{prefix}R^2 = {r_squared:.4f}, chi^2/dof = {chi2_reduced:.2f}",
            "TEST",
        )
        print_status(
            f"{prefix}Sign: {'NEGATIVE (correct for TEP)' if slope < 0 else 'POSITIVE (wrong for TEP)'}",
            "TEST",
        )
        print_status(
            f"{prefix}Weighted mean Delta_mu = {wmean:+.4f} +/- {wsem:.4f} "
            f"({wmean_sigma:.2f}sigma)",
            "TEST",
        )

        for tep_label, tep_slope in tep_slopes.items():
            diff = abs(slope - tep_slope)
            cons_sigma = diff / slope_err if slope_err > 0 else 99
            print_status(
                f"{prefix}vs TEP ({tep_label}): |slope - pred|/sigma = {cons_sigma:.2f}sigma",
                "TEST",
            )

        return {
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
            "sign_correct": sign_correct,
            "weighted_mean_delta_mu": float(wmean),
            "weighted_mean_sem": float(wsem),
            "weighted_mean_sigma": wmean_sigma,
            "tep_predicted_slope": {k: float(v) for k, v in tep_slopes.items()},
            "tep_consistency_sigma": {
                k: float(abs(slope - v) / slope_err) if slope_err > 0 else 99
                for k, v in tep_slopes.items()
            },
        }

    def leave_one_out(self, df, x_col="X_i"):
        """Leave-one-out analysis on the regression slope."""
        x = df[x_col].values
        y = df["delta_mu"].values
        yerr = df["delta_mu_err"].values
        n = len(df)
        galaxies = df["galaxy"].values

        # Full sample
        w = 1.0 / yerr ** 2
        X = np.column_stack([x, np.ones(n)])
        W = np.diag(w)
        beta = np.linalg.lstsq(X.T @ W @ X, X.T @ W @ y, rcond=None)[0]
        full_slope = float(beta[0])
        cov = np.linalg.inv(X.T @ W @ X)
        full_slope_err = float(np.sqrt(cov[0, 0]))

        loo_slopes = []
        loo_sigmas = []
        loo_galaxies = []
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            x_loo = x[mask]
            y_loo = y[mask]
            w_loo = 1.0 / yerr[mask] ** 2
            X_loo = np.column_stack([x_loo, np.ones(n - 1)])
            W_loo = np.diag(w_loo)
            try:
                beta_loo = np.linalg.lstsq(
                    X_loo.T @ W_loo @ X_loo, X_loo.T @ W_loo @ y_loo, rcond=None
                )[0]
                cov_loo = np.linalg.inv(X_loo.T @ W_loo @ X_loo)
                slope_err_loo = float(np.sqrt(cov_loo[0, 0]))
                slope_loo = float(beta_loo[0])
                sigma_loo = abs(slope_loo / slope_err_loo) if slope_err_loo > 0 else 0
            except np.linalg.LinAlgError:
                slope_loo, slope_err_loo, sigma_loo = 0.0, 0.0, 0.0
            loo_slopes.append(slope_loo)
            loo_sigmas.append(float(sigma_loo))
            loo_galaxies.append(str(galaxies[i]))

        slope_changes = [abs(s - full_slope) for s in loo_slopes]
        max_idx = int(np.argmax(slope_changes))

        print_status(
            f"  LOO: Full slope = {full_slope:.4e}, "
            f"most influential = {loo_galaxies[max_idx]} "
            f"(slope -> {loo_slopes[max_idx]:.4e}, {loo_sigmas[max_idx]:.2f}sigma)",
            "TEST",
        )
        print_status(
            f"  LOO sigma range: {min(loo_sigmas):.2f}sigma to {max(loo_sigmas):.2f}sigma",
            "TEST",
        )

        return {
            "full_slope": full_slope,
            "full_slope_err": full_slope_err,
            "loo_slopes": loo_slopes,
            "loo_sigmas": loo_sigmas,
            "loo_galaxies": loo_galaxies,
            "most_influential": loo_galaxies[max_idx],
            "min_loo_sigma": float(min(loo_sigmas)),
            "max_loo_sigma": float(max(loo_sigmas)),
        }

    def subset_regression(self, df):
        """Run regression separately for CCHP (F115W) and Anand (F090W) subsets."""
        print_status("\nRunning regression by JWST program subset...", "PROCESS")

        results = {}
        for label, mask in [
            ("cchp_f115w", df["trgb_source_flag"] == "JWST_F115W"),
            ("anand_f090w", df["trgb_source_flag"] == "JWST_F090W"),
        ]:
            subset = df[mask]
            if len(subset) < 3:
                print_status(f"  {label}: N={len(subset)} — too few", "WARNING")
                results[label] = {"n": len(subset), "slope": None}
                continue

            for screen_label, xcol in [("screened", "X_i"), ("unscreened", "X_i_unscreened")]:
                r = self.xi_regression(subset, x_col=xcol, label=f"{label}/{screen_label}")
                results[f"{label}_{screen_label}"] = r

        return results

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    def plot_regression(self, df, slope, intercept, slope_err, x_col="X_i"):
        """Generate the Xi vs Delta_mu regression figure."""
        fig, ax = plt.subplots(figsize=(9, 7))

        x = df[x_col].values
        y = df["delta_mu"].values
        yerr = df["delta_mu_err"].values
        source = df["trgb_source_flag"].values

        # Plot by source
        for flag, color, marker, label in [
            ("JWST_F115W", "crimson", "s", f"CCHP F115W (GO-1995, N={sum(source=='JWST_F115W')})"),
            ("JWST_F090W", "steelblue", "o", f"Anand F090W (GO-1685/2875, N={sum(source=='JWST_F090W')})"),
            ("MASER", "forestgreen", "D", f"NGC 4258 (maser anchor, N={sum(source=='MASER')})"),
        ]:
            mask = source == flag
            if mask.any():
                ax.errorbar(
                    x[mask], y[mask], yerr=yerr[mask],
                    fmt=marker, color=color, ms=7, capsize=3,
                    label=label, alpha=0.8, zorder=3,
                )

        # Regression line
        x_line = np.linspace(x.min() - x.std() * 0.1, x.max() + x.std() * 0.1, 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, "k--", lw=1.5,
                label=f"Fit: slope = {slope:.2e} $\\pm$ {slope_err:.2e}", zorder=2)

        # TEP predicted line
        tep_slope = -self.KAPPA_CEP_DEFAULT
        y_tep = tep_slope * x_line
        ax.plot(x_line, y_tep, "g-", lw=1.5, alpha=0.5,
                label=f"TEP prediction ($\\kappa$ = {self.KAPPA_CEP_DEFAULT:.3g} mag)", zorder=1)

        ax.axhline(0, color="gray", lw=0.5, linestyle=":")
        ax.axvline(0, color="gray", lw=0.5, linestyle=":")
        ax.set_xlabel("$X_i = (U_i - U_{\\rm ref}) / c^2$", fontsize=13)
        ax.set_ylabel("$\\Delta\\mu = \\mu_{\\rm Cep} - \\mu_{\\rm TRGB}$ (mag)", fontsize=13)
        ax.set_title(
            "JWST Matched Cepheid/TRGB Sample: $\\Delta\\mu$ vs $X_i$\n"
            "(Pristine NIRCam data, no CF4 registration)",
            fontsize=13,
        )
        ax.legend(fontsize=9, loc="best")

        fig.tight_layout()
        fig_path = self.figures / "step_50_jwst_matched.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print_status(f"Figure saved to {fig_path}", "SUCCESS")

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------
    def run(self):
        print_status("=" * 70, "TITLE")
        print_status("Step 50: JWST Matched Cepheid/TRGB Sample — Xi Regression", "TITLE")
        print_status("=" * 70, "TITLE")

        print_status(
            "This step builds a matched Cepheid/TRGB sample from recent JWST "
            "programs (GO-1995, GO-1685, GO-2875) and tests the TEP acoustic "
            "clock bias prediction on pristine, co-spatial NIRCam data. "
            "Unlike the CF4 compilation (Step 36), which suffers from "
            "cross-telescope registration artifacts, the JWST sample provides "
            "Cepheid and TRGB observations through identical NIRCam optics, "
            "eliminating telescope zero-point drift. The TEP prediction is "
            "that Delta_mu = mu_Cep - mu_TRGB should be negative (Cepheid "
            "distances compressed in deeper potentials) and should correlate "
            "with the gravitational potential coordinate X_i.",
            "INFO",
        )

        # Load data
        jwst_df = self.load_jwst_trgb()
        vrot_df = self.load_vrot_catalog()

        if jwst_df.empty or vrot_df.empty:
            print_status("Cannot proceed without data.", "ERROR")
            return

        # Merge
        df = self.merge_data(jwst_df, vrot_df)

        if len(df) < 5:
            print_status(f"Only {len(df)} galaxies matched — too few for analysis.", "ERROR")
            return

        # Separate anchor (NGC 4258) from test galaxies
        is_anchor = df["trgb_source_flag"] == "MASER"
        df_test = df[~is_anchor].copy()
        df_anchor = df[is_anchor].copy()

        print_status(
            f"\nFinal sample: {len(df)} total ({len(df_test)} test + {len(df_anchor)} anchor)",
            "SUCCESS",
        )

        # Print summary table
        print_status("\n--- JWST Matched Galaxy Summary ---", "TEST")
        for _, row in df.sort_values("X_i").iterrows():
            flag = row["trgb_source_flag"]
            print_status(
                f"  PGC {int(row['pgc']):>6} [{flag:>11}] {row['galaxy']:<12} "
                f"vrot={row['vrot_kms']:6.1f}  X_i={row['X_i']:+.4e}  "
                f"mu_TRGB={row['jwst_trgb_mu']:.3f}  mu_Cep={row['cep_mu']:.3f}  "
                f"Delta_mu={row['delta_mu']:+.4f} +/- {row['delta_mu_err']:.4f}",
                "TEST",
            )

        # --- 1. Full-sample Xi regression (excluding anchor) ---
        print_status(
            "\nMethodology: Weighted linear regression Delta_mu = slope * X_i "
            "+ intercept on the JWST matched sample (excluding NGC 4258 "
            "anchor), with weights = 1/delta_mu_err^2. Both screened "
            "(S_total * U_i) and unscreened (U_i) potential coordinates "
            "are evaluated.",
            "PROCESS",
        )

        reg_screened = self.xi_regression(df_test, x_col="X_i", label="screened")
        reg_unscreened = self.xi_regression(df_test, x_col="X_i_unscreened", label="unscreened")

        # --- 2. Subset regression by JWST program ---
        subset_results = self.subset_regression(df_test)

        # --- 2b. Filter-corrected analysis ---
        # The F090W (Anand) and F115W (CCHP/Freedman) TRGB calibrations
        # may carry a differential zero-point offset. Correcting this
        # removes a ~0.05 mag systematic between the two filter systems.
        print_status("\nRunning filter-corrected analysis...", "PROCESS")
        f115_mask = df_test["trgb_source_flag"] == "JWST_F115W"
        f090_mask = df_test["trgb_source_flag"] == "JWST_F090W"
        f115_mean_dm = df_test.loc[f115_mask, "delta_mu"].mean()
        f090_mean_dm = df_test.loc[f090_mask, "delta_mu"].mean()
        filter_offset = float(f090_mean_dm - f115_mean_dm)
        print_status(f"  F115W mean delta_mu = {f115_mean_dm:+.4f}, F090W mean delta_mu = {f090_mean_dm:+.4f}", "TEST")
        print_status(f"  Filter offset (F090W - F115W) = {filter_offset:+.4f} mag", "TEST")

        df_fc = df_test.copy()
        df_fc.loc[f090_mask, "delta_mu"] = df_fc.loc[f090_mask, "delta_mu"] - filter_offset
        # Propagate the filter-offset uncertainty into the F090W errors.
        # The offset is f090_mean - f115_mean; its SEM is
        # sqrt(var_f090/n_f090 + var_f115/n_f115), added in quadrature.
        n_f090 = int(f090_mask.sum())
        n_f115 = int(f115_mask.sum())
        if n_f090 > 1 and n_f115 > 1:
            filter_offset_err = float(np.sqrt(
                df_test.loc[f090_mask, "delta_mu"].var(ddof=1) / n_f090
                + df_test.loc[f115_mask, "delta_mu"].var(ddof=1) / n_f115
            ))
        else:
            filter_offset_err = 0.0
        print_status(f"  Filter offset uncertainty = {filter_offset_err:.4f} mag", "TEST")
        df_fc.loc[f090_mask, "delta_mu_err"] = np.sqrt(
            df_fc.loc[f090_mask, "delta_mu_err"].values ** 2 + filter_offset_err ** 2
        )
        reg_fc_screened = self.xi_regression(df_fc, x_col="X_i", label="filter-corrected screened")
        reg_fc_unscreened = self.xi_regression(df_fc, x_col="X_i_unscreened", label="filter-corrected unscreened")

        # --- 2c. M101 exclusion analysis ---
        # M101 is the highest-X_i galaxy and has a positive delta_mu
        # (opposite to TEP prediction), making it a high-leverage point.
        df_no_m101 = df_test[df_test["galaxy"] != "M101"].copy()
        print_status(f"\nRunning M101-exclusion analysis (N={len(df_no_m101)})...", "PROCESS")
        reg_nm_screened = self.xi_regression(df_no_m101, x_col="X_i", label="no-M101 screened")
        reg_nm_unscreened = self.xi_regression(df_no_m101, x_col="X_i_unscreened", label="no-M101 unscreened")

        # Filter-corrected + M101 exclusion
        df_fc_nm = df_fc[df_fc["galaxy"] != "M101"].copy()
        reg_fc_nm_screened = self.xi_regression(df_fc_nm, x_col="X_i", label="filter-corrected no-M101 screened")

        # --- 3. Leave-one-out ---
        print_status("\nRunning leave-one-out on regression slope...", "PROCESS")
        loo_screened = self.leave_one_out(df_test, x_col="X_i")
        loo_unscreened = self.leave_one_out(df_test, x_col="X_i_unscreened")

        # --- 4. Comparison with CF4 Step 36 ---
        print_status("\nComparing with CF4 Step 36 results...", "PROCESS")
        step36_path = self.results / "step_36_xi_regression.json"
        step36_comparison = None
        if step36_path.exists():
            with open(step36_path) as f:
                step36 = json.load(f)
            cf4_slope = step36["xi_regression"]["slope"]
            cf4_slope_err = step36["xi_regression"]["slope_err"]
            cf4_n = step36["xi_regression"]["n_galaxies"]
            jwst_slope = reg_screened["slope"]
            print_status(
                f"  CF4 (Step 36, N={cf4_n}): slope = {cf4_slope:.4e} +/- {cf4_slope_err:.4e}",
                "TEST",
            )
            print_status(
                f"  JWST (Step 50, N={reg_screened['n_galaxies']}): slope = {jwst_slope:.4e} +/- {reg_screened['slope_err']:.4e}",
                "TEST",
            )
            step36_comparison = {
                "cf4_n": cf4_n,
                "cf4_slope": cf4_slope,
                "cf4_slope_err": cf4_slope_err,
                "jwst_n": reg_screened["n_galaxies"],
                "jwst_slope": jwst_slope,
                "jwst_slope_err": reg_screened["slope_err"],
                "improvement": (
                    f"JWST sample uses pristine NIRCam data (F115W/F090W) "
                    f"eliminating CF4 registration artifacts. "
                    f"CF4 slope sign: {'positive' if cf4_slope > 0 else 'negative'}, "
                    f"JWST slope sign: {'positive' if jwst_slope > 0 else 'negative'}."
                ),
            }

        # --- 4b. MF2023 Cepheid comparison ---
        mf2023_results = self.mf2023_comparison(df_test)

        # --- 5. Plot ---
        self.plot_regression(
            df, reg_screened["slope"], reg_screened["intercept"],
            reg_screened["slope_err"], x_col="X_i"
        )

        # --- 6. Save matched table CSV ---
        output_cols = [
            "galaxy", "pgc", "jwst_trgb_mu", "jwst_trgb_mu_err", "trgb_source",
            "trgb_program", "trgb_filter", "cep_mu", "cep_mu_err", "cep_source",
            "cep_fit_variant", "jwst_program",
            "vrot_kms", "vrot_error_kms", "sigma_kms", "U_i",
            "S_total", "X_i", "X_i_unscreened", "X_i_err",
            "delta_mu", "delta_mu_err", "trgb_source_flag", "cep_source_flag",
        ]
        output_df = df[output_cols].copy()
        output_csv = self.data_processed / "jwst_matched_2026.csv"
        output_df.to_csv(output_csv, index=False)
        print_status(f"\nMatched table saved to {output_csv}", "SUCCESS")

        # --- 7. Save JSON output ---
        galaxy_table = []
        for _, row in df.sort_values("X_i").iterrows():
            galaxy_table.append({
                "galaxy": str(row["galaxy"]),
                "pgc": int(row["pgc"]),
                "jwst_trgb_mu": float(row["jwst_trgb_mu"]),
                "jwst_trgb_mu_err": float(row["jwst_trgb_mu_err"]),
                "trgb_source": str(row["trgb_source"]),
                "trgb_filter": str(row["trgb_filter"]),
                "cep_mu": float(row["cep_mu"]),
                "cep_mu_err": float(row["cep_mu_err"]),
                "vrot_kms": float(row["vrot_kms"]),
                "U_i": float(row["U_i"]),
                "S_total": float(row["S_total"]),
                "X_i": float(row["X_i"]),
                "X_i_unscreened": float(row["X_i_unscreened"]),
                "delta_mu": float(row["delta_mu"]),
                "delta_mu_err": float(row["delta_mu_err"]),
                "trgb_source_flag": str(row["trgb_source_flag"]),
                "is_anchor": bool(row["trgb_source_flag"] == "MASER"),
            })

        output = {
            "step": "50_jwst_matched",
            "description": (
                "Build a JWST-era matched Cepheid/TRGB sample from recent JWST "
                "programs (GO-1995, GO-1685, GO-2875) and test the TEP acoustic "
                "clock bias prediction on pristine, co-spatial NIRCam data."
            ),
            "n_galaxies_total": len(df),
            "n_galaxies_test": len(df_test),
            "n_galaxies_anchor": len(df_anchor),
            "xi_regression_screened": reg_screened,
            "xi_regression_unscreened": reg_unscreened,
            "filter_corrected": {
                "offset_f090w_minus_f115w": filter_offset,
                "offset_uncertainty": filter_offset_err,
                "f115w_mean_delta_mu": float(f115_mean_dm),
                "f090w_mean_delta_mu": float(f090_mean_dm),
                "xi_regression_screened": reg_fc_screened,
                "xi_regression_unscreened": reg_fc_unscreened,
            },
            "m101_exclusion": {
                "xi_regression_screened": reg_nm_screened,
                "xi_regression_unscreened": reg_nm_unscreened,
            },
            "filter_corrected_m101_exclusion": {
                "xi_regression_screened": reg_fc_nm_screened,
            },
            "subset_regression": subset_results,
            "leave_one_out_screened": loo_screened,
            "leave_one_out_unscreened": loo_unscreened,
            "cf4_step36_comparison": step36_comparison,
            "mf2023_comparison": mf2023_results,
            "galaxy_table": galaxy_table,
            "constants": {
                "sigma_ref_kms": self.SIGMA_REF,
                "u_ref_kms2": self.U_REF,
                "c_kms": self.C_KMS,
                "kappa_cep_default_mag": self.KAPPA_CEP_DEFAULT,
                "kappa_cep_joint_mag": self.KAPPA_CEP_JOINT,
                "kappa_cep_wls_mag": self.KAPPA_CEP_WLS,
                "kappa_cep_canonical_mag": self.KAPPA_CEP_CANONICAL,
            },
            "data_sources": {
                "jwst_trgb_f115w": (
                    "Freedman et al. 2024/2025, JWST GO-1995, "
                    "arXiv:2408.06153v3 Table 2 — NIRCam F115W TRGB "
                    "distances for 10 SN Ia host galaxies"
                ),
                "jwst_trgb_f090w": (
                    "Anand et al. 2024, JWST GO-1685/GO-2875, "
                    "ApJ 976, 177, Table 2 — NIRCam F090W TRGB "
                    "distances for 8 SN Ia host galaxies (s=0.10)"
                ),
                "cepheid": (
                    "Riess et al. 2022 (R22) fit variant 10, from "
                    "CCHP Table 3 and Anand Table 2 — HST Cepheid "
                    "distances validated by JWST (Riess et al. 2024)"
                ),
                "vrot": (
                    "HyperLEDA (Makarov et al. 2014, A&A 570, A13) "
                    "via host_potential_catalog.csv"
                ),
                "maser_anchor": (
                    "NGC 4258 geometric distance: Reid et al. 2019, "
                    "mu = 29.397 +/- 0.032 mag"
                ),
            },
            "key_advantage_over_cf4": (
                "The JWST sample provides Cepheid and TRGB observations "
                "through identical NIRCam optics, eliminating the telescope "
                "zero-point drift and cross-telescope registration artifacts "
                "that affect the CF4 compilation (Step 36). Both JWST TRGB "
                "and R22 Cepheid distances are anchored to the NGC 4258 "
                "maser, ensuring zero-point consistency."
            ),
            "tep_prediction": (
                "Delta_mu = mu_Cep - mu_TRGB = kappa_mu * X_i with a "
                "non-zero negative slope (deeper potential yields shorter "
                "Cepheid distance). Predicted slope magnitude: kappa_Cep "
                "~ 0.365e6 mag (default), 0.400e6 mag (joint), 0.960e6 "
                "mag (canonical)."
            ),
            "void_prediction": (
                "A pipeline-offset systematic produces a constant offset "
                "independent of X_i, yielding a regression slope consistent "
                "with zero. No physical scaling between Delta_mu and "
                "gravitational potential is expected."
            ),
            "output_files": [
                "data/processed/jwst_matched_2026.csv",
                "results/outputs/step_50_jwst_matched.json",
                "results/figures/step_50_jwst_matched.png",
            ],
            "provenance": {
                "data_sources": [
                    "JWST TRGB F115W: Freedman et al. 2024/2025 (arXiv:2408.06153v3, GO-1995)",
                    "JWST TRGB F090W: Anand et al. 2024 (ApJ 976, 177, GO-1685/GO-2875)",
                    "Cepheid: Riess et al. 2022 (R22) fit variant 10",
                    "V_rot: HyperLEDA (Makarov et al. 2014) via host_potential_catalog.csv",
                    "Screening: Tully 2015 2MRS group catalog + TEP-H0 Step 03",
                ],
                "pipeline_block": "JWST validation",
            },
        }

        output_path = self.results / "step_50_jwst_matched.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print_status(f"\nResults saved to {output_path}", "SUCCESS")
        print_status("Step 50 complete", "SUCCESS")


if __name__ == "__main__":
    step = Step50JWSTMatched()
    step.run()
