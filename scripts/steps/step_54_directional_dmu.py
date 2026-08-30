#!/usr/bin/env python3
"""
Step 54: Directional Δμ Analysis — Bulk Flow as Temporal Topology Gradient
=============================================================================
Test whether the Cepheid-TRGB distance modulus offset Δμ shows a directional
pattern aligned with the CMB dipole, as predicted by TEP if the "bulk flow"
is a dipolar temporal topology gradient rather than kinetic motion.

Discriminating test:
  - TEP prediction: Δμ = κ·X_i + D·(n̂·n̂_CMB) + const
    The acoustic clock (Cepheid) is biased by the temporal shear gradient;
    the nuclear candle (TRGB) is not.  The dipolar amplitude D should be
    non-zero and aligned with the CMB dipole.
  - Kinematic prediction: Δμ = κ·X_i + const (no directional term)
    Luminosity distances are direction-independent; peculiar velocities
    affect redshift-based distances, not luminosity distances.

Analyses:
  1. Primary correlation: r(Δμ, cmb_dot) with permutation significance
  2. 3-D vector dipole fit to Δμ
  3. Partial correlations controlling for X_i and R22 membership
  4. Joint model Δμ = κ·X_i + D·cmb_dot + const with F-test
  5. Subsample analysis (CF4 vs JWST, R22 vs non-R22)
  6. Indicator-specific bulk flow comparison (TRGB vs Cepheid vs TF vs FP)
  7. Sky-coverage diagnostic figure

Outputs:
    results/outputs/step_54_directional_dmu.json
    results/figures/step_54_directional_dmu.png
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.projections.geo import AitoffAxes
from scipy import stats as sp_stats
from numpy.linalg import lstsq
from astropy.coordinates import SkyCoord
import astropy.units as u

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status

# CMB dipole (Planck 2018)
CMB_DIPOLE_GAL_L = 264.021
CMB_DIPOLE_GAL_B = 48.253
C_KMS = 299792.458


def ra_dec_to_unit_vectors(ra_deg, dec_deg):
    ra_rad = np.radians(ra_deg)
    dec_rad = np.radians(dec_deg)
    return np.column_stack([
        np.cos(dec_rad) * np.cos(ra_rad),
        np.cos(dec_rad) * np.sin(ra_rad),
        np.sin(dec_rad),
    ])


def cmb_dipole_unit_vector():
    c = SkyCoord(l=CMB_DIPOLE_GAL_L * u.deg, b=CMB_DIPOLE_GAL_B * u.deg,
                 frame="galactic").icrs
    return np.array([
        np.cos(np.radians(c.dec.deg)) * np.cos(np.radians(c.ra.deg)),
        np.cos(np.radians(c.dec.deg)) * np.sin(np.radians(c.ra.deg)),
        np.sin(np.radians(c.dec.deg)),
    ])


class Step54DirectionalDmu:
    """Step 54: Directional Δμ analysis — the discriminating test."""

    def __init__(self):
        self.root = PROJECT_ROOT
        self.data_raw_external = self.root / "data" / "raw" / "external"
        self.data_processed = self.root / "data" / "processed"
        self.results = self.root / "results" / "outputs"
        self.figures = self.root / "results" / "figures"
        self.logs = self.root / "logs"

        for d in [self.results, self.figures, self.logs]:
            d.mkdir(parents=True, exist_ok=True)

        self.logger = TEPLogger(
            "step_54",
            log_file_path=self.logs / "step_54_directional_dmu.log",
        )
        set_step_logger(self.logger)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def load_sample(self):
        """Load the compiled directional sample from Step 53."""
        print_status("Loading directional Cepheid-TRGB sample...", "PROCESS")
        path = self.data_processed / "directional_ceph_trgb_sample.csv"
        if not path.exists():
            print_status(f"Sample not found at {path}. Run step_53 first.", "ERROR")
            return pd.DataFrame()

        df = pd.read_csv(path)
        print_status(f"  {len(df)} galaxies loaded", "SUCCESS")
        return df

    def load_cf4_for_bulk_flow(self):
        """Load CF4 table2 for indicator-specific bulk flow comparison."""
        print_status("Loading CF4 table2 for bulk flow comparison...", "PROCESS")
        path = self.data_raw_external / "cf4_table2.dat"
        if not path.exists():
            print_status(f"CF4 table2 not found at {path}", "WARNING")
            return None

        colspecs = [
            (0, 7), (22, 27), (28, 34), (35, 40),
            (41, 47), (48, 52), (53, 59), (60, 64),
            (65, 71), (72, 76), (77, 83), (84, 89),
            (90, 96), (97, 101), (102, 107), (108, 112),
            (113, 119), (120, 125), (126, 131), (132, 136),
            (137, 145), (146, 154), (155, 163), (164, 172),
            (173, 181), (182, 190),
        ]
        names = ["PGC", "Vcmb", "DM", "e_DM",
                 "DMsnIa", "e_DMsnIa", "DMtf", "e_DMtf",
                 "DMfp", "e_DMfp", "DMsbf", "e_DMsbf",
                 "DMsnII", "e_DMsnII", "DMtrgb", "e_DMtrgb",
                 "DMceph", "e_DMceph", "DMmas", "e_DMmas",
                 "RAdeg", "DEdeg", "GLON", "GLAT", "SGL", "SGB"]
        df = pd.read_fwf(path, colspecs=colspecs, names=names, header=None)
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["PGC"] = df["PGC"].astype("Int64")
        print_status(f"  {len(df)} CF4 galaxies", "SUCCESS")
        return df

    # ------------------------------------------------------------------
    # Analysis 1: Primary directional correlation
    # ------------------------------------------------------------------
    def directional_correlation(self, df):
        """Test r(Δμ, cmb_dot) with permutation significance."""
        print_status("\n--- Analysis 1: Primary directional correlation ---", "PROCESS")

        valid = df["delta_mu"].notna() & df["cmb_dot"].notna()
        dmu = df.loc[valid, "delta_mu"].values
        cmb_dot = df.loc[valid, "cmb_dot"].values
        n = len(dmu)

        r_obs, p_pearson = sp_stats.pearsonr(cmb_dot, dmu)
        rho_obs, p_spearman = sp_stats.spearmanr(cmb_dot, dmu)

        # Permutation test (100 000 permutations)
        np.random.seed(42)
        n_perm = 100000
        r_perm = np.zeros(n_perm)
        for i in range(n_perm):
            idx = np.random.permutation(n)
            r_perm[i], _ = sp_stats.pearsonr(cmb_dot, dmu[idx])
        p_perm_two = (np.abs(r_perm) >= np.abs(r_obs)).mean()
        p_perm_one = (r_perm >= r_obs).mean()

        # Split by direction
        toward = cmb_dot > 0
        away = cmb_dot <= 0
        dmu_toward = dmu[toward]
        dmu_away = dmu[away]

        # Permutation test for mean difference
        diff_obs = dmu_toward.mean() - dmu_away.mean()
        diff_perm = np.zeros(n_perm)
        for i in range(n_perm):
            idx = np.random.permutation(n)
            dmu_s = dmu[idx]
            diff_perm[i] = dmu_s[toward].mean() - dmu_s[away].mean()
        p_diff = (diff_perm >= diff_obs).mean()
        sigma_diff = abs(diff_obs - diff_perm.mean()) / diff_perm.std()

        result = {
            "n": int(n),
            "pearson_r": float(r_obs),
            "pearson_p": float(p_pearson),
            "spearman_rho": float(rho_obs),
            "spearman_p": float(p_spearman),
            "permutation_p_two_sided": float(p_perm_two),
            "permutation_p_one_sided": float(p_perm_one),
            "n_toward": int(toward.sum()),
            "n_away": int(away.sum()),
            "mean_dmu_toward": float(dmu_toward.mean()),
            "sem_dmu_toward": float(dmu_toward.std() / np.sqrt(toward.sum())),
            "mean_dmu_away": float(dmu_away.mean()),
            "sem_dmu_away": float(dmu_away.std() / np.sqrt(away.sum())),
            "mean_difference": float(diff_obs),
            "mean_difference_sem": float(np.sqrt(
                dmu_toward.std()**2 / toward.sum() +
                dmu_away.std()**2 / away.sum()
            )),
            "permutation_p_diff": float(p_diff),
            "permutation_sigma_diff": float(sigma_diff),
        }

        print_status(f"  N={n}: r(Δμ, cmb_dot) = {r_obs:+.4f} (p={p_pearson:.4f})", "TEST")
        print_status(f"  Permutation: p_two={p_perm_two:.4f}, p_one={p_perm_one:.4f}", "TEST")
        print_status(f"  Toward CMB (N={toward.sum()}): Δμ = {dmu_toward.mean():.3f} ± {dmu_toward.std()/np.sqrt(toward.sum()):.3f}", "TEST")
        print_status(f"  Away from CMB (N={away.sum()}): Δμ = {dmu_away.mean():.3f} ± {dmu_away.std()/np.sqrt(away.sum()):.3f}", "TEST")
        print_status(f"  Difference: {diff_obs:.3f} mag ({sigma_diff:.2f}σ, p={p_diff:.4f})", "TEST")

        return result

    # ------------------------------------------------------------------
    # Analysis 2: 3-D vector dipole fit
    # ------------------------------------------------------------------
    def dipole_fit_3d(self, df):
        """Fit a 3-D vector dipole to Δμ and compare direction with CMB dipole."""
        print_status("\n--- Analysis 2: 3-D vector dipole fit ---", "PROCESS")

        valid = df["delta_mu"].notna() & df["ra"].notna() & df["dec"].notna()
        dmu = df.loc[valid, "delta_mu"].values
        g3d = ra_dec_to_unit_vectors(df.loc[valid, "ra"].values,
                                      df.loc[valid, "dec"].values)
        n = len(dmu)

        # Fit Δμ = D·n̂ + const
        X = np.column_stack([g3d, np.ones(n)])
        beta = lstsq(X, dmu, rcond=None)[0]
        D_vec = beta[:3]
        D_mag = np.linalg.norm(D_vec)

        # Direction of fitted dipole
        D_ra = np.degrees(np.arctan2(D_vec[1], D_vec[0])) % 360
        D_dec = np.degrees(np.arcsin(D_vec[2] / max(D_mag, 1e-10)))
        D_coord = SkyCoord(ra=D_ra * u.deg, dec=D_dec * u.deg, frame="icrs")
        D_gal = D_coord.galactic

        # CMB dipole
        cmb_coord = SkyCoord(l=CMB_DIPOLE_GAL_L * u.deg, b=CMB_DIPOLE_GAL_B * u.deg,
                             frame="galactic").icrs
        sep = D_coord.separation(cmb_coord)

        # Permutation significance
        np.random.seed(42)
        null_mags = []
        for _ in range(10000):
            idx = np.random.choice(n, n, replace=False)
            beta_null = lstsq(X, dmu[idx], rcond=None)[0]
            null_mags.append(np.linalg.norm(beta_null[:3]))
        null_mags = np.array(null_mags)
        p_dipole = (null_mags >= D_mag).mean()
        sigma_dipole = (D_mag - null_mags.mean()) / null_mags.std()

        # Equivalent velocity
        v_equiv = D_mag * np.log(10) / 5 * C_KMS

        result = {
            "n": int(n),
            "dipole_amplitude_mag": float(D_mag),
            "dipole_ra_deg": float(D_ra),
            "dipole_dec_deg": float(D_dec),
            "dipole_gal_l": float(D_gal.l.deg),
            "dipole_gal_b": float(D_gal.b.deg),
            "cmb_dipole_gal_l": CMB_DIPOLE_GAL_L,
            "cmb_dipole_gal_b": CMB_DIPOLE_GAL_B,
            "angular_separation_deg": float(sep.deg),
            "equivalent_velocity_kms": float(v_equiv),
            "permutation_p": float(p_dipole),
            "permutation_sigma": float(sigma_dipole),
        }

        print_status(f"  Dipole amplitude: {D_mag:.4f} mag ({v_equiv:.0f} km/s equiv)", "TEST")
        print_status(f"  Direction: l={D_gal.l.deg:.1f}°, b={D_gal.b.deg:+.1f}°", "TEST")
        print_status(f"  CMB dipole: l={CMB_DIPOLE_GAL_L:.1f}°, b={CMB_DIPOLE_GAL_B:+.1f}°", "TEST")
        print_status(f"  Angular separation: {sep.deg:.1f}°", "TEST")
        print_status(f"  Permutation: p={p_dipole:.4f} ({sigma_dipole:.2f}σ)", "TEST")

        return result

    # ------------------------------------------------------------------
    # Analysis 3: Partial correlations
    # ------------------------------------------------------------------
    def partial_correlations(self, df):
        """Partial correlations controlling for X_i and R22 membership."""
        print_status("\n--- Analysis 3: Partial correlations ---", "PROCESS")

        valid = df["delta_mu"].notna() & df["cmb_dot"].notna() & df["X_i"].notna()
        dmu = df.loc[valid, "delta_mu"].values
        cmb_dot = df.loc[valid, "cmb_dot"].values
        xi = df.loc[valid, "X_i"].values
        r22 = df.loc[valid, "r22_matched"].astype(float).values if "r22_matched" in df.columns else np.zeros(valid.sum())
        n = len(dmu)

        results = {}

        # Simple correlations
        r_dmu_cmb, p_dmu_cmb = sp_stats.pearsonr(cmb_dot, dmu)
        r_dmu_xi, p_dmu_xi = sp_stats.pearsonr(xi, dmu)
        r_xi_cmb, p_xi_cmb = sp_stats.pearsonr(xi, cmb_dot)
        results["simple"] = {
            "r_dmu_cmb": float(r_dmu_cmb), "p_dmu_cmb": float(p_dmu_cmb),
            "r_dmu_xi": float(r_dmu_xi), "p_dmu_xi": float(p_dmu_xi),
            "r_xi_cmb": float(r_xi_cmb), "p_xi_cmb": float(p_xi_cmb),
        }

        # Partial: Δμ vs cmb_dot controlling for X_i
        Z = np.column_stack([np.ones(n), xi])
        by = lstsq(Z, dmu, rcond=None)[0]
        bz = lstsq(Z, cmb_dot, rcond=None)[0]
        y_resid = dmu - Z @ by
        z_resid = cmb_dot - Z @ bz
        r_partial_xi, p_partial_xi = sp_stats.pearsonr(z_resid, y_resid)
        results["partial_dmu_cmb_given_xi"] = {
            "r": float(r_partial_xi), "p": float(p_partial_xi),
        }

        # Partial: Δμ vs cmb_dot controlling for R22
        Z_r22 = np.column_stack([np.ones(n), r22])
        by_r22 = lstsq(Z_r22, dmu, rcond=None)[0]
        bz_r22 = lstsq(Z_r22, cmb_dot, rcond=None)[0]
        y_resid_r22 = dmu - Z_r22 @ by_r22
        z_resid_r22 = cmb_dot - Z_r22 @ bz_r22
        r_partial_r22, p_partial_r22 = sp_stats.pearsonr(z_resid_r22, y_resid_r22)
        results["partial_dmu_cmb_given_r22"] = {
            "r": float(r_partial_r22), "p": float(p_partial_r22),
        }

        # Partial: Δμ vs cmb_dot controlling for X_i AND R22
        Z_both = np.column_stack([np.ones(n), xi, r22])
        by_both = lstsq(Z_both, dmu, rcond=None)[0]
        bz_both = lstsq(Z_both, cmb_dot, rcond=None)[0]
        y_resid_both = dmu - Z_both @ by_both
        z_resid_both = cmb_dot - Z_both @ bz_both
        r_partial_both, p_partial_both = sp_stats.pearsonr(z_resid_both, y_resid_both)
        results["partial_dmu_cmb_given_xi_r22"] = {
            "r": float(r_partial_both), "p": float(p_partial_both),
        }

        # Partial: Δμ vs X_i controlling for cmb_dot
        Z_cmb = np.column_stack([np.ones(n), cmb_dot])
        by_cmb = lstsq(Z_cmb, dmu, rcond=None)[0]
        bx_cmb = lstsq(Z_cmb, xi, rcond=None)[0]
        y_resid_cmb = dmu - Z_cmb @ by_cmb
        x_resid_cmb = xi - Z_cmb @ bx_cmb
        r_partial_xi_given_cmb, p_partial_xi_given_cmb = sp_stats.pearsonr(x_resid_cmb, y_resid_cmb)
        results["partial_dmu_xi_given_cmb"] = {
            "r": float(r_partial_xi_given_cmb), "p": float(p_partial_xi_given_cmb),
        }

        print_status(f"  r(Δμ, cmb_dot) = {r_dmu_cmb:+.4f} (p={p_dmu_cmb:.4f})", "TEST")
        print_status(f"  r(Δμ, X_i)     = {r_dmu_xi:+.4f} (p={p_dmu_xi:.4f})", "TEST")
        print_status(f"  r(X_i, cmb_dot)= {r_xi_cmb:+.4f} (p={p_xi_cmb:.4f})", "TEST")
        print_status(f"  Partial r(Δμ, cmb_dot | X_i)     = {r_partial_xi:+.4f} (p={p_partial_xi:.4f})", "TEST")
        print_status(f"  Partial r(Δμ, cmb_dot | R22)     = {r_partial_r22:+.4f} (p={p_partial_r22:.4f})", "TEST")
        print_status(f"  Partial r(Δμ, cmb_dot | X_i,R22) = {r_partial_both:+.4f} (p={p_partial_both:.4f})", "TEST")
        print_status(f"  Partial r(Δμ, X_i | cmb_dot)     = {r_partial_xi_given_cmb:+.4f} (p={p_partial_xi_given_cmb:.4f})", "TEST")

        return results

    # ------------------------------------------------------------------
    # Analysis 4: Joint model with F-test
    # ------------------------------------------------------------------
    def joint_model(self, df):
        """Fit Δμ = κ·X_i + D·cmb_dot + const and F-test the directional term."""
        print_status("\n--- Analysis 4: Joint model with F-test ---", "PROCESS")

        valid = df["delta_mu"].notna() & df["cmb_dot"].notna() & df["X_i"].notna()
        dmu = df.loc[valid, "delta_mu"].values
        cmb_dot = df.loc[valid, "cmb_dot"].values
        xi = df.loc[valid, "X_i"].values
        n = len(dmu)

        # TEP model: Δμ = κ·X_i + D·cmb_dot + const
        X_tep = np.column_stack([xi, cmb_dot, np.ones(n)])
        beta_tep = lstsq(X_tep, dmu, rcond=None)[0]
        r2_tep = 1 - np.sum((dmu - X_tep @ beta_tep) ** 2) / np.sum((dmu - dmu.mean()) ** 2)

        # Kinematic model: Δμ = κ·X_i + const
        X_kin = np.column_stack([xi, np.ones(n)])
        beta_kin = lstsq(X_kin, dmu, rcond=None)[0]
        r2_kin = 1 - np.sum((dmu - X_kin @ beta_kin) ** 2) / np.sum((dmu - dmu.mean()) ** 2)

        # Direction-only model: Δμ = D·cmb_dot + const
        X_dir = np.column_stack([cmb_dot, np.ones(n)])
        beta_dir = lstsq(X_dir, dmu, rcond=None)[0]
        r2_dir = 1 - np.sum((dmu - X_dir @ beta_dir) ** 2) / np.sum((dmu - dmu.mean()) ** 2)

        # F-test for the directional term
        f_stat = ((r2_tep - r2_kin) / 1) / ((1 - r2_tep) / (n - 3))
        p_ftest = sp_stats.f.sf(f_stat, 1, n - 3)

        # Unique R² contributions
        r2_xi_only = 1 - np.sum((dmu - np.column_stack([xi, np.ones(n)]) @ lstsq(
            np.column_stack([xi, np.ones(n)]), dmu, rcond=None)[0]) ** 2) / np.sum((dmu - dmu.mean()) ** 2)
        r2_cmb_only = r2_dir

        result = {
            "n": int(n),
            "tep_model": {
                "kappa": float(beta_tep[0]),
                "D_dipolar": float(beta_tep[1]),
                "const": float(beta_tep[2]),
                "r_squared": float(r2_tep),
            },
            "kinematic_model": {
                "kappa": float(beta_kin[0]),
                "const": float(beta_kin[1]),
                "r_squared": float(r2_kin),
            },
            "direction_only_model": {
                "D_dipolar": float(beta_dir[0]),
                "const": float(beta_dir[1]),
                "r_squared": float(r2_dir),
            },
            "f_test_directional": {
                "f_stat": float(f_stat),
                "p_value": float(p_ftest),
                "delta_r_squared": float(r2_tep - r2_kin),
            },
            "unique_r2": {
                "direction_beyond_xi": float(r2_tep - r2_kin),
                "xi_beyond_direction": float(r2_tep - r2_dir),
            },
            "dipolar_amplitude_mag": float(beta_tep[1]),
            "equivalent_velocity_kms": float(abs(beta_tep[1]) * np.log(10) / 5 * C_KMS),
        }

        print_status(f"  TEP model (X_i + direction): R² = {r2_tep:.3f}", "TEST")
        print_status(f"    κ = {beta_tep[0]:+.2e}, D = {beta_tep[1]:+.3f} mag", "TEST")
        print_status(f"  Kinematic model (X_i only):  R² = {r2_kin:.3f}", "TEST")
        print_status(f"  Direction only:              R² = {r2_dir:.3f}", "TEST")
        print_status(f"  F-test for direction: F={f_stat:.2f}, p={p_ftest:.3f}", "TEST")
        print_status(f"  ΔR² from direction = {r2_tep - r2_kin:.3f}", "TEST")

        return result

    # ------------------------------------------------------------------
    # Analysis 5: Subsample analysis
    # ------------------------------------------------------------------
    def subsample_analysis(self, df):
        """Analyze directional signal in subsamples."""
        print_status("\n--- Analysis 5: Subsample analysis ---", "PROCESS")

        results = {}

        # By sample origin
        if "sample" in df.columns:
            for label in df["sample"].unique():
                sub = df[df["sample"] == label]
                valid = sub["delta_mu"].notna() & sub["cmb_dot"].notna()
                if valid.sum() >= 5:
                    r, p = sp_stats.pearsonr(sub.loc[valid, "cmb_dot"],
                                              sub.loc[valid, "delta_mu"])
                    results[f"sample_{label}"] = {
                        "n": int(valid.sum()),
                        "r_dmu_cmb": float(r),
                        "p_value": float(p),
                        "mean_dmu": float(sub.loc[valid, "delta_mu"].mean()),
                    }
                    print_status(f"  {label} (N={valid.sum()}): r={r:+.3f} (p={p:.3f}), mean Δμ={sub.loc[valid, 'delta_mu'].mean():.3f}", "TEST")

        # By R22 membership
        if "r22_matched" in df.columns:
            for label, mask in [("R22", df["r22_matched"] == True),
                                ("non-R22", df["r22_matched"] == False)]:
                sub = df[mask]
                valid = sub["delta_mu"].notna() & sub["cmb_dot"].notna()
                if valid.sum() >= 5:
                    r, p = sp_stats.pearsonr(sub.loc[valid, "cmb_dot"],
                                              sub.loc[valid, "delta_mu"])
                    r_xi, p_xi = sp_stats.pearsonr(
                        sub.loc[valid, "X_i"],
                        sub.loc[valid, "delta_mu"]) if sub.loc[valid, "X_i"].notna().all() else (np.nan, np.nan)
                    results[f"r22_{label}"] = {
                        "n": int(valid.sum()),
                        "r_dmu_cmb": float(r),
                        "p_value": float(p),
                        "r_dmu_xi": float(r_xi),
                        "p_xi": float(p_xi),
                        "mean_dmu": float(sub.loc[valid, "delta_mu"].mean()),
                        "mean_cmb_dot": float(sub.loc[valid, "cmb_dot"].mean()),
                    }
                    print_status(f"  {label} (N={valid.sum()}): r(Δμ,cmb)={r:+.3f} (p={p:.3f}), r(Δμ,X_i)={r_xi:+.3f}", "TEST")

        return results

    # ------------------------------------------------------------------
    # Analysis 6: Indicator-specific bulk flow
    # ------------------------------------------------------------------
    def bulk_flow_by_indicator(self, cf4):
        """Compare bulk flow dipole amplitude/direction by distance indicator."""
        print_status("\n--- Analysis 6: Indicator-specific bulk flow ---", "PROCESS")

        if cf4 is None:
            return {}

        cmb_coord = SkyCoord(l=CMB_DIPOLE_GAL_L * u.deg, b=CMB_DIPOLE_GAL_B * u.deg,
                             frame="galactic").icrs
        H0 = 73.04

        indicators = {
            "TRGB_nuclear_candle": ("DMtrgb", "e_DMtrgb"),
            "Cepheid_acoustic_clock": ("DMceph", "e_DMceph"),
            "TullyFisher_rotation": ("DMtf", "e_DMtf"),
            "FundamentalPlane_dispersion": ("DMfp", "e_DMfp"),
            "SNeIa_diffusion": ("DMsnIa", "e_DMsnIa"),
            "Combined_all": ("DM", "e_DM"),
        }

        results = {}
        print(f"  {'Indicator':35s} | {'N':>6s} | {'V_dip':>7s} | {'l':>6s} | {'b':>6s} | {'sep_CMB':>7s}")
        print(f"  {'-'*80}")

        for label, (dm_col, err_col) in indicators.items():
            valid = cf4[dm_col].notna() & cf4["RAdeg"].notna() & cf4["Vcmb"].notna()
            D_mpc = 10 ** ((cf4.loc[valid, dm_col] - 25) / 5)
            near = (D_mpc > 0) & (D_mpc < 100)

            if near.sum() < 10:
                continue

            ra = cf4.loc[valid, "RAdeg"].values[near]
            dec = cf4.loc[valid, "DEdeg"].values[near]
            V_cmb = cf4.loc[valid, "Vcmb"].values[near]
            D = D_mpc.values[near]
            V_pec = V_cmb - H0 * D

            g3d = ra_dec_to_unit_vectors(ra, dec)
            w = 1.0 / np.maximum(D, 1.0) ** 2
            W = np.diag(w)
            X = np.column_stack([g3d, np.ones(len(g3d))])
            try:
                beta = lstsq(X.T @ W @ X, X.T @ W @ V_pec, rcond=None)[0]
                D_vec = beta[:3]
                D_mag = np.linalg.norm(D_vec)
                D_ra = np.degrees(np.arctan2(D_vec[1], D_vec[0])) % 360
                D_dec = np.degrees(np.arcsin(D_vec[2] / max(D_mag, 1e-10)))
                D_coord = SkyCoord(ra=D_ra * u.deg, dec=D_dec * u.deg, frame="icrs")
                D_gal = D_coord.galactic
                sep = D_coord.separation(cmb_coord)

                results[label] = {
                    "n": int(near.sum()),
                    "v_dipole_kms": float(D_mag),
                    "gal_l": float(D_gal.l.deg),
                    "gal_b": float(D_gal.b.deg),
                    "sep_from_cmb_deg": float(sep.deg),
                }
                print(f"  {label:35s} | {near.sum():6d} | {D_mag:6.0f} | {D_gal.l.deg:5.1f} | {D_gal.b.deg:+5.1f} | {sep.deg:6.1f}°")
            except Exception:
                pass

        # Key comparison: Cepheid vs TRGB
        if "TRGB_nuclear_candle" in results and "Cepheid_acoustic_clock" in results:
            trgb = results["TRGB_nuclear_candle"]
            cep = results["Cepheid_acoustic_clock"]
            results["cepheid_excess_kms"] = cep["v_dipole_kms"] - trgb["v_dipole_kms"]
            print_status(f"  Cepheid excess: {cep['v_dipole_kms'] - trgb['v_dipole_kms']:.0f} km/s "
                        f"(Cepheid {cep['v_dipole_kms']:.0f} vs TRGB {trgb['v_dipole_kms']:.0f})", "TEST")

        return results

    # ------------------------------------------------------------------
    # Analysis 7: Sky coverage figure
    # ------------------------------------------------------------------
    def make_sky_figure(self, df):
        """Generate a sky-coverage figure showing Δμ color-coded by direction."""
        print_status("\n--- Generating sky-coverage figure ---", "PROCESS")

        valid = df["delta_mu"].notna() & df["ra"].notna() & df["dec"].notna() & df["cmb_dot"].notna()
        if valid.sum() < 5:
            print_status("  Insufficient data for figure", "WARNING")
            return

        sub = df.loc[valid].copy()
        coords = SkyCoord(ra=sub["ra"].values * u.deg, dec=sub["dec"].values * u.deg,
                          frame="icrs").galactic

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Panel 1: Aitoff sky plot, color-coded by Δμ
        ax1 = fig.add_subplot(221, projection="aitoff")
        l_rad = np.radians(np.where(coords.l.deg > 180, coords.l.deg - 360, coords.l.deg))
        b_rad = np.radians(coords.b.deg)
        sc = ax1.scatter(l_rad, b_rad, c=sub["delta_mu"].values, cmap="RdBu_r",
                         vmin=-0.2, vmax=0.2, s=60, edgecolors="k", linewidths=0.5, zorder=3)
        # CMB dipole
        cmb_l = np.radians(CMB_DIPOLE_GAL_L - 360 if CMB_DIPOLE_GAL_L > 180 else CMB_DIPOLE_GAL_L)
        cmb_b = np.radians(CMB_DIPOLE_GAL_B)
        ax1.scatter([cmb_l], [cmb_b], marker="*", c="gold", s=200, edgecolors="k",
                    linewidths=1, zorder=5, label="CMB dipole")
        ax1.set_title("Δμ = μ_Cep − μ_TRGB (galactic)", fontsize=11)
        ax1.legend(fontsize=8, loc="lower left")
        ax1.grid(True, alpha=0.3)
        plt.colorbar(sc, ax=ax1, orientation="vertical", shrink=0.7, label="Δμ (mag)")

        # Panel 2: Δμ vs cmb_dot
        ax2 = axes[0, 1]
        colors = {"CF4": "C0", "JWST": "C1", "R22_x_CF4TRGB": "C2", "KP_x_CF4TRGB": "C3"}
        for sample in sub["sample"].unique():
            mask = sub["sample"] == sample
            ax2.scatter(sub.loc[mask, "cmb_dot"], sub.loc[mask, "delta_mu"],
                       label=sample, c=colors.get(sample, "gray"), s=50, edgecolors="k", linewidths=0.5)
        # Fit line
        x = sub["cmb_dot"].values
        y = sub["delta_mu"].values
        beta = lstsq(np.column_stack([x, np.ones(len(x))]), y, rcond=None)[0]
        xx = np.linspace(x.min(), x.max(), 100)
        ax2.plot(xx, beta[0] * xx + beta[1], "k--", alpha=0.5)
        r, p = sp_stats.pearsonr(x, y)
        ax2.set_xlabel("CMB dipole projection (cos θ)")
        ax2.set_ylabel("Δμ (mag)")
        ax2.set_title(f"Directional Δμ signal: r={r:+.3f} (p={p:.3f})", fontsize=11)
        ax2.legend(fontsize=8)
        ax2.axhline(0, color="gray", linestyle=":", alpha=0.5)
        ax2.axvline(0, color="gray", linestyle=":", alpha=0.5)

        # Panel 3: Δμ vs X_i
        ax3 = axes[1, 0]
        valid_xi = sub["X_i"].notna()
        if valid_xi.any():
            for sample in sub.loc[valid_xi, "sample"].unique():
                mask = (sub["sample"] == sample) & valid_xi
                ax3.scatter(sub.loc[mask, "X_i"], sub.loc[mask, "delta_mu"],
                           label=sample, c=colors.get(sample, "gray"), s=50, edgecolors="k", linewidths=0.5)
            x = sub.loc[valid_xi, "X_i"].values
            y = sub.loc[valid_xi, "delta_mu"].values
            beta = lstsq(np.column_stack([x, np.ones(len(x))]), y, rcond=None)[0]
            xx = np.linspace(x.min(), x.max(), 100)
            ax3.plot(xx, beta[0] * xx + beta[1], "k--", alpha=0.5)
            r, p = sp_stats.pearsonr(x, y)
            ax3.set_title(f"X_i correlation: r={r:+.3f} (p={p:.3f})", fontsize=11)
        ax3.set_xlabel("X_i (dimensionless)")
        ax3.set_ylabel("Δμ (mag)")
        ax3.axhline(0, color="gray", linestyle=":", alpha=0.5)

        # Panel 4: Histogram by direction
        ax4 = axes[1, 1]
        toward = sub["cmb_dot"] > 0
        away = sub["cmb_dot"] <= 0
        bins = np.linspace(-0.3, 0.2, 15)
        ax4.hist(sub.loc[away, "delta_mu"], bins=bins, alpha=0.6, color="C0",
                label=f"Away from CMB (N={away.sum()})", edgecolor="k")
        ax4.hist(sub.loc[toward, "delta_mu"], bins=bins, alpha=0.6, color="C1",
                label=f"Toward CMB (N={toward.sum()})", edgecolor="k")
        ax4.axvline(sub.loc[away, "delta_mu"].mean(), color="C0", linestyle="--", linewidth=2)
        ax4.axvline(sub.loc[toward, "delta_mu"].mean(), color="C1", linestyle="--", linewidth=2)
        ax4.set_xlabel("Δμ (mag)")
        ax4.set_ylabel("Count")
        ax4.set_title("Δμ distribution by CMB dipole direction", fontsize=11)
        ax4.legend(fontsize=9)

        fig.suptitle("Step 54: Directional Δμ — Bulk Flow as Temporal Topology Gradient",
                     fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        fig_path = self.figures / "step_54_directional_dmu.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print_status(f"  Figure saved to {fig_path}", "SUCCESS")

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------
    def run(self):
        print_status("Step 54: Directional Δμ Analysis", "TITLE")
        print_status("Bulk flow as temporal topology gradient vs kinetic motion", "PROCESS")

        df = self.load_sample()
        if df.empty:
            print_status("No data available. Exiting.", "ERROR")
            return

        cf4 = self.load_cf4_for_bulk_flow()

        # Run all analyses
        corr_result = self.directional_correlation(df)
        dipole_result = self.dipole_fit_3d(df)
        partial_result = self.partial_correlations(df)
        joint_result = self.joint_model(df)
        subsample_result = self.subsample_analysis(df)
        bulk_flow_result = self.bulk_flow_by_indicator(cf4)

        # Generate figure
        self.make_sky_figure(df)

        # Compile output
        output = {
            "step": "54_directional_dmu",
            "description": "Directional Δμ analysis: bulk flow as temporal topology gradient",
            "tep_prediction": "Δμ shows dipolar pattern aligned with CMB dipole (acoustic clock biased by temporal shear gradient)",
            "kinematic_prediction": "No directional Δμ pattern (luminosity distances are direction-independent)",
            "n_galaxies": int(len(df)),
            "n_with_cmb_dot": int(df["cmb_dot"].notna().sum()),
            "primary_correlation": corr_result,
            "dipole_fit_3d": dipole_result,
            "partial_correlations": partial_result,
            "joint_model": joint_result,
            "subsample_analysis": subsample_result,
            "bulk_flow_by_indicator": bulk_flow_result,
            "interpretation": {
                "tep_signal_detected": corr_result["permutation_p_one_sided"] < 0.05,
                "significance_sigma": corr_result["permutation_sigma_diff"],
                "dipole_aligned_with_cmb": dipole_result["angular_separation_deg"] < 45,
                "cepheid_bulk_flow_excess_kms": bulk_flow_result.get("cepheid_excess_kms", None),
                "discriminating_observable": "Δμ directional signal: predicted by TEP, not by kinematic model",
            },
        }

        output_path = self.results / "step_54_directional_dmu.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print_status(f"\nResults saved to {output_path}", "SUCCESS")


if __name__ == "__main__":
    step = Step54DirectionalDmu()
    step.run()
