#!/usr/bin/env python3
r"""
Step 55: Directional Pantheon+ Hemisphere Split
=================================================
Test whether Pantheon+ Hubble residuals exhibit a directional pattern
aligned with the CMB dipole, as predicted by TEP if the disformal
spatial shear is tied to the cosmic rest frame.

Under TEP, the Cepheid clock bias is encoded in the global zero-point
$M_B$ rather than distributed across redshift. However, the Cepheid
calibrators are not isotropically distributed — they are preferentially
located toward the CMB dipole (mean $\cos\theta \approx +0.46$ for the
JWST sample). If the disformal transport coupling $\bar\epsilon_0$
varies directionally with the temporal shear gradient, the calibrated
$M_B$ inherits a directional bias, which propagates into the Hubble
residuals of all Pantheon+ SNe.

The kinematic void model predicts no directional Hubble residual
pattern (luminosity distances are direction-independent; the void
produces a monopole, not a dipole).

Analyses:
  1. Hemisphere split: Hubble residuals for CMB-aligned vs anti-aligned
  2. Correlation r(HR, cos θ_CMB) with permutation significance
  3. 3-D dipole fit to Hubble residuals
  4. H0 by hemisphere (separate fits)
  5. Joint model HR = a + b·cos θ + c·z with F-test
  6. Sky-coverage figure

Outputs:
    results/outputs/step_55_directional_pantheon.json
    results/figures/step_55_directional_pantheon.png
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
from numpy.linalg import lstsq
from astropy.coordinates import SkyCoord
import astropy.units as u

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status

CMB_DIPOLE_GAL_L = 264.021
CMB_DIPOLE_GAL_B = 48.253
C_KMS = 299792.458


def ra_dec_to_unit_vectors(ra_deg, dec_deg):
    ra_rad = np.radians(np.asarray(ra_deg, dtype=float))
    dec_rad = np.radians(np.asarray(dec_deg, dtype=float))
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
    ], dtype=float)


class Step55DirectionalPantheon:
    """Step 55: Directional Pantheon+ hemisphere split test."""

    def __init__(self):
        self.root = PROJECT_ROOT
        self.data_raw = self.root / "data" / "raw"
        self.data_raw_external = self.root / "data" / "raw" / "external"
        self.data_processed = self.root / "data" / "processed"
        self.results = self.root / "results" / "outputs"
        self.figures = self.root / "results" / "figures"
        self.logs = self.root / "logs"

        for d in [self.results, self.figures, self.logs]:
            d.mkdir(parents=True, exist_ok=True)

        self.logger = TEPLogger(
            "step_55",
            log_file_path=self.logs / "step_55_directional_pantheon.log",
        )
        set_step_logger(self.logger)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def load_pantheon(self):
        """Load Pantheon+ with coordinates and compute Hubble residuals."""
        print_status("Loading Pantheon+ sample...", "PROCESS")
        path = self.data_raw / "Pantheon+SH0ES.dat"
        if not path.exists():
            print_status(f"Pantheon+ not found at {path}", "ERROR")
            return pd.DataFrame()

        df = pd.read_csv(path, sep=r"\s+")
        df = df.rename(columns={"zCMB": "zcmb", "zHEL": "zhel",
                                "RA": "ra", "DEC": "dec"})

        # Hubble-flow subset
        hf = df[(df["zcmb"] >= 0.05) & (df["IS_CALIBRATOR"] == 0)].copy()
        print_status(f"  {len(hf)} Hubble-flow SNe (z >= 0.05)", "SUCCESS")

        # Compute Hubble residuals as residuals from the best-fit Hubble line.
        # m_b_corr - MU_SH0ES = M_B (absolute magnitude). The Hubble residual
        # is the scatter around the mean M_B — this is the standard definition.
        hf["M_B"] = hf["m_b_corr"] - hf["MU_SH0ES"]
        hf["HR"] = hf["M_B"] - hf["M_B"].mean()
        hf["HR_err"] = hf["m_b_corr_err_DIAG"]

        # Filter out zero/infinite errors
        valid = (hf["HR_err"] > 0) & np.isfinite(hf["HR_err"]) & np.isfinite(hf["HR"])
        n_bad = (~valid).sum()
        if n_bad > 0:
            print_status(f"  Filtering {n_bad} SNe with invalid errors", "WARNING")
            hf = hf[valid].copy()

        # CMB dipole projection
        cmb_vec = cmb_dipole_unit_vector()
        gal_vecs = ra_dec_to_unit_vectors(hf["ra"].values, hf["dec"].values)
        hf["cmb_dot"] = gal_vecs @ cmb_vec

        # Angular separation
        coords = SkyCoord(ra=hf["ra"].values * u.deg, dec=hf["dec"].values * u.deg,
                          frame="icrs")
        cmb_coord = SkyCoord(l=CMB_DIPOLE_GAL_L * u.deg, b=CMB_DIPOLE_GAL_B * u.deg,
                             frame="galactic").icrs
        hf["cmb_sep_deg"] = coords.separation(cmb_coord).deg

        print_status(f"  CMB dipole projection computed for {len(hf)} SNe", "SUCCESS")
        print_status(f"  Mean cmb_dot = {hf['cmb_dot'].mean():.3f}", "INFO")
        print_status(f"  Toward CMB: {(hf['cmb_dot'] > 0).sum()}, Away: {(hf['cmb_dot'] <= 0).sum()}", "INFO")

        return hf

    def load_covariance(self):
        """Load the Pantheon+ STAT+SYS covariance matrix."""
        print_status("Loading Pantheon+ covariance matrix...", "PROCESS")
        path = self.data_raw / "Pantheon+SH0ES_STAT+SYS.cov"
        if not path.exists():
            print_status(f"Covariance not found at {path}", "WARNING")
            return None

        # Read the covariance matrix
        with open(path) as f:
            n = int(f.readline().strip())
            cov = np.zeros((n, n))
            for i in range(n):
                vals = f.readline().split()
                cov[i, :] = [float(v) for v in vals]

        print_status(f"  Loaded {n}x{n} covariance matrix", "SUCCESS")
        return cov

    # ------------------------------------------------------------------
    # Analysis 1: Hemisphere split
    # ------------------------------------------------------------------
    def hemisphere_split(self, df):
        """Test Hubble residuals for CMB-aligned vs anti-aligned hemispheres."""
        print_status("\n--- Analysis 1: Hemisphere split ---", "PROCESS")

        toward = df["cmb_dot"] > 0
        away = df["cmb_dot"] <= 0

        hr_toward = df.loc[toward, "HR"].values
        hr_away = df.loc[away, "HR"].values

        # Weighted means
        w_toward = 1.0 / df.loc[toward, "HR_err"].values ** 2
        w_away = 1.0 / df.loc[away, "HR_err"].values ** 2

        mean_toward = np.sum(hr_toward * w_toward) / np.sum(w_toward)
        sem_toward = np.sqrt(1.0 / np.sum(w_toward))
        mean_away = np.sum(hr_away * w_away) / np.sum(w_away)
        sem_away = np.sqrt(1.0 / np.sum(w_away))

        diff = mean_toward - mean_away
        diff_err = np.sqrt(sem_toward ** 2 + sem_away ** 2)
        sigma = abs(diff) / diff_err if diff_err > 0 else 0

        # Permutation test
        np.random.seed(42)
        n_perm = 100000
        all_hr = df["HR"].values
        all_w = 1.0 / df["HR_err"].values ** 2
        n = len(all_hr)
        diff_perm = np.zeros(n_perm)
        for i in range(n_perm):
            idx = np.random.permutation(n)
            hr_s = all_hr[idx]
            d_t = np.sum(hr_s[toward.values] * all_w[toward.values]) / np.sum(all_w[toward.values])
            d_a = np.sum(hr_s[away.values] * all_w[away.values]) / np.sum(all_w[away.values])
            diff_perm[i] = d_t - d_a
        p_perm = (diff_perm >= diff).mean() if diff > 0 else (diff_perm <= diff).mean()

        # Equivalent H0 difference
        # HR = m_b - mu => delta_HR corresponds to delta_mu => delta_H0/H0 = -delta_mu / 5 / log10(e)
        dH0_equiv = -diff * 5 * np.log10(np.e)  # in units of H0 fraction
        dH0_kms = dH0_equiv * 73.0  # km/s/Mpc

        result = {
            "n_toward": int(toward.sum()),
            "n_away": int(away.sum()),
            "mean_hr_toward": float(mean_toward),
            "sem_hr_toward": float(sem_toward),
            "mean_hr_away": float(mean_away),
            "sem_hr_away": float(sem_away),
            "difference": float(diff),
            "difference_err": float(diff_err),
            "sigma": float(sigma),
            "permutation_p": float(p_perm),
            "equivalent_dH0_fraction": float(dH0_equiv),
            "equivalent_dH0_kms": float(dH0_kms),
        }

        print_status(f"  Toward CMB (N={toward.sum()}): HR = {mean_toward:+.4f} ± {sem_toward:.4f}", "TEST")
        print_status(f"  Away from CMB (N={away.sum()}): HR = {mean_away:+.4f} ± {sem_away:.4f}", "TEST")
        print_status(f"  Difference: {diff:+.4f} ± {diff_err:.4f} ({sigma:.2f}σ, p={p_perm:.4f})", "TEST")
        print_status(f"  Equivalent ΔH0: {dH0_kms:.2f} km/s/Mpc", "TEST")

        return result

    # ------------------------------------------------------------------
    # Analysis 2: Correlation with CMB dipole
    # ------------------------------------------------------------------
    def directional_correlation(self, df):
        """Test r(HR, cmb_dot) with permutation significance."""
        print_status("\n--- Analysis 2: Directional correlation ---", "PROCESS")

        hr = df["HR"].values
        cmb_dot = df["cmb_dot"].values
        n = len(hr)

        r_obs, p_pearson = sp_stats.pearsonr(cmb_dot, hr)

        # Permutation test
        np.random.seed(42)
        n_perm = 100000
        r_perm = np.zeros(n_perm)
        for i in range(n_perm):
            idx = np.random.permutation(n)
            r_perm[i], _ = sp_stats.pearsonr(cmb_dot, hr[idx])
        p_perm_two = (np.abs(r_perm) >= np.abs(r_obs)).mean()
        p_perm_one = (r_perm >= r_obs).mean() if r_obs > 0 else (r_perm <= r_obs).mean()

        # Weighted correlation
        w = 1.0 / df["HR_err"].values ** 2
        wmean_x = np.sum(w * cmb_dot) / np.sum(w)
        wmean_y = np.sum(w * hr) / np.sum(w)
        wcov = np.sum(w * (cmb_dot - wmean_x) * (hr - wmean_y)) / np.sum(w)
        wvar_x = np.sum(w * (cmb_dot - wmean_x) ** 2) / np.sum(w)
        wvar_y = np.sum(w * (hr - wmean_y) ** 2) / np.sum(w)
        r_weighted = wcov / np.sqrt(wvar_x * wvar_y) if wvar_x * wvar_y > 0 else 0

        result = {
            "n": int(n),
            "pearson_r": float(r_obs),
            "pearson_p": float(p_pearson),
            "weighted_r": float(r_weighted),
            "permutation_p_two_sided": float(p_perm_two),
            "permutation_p_one_sided": float(p_perm_one),
        }

        print_status(f"  r(HR, cmb_dot) = {r_obs:+.4f} (p={p_pearson:.4f})", "TEST")
        print_status(f"  Weighted r = {r_weighted:+.4f}", "TEST")
        print_status(f"  Permutation: p_two={p_perm_two:.4f}, p_one={p_perm_one:.4f}", "TEST")

        return result

    # ------------------------------------------------------------------
    # Analysis 3: 3-D dipole fit
    # ------------------------------------------------------------------
    def dipole_fit_3d(self, df):
        """Fit a 3-D vector dipole to Hubble residuals."""
        print_status("\n--- Analysis 3: 3-D vector dipole fit ---", "PROCESS")

        hr = np.asarray(df["HR"].values, dtype=float)
        g3d = ra_dec_to_unit_vectors(df["ra"].values, df["dec"].values)
        n = len(hr)

        # Weighted fit
        w = 1.0 / (np.asarray(df["HR_err"].values, dtype=float) ** 2)
        X = np.column_stack([g3d, np.ones(n)])
        XtW = X.T * w[None, :]
        XtWX = XtW @ X
        try:
            beta = lstsq(XtWX, XtW @ hr, rcond=None)[0]
        except Exception:
            beta = lstsq(X, hr, rcond=None)[0]

        D_vec = beta[:3]
        D_mag = np.linalg.norm(D_vec)

        D_ra = np.degrees(np.arctan2(D_vec[1], D_vec[0])) % 360
        D_dec = np.degrees(np.arcsin(D_vec[2] / max(D_mag, 1e-10)))
        D_coord = SkyCoord(ra=D_ra * u.deg, dec=D_dec * u.deg, frame="icrs")
        D_gal = D_coord.galactic

        cmb_coord = SkyCoord(l=CMB_DIPOLE_GAL_L * u.deg, b=CMB_DIPOLE_GAL_B * u.deg,
                             frame="galactic").icrs
        sep = D_coord.separation(cmb_coord)

        # Permutation significance
        np.random.seed(42)
        null_mags = []
        for _ in range(10000):
            idx = np.random.choice(n, n, replace=False)
            try:
                beta_null = lstsq(XtWX, XtW @ hr[idx], rcond=None)[0]
            except Exception:
                beta_null = lstsq(X, hr[idx], rcond=None)[0]
            null_mags.append(np.linalg.norm(beta_null[:3]))
        null_mags = np.array(null_mags)
        p_dipole = (null_mags >= D_mag).mean()
        sigma_dipole = (D_mag - null_mags.mean()) / null_mags.std() if null_mags.std() > 0 else 0

        # Equivalent velocity
        v_equiv = D_mag * np.log(10) / 5 * C_KMS

        result = {
            "n": int(n),
            "dipole_amplitude_mag": float(D_mag),
            "dipole_ra_deg": float(D_ra),
            "dipole_dec_deg": float(D_dec),
            "dipole_gal_l": float(D_gal.l.deg),
            "dipole_gal_b": float(D_gal.b.deg),
            "angular_separation_deg": float(sep.deg),
            "equivalent_velocity_kms": float(v_equiv),
            "permutation_p": float(p_dipole),
            "permutation_sigma": float(sigma_dipole),
        }

        print_status(f"  Dipole amplitude: {D_mag:.4f} mag ({v_equiv:.0f} km/s equiv)", "TEST")
        print_status(f"  Direction: l={D_gal.l.deg:.1f}°, b={D_gal.b.deg:+.1f}°", "TEST")
        print_status(f"  Angular separation from CMB: {sep.deg:.1f}°", "TEST")
        print_status(f"  Permutation: p={p_dipole:.4f} ({sigma_dipole:.2f}σ)", "TEST")

        return result

    # ------------------------------------------------------------------
    # Analysis 4: H0 by hemisphere
    # ------------------------------------------------------------------
    def h0_by_hemisphere(self, df):
        """Fit H0 separately for each hemisphere using calibrated M_B."""
        print_status("\n--- Analysis 4: H0 by hemisphere ---", "PROCESS")

        # The Hubble residual HR = M_B - <M_B>. If there's a directional
        # bias in M_B, it would show up as a difference in the mean HR
        # by hemisphere. The equivalent H0 shift is:
        # ΔH0/H0 = -ΔHR * ln(10) / 5
        # We compute the weighted mean HR per hemisphere and convert.

        results = {}
        H0_ref = 73.04  # SH0ES reference
        for label, mask in [("toward", df["cmb_dot"] > 0),
                            ("away", df["cmb_dot"] <= 0),
                            ("all", np.ones(len(df), dtype=bool))]:
            sub = df[mask]
            hr = sub["HR"].values
            w = 1.0 / sub["HR_err"].values ** 2
            mean_hr = np.sum(w * hr) / np.sum(w)
            sem_hr = np.sqrt(1.0 / np.sum(w))

            # Equivalent H0: since HR = M_B - <M_B>, a positive HR means
            # the SN is fainter than average, corresponding to a lower
            # inferred H0 for that hemisphere.
            H0 = H0_ref * (1 - mean_hr * np.log(10) / 5)
            H0_err = H0_ref * sem_hr * np.log(10) / 5

            results[label] = {
                "n": int(mask.sum()),
                "mean_hr": float(mean_hr),
                "sem_hr": float(sem_hr),
                "H0": float(H0),
                "H0_err": float(H0_err),
            }
            print_status(f"  {label} (N={mask.sum()}): HR = {mean_hr:+.5f} ± {sem_hr:.5f}, H0 = {H0:.2f} ± {H0_err:.2f} km/s/Mpc", "TEST")

        dH0 = results["toward"]["H0"] - results["away"]["H0"]
        dH0_err = np.sqrt(results["toward"]["H0_err"] ** 2 + results["away"]["H0_err"] ** 2)
        results["difference"] = {
            "dH0": float(dH0),
            "dH0_err": float(dH0_err),
            "sigma": float(abs(dH0) / dH0_err) if dH0_err > 0 else 0,
        }
        print_status(f"  ΔH0 (toward - away) = {dH0:+.3f} ± {dH0_err:.3f} km/s/Mpc ({abs(dH0)/dH0_err:.2f}σ)", "TEST")

        return results

    # ------------------------------------------------------------------
    # Analysis 5: Joint model with redshift control
    # ------------------------------------------------------------------
    def joint_model(self, df):
        """Fit HR = a + b·cos θ + c·z and F-test the directional term."""
        print_status("\n--- Analysis 5: Joint model with F-test ---", "PROCESS")

        hr = np.asarray(df["HR"].values, dtype=float)
        cmb_dot = np.asarray(df["cmb_dot"].values, dtype=float)
        z = np.asarray(df["zcmb"].values, dtype=float)
        n = len(hr)

        # Full model: HR = a + b·cos θ + c·z
        X_full = np.column_stack([cmb_dot, z, np.ones(n)])
        beta_full = lstsq(X_full, hr, rcond=None)[0]
        ss_tot = np.sum((hr - hr.mean()) ** 2)
        ss_tot = max(ss_tot, 1e-12)
        r2_full = 1.0 - np.sum((hr - X_full @ beta_full) ** 2) / ss_tot

        # Reduced model: HR = a + c·z
        X_red = np.column_stack([z, np.ones(n)])
        beta_red = lstsq(X_red, hr, rcond=None)[0]
        r2_red = 1.0 - np.sum((hr - X_red @ beta_red) ** 2) / ss_tot

        # F-test
        f_stat = ((r2_full - r2_red) / 1) / ((1 - r2_full) / (n - 3))
        p_ftest = sp_stats.f.sf(f_stat, 1, n - 3)

        result = {
            "n": int(n),
            "full_model": {
                "b_directional": float(beta_full[0]),
                "c_redshift": float(beta_full[1]),
                "const": float(beta_full[2]),
                "r_squared": float(r2_full),
            },
            "reduced_model": {
                "c_redshift": float(beta_red[0]),
                "const": float(beta_red[1]),
                "r_squared": float(r2_red),
            },
            "f_test": {
                "f_stat": float(f_stat),
                "p_value": float(p_ftest),
                "delta_r_squared": float(r2_full - r2_red),
            },
        }

        print_status(f"  Full model (direction + z): R² = {r2_full:.4f}, b = {beta_full[0]:+.4f}", "TEST")
        print_status(f"  Reduced model (z only): R² = {r2_red:.4f}", "TEST")
        print_status(f"  F-test: F={f_stat:.2f}, p={p_ftest:.4f}, ΔR²={r2_full-r2_red:.4f}", "TEST")

        return result

    # ------------------------------------------------------------------
    # Analysis 6: Sky figure
    # ------------------------------------------------------------------
    def make_figure(self, df):
        """Generate sky-coverage figure."""
        print_status("\n--- Generating sky figure ---", "PROCESS")

        coords = SkyCoord(ra=df["ra"].values * u.deg, dec=df["dec"].values * u.deg,
                          frame="icrs").galactic

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Panel 1: Aitoff sky plot
        ax1 = fig.add_subplot(221, projection="aitoff")
        l_rad = np.radians(np.where(coords.l.deg > 180, coords.l.deg - 360, coords.l.deg))
        b_rad = np.radians(coords.b.deg)
        sc = ax1.scatter(l_rad, b_rad, c=df["HR"].values, cmap="RdBu_r",
                         vmin=-0.3, vmax=0.3, s=5, alpha=0.6, zorder=3)
        cmb_l = np.radians(CMB_DIPOLE_GAL_L - 360 if CMB_DIPOLE_GAL_L > 180 else CMB_DIPOLE_GAL_L)
        cmb_b = np.radians(CMB_DIPOLE_GAL_B)
        ax1.scatter([cmb_l], [cmb_b], marker="*", c="gold", s=200, edgecolors="k",
                    linewidths=1, zorder=5, label="CMB dipole")
        ax1.set_title("Pantheon+ Hubble residuals (galactic)", fontsize=11)
        ax1.legend(fontsize=8, loc="lower left")
        ax1.grid(True, alpha=0.3)
        plt.colorbar(sc, ax=ax1, orientation="vertical", shrink=0.7, label="HR (mag)")

        # Panel 2: HR vs cmb_dot
        ax2 = axes[0, 1]
        toward = df["cmb_dot"] > 0
        ax2.scatter(df.loc[toward, "cmb_dot"], df.loc[toward, "HR"],
                   c="C1", s=10, alpha=0.5, label=f"Toward (N={toward.sum()})")
        ax2.scatter(df.loc[~toward, "cmb_dot"], df.loc[~toward, "HR"],
                   c="C0", s=10, alpha=0.5, label=f"Away (N={(~toward).sum()})")
        # Binned means
        bins = np.linspace(-1, 1, 11)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        bin_means = []
        bin_errs = []
        for i in range(len(bins) - 1):
            mask = (df["cmb_dot"] >= bins[i]) & (df["cmb_dot"] < bins[i+1])
            if mask.sum() > 0:
                w = 1.0 / df.loc[mask, "HR_err"].values ** 2
                m = np.sum(df.loc[mask, "HR"].values * w) / np.sum(w)
                e = np.sqrt(1.0 / np.sum(w))
                bin_means.append(m)
                bin_errs.append(e)
            else:
                bin_means.append(np.nan)
                bin_errs.append(np.nan)
        ax2.errorbar(bin_centers, bin_means, yerr=bin_errs, fmt="ko-",
                    markersize=6, linewidth=2, capsize=3, zorder=5)
        r, p = sp_stats.pearsonr(df["cmb_dot"].values, df["HR"].values)
        ax2.set_xlabel("CMB dipole projection (cos θ)")
        ax2.set_ylabel("Hubble residual (mag)")
        ax2.set_title(f"Directional HR: r={r:+.4f} (p={p:.4f})", fontsize=11)
        ax2.legend(fontsize=8)
        ax2.axhline(0, color="gray", linestyle=":", alpha=0.5)
        ax2.axvline(0, color="gray", linestyle=":", alpha=0.5)

        # Panel 3: H0 by hemisphere
        ax3 = axes[1, 0]
        labels = ["Toward\nCMB", "Away from\nCMB", "All"]
        h0_vals = []
        h0_errs = []
        H0_ref = 73.04
        for mask in [toward, ~toward, np.ones(len(df), dtype=bool)]:
            sub = df[mask]
            hr = sub["HR"].values
            w = 1.0 / sub["HR_err"].values ** 2
            mean_hr = np.sum(w * hr) / np.sum(w)
            sem_hr = np.sqrt(1.0 / np.sum(w))
            H0 = H0_ref * (1 - mean_hr * np.log(10) / 5)
            H0_err = H0_ref * sem_hr * np.log(10) / 5
            h0_vals.append(H0)
            h0_errs.append(H0_err)
        ax3.bar(labels, h0_vals, yerr=h0_errs, color=["C1", "C0", "gray"],
               capsize=5, edgecolor="k")
        ax3.set_ylabel("H0 (km/s/Mpc)")
        ax3.set_title("H0 by CMB hemisphere", fontsize=11)
        ax3.axhline(73.0, color="red", linestyle="--", alpha=0.5, label="R22 H0")
        ax3.axhline(69.8, color="blue", linestyle="--", alpha=0.5, label="TRGB H0")
        ax3.legend(fontsize=8)

        # Panel 4: Histogram
        ax4 = axes[1, 1]
        ax4.hist(df.loc[~toward, "HR"], bins=50, alpha=0.6, color="C0",
                label=f"Away (N={(~toward).sum()})", edgecolor="k", density=True)
        ax4.hist(df.loc[toward, "HR"], bins=50, alpha=0.6, color="C1",
                label=f"Toward (N={toward.sum()})", edgecolor="k", density=True)
        ax4.axvline(df.loc[~toward, "HR"].mean(), color="C0", linestyle="--", linewidth=2)
        ax4.axvline(df.loc[toward, "HR"].mean(), color="C1", linestyle="--", linewidth=2)
        ax4.set_xlabel("Hubble residual (mag)")
        ax4.set_ylabel("Density")
        ax4.set_title("HR distribution by hemisphere", fontsize=11)
        ax4.legend(fontsize=9)

        fig.suptitle("Step 55: Directional Pantheon+ — CMB Hemisphere Split",
                     fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        fig_path = self.figures / "step_55_directional_pantheon.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print_status(f"  Figure saved to {fig_path}", "SUCCESS")

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------
    def run(self):
        print_status("Step 55: Directional Pantheon+ Hemisphere Split", "TITLE")
        print_status("Testing for CMB-dipole-aligned Hubble residual pattern", "PROCESS")

        df = self.load_pantheon()
        if df.empty:
            print_status("No data. Exiting.", "ERROR")
            return

        # Run analyses
        hemi_result = self.hemisphere_split(df)
        corr_result = self.directional_correlation(df)
        dipole_result = self.dipole_fit_3d(df)
        h0_result = self.h0_by_hemisphere(df)
        joint_result = self.joint_model(df)

        # Figure
        self.make_figure(df)

        # Compile output
        output = {
            "step": "55_directional_pantheon",
            "description": "Directional Pantheon+ hemisphere split: CMB-dipole-aligned Hubble residuals",
            "tep_prediction": "If disformal shear is tied to cosmic rest frame, H0 residuals show dipole along CMB axis",
            "kinematic_prediction": "No directional HR pattern (luminosity distances are direction-independent; void produces monopole)",
            "n_sne": int(len(df)),
            "hemisphere_split": hemi_result,
            "directional_correlation": corr_result,
            "dipole_fit_3d": dipole_result,
            "h0_by_hemisphere": h0_result,
            "joint_model": joint_result,
            "interpretation": {
                "directional_signal_detected": corr_result["permutation_p_one_sided"] < 0.05,
                "hemisphere_split_significant": hemi_result["permutation_p"] < 0.05,
                "dipole_aligned_with_cmb": dipole_result["angular_separation_deg"] < 45,
                "h0_hemisphere_difference_kms": h0_result["difference"]["dH0"],
                "h0_hemisphere_difference_sigma": h0_result["difference"]["sigma"],
            },
        }

        output_path = self.results / "step_55_directional_pantheon.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print_status(f"\nResults saved to {output_path}", "SUCCESS")


if __name__ == "__main__":
    step = Step55DirectionalPantheon()
    step.run()
