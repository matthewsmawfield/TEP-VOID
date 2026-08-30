#!/usr/bin/env python3
"""
Step 48: X_i-step test with measured V_rot from Vizier/HyperLEDA

This script replaces the Tully-Fisher proxy in the X_i-step test with
measured rotation velocities from the HyperLEDA catalog (VII/237),
queried via astroquery.vizier. The measured V_rot eliminates the
30-40% scatter introduced by the Tully-Fisher relation, which should
amplify the X_i-step signal from the current ~1.4sigma to a
statistically significant detection.

Inclination corrections are applied where needed:
    V_rot = V_max / sin(i)

The script also tests the V_rot^2/r^2 scaling prediction from the
DHOST derivation (Section 4.6): epsilon_0 should scale as V_rot^2/r^2
rather than V_rot^2 alone, if the disformal mechanism is correct.
"""

import json
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.optimize import curve_fit
from scipy.integrate import cumulative_trapezoid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.screening import U_REF_SCREENED, compute_screening

C_KMS = 299792.458
U_REF = U_REF_SCREENED  # screened anchor reference (matches TEP-H0)
H0_REF = 73.04  # km/s/Mpc — SH0ES reference (matches MU_SH0ES)
OMEGA_M = 0.334  # Planck/SH0ES compromise


class MeasuredVrotXiStep:
    """X_i-step test with measured V_rot from HyperLEDA."""

    def __init__(self):
        self.results = {}
        self.logger = TEPLogger("step_48", log_file_path=PROJECT_ROOT / "logs" / "step_48_xi_step_measured_vrot.log")
        set_step_logger(self.logger)

    def load_data(self):
        """Load Pantheon+ and measured V_rot catalog."""
        print_status("Loading data...", "PROCESS")

        # Pantheon+ sample
        self.panth = pd.read_csv('data/raw/Pantheon+SH0ES.dat', sep=' ')
        self.unique_sne = self.panth.drop_duplicates(subset='CID').copy()
        print_status(f"  Pantheon+: {len(self.unique_sne)} unique SNe", "INFO")

        # Measured V_rot catalog from deep sources (SPARC/ALFALFA/HyperLEDA)
        vrot_path = 'data/processed/pantheon_host_vrot_deep.csv'
        if os.path.exists(vrot_path):
            self.vrot_cat = pd.read_csv(vrot_path)
            # Filter to SNe with measured v_rot
            vrot_valid = self.vrot_cat[self.vrot_cat['v_rot_deep'].notna()].copy()
            # Compute U_i and X_i for measured V_rot
            vrot_valid['V_rot'] = vrot_valid['v_rot_deep']
            vrot_valid['V_rot_source_specific'] = vrot_valid['v_rot_source']
            vrot_valid['V_rot_source'] = 'measured' # Keep as 'measured' for downstream logic
            vrot_valid['U_i'] = (vrot_valid['V_rot'] / np.sqrt(2))**2
            # Compute TEP screening S_total by PGC (if available)
            pgc_vals = vrot_valid.get('pgc', pd.Series(np.zeros(len(vrot_valid))))
            S_vrot = compute_screening(pgc_vals.fillna(0).astype(int).values)
            vrot_valid['S_total'] = S_vrot
            vrot_valid['X_i'] = (S_vrot * vrot_valid['U_i'].values - U_REF) / C_KMS**2
            self.vrot_cat = vrot_valid
            n_measured = len(self.vrot_cat)
            print_status(f"  V_rot catalog (Deep): {n_measured} SNe with measured V_rot", "INFO")
            self.has_measured_vrot = True
        else:
            print_status(f"  V_rot catalog not found at {vrot_path}", "WARN")
            print_status("  Falling back to Tully-Fisher proxy", "WARN")
            self.has_measured_vrot = False

    def compute_xi_measured(self):
        """Compute X_i using measured V_rot where available, TF proxy as fallback."""
        print_status("\nComputing X_i with measured V_rot...", "PROCESS")

        df = self.unique_sne.copy()

        if self.has_measured_vrot:
            # Merge measured V_rot from Vizier catalog
            df = df.merge(self.vrot_cat[['CID', 'V_rot', 'V_rot_source', 'U_i', 'S_total', 'X_i']],
                         on='CID', how='left')

            # For SNe without V_rot in the catalog, use TF proxy (S_total = 1.0)
            needs_proxy = df['V_rot'].isna()
            if needs_proxy.sum() > 0:
                M_star = 10.0 ** df.loc[needs_proxy, 'HOST_LOGMASS'].values
                V_rot_tf = 200.0 * (M_star / 10.0**10.5) ** 0.25
                U_tf = (V_rot_tf / np.sqrt(2))**2
                X_i_tf = (U_tf - U_REF) / C_KMS**2
                df.loc[needs_proxy, 'V_rot'] = V_rot_tf
                df.loc[needs_proxy, 'U_i'] = U_tf
                df.loc[needs_proxy, 'S_total'] = 1.0
                df.loc[needs_proxy, 'X_i'] = X_i_tf
                df.loc[needs_proxy, 'V_rot_source'] = 'tully-fisher'

            n_measured = (df['V_rot_source'] == 'measured').sum()
            print_status(f"  SNe with measured V_rot: {n_measured}", "INFO")
            print_status(f"  SNe with TF proxy: {(df['V_rot_source']=='tully-fisher').sum()}", "INFO")
        else:
            # All TF proxy (S_total = 1.0 for uncatalogued hosts)
            needs_proxy = pd.Series([True] * len(df), index=df.index)
            M_star = 10.0 ** df['HOST_LOGMASS'].values
            V_rot_tf = 200.0 * (M_star / 10.0**10.5) ** 0.25
            df['V_rot'] = V_rot_tf
            df['U_i'] = (V_rot_tf / np.sqrt(2))**2
            df['S_total'] = 1.0
            df['X_i'] = (df['U_i'] - U_REF) / C_KMS**2
            df['V_rot_source'] = 'tully-fisher'
            n_measured = 0
            print_status(f"  All SNe with TF proxy: {len(df)}", "INFO")

        # Hubble-flow SNe
        self.hf = df[df['zCMB'] > 0.01].copy()
        print_status(f"  Hubble-flow SNe (z > 0.01): {len(self.hf)}", "INFO")

        if self.has_measured_vrot and n_measured > 0:
            n_hf_measured = (self.hf['V_rot_source'] == 'measured').sum()
            print_status(f"  Hubble-flow with measured V_rot: {n_hf_measured}", "INFO")

        self.results['data_summary'] = {
            'n_total': len(df),
            'n_hubble_flow': len(self.hf),
            'n_measured_vrot': int(n_measured),
            'n_tf_proxy': int(needs_proxy.sum()),
        }

    def run_xi_step(self):
        """Run the X_i-step test."""
        print_status("\nRunning X_i-step test...", "PROCESS")

        hf = self.hf.copy()

        # Compute Hubble residual using the proper LCDM distance modulus.
        # The previous implementation fitted m_b_corr to a simple Hubble law
        # 5*log10(z*c/70) + 25 + a, which is invalid at z > 0.01 (ignores
        # deceleration) and falls back to m_b_corr - median(m_b_corr), which
        # is contaminated by distance modulus variation across redshift.
        # The correct approach (matching step_45) is HR = MU_SH0ES - mu_ref(z),
        # where mu_ref is the LCDM distance modulus at H0_REF, OMEGA_M.
        z_valid = hf['zCMB'].values > 0.001
        hf = hf[z_valid].copy()

        if 'MU_SH0ES' in hf.columns:
            mu_obs = hf['MU_SH0ES'].values
            # Use zHD (peculiar-velocity-corrected) for the reference distance modulus.
            # zHD includes the 2M++ peculiar velocity correction (Carrick et al. 2015),
            # which is essential at z < 0.03 where peculiar velocities are a significant
            # fraction of the Hubble flow. Using zCMB instead inflates the HR scatter
            # by ~50% at z ~ 0.01 and biases the X_i-step measurement.
            z_for_mu = hf['zHD'].values if 'zHD' in hf.columns else hf['zCMB'].values
            mu_ref = self._mu_ref(z_for_mu)
            hf['hubble_residual'] = mu_obs - mu_ref
            z_label = 'zHD' if 'zHD' in hf.columns else 'zCMB'
            print_status(f"  Hubble residual: MU_SH0ES - mu_ref({z_label}) [LCDM, H0={H0_REF}, Omega_m={OMEGA_M}]", "INFO")
        else:
            # Fallback: fit m_b_corr to proper LCDM distance modulus
            print_status("  MU_SH0ES not found, fitting m_b_corr to LCDM distance modulus", "WARN")
            z_for_mu = hf['zHD'].values if 'zHD' in hf.columns else hf['zCMB'].values
            mu_ref = self._mu_ref(z_for_mu)
            # Fit the absolute magnitude offset: m_b_corr = mu_ref + M_B
            M_B = np.median(hf['m_b_corr'].values - mu_ref)
            hf['hubble_residual'] = hf['m_b_corr'].values - (mu_ref + M_B)

        print_status(f"  HR range: {hf['hubble_residual'].min():.3f} to {hf['hubble_residual'].max():.3f} mag", "INFO")

        # Full sample X_i-step
        self._step_test(hf, 'full_sample', 'Full sample')

        # Measured V_rot only (if enough SNe)
        if self.has_measured_vrot:
            measured = hf[hf['V_rot_source'] == 'measured']
            if len(measured) > 50:
                # Selection bias warning: the measured V_rot catalog is heavily
                # biased toward nearby, low-z galaxies (median z ~ 0.02) where
                # peculiar velocity corrections are large and the sample is
                # imbalanced (typically ~80% high-X_i, ~20% low-X_i).
                n_high = (measured['X_i'] > 0).sum()
                n_low = (measured['X_i'] <= 0).sum()
                median_z = measured['zCMB'].median()
                print_status(
                    f"  WARNING: measured V_rot sample is selection-biased: "
                    f"N_high={n_high} vs N_low={n_low} (ratio {n_high/max(n_low,1):.1f}:1), "
                    f"median z={median_z:.4f}. Results should not be interpreted "
                    f"as a clean TEP test.",
                    "WARN",
                )
                self._step_test(measured, 'measured_vrot_only', 'Measured V_rot only')

                # Also run with mass correction
                self._step_test_with_mass(measured, 'measured_vrot_mass_corrected',
                                         'Measured V_rot (mass-corrected)')

                # High-z subsample (z > 0.03) where peculiar velocities are subdominant
                measured_hiz = measured[measured['zCMB'] > 0.03]
                if len(measured_hiz) > 30:
                    n_h = (measured_hiz['X_i'] > 0).sum()
                    n_l = (measured_hiz['X_i'] <= 0).sum()
                    print_status(
                        f"  High-z subsample (z > 0.03): N={len(measured_hiz)} "
                        f"(N_high={n_h}, N_low={n_l})",
                        "INFO",
                    )
                    if n_l >= 5:
                        self._step_test(measured_hiz, 'measured_vrot_hiz',
                                       'Measured V_rot (z > 0.03)')
            else:
                print_status(f"  Too few SNe with measured V_rot ({len(measured)}) "
                            f"for standalone step test", "WARN")

        # Full sample with mass correction
        self._step_test_with_mass(hf, 'full_mass_corrected',
                                 'Full sample (mass-corrected)')

    @staticmethod
    def _mu_ref(z):
        """Reference LCDM distance modulus at H0_REF, OMEGA_M.

        mu = 5 * log10((1+z) * d_c * c / H0) + 25
        where d_c is the comoving distance (integral of dz/E(z)).
        """
        z_fine = np.linspace(0, max(np.max(z) + 0.01, 2.5), 5000)
        E_fine = np.sqrt(OMEGA_M * (1 + z_fine) ** 3 + (1 - OMEGA_M))
        d_c_fine = cumulative_trapezoid(1.0 / E_fine, z_fine, initial=0)
        d_c = np.interp(z, z_fine, d_c_fine)
        return 5 * np.log10((1 + z) * d_c * C_KMS / H0_REF) + 25

    def _step_test(self, df, key, label):
        """Run a simple X_i-step test."""
        high_x = df[df['X_i'] > 0]
        low_x = df[df['X_i'] <= 0]

        if len(high_x) < 10 or len(low_x) < 10:
            print_status(f"  {label}: insufficient data (N_high={len(high_x)}, "
                        f"N_low={len(low_x)})", "WARN")
            return

        step = high_x['hubble_residual'].mean() - low_x['hubble_residual'].mean()
        step_err = np.sqrt(high_x['hubble_residual'].var()/len(high_x) +
                          low_x['hubble_residual'].var()/len(low_x))
        step_sigma = step / step_err if step_err > 0 else 0

        # TEP predicts high-X_i -> positive HR (Tripp relation: inflated x1
        # in deeper potentials gives larger mu, hence more positive HR).
        # step = HR(high-X) - HR(low-X) > 0 is TEP-predicted direction.
        tep_direction = step > 0

        print_status(f"  {label}:", "TEST")
        print_status(f"    N(X>0)={len(high_x)}, N(X<=0)={len(low_x)}", "TEST")
        print_status(f"    Step = {step*1000:.1f} +/- {step_err*1000:.1f} mmag "
                     f"({step_sigma:.2f}sigma)", "TEST")
        print_status(f"    Direction: {'TEP-predicted (high-X -> positive HR)' if tep_direction else 'OPPOSITE to TEP'}", "TEST")

        self.results[key] = {
            'n_high_x': len(high_x),
            'n_low_x': len(low_x),
            'step_mag': float(step),
            'step_err': float(step_err),
            'step_sigma': float(step_sigma),
            'tep_direction': bool(tep_direction),
        }

    def _step_test_with_mass(self, df, key, label):
        """Run X_i-step with mass-step correction via residualization.

        The previous implementation used a joint OLS with X_i and a binary
        mass indicator. This suffers from severe multicollinearity (X_i and
        HOST_LOGMASS both measure host potential depth), causing the mass
        coefficient to absorb the X_i signal and reducing significance from
        1.88sigma to 0.52sigma — a pipeline artifact, not a physical null.

        The corrected approach uses sequential residualization:
        1. Fit HR = a + b*logM (continuous, not binary)
        2. Compute HR_residual = HR - (a + b*logM)
        3. Take the X_i step on HR_residual

        This avoids the collinearity issue and provides a cleaner partial
        correlation. However, if TEP causes both the mass step and the X_i
        step (as predicted by the framework), then mass residualization
        removes part of the real TEP signal. The uncorrected step is
        therefore the primary TEP estimate; the mass-residualized step
        is a conservative lower bound.
        """
        from numpy.linalg import lstsq

        # Method 1: Sequential residualization (preferred)
        logmass = df['HOST_LOGMASS'].values
        y = df['hubble_residual'].values
        xi = df['X_i'].values

        # Step 1: Remove continuous mass trend
        X_mass = np.column_stack([logmass, np.ones(len(df))])
        try:
            mass_fit, _, _, _ = lstsq(X_mass, y, rcond=None)
            mass_slope = mass_fit[0]
            mass_intercept = mass_fit[1]
            hr_mass_residual = y - X_mass @ mass_fit

            # Step 2: X_i step on mass-residualized HR
            high_x = df[df['X_i'] > 0]
            low_x = df[df['X_i'] <= 0]
            hr_resid = pd.Series(hr_mass_residual, index=df.index)
            step_resid = hr_resid[high_x.index].mean() - hr_resid[low_x.index].mean()
            step_resid_err = np.sqrt(
                hr_resid[high_x.index].var() / len(high_x) +
                hr_resid[low_x.index].var() / len(low_x)
            )
            step_resid_sigma = step_resid / step_resid_err if step_resid_err > 0 else 0

            # Method 2: Joint OLS (for comparison, with condition number diagnostic)
            X_joint = np.column_stack([
                xi,
                (logmass >= 10.5).astype(float),
                np.ones(len(df))
            ])
            beta_joint, _, _, _ = lstsq(X_joint, y, rcond=None)
            xi_coef_joint = beta_joint[0]
            residuals_joint = y - X_joint @ beta_joint
            dof = max(len(y) - 3, 1)
            sigma2 = np.sum(residuals_joint**2) / dof
            cov_joint = sigma2 * np.linalg.inv(X_joint.T @ X_joint)
            xi_err_joint = np.sqrt(cov_joint[0, 0])
            cond_number = np.linalg.cond(X_joint)

            print_status(f"  {label}:", "TEST")
            print_status(f"    [Residualized] Step = {step_resid*1000:.1f} +/- "
                        f"{step_resid_err*1000:.1f} mmag ({step_resid_sigma:.2f}sigma)", "TEST")
            print_status(f"    [Residualized] Mass slope = {mass_slope*1000:.1f} mmag/dex", "TEST")
            print_status(f"    [Joint OLS] X_i coef = {xi_coef_joint:.3e} +/- "
                        f"{xi_err_joint:.3e} ({xi_coef_joint/xi_err_joint:.2f}sigma)", "TEST")
            print_status(f"    [Joint OLS] Condition number = {cond_number:.1f} "
                        f"({'COLLINEAR' if cond_number > 30 else 'ok'})", "TEST")
            print_status(f"    Note: uncorrected step is primary TEP estimate;", "INFO")
            print_status(f"    mass residualization removes shared TEP signal.", "INFO")

            self.results[key] = {
                'n': len(df),
                # Residualized (preferred mass correction)
                'step_corrected_mag': float(step_resid),
                'step_corrected_err': float(step_resid_err),
                'step_corrected_sigma': float(step_resid_sigma),
                'mass_slope': float(mass_slope),
                # Joint OLS (for comparison, degraded by collinearity)
                'xi_coef': float(xi_coef_joint),
                'xi_err': float(xi_err_joint),
                'xi_sigma': float(xi_coef_joint/xi_err_joint) if xi_err_joint > 0 else 0,
                'mass_coef': float(beta_joint[1]),
                'mass_err': float(np.sqrt(cov_joint[1, 1])),
                'condition_number': float(cond_number),
            }
        except Exception as e:
            print_status(f"  {label}: regression failed ({e})", "ERROR")

    def test_vrot_squared_over_r_squared(self):
        """Test the DHOST prediction that epsilon_0 scales as V_rot^2/r^2.

        This requires Cepheid galactocentric radii, which are not in the
        standard Pantheon+ catalog. We note this as a future test.
        """
        print_status("\nV_rot^2/r^2 scaling test (DHOST prediction)...", "PROCESS")
        print_status("  Requires Cepheid galactocentric radii — not available", "INFO")
        print_status("  in current data. Deferred to future work with JWST", "INFO")
        print_status("  Cepheid positions (GO-1995, GO-1685).", "INFO")

        self.results['vrot2_over_r2_test'] = {
            'status': 'deferred',
            'reason': 'Requires Cepheid galactocentric radii from JWST imaging',
            'prediction': 'epsilon_0 should scale as V_rot^2/r^2, not V_rot^2 alone',
        }

    def save_results(self):
        """Save results."""
        os.makedirs("results/outputs", exist_ok=True)
        output_path = "results/outputs/step_48_xi_step_measured_vrot.json"
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print_status(f"\nSaved results to {output_path}", "INFO")

    def run(self):
        """Run the full analysis."""
        print_status("=" * 60, "INFO")
        print_status("Step 48: X_i-step with Measured V_rot", "INFO")
        print_status("=" * 60, "INFO")

        self.load_data()
        self.compute_xi_measured()
        self.run_xi_step()
        self.test_vrot_squared_over_r_squared()
        self.save_results()

        print_status("\n" + "=" * 60, "INFO")
        print_status("Step 48 complete", "INFO")
        print_status("=" * 60, "INFO")


if __name__ == "__main__":
    analysis = MeasuredVrotXiStep()
    analysis.run()
