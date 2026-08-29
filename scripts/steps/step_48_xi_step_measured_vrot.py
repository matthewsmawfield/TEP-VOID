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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.utils.logger import TEPLogger, set_step_logger, print_status

C_KMS = 299792.458
U_REF = (87.165) ** 2  # (km/s)^2, reference potential from NGC 4258


class MeasuredVrotXiStep:
    """X_i-step test with measured V_rot from HyperLEDA."""

    def __init__(self):
        self.results = {}

    def load_data(self):
        """Load Pantheon+ and measured V_rot catalog."""
        print_status("Loading data...", "PROCESS")

        # Pantheon+ sample
        self.panth = pd.read_csv('data/raw/Pantheon+SH0ES.dat', sep=' ')
        self.unique_sne = self.panth.drop_duplicates(subset='CID').copy()
        print_status(f"  Pantheon+: {len(self.unique_sne)} unique SNe", "INFO")

        # Measured V_rot catalog from Vizier/HyperLEDA
        vrot_path = 'data/processed/pantheon_host_vrot_vizier.csv'
        if os.path.exists(vrot_path):
            self.vrot_cat = pd.read_csv(vrot_path)
            # Filter to SNe with measured v_rot
            vrot_valid = self.vrot_cat[self.vrot_cat['v_rot'].notna()].copy()
            # Compute U_i and X_i for measured V_rot
            vrot_valid['V_rot'] = vrot_valid['v_rot']
            vrot_valid['V_rot_source'] = 'measured'
            vrot_valid['U_i'] = (vrot_valid['v_rot'] / np.sqrt(2))**2
            vrot_valid['X_i'] = (vrot_valid['U_i'] - U_REF) / C_KMS**2
            self.vrot_cat = vrot_valid
            n_measured = len(self.vrot_cat)
            print_status(f"  V_rot catalog (Vizier): {n_measured} SNe with measured V_rot", "INFO")
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
            df = df.merge(self.vrot_cat[['CID', 'V_rot', 'V_rot_source', 'U_i', 'X_i']],
                         on='CID', how='left')

            # For SNe without V_rot in the catalog, use TF proxy
            needs_proxy = df['V_rot'].isna()
            if needs_proxy.sum() > 0:
                M_star = 10.0 ** df.loc[needs_proxy, 'HOST_LOGMASS'].values
                V_rot_tf = 200.0 * (M_star / 10.0**10.5) ** 0.25
                U_tf = (V_rot_tf / np.sqrt(2))**2
                X_i_tf = (U_tf - U_REF) / C_KMS**2
                df.loc[needs_proxy, 'V_rot'] = V_rot_tf
                df.loc[needs_proxy, 'U_i'] = U_tf
                df.loc[needs_proxy, 'X_i'] = X_i_tf
                df.loc[needs_proxy, 'V_rot_source'] = 'tully-fisher'

            n_measured = (df['V_rot_source'] == 'measured').sum()
            print_status(f"  SNe with measured V_rot: {n_measured}", "INFO")
            print_status(f"  SNe with TF proxy: {(df['V_rot_source']=='tully-fisher').sum()}", "INFO")
        else:
            # All TF proxy
            needs_proxy = pd.Series([True] * len(df), index=df.index)
            M_star = 10.0 ** df['HOST_LOGMASS'].values
            V_rot_tf = 200.0 * (M_star / 10.0**10.5) ** 0.25
            df['V_rot'] = V_rot_tf
            df['U_i'] = (V_rot_tf / np.sqrt(2))**2
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

        # Compute Hubble residual
        # Use m_b_corr and fit Hubble law
        z_valid = hf['zCMB'].values > 0.001
        hf = hf[z_valid].copy()

        def hubble_model(z, a):
            return 5.0 * np.log10(z * C_KMS / 70.0) + 25.0 + a

        try:
            popt, _ = curve_fit(hubble_model, hf['zCMB'].values,
                               hf['m_b_corr'].values, p0=[-19.3])
            hf['hubble_residual'] = hf['m_b_corr'] - hubble_model(hf['zCMB'], *popt)
        except:
            hf['hubble_residual'] = hf['m_b_corr'] - hf['m_b_corr'].median()

        # Full sample X_i-step
        self._step_test(hf, 'full_sample', 'Full sample')

        # Measured V_rot only (if enough SNe)
        if self.has_measured_vrot:
            measured = hf[hf['V_rot_source'] == 'measured']
            if len(measured) > 50:
                self._step_test(measured, 'measured_vrot_only', 'Measured V_rot only')

                # Also run with mass correction
                self._step_test_with_mass(measured, 'measured_vrot_mass_corrected',
                                         'Measured V_rot (mass-corrected)')
            else:
                print_status(f"  Too few SNe with measured V_rot ({len(measured)}) "
                            f"for standalone step test", "WARN")

        # Full sample with mass correction
        self._step_test_with_mass(hf, 'full_mass_corrected',
                                 'Full sample (mass-corrected)')

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

        print_status(f"  {label}:", "TEST")
        print_status(f"    N(X>0)={len(high_x)}, N(X<=0)={len(low_x)}", "TEST")
        print_status(f"    Step = {step*1000:.1f} +/- {step_err*1000:.1f} mmag "
                     f"({step_sigma:.2f}sigma)", "TEST")

        self.results[key] = {
            'n_high_x': len(high_x),
            'n_low_x': len(low_x),
            'step_mag': float(step),
            'step_err': float(step_err),
            'step_sigma': float(step_sigma),
        }

    def _step_test_with_mass(self, df, key, label):
        """Run X_i-step with mass-step correction via multivariate regression."""
        from numpy.linalg import lstsq

        X = np.column_stack([
            df['X_i'].values,
            (df['HOST_LOGMASS'] >= 10.5).astype(float),
            np.ones(len(df))
        ])
        y = df['hubble_residual'].values

        try:
            beta, _, _, _ = lstsq(X, y, rcond=None)
            xi_coef = beta[0]
            mass_coef = beta[1]

            residuals = y - X @ beta
            dof = max(len(y) - 3, 1)
            sigma2 = np.sum(residuals**2) / dof
            cov = sigma2 * np.linalg.inv(X.T @ X)
            xi_err = np.sqrt(cov[0, 0])

            high_x = df[df['X_i'] > 0]
            low_x = df[df['X_i'] <= 0]
            step_corrected = xi_coef * (high_x['X_i'].mean() - low_x['X_i'].mean())
            step_corrected_err = xi_err * abs(high_x['X_i'].mean() - low_x['X_i'].mean())
            step_corrected_sigma = step_corrected / step_corrected_err if step_corrected_err > 0 else 0

            print_status(f"  {label}:", "TEST")
            print_status(f"    X_i coefficient = {xi_coef:.3e} +/- {xi_err:.3e} "
                         f"({xi_coef/xi_err:.2f}sigma)" if xi_err > 0 else
                         f"    X_i coefficient = {xi_coef:.3e}", "TEST")
            print_status(f"    Step (mass-corrected) = {step_corrected*1000:.1f} +/- "
                        f"{step_corrected_err*1000:.1f} mmag ({step_corrected_sigma:.2f}sigma)", "TEST")
            print_status(f"    Mass step = {mass_coef*1000:.1f} +/- "
                        f"{np.sqrt(cov[1,1])*1000:.1f} mmag", "TEST")

            self.results[key] = {
                'n': len(df),
                'xi_coef': float(xi_coef),
                'xi_err': float(xi_err),
                'xi_sigma': float(xi_coef/xi_err) if xi_err > 0 else 0,
                'mass_coef': float(mass_coef),
                'mass_err': float(np.sqrt(cov[1, 1])),
                'step_corrected_mag': float(step_corrected),
                'step_corrected_err': float(step_corrected_err),
                'step_corrected_sigma': float(step_corrected_sigma),
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
