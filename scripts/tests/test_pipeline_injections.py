#!/usr/bin/env python3
"""
Synthetic Known-Answer Test Suite for TEP-VOID

Tests:
1. Test A (Void Falsification Pipeline): Injects KBC signal into Pantheon+ 
   and tests recovery via exact delta_chi^2 / marginalized-zero-point machinery.
2. Test B (kappa_Cep Regression): Injects kappa signal into synthetic host data
   and tests recovery via WLS in H0-space.
"""

import sys
from pathlib import Path
import numpy as np
from scipy import integrate
import statsmodels.api as sm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Import machinery from step 32
from scripts.steps.step_32_redshift_decay_profile import Step32RedshiftDecayProfile

def test_A_void_falsification():
    print("\n" + "="*60)
    print("Test A: Void-Falsification Pipeline (Synthetic Pantheon+)")
    print("="*60)
    
    step32 = Step32RedshiftDecayProfile()
    
    # 1. Load Pantheon+ Z grid and Covariance
    dat_path = PROJECT_ROOT / "data" / "raw" / "Pantheon+SH0ES.dat"
    cov_path = PROJECT_ROOT / "data" / "raw" / "Pantheon+SH0ES_STAT+SYS.cov"
    
    if not dat_path.exists() or not cov_path.exists():
        print("Data files not found. Using diagonal synthetic covariance.")
        N = 1000
        z_array = np.linspace(0.01, 1.0, N)
        cov_full = np.diag(np.ones(N) * 0.15**2)
    else:
        import pandas as pd
        df = pd.read_csv(dat_path, delim_whitespace=True)
        # Filter z > 0.05
        mask = df['zHD'] >= 0.05
        z_array = df.loc[mask, 'zHD'].values
        N = len(z_array)
        
        with open(cov_path) as f:
            n_tot = int(f.readline().strip())
            data = np.fromstring(f.read(), sep="\n")
        cov_full_all = data[:n_tot * n_tot].reshape(n_tot, n_tot)
        
        # Submatrix
        idx = np.where(mask)[0]
        cov_full = cov_full_all[np.ix_(idx, idx)]
        
    print(f"Loaded {N} SNe with {cov_full.shape} covariance.")
    
    # Precompute LCDM mu
    mu_lcdm = np.zeros(N)
    for i, z in enumerate(z_array):
        integral, _ = integrate.quad(lambda zp: 1.0 / step32._E(zp), 0, z)
        d_L = (1 + z) * step32.C_KMS * integral / 73.0
        mu_lcdm[i] = 5.0 * np.log10(d_L) + 25.0
        
    # KBC Signal (from step 32 Gaussian model approximation)
    delta_h0 = 73.0 - 67.4
    h0_kbc = 67.4 + delta_h0 * np.exp(-z_array**2 / (2 * 0.82**2))
    
    # Compute the expected mu_kbc if H0(z) varies like h0_kbc
    mu_kbc = np.zeros(N)
    for i, z in enumerate(z_array):
        integral, _ = integrate.quad(lambda zp: 1.0 / step32._E(zp), 0, z)
        # Here we approximate the luminosity distance using the local effective H0(z)
        d_L = (1 + z) * step32.C_KMS * integral / h0_kbc[i]
        mu_kbc[i] = 5.0 * np.log10(d_L) + 25.0
        
    # Inject 1: KBC Signal
    np.random.seed(42)
    noise = np.zeros(N)
    mu_obs_kbc = mu_kbc + noise
    
    # Inject 2: Flat (Null) Signal
    mu_obs_flat = mu_lcdm + noise
    
    def evaluate_pipeline(mu_obs):
        # We use the unbinned 1701x1701 (or NxN) covariance directly
        chi2_flat = step32._chi2_mu_space_marginalized(mu_obs, mu_lcdm, cov_full)
        chi2_kbc = step32._chi2_mu_space_marginalized(mu_obs, mu_kbc, cov_full)
        delta_chi2 = chi2_flat - chi2_kbc
        return delta_chi2
        
    delta_chi2_inj_kbc = evaluate_pipeline(mu_obs_kbc)
    delta_chi2_inj_flat = evaluate_pipeline(mu_obs_flat)
    
    print(f"Injected KBC signal  => Recovered Delta_Chi^2 (Flat - KBC) = {delta_chi2_inj_kbc:.2f} (Expected > 0, strongly prefers KBC)")
    print(f"Injected Null signal => Recovered Delta_Chi^2 (Flat - KBC) = {delta_chi2_inj_flat:.2f} (Expected ~ 0, no false preference)")
    
    assert delta_chi2_inj_kbc > 10, "Failed to recover strong preference for KBC when injected!"
    assert abs(delta_chi2_inj_flat) < 15, "Spurious rejection found in null test!"
    print("Test A PASSED.\n")


def test_B_kappa_regression():
    print("="*60)
    print("Test B: kappa_Cep Regression (Synthetic Host Data)")
    print("="*60)
    
    N_hosts = 33
    np.random.seed(123)
    # Real X_i distribution is typical of galaxies V_rot in 50-300 km/s
    X_i = np.random.uniform(0, 1.5e-7, N_hosts) 
    
    # H0-space machinery:
    # dH0/dmu = -H0 * ln(10)/5 
    # Delta_mu = - kappa X_i
    # => Delta H0 = kappa * (H0 ln 10 / 5) * X_i
    H0 = 73.0
    factor = H0 * np.log(10) / 5.0
    
    sigma_v = 150.0 # km/s
    # Velocity dispersion mapped to H0 uncertainty at typical cz ~ 5000 km/s:
    # sigma_H0 = H0 * (sigma_v / cz)
    # Average H0 err ~ 2.0 km/s/Mpc
    sigma_H0 = 2.0 
    weights = 1.0 / (sigma_H0**2)
    
    def run_wls(kappa_inj):
        # Inject Delta H0
        dH0 = kappa_inj * factor * X_i
        # Add noise
        noise = np.random.normal(0, sigma_H0, N_hosts)
        obs_dH0 = dH0 + noise
        
        # WLS regression without intercept (since Delta H0 = 0 at X_i = 0)
        # Actually, step_44 and step_36 regressions fit an intercept too for robustness
        X_design = sm.add_constant(X_i)
        wls_model = sm.WLS(obs_dH0, X_design, weights=np.ones(N_hosts)*weights).fit()
        
        hat_kappa = wls_model.params[1] / factor
        err_kappa = wls_model.bse[1] / factor
        return hat_kappa, err_kappa
        
    test_cases = [0.0, 0.4e6, 0.8e6]
    n_realizations = 100
    for k in test_cases:
        hat_k_list = []
        err_k_list = []
        for _ in range(n_realizations):
            hat, err = run_wls(k)
            hat_k_list.append(hat)
            err_k_list.append(err)
            
        mean_hat_k = np.mean(hat_k_list)
        std_hat_k = np.std(hat_k_list)
        mean_err_k = np.mean(err_k_list)
        
        print(f"Injected kappa = {k/1e6:.2f}e6 => Recovered mean kappa = {mean_hat_k/1e6:.2f}e6 +/- {mean_err_k/1e6:.2f}e6 (scatter: {std_hat_k/1e6:.2f}e6)")
        
        # Pass condition: recovered brackets injected
        assert abs(mean_hat_k - k) < 3 * mean_err_k, f"Recovery failed for {k}"
        if k == 0:
            assert abs(mean_hat_k) < 3 * mean_err_k, "Null injection returned false positive!"
            
    print("Test B PASSED.\n")


if __name__ == "__main__":
    test_A_void_falsification()
    test_B_kappa_regression()
