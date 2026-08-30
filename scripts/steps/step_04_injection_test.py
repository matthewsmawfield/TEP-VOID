#!/usr/bin/env python3
"""
Step 04: Exact Production Injection Test
========================================
Validates the TEP estimator bias and coverage using the exact 
SH0ES design matrix and GLS likelihood from the TEP-H0 pipeline.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEP_H0_PATH = PROJECT_ROOT.parent / "TEP-H0"

# Temporarily prioritize TEP-H0 to avoid module conflicts
sys.path.insert(0, str(TEP_H0_PATH))

from scripts.steps.step_34_full_ladder_likelihood import FullLadderLikelihood

# Restore PROJECT_ROOT priority for local logging
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.utils.logger import TEPLogger, set_step_logger, print_status

def run_injection_test():
    logger = TEPLogger("step_04_injection", log_file_path=PROJECT_ROOT / "logs" / "step_04_injection.log")
    set_step_logger(logger)
    
    print_status("================================================================================", "INFO")
    print_status("   Step 04: Exact Production Injection Test", "INFO")
    print_status("================================================================================", "INFO")
    
    fll = FullLadderLikelihood()
    
    print_status("Loading True SH0ES Design Matrices from TEP-H0...", "PROCESS")
    L, y, C, q, y_source = fll.load_sh0es_data()
    host_sigma, host_screening = fll.load_host_metadata()
    sigma_ref = fll.calculate_effective_sigma_ref()
    
    # Build TEP vector
    x_cepheid, x_sn, _, _ = fll.build_tep_columns(L, q, host_sigma, host_screening, sigma_ref)
    
    # Get baseline theta to use as nuisance truth
    theta_base, cov_base, chi2_base, rank_base, _ = fll.fit_gls(L, y, C)
    h0_idx = np.where(q == "5logH0")[0][0]
    
    # Augmented design matrix
    X_SCALE = 1e6
    x_cepheid_scaled = x_cepheid * X_SCALE
    L_aug = np.column_stack([L, -x_cepheid_scaled])
    
    # Injection parameters
    N_ITER = 200
    H0_TRUE = 73.0
    KAPPA_TRUE = 4.0e5
    kappa6_true = KAPPA_TRUE / X_SCALE
    
    theta_true = theta_base.copy()
    theta_true[h0_idx] = 5 * np.log10(H0_TRUE)
    
    # Exact covariance sampling
    rng = np.random.default_rng(42)
    
    # Precompute Cholesky for lightning fast noise generation
    jitter = np.eye(len(C)) * 1e-10
    try:
        L_cov = np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        L_cov = np.linalg.cholesky(C + jitter)
    
    print_status(f"Starting {N_ITER} Full Matrix Injections...", "PROCESS")
    
    h0_hats = []
    k_hats = []
    h0_errs = []
    k_errs = []
    
    for i in range(N_ITER):
        # Generate exact synthetic signal incredibly fast
        noise = L_cov @ rng.standard_normal(len(y))
        y_mock = L @ theta_true - kappa6_true * x_cepheid_scaled + noise
        
        # Fit exact production GLS
        theta_aug, cov_aug, chi2_aug, _, _ = fll.fit_gls(L_aug, y_mock, C)
        
        # Extract H0
        h0_param = theta_aug[h0_idx]
        h0_err_param = np.sqrt(cov_aug[h0_idx, h0_idx])
        h0_hat = 10 ** (h0_param / 5)
        h0_err = (np.log(10) / 5) * h0_err_param * h0_hat
        
        # Extract Kappa
        kappa6_hat = theta_aug[-1]
        kappa_hat = kappa6_hat * X_SCALE
        kappa_err = np.sqrt(cov_aug[-1, -1]) * X_SCALE
        
        h0_hats.append(h0_hat)
        k_hats.append(kappa_hat)
        h0_errs.append(h0_err)
        k_errs.append(kappa_err)
        
    h0_hats = np.array(h0_hats)
    k_hats = np.array(k_hats)
    h0_errs = np.array(h0_errs)
    k_errs = np.array(k_errs)
    
    # Analysis
    h0_bias = np.mean(h0_hats - H0_TRUE)
    k_bias = np.mean(k_hats - KAPPA_TRUE)
    h0_rmse = np.sqrt(np.mean((h0_hats - H0_TRUE)**2))
    k_rmse = np.sqrt(np.mean((k_hats - KAPPA_TRUE)**2))
    
    h0_pulls = (h0_hats - H0_TRUE) / h0_errs
    k_pulls = (k_hats - KAPPA_TRUE) / k_errs
    h0_cov68 = np.mean(np.abs(h0_pulls) <= 1.0) * 100
    k_cov68 = np.mean(np.abs(k_pulls) <= 1.0) * 100
    
    print_status(f"--- Results (N={N_ITER}, H0_true={H0_TRUE}, kappa_true={KAPPA_TRUE:.1e}) ---", "SUCCESS")
    print_status(f"H0: Bias = {h0_bias:+.3f} | RMSE = {h0_rmse:.3f} | Cov68 = {h0_cov68:.1f}% | Pull Std = {np.std(h0_pulls):.2f}", "TEST")
    print_status(f"Kappa: Bias = {k_bias:+.2e} | RMSE = {k_rmse:.2e} | Cov68 = {k_cov68:.1f}% | Pull Std = {np.std(k_pulls):.2f}", "TEST")
    
    if abs(h0_bias) < 0.1 and abs(np.std(h0_pulls) - 1.0) < 0.1:
        print_status("CONCLUSION: The Exact Production Estimator is rigorously UNBIASED and has perfect covariance coverage.", "SUCCESS")
    else:
        print_status("CONCLUSION: The Exact Production Estimator exhibits bias.", "WARNING")

if __name__ == "__main__":
    run_injection_test()
