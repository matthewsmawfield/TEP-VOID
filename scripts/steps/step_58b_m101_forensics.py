#!/usr/bin/env python3
"""
Gate B: Predictor Integrity (M101 Forensics - Validation Pass)
Step 58: M101 Localized Predictor Reconstruction & Out-of-Sample Audit
======================================================================
Tests whether V^4/R^2 is a superior global descriptor of the optical-NIR Cepheid
offset, comparing M1 (X=V^2) vs M2 (X=V^4/R^2). Both models are fit to the dataset
*excluding* M101, then used to predict M101 out-of-sample.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status

def student_t_nll(params, x, y, yerr, nu=4.0):
    slope, intercept, log_scale = params
    scale = np.exp(log_scale)
    mu = slope * x + intercept
    total_err = np.sqrt(yerr**2 + scale**2)
    z = (y - mu) / total_err
    return np.sum(0.5 * (nu + 1) * np.log1p(z**2 / nu) + np.log(total_err))

def fit_model(x, y, yerr):
    w = 1.0 / yerr**2
    slope_guess = np.sum(w * x * y) / np.sum(w * x**2)
    initial_params = [slope_guess, np.mean(y), -2.0]
    res = minimize(student_t_nll, initial_params, args=(x, y, yerr), method='Nelder-Mead')
    return res.x, res.fun

def student_t_logpdf(y, mu, yerr, scale, nu=4.0):
    total_err = np.sqrt(yerr**2 + scale**2)
    z = (y - mu) / total_err
    return -(0.5 * (nu + 1) * np.log1p(z**2 / nu) + np.log(total_err))

def get_deprojected_radii(ra, dec, ra_c, dec_c, pa_deg, inc_deg, D_Mpc):
    ra_rad, dec_rad = np.radians(ra), np.radians(dec)
    ra_c_rad, dec_c_rad = np.radians(ra_c), np.radians(dec_c)
    pa_rad, inc_rad = np.radians(pa_deg), np.radians(inc_deg)
    
    x = (ra_rad - ra_c_rad) * np.cos(dec_c_rad)
    y = dec_rad - dec_c_rad
    
    x_prime = x * np.cos(pa_rad) + y * np.sin(pa_rad)
    y_prime = -x * np.sin(pa_rad) + y * np.cos(pa_rad)
    
    y_deproj = y_prime / np.cos(inc_rad)
    r_rad = np.sqrt(x_prime**2 + y_deproj**2)
    return r_rad * D_Mpc * 1000.0

def run():
    logger = TEPLogger("step_58", log_file_path=PROJECT_ROOT / "logs" / "step_58_m101_forensics.log")
    set_step_logger(logger)
    
    print_status("================================================================================", "INFO")
    print_status("   Step 58: M101 Predictor Integrity Out-of-Sample Audit (Gate B)", "INFO")
    print_status("================================================================================", "INFO")
    
    # 1. M101 Deprojection
    path = PROJECT_ROOT / "data" / "raw" / "external" / "J_ApJ_830_10_table5.csv"
    if not path.exists():
        print_status(f"Hoffmann 2016 data not found at {path}", "ERROR")
        return
        
    df_hoff = pd.read_csv(path)
    df_m101_hoff = df_hoff[df_hoff["Gal"].str.contains("M101")].copy()
    
    # M101 Physical properties
    RA_C, DEC_C = 210.8025, 54.3492
    PA, INC, D_MPC = 39.0, 18.0, 6.4
    
    df_m101_hoff["R_kpc"] = get_deprojected_radii(
        df_m101_hoff["RAJ2000"].values, df_m101_hoff["DEJ2000"].values,
        RA_C, DEC_C, PA, INC, D_MPC
    )
    
    # Simplified Flat Rotation Curve V_flat = 240, R_d = 2.0
    df_m101_hoff["V_rot_local"] = 240.0 * (1 - np.exp(-df_m101_hoff["R_kpc"] / 2.0))
    
    c = 299792.458
    df_m101_hoff["X_V2"] = (df_m101_hoff["V_rot_local"] / c)**2
    # Normalize V^4/R^2 so it matches the scale of V^2 for easier slope comparison
    norm = (10.0 / 200.0)**2 / c**2
    df_m101_hoff["X_V4_R2"] = (df_m101_hoff["V_rot_local"]**4 / df_m101_hoff["R_kpc"]**2) * norm
    
    mean_X_V2_m101 = df_m101_hoff["X_V2"].mean()
    mean_X_V4_m101 = df_m101_hoff["X_V4_R2"].mean()
    
    print_status("\n--- M101 Deprojection ---", "PROCESS")
    print_status(f"Average Galactocentric Radius: {df_m101_hoff['R_kpc'].mean():.2f} kpc", "TEST")
    print_status(f"Average Local V_rot: {df_m101_hoff['V_rot_local'].mean():.2f} km/s", "TEST")
    print_status(f"Mean Localized V^2/c^2 Predictor (M101): {mean_X_V2_m101:.2e}", "TEST")
    print_status(f"Mean Localized V^4/R^2 Predictor (M101): {mean_X_V4_m101:.2e}", "TEST")
    
    # 2. Out-of-Sample Prediction Test
    # Load Band-Dependence dataset (MF2023 VIH vs VI)
    # We will approximate V^4/R^2 for the other galaxies.
    # For a fair test, we assume a typical R_eff ~ 5 kpc for spirals, V ~ V_rot
    # In a real scenario we'd deproject Cepheids for all galaxies, but here we do a bulk approximation.
    print_status("\n--- Out-of-Sample Regression Test ---", "PROCESS")
    
    bd_path = PROJECT_ROOT / "data" / "processed" / "band_dependence_matched.csv"
    if not bd_path.exists():
        print_status("Band dependence matched catalog not found. Run step 49.", "ERROR")
        return
        
    df_bd = pd.read_csv(bd_path)
    df_bd = df_bd[df_bd["sample"] == "primary_MF2023"].copy()
    
    # Extract M101
    m101_mask = df_bd["galaxy"].str.contains("M101|M 101|NGC 5457")
    df_train = df_bd[~m101_mask].copy()
    df_test = df_bd[m101_mask].copy()
    
    if len(df_test) == 0:
        print_status("M101 not found in band dependence sample.", "ERROR")
        return
        
    # Bulk V^4/R^2 approximation for training set (assuming R_eff = 5 kpc)
    # df_train["X_i"] is already (V_rot/c)^2
    # So V^4/R^2 * norm = (V_rot/c)^4 * c^2 / R^2 * norm
    df_train["X_V2"] = df_train["X_i"]
    df_train["X_V4_R2"] = df_train["X_V2"]**2 * c**4 / (5.0**2) * norm
    
    y_train = df_train["delta_mu_band"].values
    yerr_train = df_train["delta_mu_band_err"].values
    
    params_M1, nll_M1 = fit_model(df_train["X_V2"].values, y_train, yerr_train)
    params_M2, nll_M2 = fit_model(df_train["X_V4_R2"].values, y_train, yerr_train)
    
    print_status(f"Training Model 1 (V^2) NLL: {nll_M1:.2f} (excluding M101)", "TEST")
    print_status(f"Training Model 2 (V^4/R^2) NLL: {nll_M2:.2f} (excluding M101)", "TEST")
    
    # Predict M101
    y_m101 = df_test["delta_mu_band"].values[0]
    yerr_m101 = df_test["delta_mu_band_err"].values[0]
    
    mu_M1 = params_M1[0] * mean_X_V2_m101 + params_M1[1]
    mu_M2 = params_M2[0] * mean_X_V4_m101 + params_M2[1]
    
    logpdf_M1 = student_t_logpdf(y_m101, mu_M1, yerr_m101, np.exp(params_M1[2]))
    logpdf_M2 = student_t_logpdf(y_m101, mu_M2, yerr_m101, np.exp(params_M2[2]))
    
    print_status(f"\nM101 True Delta_mu: {y_m101:.4f} ± {yerr_m101:.4f}", "TEST")
    print_status(f"M1 Prediction (V^2): {mu_M1:.4f} (Residual: {y_m101 - mu_M1:+.4f})", "TEST")
    print_status(f"M2 Prediction (V^4/R^2): {mu_M2:.4f} (Residual: {y_m101 - mu_M2:+.4f})", "TEST")
    
    print_status(f"\nM101 Predictive Log-Likelihood (V^2): {logpdf_M1:.2f}", "TEST")
    print_status(f"M101 Predictive Log-Likelihood (V^4/R^2): {logpdf_M2:.2f}", "TEST")
    print_status(f"Delta Predictive LL: {logpdf_M2 - logpdf_M1:+.2f}", "TEST")
    
if __name__ == "__main__":
    run()
