#!/usr/bin/env python3
"""
Step 57: Differential Dipole Test (Gate D - Second-Order Audit)
===============================================================
Tests the CMB-Directional Dipole (Delta_mu) using a Cartesian
dipole vector model with sample-specific intercepts and heteroscedastic
parametric bootstrap.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from scipy.optimize import minimize
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status

# CMB Dipole Direction
CMB_L = 264.0
CMB_B = 48.0
CMB_NX = np.cos(np.radians(CMB_B)) * np.cos(np.radians(CMB_L))
CMB_NY = np.cos(np.radians(CMB_B)) * np.sin(np.radians(CMB_L))
CMB_NZ = np.sin(np.radians(CMB_B))

def nll_m_x(params, df, nu=4.0):
    """Model M_X: Base TEP model (no dipole)
    params = [kappa, alpha_1, ..., alpha_N, log_scale]
    """
    samples = df['sample'].unique()
    kappa = params[0]
    log_scale = np.clip(params[-1], -20.0, 10.0)
    scale = np.exp(log_scale)
    
    mu = np.zeros(len(df))
    for i, s in enumerate(samples):
        mask = df['sample'] == s
        mu[mask] = params[1+i] - kappa * df.loc[mask, 'X_i'].values
        
    y = df['delta_mu'].values
    yerr = df['delta_mu_err'].values
    total_err = np.sqrt(yerr**2 + scale**2)
    z = (y - mu) / total_err
    return np.sum(0.5 * (nu + 1) * np.log1p(z**2 / nu) + np.log(total_err))

def nll_m_cmb(params, df, nu=4.0):
    """Model M_CMB: Dipole fixed to CMB direction
    params = [kappa, D, alpha_1, ..., alpha_N, log_scale]
    """
    samples = df['sample'].unique()
    kappa = params[0]
    D = params[1]
    log_scale = np.clip(params[-1], -20.0, 10.0)
    scale = np.exp(log_scale)
    
    mu = np.zeros(len(df))
    for i, s in enumerate(samples):
        mask = df['sample'] == s
        mu[mask] = params[2+i] - kappa * df.loc[mask, 'X_i'].values
        
    nx = df['nx'].values
    ny = df['ny'].values
    nz = df['nz'].values
    
    cos_theta = nx*CMB_NX + ny*CMB_NY + nz*CMB_NZ
    mu += D * cos_theta
    
    y = df['delta_mu'].values
    yerr = df['delta_mu_err'].values
    total_err = np.sqrt(yerr**2 + scale**2)
    z = (y - mu) / total_err
    return np.sum(0.5 * (nu + 1) * np.log1p(z**2 / nu) + np.log(total_err))

def nll_m_3d(params, df, nu=4.0):
    """Model M_3D: Freely varying 3D Cartesian dipole
    params = [kappa, dx, dy, dz, alpha_1, ..., alpha_N, log_scale]
    """
    samples = df['sample'].unique()
    kappa = params[0]
    dx, dy, dz = params[1:4]
    log_scale = np.clip(params[-1], -20.0, 10.0)
    scale = np.exp(log_scale)
    
    mu = np.zeros(len(df))
    for i, s in enumerate(samples):
        mask = df['sample'] == s
        mu[mask] = params[4+i] - kappa * df.loc[mask, 'X_i'].values
        
    nx = df['nx'].values
    ny = df['ny'].values
    nz = df['nz'].values
    
    mu += dx*nx + dy*ny + dz*nz
    
    y = df['delta_mu'].values
    yerr = df['delta_mu_err'].values
    total_err = np.sqrt(yerr**2 + scale**2)
    z = (y - mu) / total_err
    return np.sum(0.5 * (nu + 1) * np.log1p(z**2 / nu) + np.log(total_err))

def angle_between(dx, dy, dz, l_deg, b_deg):
    vec1 = np.array([dx, dy, dz])
    norm = np.linalg.norm(vec1)
    if norm < 1e-10:
        return 90.0
    vec1 = vec1 / norm
    
    l_rad = np.radians(l_deg)
    b_rad = np.radians(b_deg)
    vec2 = np.array([
        np.cos(b_rad) * np.cos(l_rad),
        np.cos(b_rad) * np.sin(l_rad),
        np.sin(b_rad)
    ])
    
    dot = np.clip(np.dot(vec1, vec2), -1.0, 1.0)
    return np.degrees(np.arccos(dot))

def run():
    logger = TEPLogger("step_57", log_file_path=PROJECT_ROOT / "logs" / "step_57_cartesian.log")
    set_step_logger(logger)
    
    print_status("================================================================================", "INFO")
    print_status("   Step 57: Differential Dipole Test (Nested Model Hierarchy)", "INFO")
    print_status("================================================================================", "INFO")
    
    df_path = PROJECT_ROOT / "data" / "processed" / "directional_ceph_trgb_sample.csv"
    if not df_path.exists():
        print_status("Directional sample not found.", "ERROR")
        return
        
    df = pd.read_csv(df_path)
    df = df[df['X_i'].notnull()].copy()
    print_status(f"Loaded {len(df)} galaxies.", "SUCCESS")
    
    # Calculate Cartesian coordinates for each galaxy
    coords = SkyCoord(ra=df['ra'].values*u.deg, dec=df['dec'].values*u.deg, frame='icrs')
    gal = coords.galactic
    l_rad = gal.l.rad
    b_rad = gal.b.rad

    df['nx'] = np.cos(b_rad) * np.cos(l_rad)
    df['ny'] = np.cos(b_rad) * np.sin(l_rad)
    df['nz'] = np.sin(b_rad)
    
    samples = df['sample'].unique()
    num_samples = len(samples)
    
    # --- 1. Fit M_X (Base Model) ---
    init_mx = [0.0] + [0.0]*num_samples + [-2.0]
    res_mx = minimize(nll_m_x, init_mx, args=(df,), method='BFGS')
    nll_mx = res_mx.fun
    kappa_mx = res_mx.x[0]
    scale_mx = np.exp(res_mx.x[-1])
    mx_alphas = res_mx.x[1:-1]
    print_status(f"M_X (Base) NLL: {nll_mx:.2f}, kappa: {kappa_mx:.2e}, sigma_t: {scale_mx:.3f}", "TEST")
    
    # --- 2. Fit M_CMB (Fixed Dipole Model) ---
    init_mcmb = [kappa_mx, 0.0] + list(mx_alphas) + [np.log(scale_mx)]
    res_mcmb = minimize(nll_m_cmb, init_mcmb, args=(df,), method='BFGS')
    nll_mcmb = res_mcmb.fun
    D_cmb = res_mcmb.x[1]
    print_status(f"M_CMB NLL: {nll_mcmb:.2f} (Delta NLL = {nll_mx - nll_mcmb:.2f}), D = {D_cmb:.3f}", "TEST")
    
    # --- 3. Fit M_3D (Free Dipole Model) ---
    init_m3d = [kappa_mx, 0.0, 0.0, 0.0] + list(mx_alphas) + [np.log(scale_mx)]
    res_m3d = minimize(nll_m_3d, init_m3d, args=(df,), method='BFGS')
    nll_m3d = res_m3d.fun
    dx, dy, dz = res_m3d.x[1:4]
    D_3d = np.sqrt(dx**2 + dy**2 + dz**2)
    l_fit = np.degrees(np.arctan2(dy, dx)) % 360
    b_fit = np.degrees(np.arcsin(dz / D_3d)) if D_3d > 1e-10 else 0
    cmb_dist = angle_between(dx, dy, dz, CMB_L, CMB_B)
    print_status(f"M_3D NLL: {nll_m3d:.2f} (Delta NLL vs M_X = {nll_mx - nll_m3d:.2f})", "TEST")
    print_status(f"Freely Preferred Dipole: D = {D_3d:.4f} at (l={l_fit:.1f}, b={b_fit:.1f})", "SUCCESS")
    print_status(f"Angular distance to CMB ({CMB_L}, {CMB_B}): {cmb_dist:.1f} degrees", "SUCCESS")
    
    # Likelihood Ratio Test (Chi-square approx)
    # 2*Delta_NLL ~ chi2(dof)
    from scipy.stats import chi2
    chi2_cmb_vs_mx = 2 * (nll_mx - nll_mcmb) # 1 dof (D)
    chi2_3d_vs_mx = 2 * (nll_mx - nll_m3d)   # 3 dof (dx, dy, dz)
    chi2_3d_vs_cmb = 2 * (nll_mcmb - nll_m3d) # 2 dof
    
    print_status(f"\n--- Likelihood Ratio Tests ---", "PROCESS")
    print_status(f"M_CMB vs M_X: 2*dNLL={chi2_cmb_vs_mx:.2f}, p-value={chi2.sf(chi2_cmb_vs_mx, 1):.4f}", "TEST")
    print_status(f"M_3D vs M_X:  2*dNLL={chi2_3d_vs_mx:.2f}, p-value={chi2.sf(chi2_3d_vs_mx, 3):.4f}", "TEST")
    print_status(f"M_3D vs M_CMB: 2*dNLL={chi2_3d_vs_cmb:.2f}, p-value={chi2.sf(chi2_3d_vs_cmb, 2):.4f}", "TEST")

    # --- 4. Heteroscedastic Parametric Bootstrap ---
    print_status("\n--- Parametric Bootstrap (Null Model M_X) ---", "PROCESS")
    
    # Generate predicted means from M_X
    mu_mx = np.zeros(len(df))
    for i, s in enumerate(samples):
        mask = df['sample'] == s
        mu_mx[mask] = mx_alphas[i] - kappa_mx * df.loc[mask, 'X_i'].values
        
    n_boots = 1000
    boot_chi2_3d_vs_mx = []
    boot_angles = []
    
    np.random.seed(42)
    df_boot = df.copy()
    
    yerr = df['delta_mu_err'].values
    total_err = np.sqrt(yerr**2 + scale_mx**2)
    
    for i in range(n_boots):
        # Generate new y values from Student-t (nu=4) centered on M_X predictions
        # stats.t.rvs(df, loc, scale, size)
        y_sim = stats.t.rvs(df=4.0, loc=mu_mx, scale=total_err)
        df_boot['delta_mu'] = y_sim
        
        # Fit M_X
        b_res_mx = minimize(nll_m_x, init_mx, args=(df_boot,), method='BFGS')
        
        # Fit M_3D
        b_init_m3d = [b_res_mx.x[0], 0.0, 0.0, 0.0] + list(b_res_mx.x[1:-1]) + [b_res_mx.x[-1]]
        b_res_m3d = minimize(nll_m_3d, b_init_m3d, args=(df_boot,), method='BFGS')
        
        d_chi2 = 2 * (b_res_mx.fun - b_res_m3d.fun)
        boot_chi2_3d_vs_mx.append(d_chi2)
        
        dx_b, dy_b, dz_b = b_res_m3d.x[1:4]
        if np.sqrt(dx_b**2 + dy_b**2 + dz_b**2) > 1e-4:
            boot_angles.append(angle_between(dx_b, dy_b, dz_b, CMB_L, CMB_B))
            
    boot_chi2_3d_vs_mx = np.array(boot_chi2_3d_vs_mx)
    boot_angles = np.array(boot_angles)
    
    p_val_boot = np.mean(boot_chi2_3d_vs_mx >= chi2_3d_vs_mx)
    print_status(f"Bootstrap p-value for M_3D vs M_X: p = {p_val_boot:.4f}", "TEST")
    
    p30 = np.mean(boot_angles <= 30.0) * 100
    p60 = np.mean(boot_angles <= 60.0) * 100
    p90 = np.mean(boot_angles <= 90.0) * 100
    
    print_status(f"Random alignment P(theta_CMB <= 30 deg): {p30:.1f}%", "TEST")
    print_status(f"Random alignment P(theta_CMB <= 60 deg): {p60:.1f}%", "TEST")
    print_status(f"Random alignment P(theta_CMB <= 90 deg): {p90:.1f}%", "TEST")

if __name__ == "__main__":
    run()
