#!/usr/bin/env python3
"""
Step 62: Pantheon+ Radial Discriminator (Gate G v2)
=====================================================
Applies the Phase F1 discriminator to Pantheon+ SNe.
Because TF suffers from conformal cancellation (q_i ~ 1), the true TEP temporal
signal should emerge purely in SNe where z_spec (core) differs from x1 (disk/halo).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from astropy.coordinates import SkyCoord
import astropy.units as u

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.utils.logger import TEPLogger, set_step_logger

CMB_L = 264.0
CMB_B = 48.0
CMB_NX = np.cos(np.radians(CMB_B)) * np.cos(np.radians(CMB_L))
CMB_NY = np.cos(np.radians(CMB_B)) * np.sin(np.radians(CMB_L))
CMB_NZ = np.sin(np.radians(CMB_B))
CMB_VEC = np.array([CMB_NX, CMB_NY, CMB_NZ])

def angle_between(v1, v2):
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

V_STAR = 10000.0  # c * z = 10000 km/s
C_KMS = 299792.458

def load_pantheon():
    df = pd.read_csv(PROJECT_ROOT / 'data/raw/Pantheon+SH0ES.dat', delim_whitespace=True)
    # Filter for Hubble flow
    df = df[(df['zHD'] > 0.01) & (df['zHD'] < 0.1)]
    
    # Calculate cartesian coordinates
    coords = SkyCoord(ra=df['RA'].values*u.deg, dec=df['DEC'].values*u.deg, frame='icrs')
    gal = coords.galactic
    df['nx'] = np.cos(gal.b.rad) * np.cos(gal.l.rad)
    df['ny'] = np.cos(gal.b.rad) * np.sin(gal.l.rad)
    df['nz'] = np.sin(gal.b.rad)
    
    df['cos_theta_cmb'] = df['nx']*CMB_NX + df['ny']*CMB_NY + df['nz']*CMB_NZ
    
    # Velocity in km/s
    df['vcmb'] = df['zCMB'] * C_KMS
    df['vhel'] = df['zHEL'] * C_KMS
    
    # Apparent magnitude error (using diagonal for simplicity)
    df['e_m'] = df['m_b_corr_err_DIAG']
    
    return df

def log_like(params, model_type, y, v, nx, ny, nz, cos_theta, err):
    alpha = params[0]
    
    if model_type == 'M0':
        y_pred = alpha * np.ones_like(y)
    elif model_type == 'MK':
        bx, by, bz = params[1:4]
        b_dot_n = bx * nx + by * ny + bz * nz
        y_pred = alpha + (5 / np.log(10)) * (b_dot_n / v)
    elif model_type == 'MT':
        T_amp = params[1]
        y_pred = alpha + T_amp * (v / V_STAR) * cos_theta
    elif model_type == 'MKT':
        T_amp = params[1]
        bx, by, bz = params[2:5]
        b_dot_n = bx * nx + by * ny + bz * nz
        y_pred = alpha + T_amp * (v / V_STAR) * cos_theta + (5 / np.log(10)) * (b_dot_n / v)
    
    chi2 = np.sum(((y - y_pred) / err)**2)
    return 0.5 * chi2

def bic(n_params, n_data, min_nll):
    return 2 * min_nll + n_params * np.log(n_data)

def fit_model(df, model_type, frame='cmb'):
    vel_col = 'vcmb' if frame == 'cmb' else 'vhel'
    # y = 5 log10(V) - m_b
    y = 5 * np.log10(df[vel_col].values) - df['m_b_corr'].values
    v = df[vel_col].values
    nx = df['nx'].values
    ny = df['ny'].values
    nz = df['nz'].values
    cos_theta = df['cos_theta_cmb'].values
    err = df['e_m'].values
    
    alpha0 = np.median(y)
    if model_type == 'M0':
        p0 = [alpha0]
    elif model_type == 'MK':
        p0 = [alpha0, 0.0, 0.0, 0.0]
    elif model_type == 'MT':
        p0 = [alpha0, 0.0]
    elif model_type == 'MKT':
        p0 = [alpha0, 0.0, 0.0, 0.0, 0.0]
        
    res = minimize(log_like, p0, args=(model_type, y, v, nx, ny, nz, cos_theta, err), method='BFGS')
    return res

def run_pantheon_discriminator():
    log = TEPLogger("step_62_pantheon", log_file_path=PROJECT_ROOT / "outputs/logs/step_62_pantheon.log")
    set_step_logger(log)
    log.info("Loading Pantheon+ Catalog...")
    df = load_pantheon()
    log.info(f"Loaded {len(df)} SNe Ia (0.01 < z < 0.1).")
    
    for frame in ['cmb', 'hel']:
        log.info(f"\n=== Global Model Comparison (Frame: {frame.upper()}) ===")
        res_m0 = fit_model(df, 'M0', frame)
        res_mk = fit_model(df, 'MK', frame)
        res_mt = fit_model(df, 'MT', frame)
        res_mkt = fit_model(df, 'MKT', frame)
        
        n = len(df)
        bic_m0 = bic(1, n, res_m0.fun)
        bic_mk = bic(4, n, res_mk.fun)
        bic_mt = bic(2, n, res_mt.fun)
        bic_mkt = bic(5, n, res_mkt.fun)
        
        log.info(f"M0 (Monopole):       BIC = {bic_m0:.1f} | NLL = {res_m0.fun:.1f}")
        log.info(f"MK (Kinematic):      BIC = {bic_mk:.1f} | NLL = {res_mk.fun:.1f} | dBIC(to M0) = {bic_mk - bic_m0:.1f}")
        log.info(f"MT (Temporal):       BIC = {bic_mt:.1f} | NLL = {res_mt.fun:.1f} | dBIC(to M0) = {bic_mt - bic_m0:.1f}")
        log.info(f"MKT (Kin+Temporal):  BIC = {bic_mkt:.1f} | NLL = {res_mkt.fun:.1f} | dBIC(to best) = {bic_mkt - min(bic_mk, bic_mt):.1f}")
        
        T_global = res_mt.x[1]
        B_kin_global = res_mk.x[1:4]
        b_kin_dot_cmb = np.dot(B_kin_global, CMB_VEC)
        log.info(f"Global TEP (MT): T = {T_global:.4f} mag | Global Kin (MK): B_|| = {b_kin_dot_cmb:.1f} km/s")

        log.info(f"\n--- Radial Bin Kinematic Fits (M_K on {frame.upper()}) ---")
        vel_col = 'vcmb' if frame == 'cmb' else 'vhel'
        
        # Bins: 3k-6k, 6k-10k, 10k-15k, 15k-20k, 20k-30k
        bins = [(3000, 6000), (6000, 10000), (10000, 15000), (15000, 20000), (20000, 30000)]
        
        for vmin, vmax in bins:
            mask = (df[vel_col] >= vmin) & (df[vel_col] < vmax)
            df_bin = df[mask]
            if len(df_bin) < 20:
                continue
                
            res_bin = fit_model(df_bin, 'MK', frame)
            bx, by, bz = res_bin.x[1:4]
            b_dot_cmb = bx*CMB_NX + by*CMB_NY + bz*CMB_NZ
            b_mag = np.linalg.norm([bx, by, bz])
            angle = angle_between(np.array([bx, by, bz]), CMB_VEC)
            
            v_center = np.median(df_bin[vel_col])
            b_app_expected = T_global * (v_center**2 / V_STAR) * (np.log(10) / 5.0)
            
            log.info(f"Bin [{vmin:5d}-{vmax:5d}] (N={len(df_bin):3d}): V={v_center:.0f}, B_|| = {b_dot_cmb:6.1f} | |B| = {b_mag:6.1f} ({angle:3.0f} deg) | TEP Pred: {b_app_expected:6.1f}")

if __name__ == '__main__':
    run_pantheon_discriminator()
