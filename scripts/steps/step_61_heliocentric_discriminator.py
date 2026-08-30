#!/usr/bin/env python3
"""
Step 61: Heliocentric Discriminator (Gate F v2 - Phase F1.b)
=====================================================
Fits the CF4 TF catalog over all radial shells using the raw Heliocentric 
velocities (V_hel) rather than V_cmb. This strips away the artificial 371 km/s 
kinematic injection added by standard cosmology to expose the raw temporal field.
"""

import sys
import gzip
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

V_STAR = 10000.0

def parse_cf4_tf():
    rows = []
    with gzip.open(PROJECT_ROOT / 'data/raw/external/cf4_table2.dat.gz', 'rt') as f:
        for line in f:
            pgc = line[0:7].strip()
            vcmb = line[22:27].strip()
            dm_tf = line[53:59].strip()
            e_dm_tf = line[60:64].strip()
            ra = line[137:145].strip()
            dec = line[146:154].strip()
            
            if dm_tf and pgc and vcmb and ra and dec:
                vcmb_val = float(vcmb)
                if vcmb_val < 2000 or vcmb_val > 20000:
                    continue
                
                dm_val = float(dm_tf)
                err_val = float(e_dm_tf)
                if err_val == 0:
                    continue
                    
                coord = SkyCoord(ra=float(ra)*u.deg, dec=float(dec)*u.deg, frame='icrs')
                gal = coord.galactic
                nx = np.cos(gal.b.rad) * np.cos(gal.l.rad)
                ny = np.cos(gal.b.rad) * np.sin(gal.l.rad)
                nz = np.sin(gal.b.rad)
                
                cos_theta = np.dot([nx, ny, nz], CMB_VEC)
                
                # Reconstruct Heliocentric Velocity
                # V_cmb = V_hel + 371 * cos_theta  => V_hel = V_cmb - 371 * cos_theta
                vhel_val = vcmb_val - 371.0 * cos_theta
                
                rows.append({
                    'pgc': pgc,
                    'vhel': vhel_val,
                    'dm': dm_val,
                    'e_dm': err_val,
                    'nx': nx, 'ny': ny, 'nz': nz,
                    'cos_theta_cmb': cos_theta
                })
    return pd.DataFrame(rows)

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

def fit_model(df, model_type):
    y = 5 * np.log10(df['vhel'].values) - df['dm'].values
    v = df['vhel'].values
    nx = df['nx'].values
    ny = df['ny'].values
    nz = df['nz'].values
    cos_theta = df['cos_theta_cmb'].values
    err = df['e_dm'].values
    
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

def run_heliocentric_discriminator():
    log = TEPLogger("step_61_helio", log_file_path=PROJECT_ROOT / "outputs/logs/step_61_helio.log")
    set_step_logger(log)
    log.info("Loading CF4 TF Catalog (Heliocentric)...")
    df = parse_cf4_tf()
    log.info(f"Loaded {len(df)} TF galaxies.")
    
    log.info("\n=== Global Model Comparison (All V_hel) ===")
    res_m0 = fit_model(df, 'M0')
    res_mk = fit_model(df, 'MK')
    res_mt = fit_model(df, 'MT')
    res_mkt = fit_model(df, 'MKT')
    
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
    log.info(f"\nGlobal TEP Amplitude (MT): T = {T_global:.4f} mag at 100 Mpc/h")
    
    B_kin_global = res_mk.x[1:4]
    B_kin_global_mag = np.linalg.norm(B_kin_global)
    b_kin_dot_cmb = np.dot(B_kin_global, CMB_VEC)
    log.info(f"Global Kinematic Flow (MK): B = {B_kin_global_mag:.1f} km/s, B_|| = {b_kin_dot_cmb:.1f} km/s")

    log.info("\n=== Radial Bin Kinematic Fits (M_K on V_hel) ===")
    bins = [(2000, 5000), (5000, 8000), (8000, 11000), (11000, 14000), (14000, 17000), (17000, 20000)]
    
    bin_centers = []
    b_parallel_fitted = []
    t_predicted = []
    
    for vmin, vmax in bins:
        mask = (df['vhel'] >= vmin) & (df['vhel'] < vmax)
        df_bin = df[mask]
        if len(df_bin) < 50:
            continue
            
        res_bin = fit_model(df_bin, 'MK')
        bx, by, bz = res_bin.x[1:4]
        b_dot_cmb = bx*CMB_NX + by*CMB_NY + bz*CMB_NZ
        b_mag = np.linalg.norm([bx, by, bz])
        angle = angle_between(np.array([bx, by, bz]), CMB_VEC)
        
        v_center = np.median(df_bin['vhel'])
        bin_centers.append(v_center)
        b_parallel_fitted.append(b_dot_cmb)
        
        b_app_expected = T_global * (v_center**2 / V_STAR) * (np.log(10) / 5.0)
        t_predicted.append(b_app_expected)
        
        log.info(f"Bin [{vmin:5d}-{vmax:5d}] (N={len(df_bin):4d}): V_med={v_center:.0f}, B_|| = {b_dot_cmb:6.1f} | |B| = {b_mag:6.1f} (Angle: {angle:4.0f} deg) | TEP Predicts: {b_app_expected:6.1f}")

    plt.figure(figsize=(8,6))
    plt.errorbar(bin_centers, b_parallel_fitted, yerr=np.ones_like(bin_centers)*50, fmt='o-', label='Fitted Kinematic B_|| (Heliocentric)', markersize=8)
    plt.plot(bin_centers, t_predicted, 'r--', label='TEP Transport Prediction (T * r^2)', linewidth=2)
    plt.axhline(0, color='k', linestyle=':')
    plt.xlabel("V_hel (km/s)")
    plt.ylabel("Apparent CMB-Aligned Bulk Flow (km/s)")
    plt.title("Radial Growth in Heliocentric Frame: Revealing the Temporal Field")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(PROJECT_ROOT / "outputs/figures/step_61_helio_radial_flow.png", dpi=300, bbox_inches='tight')
    log.info(f"Saved radial plot to outputs/figures/step_61_helio_radial_flow.png")

run = run_heliocentric_discriminator

if __name__ == '__main__':
    run_heliocentric_discriminator()
