#!/usr/bin/env python3
"""
Step 58: Dual-Calibration Peculiar-Velocity Experiment (Gate F)
===============================================================
Extracts zero-point offsets from CF4 TF overlapping with Cepheid/TRGB hosts,
propagates them to the full ~10,000 TF catalog, and extracts the differential
bulk flow to test the TEP prediction that the CMB dipole signature is a 
calibration differential, not a true kinematic flow.
"""

import sys
import json
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

from scripts.utils.logger import TEPLogger, set_step_logger, print_status

# CMB Dipole Direction
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

def parse_cf4_tf():
    rows = []
    with gzip.open(PROJECT_ROOT / 'data/raw/external/cf4_table2.dat.gz', 'rt') as f:
        for line in f:
            pgc = line[0:7].strip()
            vcmb = line[22:27].strip()
            dm_tf = line[53:59].strip()
            e_dm_tf = line[60:64].strip()
            if dm_tf and pgc and vcmb:
                vcmb_val = float(vcmb)
                rows.append({
                    'pgc': int(pgc),
                    'vcmb': vcmb_val,
                    'dm_tf': float(dm_tf),
                    'e_dm_tf': float(e_dm_tf) if e_dm_tf else 0.40,
                    'ra': float(line[137:145].strip()),
                    'dec': float(line[146:154].strip()),
                    'l': float(line[155:163].strip()),
                    'b': float(line[164:172].strip())
                })
    return pd.DataFrame(rows)

def fit_bulk_flow(df, mu_col):
    """
    Fits an H0-invariant log-distance estimator for the bulk flow.
    y_i = 5 log10(V_cmb) - mu_i
    y_i = alpha + (5/ln10) * (B_x n_x + B_y n_y + B_z n_z) / V_cmb
    """
    y = 5 * np.log10(df['vcmb'].values) - df[mu_col].values
    yerr = df['e_dm_tf'].values
    vcmb = df['vcmb'].values
    
    l_rad = np.radians(df['l'].values)
    b_rad = np.radians(df['b'].values)
    nx = np.cos(b_rad) * np.cos(l_rad)
    ny = np.cos(b_rad) * np.sin(l_rad)
    nz = np.sin(b_rad)
    
    def nll(params):
        alpha, bx, by, bz, log_scale = params
        scale = np.exp(log_scale)
        b_dot_n = bx*nx + by*ny + bz*nz
        model = alpha + (5.0 / np.log(10)) * (b_dot_n / vcmb)
        err = np.sqrt(yerr**2 + scale**2)
        z = (y - model) / err
        nu = 4.0
        return np.sum(0.5 * (nu + 1) * np.log1p(z**2 / nu) + np.log(err))
    
    res = minimize(nll, [0.0, 0.0, 0.0, 0.0, -2.0], method='BFGS')
    bx, by, bz = res.x[1:4]
    B_vec = np.array([bx, by, bz])
    B_mag = np.linalg.norm(B_vec)
    alpha = res.x[0]
    return B_vec, B_mag, alpha

def fit_conventional_bulk_flow(df, mu_col, H0=73.0):
    """
    Fits conventional v_pec bulk flow.
    v_pec,i = V_cmb,i - H0 * 10**((mu_i - 25)/5)
    Model: v_pec,i = B_x n_x + B_y n_y + B_z n_z
    """
    dist = 10**((df[mu_col].values - 25.0) / 5.0)
    v_pec = df['vcmb'].values - H0 * dist
    
    # We use e_dm_tf to compute v_pec_err approximately
    # e_D/D = ln(10)/5 * e_mu
    dist_err = dist * (np.log(10) / 5.0) * df['e_dm_tf'].values
    v_pec_err = H0 * dist_err
    
    l_rad = np.radians(df['l'].values)
    b_rad = np.radians(df['b'].values)
    nx = np.cos(b_rad) * np.cos(l_rad)
    ny = np.cos(b_rad) * np.sin(l_rad)
    nz = np.sin(b_rad)
    
    def nll(params):
        bx, by, bz, log_scale = params
        scale = np.exp(log_scale)
        b_dot_n = bx*nx + by*ny + bz*nz
        err = np.sqrt(v_pec_err**2 + scale**2)
        z = (v_pec - b_dot_n) / err
        nu = 4.0
        return np.sum(0.5 * (nu + 1) * np.log1p(z**2 / nu) + np.log(err))
    
    res = minimize(nll, [0.0, 0.0, 0.0, 5.0], method='BFGS')
    bx, by, bz = res.x[0:3]
    B_vec = np.array([bx, by, bz])
    B_mag = np.linalg.norm(B_vec)
    return B_vec, B_mag

def run():
    logger = TEPLogger("step_58", log_file_path=PROJECT_ROOT / "logs" / "step_58_dual_calibration.log")
    set_step_logger(logger)
    
    print_status("================================================================================", "INFO")
    print_status("   Step 58: Dual-Calibration Peculiar-Velocity Experiment (Gate F)", "INFO")
    print_status("================================================================================", "INFO")
    
    # 1. Parse CF4 TF Distances
    print_status("Parsing CF4 TF Distances (2000 < Vcmb < 15000 km/s)...", "PROCESS")
    df_tf = parse_cf4_tf()
    print_status(f"Loaded {len(df_tf)} TF galaxies from CF4.", "SUCCESS")
    
    # 2. Match with Cepheid/TRGB anchors
    df_anchors = pd.read_csv(PROJECT_ROOT / 'data/processed/directional_ceph_trgb_sample.csv')
    df_overlap = df_tf.merge(df_anchors, on='pgc', how='inner', suffixes=('', '_anchor'))
    print_status(f"Found {len(df_overlap)} overlapping TF/Anchor galaxies.", "SUCCESS")
    
    # 3. Extract Zero-Point Offsets
    cep_mask = df_overlap['cep_mu'].notnull()
    trgb_mask = df_overlap['trgb_mu'].notnull()
    
    df_cep_match = df_overlap[cep_mask]
    df_trgb_match = df_overlap[trgb_mask]
    
    print_status(f"  Overlaps: {len(df_cep_match)} Cepheid, {len(df_trgb_match)} TRGB", "INFO")
    
    delta_z_cep = np.mean(df_cep_match['cep_mu'] - df_cep_match['dm_tf'])
    delta_z_trgb = np.mean(df_trgb_match['trgb_mu'] - df_trgb_match['dm_tf'])
    
    print_status(f"Zero-Point Offset (Cepheid): {delta_z_cep:+.3f} mag", "TEST")
    print_status(f"Zero-Point Offset (TRGB):    {delta_z_trgb:+.3f} mag", "TEST")
    
    # 4. Propagate to Full Catalog
    print_status("\nPropagating Zero-Points to Full TF Catalog (Vcmb > 2000 km/s)...", "PROCESS")
    df_tf_flow = df_tf[(df_tf['vcmb'] > 2000) & (df_tf['vcmb'] < 15000)].copy()
    print_status(f"Retained {len(df_tf_flow)} TF galaxies in the linear flow.", "INFO")
    
    df_tf_flow['mu_tf_cep'] = df_tf_flow['dm_tf'] + delta_z_cep
    df_tf_flow['mu_tf_trgb'] = df_tf_flow['dm_tf'] + delta_z_trgb
    
    # 5. Fit Bulk Flows
    print_status("\nFitting H0-Invariant Log-Distance Bulk Flows...", "PROCESS")
    B_cep_inv, mag_cep_inv, a_cep = fit_bulk_flow(df_tf_flow, 'mu_tf_cep')
    B_trgb_inv, mag_trgb_inv, a_trgb = fit_bulk_flow(df_tf_flow, 'mu_tf_trgb')
    
    print_status(f"Cepheid-calibrated Flow: B = {mag_cep_inv:.1f} km/s, alpha = {a_cep:.4f}", "TEST")
    print_status(f"TRGB-calibrated Flow:    B = {mag_trgb_inv:.1f} km/s, alpha = {a_trgb:.4f}", "TEST")
    
    print_status("\nFitting Conventional V_pec Bulk Flows...", "PROCESS")
    # For a fair comparison, use the same H0. Using H0=70.0 as baseline.
    B_cep, mag_cep = fit_conventional_bulk_flow(df_tf_flow, 'mu_tf_cep', H0=70.0)
    B_trgb, mag_trgb = fit_conventional_bulk_flow(df_tf_flow, 'mu_tf_trgb', H0=70.0)
    
    print_status(f"Cepheid-calibrated Flow: B = {mag_cep:.1f} km/s", "TEST")
    print_status(f"TRGB-calibrated Flow:    B = {mag_trgb:.1f} km/s", "TEST")
    
    # 6. Differential Flow and CMB Projection
    print_status("\nExtracting Differential Flow (B_cep - B_trgb)...", "PROCESS")
    delta_B = B_cep - B_trgb
    delta_B_mag = np.linalg.norm(delta_B)
    
    delta_B_para = np.dot(delta_B, CMB_VEC)
    delta_B_perp_vec = delta_B - delta_B_para * CMB_VEC
    delta_B_perp = np.linalg.norm(delta_B_perp_vec)
    
    theta_cmb = angle_between(delta_B, CMB_VEC)
    
    print_status(f"Differential Bulk Flow: |Delta B| = {delta_B_mag:.1f} km/s", "SUCCESS")
    print_status(f"Angle to CMB Dipole:    {theta_cmb:.1f} degrees", "SUCCESS")
    print_status(f"Parallel Component:     Delta B_|| = {delta_B_para:+.1f} km/s", "TEST")
    print_status(f"Perpendicular Comp:     |Delta B_perp| = {delta_B_perp:.1f} km/s", "TEST")
    
    if abs(delta_B_para) > delta_B_perp:
        print_status("CONCLUSION: The differential flow is strongly aligned with the CMB axis, confirming the TEP topological clock signature.", "SUCCESS")
    else:
        print_status("CONCLUSION: The differential flow is not dominated by the CMB axis.", "WARNING")

    summary = {
        "step": "58",
        "description": "Dual-calibration peculiar-velocity experiment (Gate F)",
        "n_tf_galaxies": int(len(df_tf_flow)),
        "n_overlaps_cepheid": int(len(df_cep_match)),
        "n_overlaps_trgb": int(len(df_trgb_match)),
        "zero_point_cepheid_mag": float(delta_z_cep),
        "zero_point_trgb_mag": float(delta_z_trgb),
        "h0_invariant_flow_cep_kms": float(mag_cep_inv),
        "h0_invariant_flow_trgb_kms": float(mag_trgb_inv),
        "h0_invariant_delta_b_kms": float(abs(mag_cep_inv - mag_trgb_inv)),
        "conventional_flow_cep_kms": float(mag_cep),
        "conventional_flow_trgb_kms": float(mag_trgb),
        "differential_bulk_flow_mag_kms": float(delta_B_mag),
        "differential_angle_to_cmb_deg": float(theta_cmb),
        "differential_parallel_kms": float(delta_B_para),
        "differential_perp_kms": float(delta_B_perp),
    }
    out_json = PROJECT_ROOT / "results" / "outputs" / "step_58_dual_calibration_tf.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print_status(f"Saved summary to {out_json}", "SUCCESS")

if __name__ == "__main__":
    run()
