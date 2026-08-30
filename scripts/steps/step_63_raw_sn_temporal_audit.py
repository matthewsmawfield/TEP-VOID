#!/usr/bin/env python3
"""
Step 63: Raw SN Temporal Gradient Audit (Gate G v3)
=====================================================
Performs a 5-stage signal accounting analysis on Pantheon+ SNe
to track the macroscopic temporal gradient (G(r)*cos(theta)) 
as it passes through the SALT standardization pipeline.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.utils.logger import TEPLogger, set_step_logger

CMB_L = 264.0
CMB_B = 48.0
CMB_NX = np.cos(np.radians(CMB_B)) * np.cos(np.radians(CMB_L))
CMB_NY = np.cos(np.radians(CMB_B)) * np.sin(np.radians(CMB_L))
CMB_NZ = np.sin(np.radians(CMB_B))
CMB_VEC = np.array([CMB_NX, CMB_NY, CMB_NZ])

C_KMS = 299792.458
V_STAR = 10000.0

cosmo = FlatLambdaCDM(H0=73.04, Om0=0.334)

def load_pantheon():
    df = pd.read_csv(PROJECT_ROOT / 'data/raw/Pantheon+SH0ES.dat', sep=r'\s+')
    df = df.drop_duplicates(subset=['CID'], keep='first')
    df = df[(df['zHD'] > 0.01) & (df['zHD'] < 0.1)]
    df = df[df['HOST_LOGMASS'] > 0].copy()
    
    coords = SkyCoord(ra=df['RA'].values*u.deg, dec=df['DEC'].values*u.deg, frame='icrs')
    gal = coords.galactic
    df['nx'] = np.cos(gal.b.rad) * np.cos(gal.l.rad)
    df['ny'] = np.cos(gal.b.rad) * np.sin(gal.l.rad)
    df['nz'] = np.sin(gal.b.rad)
    
    df['cos_theta_cmb'] = df['nx']*CMB_NX + df['ny']*CMB_NY + df['nz']*CMB_NZ
    
    df['vcmb'] = df['zCMB'] * C_KMS
    df['temporal_predictor'] = (df['vcmb'] / V_STAR) * df['cos_theta_cmb']
    df['mu_bg'] = cosmo.distmod(df['zCMB']).value
    
    df['raw_mag_resid'] = df['mB'] - df['mu_bg']
    df['std_mag_resid'] = df['m_b_corr'] - df['mu_bg']
    df['salt_correction'] = df['mB'] - df['m_b_corr']
    df['pv_correction'] = C_KMS * (df['zCMB'] - df['zHD']) / (1 + df['zCMB'])
    
    return df

def fit_target(df, target_col, use_weights=False, err_col=None):
    X = pd.DataFrame({
        'temporal': df['temporal_predictor'],
        'zCMB': df['zCMB'],
        'mass': df['HOST_LOGMASS']
    })
    surveys = pd.get_dummies(df['IDSURVEY'], drop_first=True, dtype=float)
    X = pd.concat([X, surveys], axis=1)
    X = sm.add_constant(X)
    
    y = df[target_col]
    
    if use_weights and err_col is not None and err_col in df.columns:
        weights = 1.0 / (df[err_col]**2 + 1e-6)
        model = sm.WLS(y, X, weights=weights)
    else:
        model = sm.OLS(y, X)
        
    res = model.fit()
    return res.params['temporal'], res.bse['temporal'], res.tvalues['temporal'], res.pvalues['temporal']

def run_permutation(df, target_col, use_weights, err_col, n_perms=1000):
    # Survey-stratified permutation of angular coordinates
    np.random.seed(42)
    original_cos = df['cos_theta_cmb'].values.copy()
    original_pred = df['temporal_predictor'].values.copy()
    
    obs_coef, _, _, _ = fit_target(df, target_col, use_weights, err_col)
    null_coefs = []
    
    for _ in range(n_perms):
        shuffled_cos = np.zeros_like(original_cos)
        for survey in df['IDSURVEY'].unique():
            mask = df['IDSURVEY'] == survey
            shuffled_cos[mask] = np.random.permutation(original_cos[mask])
        
        df['temporal_predictor'] = (df['vcmb'] / V_STAR) * shuffled_cos
        c, _, _, _ = fit_target(df, target_col, use_weights, err_col)
        null_coefs.append(c)
        
    df['cos_theta_cmb'] = original_cos
    df['temporal_predictor'] = original_pred
    
    null_coefs = np.array(null_coefs)
    p_perm = np.mean(np.abs(null_coefs) >= np.abs(obs_coef))
    return p_perm

def run_audit():
    log = TEPLogger("step_63_temporal_audit", log_file_path=PROJECT_ROOT / "logs/step_63_raw_sn_temporal_audit.log")
    set_step_logger(log)
    
    df = load_pantheon()
    log.info(f"Loaded {len(df)} SNe Ia (Hubble flow).")
    
    targets = [
        {'name': '1. Raw Stretch (x1)', 'col': 'x1', 'err': 'x1ERR'},
        {'name': '2. Raw Mag Residual', 'col': 'raw_mag_resid', 'err': 'mBERR'},
        {'name': '3. SALT Correction (mB-corr)', 'col': 'salt_correction', 'err': None},
        {'name': '4. Std Mag Residual', 'col': 'std_mag_resid', 'err': 'm_b_corr_err_DIAG'},
        {'name': '5. PV Correction (v_pec, km/s)', 'col': 'pv_correction', 'err': None}
    ]
    
    log.info("\n=== ACCOUNTING IDENTITY (Unweighted OLS) ===")
    log.info("Unweighted OLS ensures exact algebraic summation across targets.")
    for t in targets:
        coef, err, tval, pval = fit_target(df, t['col'], use_weights=False)
        log.info(f"{t['name']:<32} : D = {coef:>8.4f} +/- {err:>6.4f} (p = {pval:.3f})")
    
    log.info("\n=== STATISTICAL SIGNIFICANCE (WLS + Permutation) ===")
    results = {}
    for t in targets:
        coef, err, tval, pval = fit_target(df, t['col'], use_weights=True, err_col=t['err'])
        p_perm = run_permutation(df, t['col'], use_weights=True, err_col=t['err'], n_perms=500)
        log.info(f"{t['name']:<32} : D = {coef:>8.4f} +/- {err:>6.4f} (p_stat = {pval:.3f}, p_perm = {p_perm:.3f})")
        results[t['col']] = {
            "name": t['name'],
            "coef": float(coef),
            "err": float(err),
            "tval": float(tval),
            "pval_stat": float(pval),
            "pval_perm": float(p_perm)
        }

    summary = {
        "step": "63",
        "description": "Gate G: raw SN temporal audit — pre-standardization magnitude residuals vs SALT3-standardized residuals",
        "n_sne": int(len(df)),
        "targets": results
    }
    out_json = PROJECT_ROOT / "results" / "outputs" / "step_63_raw_sn_temporal_audit.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Saved summary to {out_json}")

run = run_audit

if __name__ == '__main__':
    run_audit()
