#!/usr/bin/env python3
"""
Step 65: Finite-Coherence Temporal Kernel Audit (Gate G v3.2)
===========================================================
Fits a family of finite-coherence temporal kernels P(D) = (1 - exp(-D/L_T)) / D * cos(theta)
to the raw magnitude residuals of Pantheon+ SNe. Finds the maximum likelihood
coherence scale L_T to determine if the signal is a structurally bound local field
or a purely kinematic velocity shift (L_T -> 0).
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

C_KMS = 299792.458

AXES = {
    'CMB': {'l': 264.0, 'b': 48.0},
    'CF4': {'l': 298.0, 'b': -7.0}
}

cosmo = FlatLambdaCDM(H0=73.04, Om0=0.334)

def setup_axes(df):
    coords = SkyCoord(ra=df['RA'].values*u.deg, dec=df['DEC'].values*u.deg, frame='icrs')
    gal = coords.galactic
    df['nx'] = np.cos(gal.b.rad) * np.cos(gal.l.rad)
    df['ny'] = np.cos(gal.b.rad) * np.sin(gal.l.rad)
    df['nz'] = np.sin(gal.b.rad)
    
    for name, ax in AXES.items():
        l_rad = np.radians(ax['l'])
        b_rad = np.radians(ax['b'])
        nx = np.cos(b_rad) * np.cos(l_rad)
        ny = np.cos(b_rad) * np.sin(l_rad)
        nz = np.sin(b_rad)
        
        df[f'cos_theta_{name}'] = df['nx']*nx + df['ny']*ny + df['nz']*nz
        
    return df

def load_pantheon():
    df = pd.read_csv(PROJECT_ROOT / 'data/raw/Pantheon+SH0ES.dat', sep=r'\s+')
    df = df.drop_duplicates(subset=['CID'], keep='first')
    df = df[(df['zHD'] > 0.01) & (df['zHD'] < 0.1)]
    df = df[df['HOST_LOGMASS'] > 0].copy()
    
    df = setup_axes(df)
    df['mu_bg'] = cosmo.distmod(df['zCMB']).value
    df['D_Mpc'] = cosmo.comoving_distance(df['zCMB']).value
    df['raw_mag_resid'] = df['mB'] - df['mu_bg']
    
    return df

def fit_kernel(df, L_T, axis_name='CMB', target='raw_mag_resid'):
    D = df['D_Mpc'].values
    cos_t = df[f'cos_theta_{axis_name}'].values
    
    # Kernel: P = (1 - exp(-D / L_T)) / D * cos(theta)
    # Note: If L_T is very small, exp(-D/L_T) -> 0, so P -> 1/D * cos(theta) (Kinematic limit)
    # If L_T is very large, 1 - exp(-D/L_T) ~ D/L_T, so P -> 1/L_T * cos(theta) ~ constant (Linear temporal limit)
    # We multiply by 100 to keep coefficients on a reasonable scale
    if L_T < 1e-5:
        P = (1.0 / D) * cos_t * 100.0
    elif L_T > 1e6:
        P = np.ones_like(D) * cos_t * (100.0 / 1e6)
    else:
        P = ((1.0 - np.exp(-D / L_T)) / D) * cos_t * 100.0
        
    X = pd.DataFrame({
        'P_kernel': P,
        'zCMB': df['zCMB'],
        'mass': df['HOST_LOGMASS']
    })
    
    surveys = pd.get_dummies(df['IDSURVEY'], drop_first=True, dtype=float)
    X = pd.concat([X, surveys], axis=1)
    X = sm.add_constant(X)
    
    y = df[target]
    model = sm.OLS(y, X)
    res = model.fit()
    
    return {
        'L_T': L_T,
        'coef': res.params['P_kernel'],
        'err': res.bse['P_kernel'],
        'tval': res.tvalues['P_kernel'],
        'pval': res.pvalues['P_kernel'],
        'loglike': res.llf,
        'aic': res.aic,
        'bic': res.bic
    }

def run_audit():
    log = TEPLogger("step_65_finite_coherence_audit", log_file_path=PROJECT_ROOT / "logs/step_65_finite_coherence_audit.log")
    set_step_logger(log)
    
    df = load_pantheon()
    log.info(f"Loaded {len(df)} SNe Ia (Hubble flow, 0.01 < zHD < 0.1).")
    
    # Grid of L_T (Mpc)
    # 0 implies pure 1/r kinematic flow
    # > 1000 implies pure linear gradient
    LT_grid = [0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 500.0, 1000.0, 5000.0]
    
    for axis in ['CMB', 'CF4']:
        log.info(f"\n=== FINITE COHERENCE KERNEL FIT ({axis} Axis) ===")
        log.info(f"{'L_T (Mpc)':>10} | {'Coef':>8} | {'P-Value':>8} | {'LogLike':>10} | {'AIC':>10} | {'BIC':>10}")
        log.info("-" * 75)
        
        results = []
        for lt in LT_grid:
            res = fit_kernel(df, lt, axis_name=axis)
            results.append(res)
            log.info(f"{lt:>10.1f} | {res['coef']:>8.4f} | {res['pval']:>8.4f} | {res['loglike']:>10.2f} | {res['aic']:>10.2f} | {res['bic']:>10.2f}")
            
        best_lt = min(results, key=lambda x: x['bic'])
        log.info(f"\n--> BEST FIT (Lowest BIC) for {axis}: L_T = {best_lt['L_T']} Mpc (BIC = {best_lt['bic']:.2f})")

    summary = {
        "step": "65",
        "description": "Finite-coherence kernel audit — grid search over L_T on Pantheon+ raw magnitude residuals",
        "n_sne": int(len(df)),
        "lt_grid": LT_grid
    }
    out_json = PROJECT_ROOT / "results" / "outputs" / "step_65_finite_coherence_audit.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Saved summary to {out_json}")

run = run_audit

if __name__ == '__main__':
    run_audit()
