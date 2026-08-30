#!/usr/bin/env python3
"""
Step 64: Mechanism-Resolved SN Temporal Audit (Gate G v3.1)
===========================================================
Simultaneously fits Temporal (G~r) and Kinematic (G~1/r) macroscopic gradients 
to the pre- and post-standardization observables of Pantheon+ SNe.
Includes injection-recovery, stratified permutations, and survey ablation.
"""

import sys
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
V_STAR = 10000.0

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
        
        # P_T scales as v / 10000
        df[f'PT_{name}'] = (df['vcmb'] / V_STAR) * df[f'cos_theta_{name}']
        # P_K scales as 10000 / v
        df[f'PK_{name}'] = (V_STAR / df['vcmb']) * df[f'cos_theta_{name}']
        
    return df

def load_pantheon():
    df = pd.read_csv(PROJECT_ROOT / 'data/raw/Pantheon+SH0ES.dat', sep=r'\s+')
    df = df.drop_duplicates(subset=['CID'], keep='first')
    df = df[(df['zHD'] > 0.01) & (df['zHD'] < 0.1)]
    df = df[df['HOST_LOGMASS'] > 0].copy()
    
    df['vcmb'] = df['zCMB'] * C_KMS
    df = setup_axes(df)
    
    df['mu_bg'] = cosmo.distmod(df['zCMB']).value
    
    # Exact Accounting Targets
    df['raw_mag_resid'] = df['mB'] - df['mu_bg']
    df['C_SALT'] = df['mB'] - df['m_b_corr']
    df['std_mag_resid'] = df['m_b_corr'] - df['mu_bg']
    
    # Mechanism Attribution Targets
    df['salt_x1'] = df['x1']
    df['salt_c'] = df['c']
    df['salt_biasCor'] = df['biasCor_m_b']
    
    # PV Correction Channel
    df['pv_correction'] = C_KMS * (df['zCMB'] - df['zHD']) / (1 + df['zCMB'])
    
    # Setup redshift bins for stratification
    df['z_bin'] = pd.cut(df['zCMB'], bins=[0.01, 0.04, 0.07, 0.10], labels=['low', 'mid', 'high'])
    df['strata'] = df['IDSURVEY'].astype(str) + '_' + df['z_bin'].astype(str)
    
    return df

def fit_simultaneous(df, target_col, axis_name='CMB', use_weights=False, err_col=None, return_model=False, exclude_survey=None):
    if exclude_survey is not None:
        df_fit = df[df['IDSURVEY'] != exclude_survey].copy()
    else:
        df_fit = df
        
    X = pd.DataFrame({
        'PT': df_fit[f'PT_{axis_name}'],
        'PK': df_fit[f'PK_{axis_name}'],
        'zCMB': df_fit['zCMB'],
        'mass': df_fit['HOST_LOGMASS']
    })
    
    surveys = pd.get_dummies(df_fit['IDSURVEY'], drop_first=True, dtype=float)
    X = pd.concat([X, surveys], axis=1)
    X = sm.add_constant(X)
    y = df_fit[target_col]
    
    if use_weights and err_col is not None and err_col in df_fit.columns:
        weights = 1.0 / (df_fit[err_col]**2 + 1e-6)
        model = sm.WLS(y, X, weights=weights)
    else:
        model = sm.OLS(y, X)
        
    res = model.fit()
    if return_model:
        return res
    
    return {
        'DT': res.params['PT'], 'DT_err': res.bse['PT'], 'DT_p': res.pvalues['PT'],
        'DK': res.params['PK'], 'DK_err': res.bse['PK'], 'DK_p': res.pvalues['PK']
    }

def run_injection_recovery(df, axis_name='CMB'):
    # Synthetic injection recovery on the true geometry
    X = pd.DataFrame({
        'PT': df[f'PT_{axis_name}'],
        'PK': df[f'PK_{axis_name}'],
        'zCMB': df['zCMB'],
        'mass': df['HOST_LOGMASS']
    })
    surveys = pd.get_dummies(df['IDSURVEY'], drop_first=True, dtype=float)
    X = pd.concat([X, surveys], axis=1)
    X = sm.add_constant(X)
    
    scenarios = [
        (0.0, 0.0),
        (0.05, 0.0),
        (0.0, 0.05),
        (0.05, 0.05)
    ]
    
    np.random.seed(42)
    results = {}
    
    for (DT_inj, DK_inj) in scenarios:
        rec_DT = []
        rec_DK = []
        for _ in range(100):
            noise = np.random.normal(0, 0.15, size=len(df))
            y_syn = DT_inj * df[f'PT_{axis_name}'] + DK_inj * df[f'PK_{axis_name}'] + noise
            res = sm.OLS(y_syn, X).fit()
            rec_DT.append(res.params['PT'])
            rec_DK.append(res.params['PK'])
            
        results[(DT_inj, DK_inj)] = {
            'DT_mean': np.mean(rec_DT), 'DT_std': np.std(rec_DT),
            'DK_mean': np.mean(rec_DK), 'DK_std': np.std(rec_DK)
        }
        
    return results

def run_stratified_permutation(df, target_col, axis_name='CMB', n_perms=1000):
    np.random.seed(42)
    original_cos = df[f'cos_theta_{axis_name}'].values.copy()
    
    obs = fit_simultaneous(df, target_col, axis_name, use_weights=False)
    obs_DT = obs['DT']
    obs_DK = obs['DK']
    
    null_DT = []
    null_DK = []
    
    for _ in range(n_perms):
        shuffled_cos = np.zeros_like(original_cos)
        for stratum in df['strata'].unique():
            mask = df['strata'] == stratum
            shuffled_cos[mask] = np.random.permutation(original_cos[mask])
            
        df[f'PT_{axis_name}'] = (df['vcmb'] / V_STAR) * shuffled_cos
        df[f'PK_{axis_name}'] = (V_STAR / df['vcmb']) * shuffled_cos
        
        res = fit_simultaneous(df, target_col, axis_name, use_weights=False)
        null_DT.append(res['DT'])
        null_DK.append(res['DK'])
        
    df[f'cos_theta_{axis_name}'] = original_cos
    df[f'PT_{axis_name}'] = (df['vcmb'] / V_STAR) * df[f'cos_theta_{axis_name}']
    df[f'PK_{axis_name}'] = (V_STAR / df['vcmb']) * df[f'cos_theta_{axis_name}']
    
    p_DT = np.mean(np.abs(null_DT) >= np.abs(obs_DT))
    p_DK = np.mean(np.abs(null_DK) >= np.abs(obs_DK))
    
    return p_DT, p_DK

def run_audit():
    log = TEPLogger("step_64_mechanism_audit", log_file_path=PROJECT_ROOT / "outputs/logs/step_64_mechanism_audit.log")
    set_step_logger(log)
    
    df = load_pantheon()
    log.info(f"Loaded {len(df)} SNe Ia (Hubble flow, 0.01 < zHD < 0.1).")
    
    log.info("\n=== 1. INJECTION RECOVERY (CMB Axis) ===")
    inj_res = run_injection_recovery(df, 'CMB')
    for (inj_T, inj_K), rec in inj_res.items():
        log.info(f"Inject (DT={inj_T:.2f}, DK={inj_K:.2f}) -> Recovered DT={rec['DT_mean']:.4f}+/-{rec['DT_std']:.4f}, DK={rec['DK_mean']:.4f}+/-{rec['DK_std']:.4f}")
        
    targets = [
        {'name': 'Raw Mag Residual', 'col': 'raw_mag_resid'},
        {'name': 'SALT C_SALT (mB - m_b_corr)', 'col': 'C_SALT'},
        {'name': 'Std Mag Residual', 'col': 'std_mag_resid'},
        {'name': 'Mechanism: x1', 'col': 'salt_x1'},
        {'name': 'Mechanism: c', 'col': 'salt_c'},
        {'name': 'Mechanism: biasCor_m_b', 'col': 'salt_biasCor'},
        {'name': 'PV Correction (km/s)', 'col': 'pv_correction'}
    ]
    
    for axis in ['CMB', 'CF4']:
        log.info(f"\n=== 2. SIMULTANEOUS FITS ({axis} Axis) ===")
        log.info(f"{'Observable':<30} | {'DT':>8} {'DT_err':>8} {'DT_p':>7} | {'DK':>8} {'DK_err':>8} {'DK_p':>7}")
        log.info("-" * 80)
        
        for t in targets:
            res = fit_simultaneous(df, t['col'], axis, use_weights=False)
            log.info(f"{t['name']:<30} | {res['DT']:>8.4f} {res['DT_err']:>8.4f} {res['DT_p']:>7.4f} | {res['DK']:>8.4f} {res['DK_err']:>8.4f} {res['DK_p']:>7.4f}")
            
        log.info("\n--- Exact Accounting Identity Check ---")
        dt_raw = fit_simultaneous(df, 'raw_mag_resid', axis)['DT']
        dt_csalt = fit_simultaneous(df, 'C_SALT', axis)['DT']
        dt_std = fit_simultaneous(df, 'std_mag_resid', axis)['DT']
        log.info(f"DT_raw ({dt_raw:.4f}) = DT_C_SALT ({dt_csalt:.4f}) + DT_std ({dt_std:.4f})  [Diff: {dt_raw - (dt_csalt+dt_std):.1e}]")

    log.info("\n=== 3. STRATIFIED PERMUTATIONS (CMB Axis, N=1000) ===")
    log.info(f"{'Observable':<30} | {'p_DT_perm':>9} | {'p_DK_perm':>9}")
    log.info("-" * 55)
    for t in targets[:3]:  # Just run on the main magnitude targets to save time
        p_DT, p_DK = run_stratified_permutation(df, t['col'], 'CMB', n_perms=1000)
        log.info(f"{t['name']:<30} | {p_DT:>9.4f} | {p_DK:>9.4f}")
        
    log.info("\n=== 4. LEAVE-ONE-SURVEY-OUT ROBUSTNESS (Raw Mag Resid, CMB Axis) ===")
    surveys = df['IDSURVEY'].unique()
    for s in surveys:
        s_count = len(df[df['IDSURVEY'] == s])
        if s_count < 5: continue
        res = fit_simultaneous(df, 'raw_mag_resid', 'CMB', exclude_survey=s)
        log.info(f"Exclude {s:<10} (N={s_count:>3}) -> DT = {res['DT']:>8.4f} (p={res['DT_p']:.4f}) | DK = {res['DK']:>8.4f} (p={res['DK_p']:.4f})")

if __name__ == '__main__':
    run_audit()
