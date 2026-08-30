#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.steps.step_49_band_dependence import Step49BandDependence

def run_loo_for_sample(step49, df, sample_name):
    print_status(f"\n--- LOO Analysis for {sample_name} ---", "PROCESS")
    
    if len(df) < 3:
        print_status("Dataset not large enough for LOO.", "ERROR")
        return
        
    base_res = step49.run_regression(df, f"Base {sample_name}")
    base_slope_wls = base_res["slope"]
    assert "student_t_slope" in base_res, "Student-t slope missing!"
    base_slope_t = base_res["student_t_slope"]
    base_sigma = base_res["slope_significance_sigma"]
    
    print_status(f"Base Model (WLS):       slope = {base_slope_wls:+.2e} ({base_sigma:.2f} sigma)", "TEST")
    print_status(f"Base Model (Student-t): slope = {base_slope_t:+.2e} +/- {base_res['student_t_slope_err']:.2e}", "TEST")
    
    loo_results = []
    for idx, row in df.iterrows():
        gal = row["galaxy"]
        df_loo = df[df.index != idx]
        res = step49.run_regression(df_loo, f"LOO {gal}", x_col="X_i", y_col="delta_mu_band", err_col="delta_mu_band_err")
        if res:
            wls_slope = res["slope"]
            assert "student_t_slope" in res, "Student-t slope missing!"
            t_slope = res["student_t_slope"]
            sigma = res["slope_significance_sigma"]
            diff_wls = (wls_slope - base_slope_wls) / abs(base_slope_wls) * 100
            diff_t = (t_slope - base_slope_t) / abs(base_slope_t) * 100 if base_slope_t != 0 else 0
            
            loo_results.append({
                "galaxy": gal, 
                "wls_slope": wls_slope, 
                "wls_diff": diff_wls,
                "t_slope": t_slope,
                "t_diff": diff_t,
                "sigma": sigma
            })
            
    loo_results_sorted = sorted(loo_results, key=lambda x: abs(x["wls_diff"]), reverse=True)
    
    print_status(f"\n--- LOO Results: {sample_name} ---", "SUCCESS")
    print_status(f"{'Excluded':<12} | {'WLS Slope':<10} ({'%Diff':>7}) | {'Student-t':<10} ({'%Diff':>7}) | WLS Sigma", "INFO")
    print_status("-" * 80, "INFO")
    for r in loo_results_sorted:
        print_status(
            f"{r['galaxy']:<12} | {r['wls_slope']:+.2e} ({r['wls_diff']:>6.1f}%) | "
            f"{r['t_slope']:+.2e} ({r['t_diff']:>6.1f}%) | {r['sigma']:.2f}σ", 
            "TEST"
        )
        
    # Summary Metrics
    min_wls = min([r['wls_slope'] for r in loo_results])
    max_wls = max([r['wls_slope'] for r in loo_results])
    min_t = min([r['t_slope'] for r in loo_results])
    max_t = max([r['t_slope'] for r in loo_results])
    
    all_wls_negative = all(r['wls_slope'] < 0 for r in loo_results)
    all_t_negative = all(r['t_slope'] < 0 for r in loo_results)
    
    print_status(f"\nSummary for {sample_name}:", "INFO")
    print_status(f"WLS LOO Range: [{min_wls:+.2e}, {max_wls:+.2e}]. All negative? {all_wls_negative}", "TEST")
    print_status(f"Student-t LOO Range: [{min_t:+.2e}, {max_t:+.2e}]. All negative? {all_t_negative}", "TEST")
    

def run():
    logger = TEPLogger("step_49b", log_file_path=PROJECT_ROOT / "logs" / "step_49b_loo.log")
    set_step_logger(logger)
    
    print_status("================================================================================", "INFO")
    print_status("   Step 49b: Band Dependence Leave-One-Out Robustness (Cross-Team Audit)", "INFO")
    print_status("================================================================================", "INFO")
    
    step49 = Step49BandDependence()
    path = PROJECT_ROOT / "data" / "processed" / "band_dependence_matched.csv"
    if not path.exists():
        print_status("Run step 49 first.", "ERROR")
        return
        
    df_matched = pd.read_csv(path)
    
    df_primary = df_matched[df_matched["sample"] == "primary_MF2023"].copy()
    df_secondary = df_matched[df_matched["sample"] == "secondary_KP_R22"].copy()
    
    run_loo_for_sample(step49, df_primary, "MF2023 (Same-Team)")
    run_loo_for_sample(step49, df_secondary, "KP vs R22 (Cross-Team)")
        
if __name__ == "__main__":
    run()
