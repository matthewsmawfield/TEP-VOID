import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path

# Re-run the exact data loading from step 71
df = pd.read_csv("data/raw/Pantheon+SH0ES.dat", sep=r"\s+")
df = df.rename(columns={"zCMB": "zcmb", "RA": "ra", "DEC": "dec"})

vrot = pd.read_csv("data/processed/pantheon_host_vrot_deep.csv")
measured = vrot[vrot["v_rot_deep"].notna() & (vrot["v_rot_deep"] > 0)].copy()
measured = measured.rename(columns={"v_rot_deep": "V_rot"})

import sys
sys.path.insert(0, str(Path(".").resolve()))
from scripts.utils.screening import compute_screening, U_REF_SCREENED

C_KMS = 299792.458
pgc = measured["pgc"].fillna(0).astype(int).values
S_total = compute_screening(pgc, Path("."))
measured["S_total"] = S_total
measured["U_i"] = (measured["V_rot"].values ** 2) / 2.0
measured["X_i"] = (S_total * measured["U_i"].values - U_REF_SCREENED) / C_KMS ** 2

df = df.merge(measured[["CID", "V_rot", "X_i"]], on="CID", how="left")
df = df[df["X_i"].notna() & np.isfinite(df["X_i"])].copy()

# Restrict to Hubble flow and valid mass
df = df[df["IS_CALIBRATOR"] == 0].copy()
df = df[df["HOST_LOGMASS"].notna() & (df["HOST_LOGMASS"] > 7)].copy()
df = df.sort_values("x1ERR").drop_duplicates(subset=["CID"], keep="first")

print(f"N = {len(df)}")

# OLS Regression for x1
y = df["x1"].values
X_z_mass = sm.add_constant(df[["zcmb", "HOST_LOGMASS"]].values)
model_z_mass = sm.OLS(y, X_z_mass).fit()

X_all = sm.add_constant(df[["zcmb", "HOST_LOGMASS", "X_i"]].values)
model_all = sm.OLS(y, X_all).fit()
model_all_hc3 = model_all.get_robustcov_results(cov_type='HC3')

print("\n--- Mz (z + logM) ---")
print(f"RSS: {model_z_mass.ssr:.2f}, R2: {model_z_mass.rsquared:.4f}, BIC: {model_z_mass.bic:.2f}")

print("\n--- MX (z + logM + X_i) ---")
print(f"RSS: {model_all.ssr:.2f}, R2: {model_all.rsquared:.4f}, BIC: {model_all.bic:.2f}")

print("\nX_i Coefficient (Standard):")
print(f"  Coef: {model_all.params[3]:.2e}, StdErr: {model_all.bse[3]:.2e}, t: {model_all.tvalues[3]:.2f}, p: {model_all.pvalues[3]:.4f}")

print("\nX_i Coefficient (HC3):")
print(f"  Coef: {model_all_hc3.params[3]:.2e}, StdErr: {model_all_hc3.bse[3]:.2e}, t: {model_all_hc3.tvalues[3]:.2f}, p: {model_all_hc3.pvalues[3]:.4f}")

