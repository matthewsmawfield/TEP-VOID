import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import linregress
from astropy.coordinates import SkyCoord
import astropy.units as u
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path('.').resolve()
df = pd.read_csv(PROJECT_ROOT / 'data/raw/Pantheon+SH0ES.dat', sep=r'\s+')
df = df[(df['zHD'] > 0.01) & (df['zHD'] < 0.1)]

CMB_L = 264.0
CMB_B = 48.0
CMB_NX = np.cos(np.radians(CMB_B)) * np.cos(np.radians(CMB_L))
CMB_NY = np.cos(np.radians(CMB_B)) * np.sin(np.radians(CMB_L))
CMB_NZ = np.sin(np.radians(CMB_B))
CMB_VEC = np.array([CMB_NX, CMB_NY, CMB_NZ])

coords = SkyCoord(ra=df['RA'].values*u.deg, dec=df['DEC'].values*u.deg, frame='icrs')
gal = coords.galactic
nx = np.cos(gal.b.rad) * np.cos(gal.l.rad)
ny = np.cos(gal.b.rad) * np.sin(gal.l.rad)
nz = np.sin(gal.b.rad)
df['cos_theta_cmb'] = nx*CMB_NX + ny*CMB_NY + nz*CMB_NZ
df['vhel'] = df['zHEL'] * 299792.458

df['temporal_pred'] = (df['vhel'] / 10000.0) * df['cos_theta_cmb']

res = linregress(df['temporal_pred'], df['x1'])
print(f"Global Correlation of x1 with Temporal Gradient:")
print(f"Slope: {res.slope:.4f} +/- {res.stderr:.4f} (p-value: {res.pvalue:.4e})")

bins = [(3000, 6000), (6000, 10000), (10000, 15000), (15000, 20000), (20000, 30000)]
for vmin, vmax in bins:
    mask = (df['vhel'] >= vmin) & (df['vhel'] < vmax)
    df_bin = df[mask]
    if len(df_bin) > 10:
        res_bin = linregress(df_bin['cos_theta_cmb'], df_bin['x1'])
        print(f"Bin [{vmin:5d}-{vmax:5d}]: Slope vs cos_theta = {res_bin.slope:.4f} (p={res_bin.pvalue:.4f})")
