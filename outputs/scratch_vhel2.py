import gzip
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path('.').resolve()
groups = {}
with gzip.open(PROJECT_ROOT / 'data/raw/external/cf4_table4.dat.gz', 'rt') as f:
    for line in f:
        pgc1 = line[0:7].strip()
        vh = line[27:32].strip()
        vcmb = line[39:44].strip()
        ra = line[83:91].strip()
        dec = line[92:100].strip()
        glon = line[101:109].strip()
        glat = line[110:118].strip()
        if pgc1 and vh and vcmb and glon and glat:
            gl = np.radians(float(glon))
            gb = np.radians(float(glat))
            nx = np.cos(gb) * np.cos(gl)
            ny = np.cos(gb) * np.sin(gl)
            nz = np.sin(gb)
            
            CMB_L = 264.0
            CMB_B = 48.0
            CMB_NX = np.cos(np.radians(CMB_B)) * np.cos(np.radians(CMB_L))
            CMB_NY = np.cos(np.radians(CMB_B)) * np.sin(np.radians(CMB_L))
            CMB_NZ = np.sin(np.radians(CMB_B))
            
            cos_theta = nx * CMB_NX + ny * CMB_NY + nz * CMB_NZ
            
            groups[pgc1] = (float(vh), float(vcmb), cos_theta)
            
print("Group 1PGC, Vh, Vcmb, Diff, 371*cos_theta")
for i, (pgc1, (vh, vcmb, cos_theta)) in enumerate(groups.items()):
    if i > 10: break
    print(f"{pgc1}: Vh={vh}, Vcmb={vcmb}, Diff={vcmb - vh:.1f}, 371*cos_theta={371*cos_theta:.1f}")
