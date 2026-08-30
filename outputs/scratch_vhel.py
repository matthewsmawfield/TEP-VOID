import gzip
from pathlib import Path

PROJECT_ROOT = Path('.').resolve()
groups = {}
with gzip.open(PROJECT_ROOT / 'data/raw/external/cf4_table4.dat.gz', 'rt') as f:
    for line in f:
        pgc1 = line[0:7].strip()
        vh = line[27:32].strip()
        vcmb = line[39:44].strip()
        if pgc1 and vh and vcmb:
            groups[pgc1] = (float(vh), float(vcmb))
            
print("Group 1PGC, Vh, Vcmb, Diff")
for i, (pgc1, (vh, vcmb)) in enumerate(groups.items()):
    if i > 10: break
    print(f"{pgc1}: Vh={vh}, Vcmb={vcmb}, Diff={vcmb - vh}")
