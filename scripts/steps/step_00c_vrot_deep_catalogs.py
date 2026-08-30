#!/usr/bin/env python3
"""
Step 00c: Deep V_rot Catalogs Cross-Match
=========================================
Expands the host potential catalog by cross-matching Pantheon+ supernovae
against deep rotation velocity catalogs directly from VizieR using astroquery.
This "de-contaminates" the low-redshift velocity field and pushes the
X_i-step testing to z > 0.03.

Catalogs:
- SPARC (Lelli+, 2016) - J/AJ/152/157
- ALFALFA a.100 (Haynes+, 2018) - J/ApJ/861/49
"""

import json
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.vizier import Vizier
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status

class Step00cDeepVrot:
    """Step 00c: Unified deep V_rot catalog for Pantheon+."""

    def __init__(self):
        self.root = PROJECT_ROOT
        self.data_raw = self.root / "data" / "raw"
        self.data_proc = self.root / "data" / "processed"
        self.results = self.root / "results" / "outputs"
        
        self.data_proc.mkdir(parents=True, exist_ok=True)
        
        self.logger = TEPLogger("step_00c", log_file_path=self.root / "logs" / "step_00c_vrot_deep_catalogs.log")
        set_step_logger(self.logger)

    def load_pantheon(self):
        pantheon_file = self.data_raw / "Pantheon+SH0ES.dat"
        df = pd.read_csv(pantheon_file, sep=r"\s+")
        self.sne = df.drop_duplicates(subset="CID")[
            ["CID", "RA", "DEC", "zCMB", "HOST_LOGMASS", "IS_CALIBRATOR"]
        ].copy()
        print_status(f"Loaded {len(self.sne)} unique Pantheon+ SNe", "INFO")

    def _crossmatch(self, sne_df, cat_coords, radius_arcsec=120):
        """Crossmatch SNe against a catalog of SkyCoords, returning best match indices."""
        sne_coords = SkyCoord(ra=sne_df['RA'].values*u.deg, dec=sne_df['DEC'].values*u.deg)
        idx, d2d, _ = sne_coords.match_to_catalog_sky(cat_coords)
        
        valid = d2d < radius_arcsec * u.arcsec
        return valid, idx[valid]

    def _safe_float(self, val):
        if val is None:
            return np.nan
        if np.ma.is_masked(val):
            return np.nan
        try:
            f = float(val)
            if np.isnan(f) or np.isinf(f):
                return np.nan
            return f
        except Exception:
            return np.nan

    def query_alfalfa(self):
        print_status("Querying ALFALFA 100% (J/ApJ/861/49) from VizieR...", "PROCESS")
        v = Vizier(columns=["RAJ2000", "DEJ2000", "W50"], row_limit=-1)
        catalogs = v.get_catalogs("J/ApJ/861/49")
        if not catalogs:
            print_status("ALFALFA catalog not found.", "ERROR")
            return pd.DataFrame()
            
        cat = catalogs[0]
        ra = cat['RAJ2000']
        dec = cat['DEJ2000']
        w50 = cat['W50']
        
        cat_coords = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))
        valid_mask, cat_indices = self._crossmatch(self.sne, cat_coords)
        
        matches = []
        valid_sne_indices = np.where(valid_mask)[0]
        for i, sn_idx in enumerate(valid_sne_indices):
            cat_idx = cat_indices[i]
            width = self._safe_float(w50[cat_idx])
            if not np.isnan(width) and width > 0:
                matches.append({
                    'CID': self.sne.iloc[sn_idx]['CID'],
                    'v_rot_alfalfa': width / 2.0,  # Uncorrected for inclination here
                    'w50_alfalfa': width
                })
        
        df_alfalfa = pd.DataFrame(matches)
        print_status(f"Matched {len(df_alfalfa)} SNe to ALFALFA", "SUCCESS")
        return df_alfalfa

    def query_sparc(self):
        print_status("Querying SPARC (J/AJ/152/157) from VizieR...", "PROCESS")
        v = Vizier(columns=["_RA", "_DE", "Vflat"], row_limit=-1)
        catalogs = v.get_catalogs("J/AJ/152/157")
        if not catalogs:
            print_status("SPARC catalog not found.", "ERROR")
            return pd.DataFrame()
            
        cat = catalogs[0]
        ra = cat['_RA']
        dec = cat['_DE']
        vflat = cat['Vflat']
        
        cat_coords = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))
        valid_mask, cat_indices = self._crossmatch(self.sne, cat_coords)
        
        matches = []
        valid_sne_indices = np.where(valid_mask)[0]
        for i, sn_idx in enumerate(valid_sne_indices):
            cat_idx = cat_indices[i]
            vf = self._safe_float(vflat[cat_idx])
            if not np.isnan(vf) and vf > 0:
                matches.append({
                    'CID': self.sne.iloc[sn_idx]['CID'],
                    'v_rot_sparc': vf
                })
                
        df_sparc = pd.DataFrame(matches)
        print_status(f"Matched {len(df_sparc)} SNe to SPARC", "SUCCESS")
        return df_sparc

    def merge_deep_catalog(self, df_alfalfa, df_sparc):
        """Merge ALFALFA, SPARC, and existing HyperLEDA into a unified V_rot catalog."""
        print_status("Merging deep V_rot sources...", "PROCESS")
        
        hyperleda_path = self.data_proc / "pantheon_host_vrot_vizier.csv"
        if not hyperleda_path.exists():
            print_status("pantheon_host_vrot_vizier.csv not found.", "WARN")
            df_unified = self.sne.copy()
            df_unified['v_rot'] = np.nan
        else:
            df_unified = pd.read_csv(hyperleda_path)
            
        if not df_alfalfa.empty:
            df_unified = df_unified.merge(df_alfalfa, on='CID', how='left')
        else:
            df_unified['v_rot_alfalfa'] = np.nan
            
        if not df_sparc.empty:
            df_unified = df_unified.merge(df_sparc, on='CID', how='left')
        else:
            df_unified['v_rot_sparc'] = np.nan
            
        def resolve_vrot(row):
            # 1. SPARC
            if pd.notna(row.get('v_rot_sparc')) and row['v_rot_sparc'] > 0:
                return row['v_rot_sparc'], 'SPARC'
            # 2. ALFALFA
            elif pd.notna(row.get('v_rot_alfalfa')) and row['v_rot_alfalfa'] > 0:
                # Need inclination correction for ALFALFA
                incl = row.get('inclination_deg', np.nan)
                if pd.notna(incl) and incl > 0:
                    sin_i = np.sin(np.radians(incl))
                    if sin_i > 0.01:
                        return row['v_rot_alfalfa'] / sin_i, 'ALFALFA'
            # 3. HyperLEDA
            v = row.get('v_rot', np.nan)
            if pd.notna(v) and v > 0:
                return v, 'HyperLEDA'
            return np.nan, 'None'
            
        vrot_unified = []
        sources = []
        for _, row in df_unified.iterrows():
            v, source = resolve_vrot(row)
            vrot_unified.append(v)
            sources.append(source)
            
        df_unified['v_rot_deep'] = vrot_unified
        df_unified['v_rot_source'] = sources
        
        n_sparc = (df_unified['v_rot_source'] == 'SPARC').sum()
        n_alf = (df_unified['v_rot_source'] == 'ALFALFA').sum()
        n_hl = (df_unified['v_rot_source'] == 'HyperLEDA').sum()
        
        print_status(f"Unified catalog V_rot sources:", "SUCCESS")
        print_status(f"  SPARC:     {n_sparc}", "INFO")
        print_status(f"  ALFALFA:   {n_alf}", "INFO")
        print_status(f"  HyperLEDA: {n_hl}", "INFO")
        print_status(f"  Total measured: {n_sparc + n_alf + n_hl}", "INFO")
        
        output_path = self.data_proc / "pantheon_host_vrot_deep.csv"
        df_unified.to_csv(output_path, index=False)
        print_status(f"Saved deep V_rot catalog to {output_path}", "SUCCESS")
        
        summary = {
            "step": "00c_vrot_deep_catalogs",
            "description": "Cross-match Pantheon+ against VizieR deep V_rot catalogs (SPARC, ALFALFA)",
            "sources": {
                "SPARC": int(n_sparc),
                "ALFALFA": int(n_alf),
                "HyperLEDA": int(n_hl),
                "Total Measured": int(n_sparc + n_alf + n_hl)
            }
        }
        with open(self.results / "step_00c_vrot_deep_catalogs_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    def run(self):
        print_status("="*70, "TITLE")
        print_status("Step 00c: Deep V_rot Catalogs Cross-Match", "TITLE")
        print_status("="*70, "TITLE")
        
        self.load_pantheon()
        
        df_alfalfa = self.query_alfalfa()
        df_sparc = self.query_sparc()
        
        self.merge_deep_catalog(df_alfalfa, df_sparc)
        print_status("Step 00c complete", "SUCCESS")

if __name__ == "__main__":
    step = Step00cDeepVrot()
    step.run()
