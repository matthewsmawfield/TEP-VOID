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
- 2MTF (Hong+, 2019) - J/MNRAS/487/2061
- WALLABY (Wang+, 2021) - J/ApJ/915/70
"""

import json
import sys
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
            ["CID", "RA", "DEC", "HOST_RA", "HOST_DEC", "zCMB", "HOST_LOGMASS", "IS_CALIBRATOR"]
        ].copy()
        
        # Use HOST_RA/HOST_DEC if valid, else fall back to RA/DEC
        valid_host = (self.sne['HOST_RA'] >= 0) & (self.sne['HOST_RA'] <= 360) & \
                     (self.sne['HOST_DEC'] >= -90) & (self.sne['HOST_DEC'] <= 90)
        self.sne['match_ra'] = np.where(valid_host, self.sne['HOST_RA'], self.sne['RA'])
        self.sne['match_dec'] = np.where(valid_host, self.sne['HOST_DEC'], self.sne['DEC'])
        
        print_status(f"Loaded {len(self.sne)} unique Pantheon+ SNe", "INFO")

    def _crossmatch(self, sne_df, cat_coords, radius_arcsec=180):
        """Crossmatch SNe against a catalog of SkyCoords."""
        sne_coords = SkyCoord(ra=sne_df['match_ra'].values*u.deg, dec=sne_df['match_dec'].values*u.deg)
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
        cat_coords = SkyCoord(ra=cat['RAJ2000'], dec=cat['DEJ2000'], unit=(u.deg, u.deg))
        valid_mask, cat_indices = self._crossmatch(self.sne, cat_coords)
        
        matches = []
        valid_sne_indices = np.where(valid_mask)[0]
        for i, sn_idx in enumerate(valid_sne_indices):
            width = self._safe_float(cat['W50'][cat_indices[i]])
            if not np.isnan(width) and width > 0:
                matches.append({'CID': self.sne.iloc[sn_idx]['CID'], 'v_rot_alfalfa': width / 2.0})
        
        df = pd.DataFrame(matches)
        print_status(f"Matched {len(df)} SNe to ALFALFA", "SUCCESS")
        return df

    def query_sparc(self):
        print_status("Querying SPARC (J/AJ/152/157) from VizieR...", "PROCESS")
        v = Vizier(columns=["_RA", "_DE", "Vflat"], row_limit=-1)
        catalogs = v.get_catalogs("J/AJ/152/157")
        if not catalogs:
            print_status("SPARC catalog not found.", "ERROR")
            return pd.DataFrame()
            
        cat = catalogs[0]
        cat_coords = SkyCoord(ra=cat['_RA'], dec=cat['_DE'], unit=(u.deg, u.deg))
        valid_mask, cat_indices = self._crossmatch(self.sne, cat_coords)
        
        matches = []
        valid_sne_indices = np.where(valid_mask)[0]
        for i, sn_idx in enumerate(valid_sne_indices):
            vf = self._safe_float(cat['Vflat'][cat_indices[i]])
            if not np.isnan(vf) and vf > 0:
                matches.append({'CID': self.sne.iloc[sn_idx]['CID'], 'v_rot_sparc': vf})
                
        df = pd.DataFrame(matches)
        print_status(f"Matched {len(df)} SNe to SPARC", "SUCCESS")
        return df
        
    def query_2mtf(self):
        print_status("Querying 2MTF (J/MNRAS/487/2061) from VizieR...", "PROCESS")
        v = Vizier(columns=["RAJ2000", "DEJ2000", "WHIc"], row_limit=-1)
        catalogs = v.get_catalogs("J/MNRAS/487/2061")
        if not catalogs:
            print_status("2MTF catalog not found.", "ERROR")
            return pd.DataFrame()
            
        cat = catalogs[0]
        cat_coords = SkyCoord(ra=cat['RAJ2000'], dec=cat['DEJ2000'], unit=(u.deg, u.deg))
        valid_mask, cat_indices = self._crossmatch(self.sne, cat_coords)
        
        matches = []
        valid_sne_indices = np.where(valid_mask)[0]
        for i, sn_idx in enumerate(valid_sne_indices):
            width = self._safe_float(cat['WHIc'][cat_indices[i]])
            if not np.isnan(width) and width > 0:
                # WHIc is already inclination-corrected HI line width. Vrot = W/2.
                matches.append({'CID': self.sne.iloc[sn_idx]['CID'], 'v_rot_2mtf': width / 2.0})
                
        df = pd.DataFrame(matches)
        print_status(f"Matched {len(df)} SNe to 2MTF", "SUCCESS")
        return df
        
    def query_wallaby(self):
        print_status("Querying WALLABY (J/ApJ/915/70) from VizieR...", "PROCESS")
        v = Vizier(columns=["_RA", "_DE", "dV/sigC"], row_limit=-1)
        catalogs = v.get_catalogs("J/ApJ/915/70")
        if not catalogs:
            print_status("WALLABY catalog not found.", "ERROR")
            return pd.DataFrame()
            
        cat = catalogs[0]
        cat_coords = SkyCoord(ra=cat['_RA'], dec=cat['_DE'], unit=(u.deg, u.deg))
        valid_mask, cat_indices = self._crossmatch(self.sne, cat_coords)
        
        matches = []
        valid_sne_indices = np.where(valid_mask)[0]
        for i, sn_idx in enumerate(valid_sne_indices):
            dv = self._safe_float(cat['dV/sigC'][cat_indices[i]])
            if not np.isnan(dv) and dv > 0:
                # Approximating velocity width from dV
                matches.append({'CID': self.sne.iloc[sn_idx]['CID'], 'v_rot_wallaby': dv / 2.0})
                
        df = pd.DataFrame(matches)
        print_status(f"Matched {len(df)} SNe to WALLABY", "SUCCESS")
        return df

    def merge_deep_catalog(self, df_alfalfa, df_sparc, df_2mtf, df_wallaby):
        print_status("Merging deep V_rot sources...", "PROCESS")
        
        hyperleda_path = self.data_proc / "pantheon_host_vrot_vizier.csv"
        if not hyperleda_path.exists():
            print_status("pantheon_host_vrot_vizier.csv not found.", "WARN")
            df_unified = self.sne.copy()
            df_unified['v_rot'] = np.nan
        else:
            df_unified = pd.read_csv(hyperleda_path)
            
        for df, col in [(df_alfalfa, 'v_rot_alfalfa'), (df_sparc, 'v_rot_sparc'), 
                        (df_2mtf, 'v_rot_2mtf'), (df_wallaby, 'v_rot_wallaby')]:
            if not df.empty:
                # Merge dropping duplicate CIDs in the incoming DF to avoid fanout
                df = df.groupby('CID').first().reset_index()
                df_unified = df_unified.merge(df, on='CID', how='left')
            else:
                df_unified[col] = np.nan
            
        def resolve_vrot(row):
            if pd.notna(row.get('v_rot_sparc')) and row['v_rot_sparc'] > 0:
                return row['v_rot_sparc'], 'SPARC'
            elif pd.notna(row.get('v_rot_alfalfa')) and row['v_rot_alfalfa'] > 0:
                incl = row.get('inclination_deg', np.nan)
                if pd.notna(incl) and incl > 0:
                    sin_i = np.sin(np.radians(incl))
                    if sin_i > 0.01:
                        return row['v_rot_alfalfa'] / sin_i, 'ALFALFA'
            elif pd.notna(row.get('v_rot_wallaby')) and row['v_rot_wallaby'] > 0:
                incl = row.get('inclination_deg', np.nan)
                if pd.notna(incl) and incl > 0:
                    sin_i = np.sin(np.radians(incl))
                    if sin_i > 0.01:
                        return row['v_rot_wallaby'] / sin_i, 'WALLABY'
            elif pd.notna(row.get('v_rot_2mtf')) and row['v_rot_2mtf'] > 0:
                # 2MTF is already inclination corrected (WHIc)
                return row['v_rot_2mtf'], '2MTF'
            
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
        
        counts = df_unified['v_rot_source'].value_counts()
        print_status(f"Unified catalog V_rot sources:", "SUCCESS")
        for source, count in counts.items():
            if source != 'None':
                print_status(f"  {source}: {count}", "INFO")
                
        total_measured = df_unified['v_rot_deep'].notna().sum()
        print_status(f"  Total measured: {total_measured}", "INFO")
        
        output_path = self.data_proc / "pantheon_host_vrot_deep.csv"
        df_unified.to_csv(output_path, index=False)
        print_status(f"Saved deep V_rot catalog to {output_path}", "SUCCESS")
        
        summary = {
            "step": "00c_vrot_deep_catalogs",
            "description": "Cross-match Pantheon+ against VizieR deep V_rot catalogs (SPARC, ALFALFA, 2MTF, WALLABY)",
            "sources": counts.to_dict()
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
        df_2mtf = self.query_2mtf()
        df_wallaby = self.query_wallaby()
        
        self.merge_deep_catalog(df_alfalfa, df_sparc, df_2mtf, df_wallaby)
        print_status("Step 00c complete", "SUCCESS")

if __name__ == "__main__":
    step = Step00cDeepVrot()
    step.run()
