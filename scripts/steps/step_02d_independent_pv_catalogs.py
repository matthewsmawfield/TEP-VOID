#!/usr/bin/env python3
import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status

class Step02dIndependentPV:
    def __init__(self):
        self.root = PROJECT_ROOT
        self.data_processed = self.root / "data" / "processed"
        self.data_raw_external = self.root / "data" / "raw" / "external"
        self.logs = self.root / "logs"

        for d in [self.data_processed, self.data_raw_external, self.logs]:
            d.mkdir(parents=True, exist_ok=True)

        self.logger = TEPLogger(
            "step_02d", log_file_path=self.logs / "step_02d_independent_pv.log"
        )
        set_step_logger(self.logger)
        self.H0_FIDUCIAL = 75.0 

    def fetch_6dfgsv(self):
        print_status("Fetching 6dFGSv (Fundamental Plane) from VizieR...", "PROCESS")
        out_path = self.data_raw_external / "6dfgsv_pv.csv"
        
        try:
            from astroquery.vizier import Vizier
            v = Vizier(row_limit=-1, columns=['6dFGS', 'RAJ2000', 'DEJ2000', 'czgal', 'czgr', 'log(Dz/DH)', 'e_log(Dz/DH)'])
            res = v.query_constraints(catalog='J/MNRAS/445/2677/table1')
            if res and len(res) > 0:
                df = res[0].to_pandas()
                df.to_csv(out_path, index=False)
                print_status(f"  Fetched {len(df)} 6dFGSv galaxies", "SUCCESS")
                return df
        except Exception as e:
            print_status(f"  Failed to fetch 6dFGSv: {e}", "ERROR")
            
        return pd.DataFrame()

    def process_6dfgsv(self, df):
        if df.empty: return pd.DataFrame()
        cz = pd.to_numeric(df['czgal'], errors='coerce')
        eta = pd.to_numeric(df['log(Dz/DH)'], errors='coerce')
        e_eta = pd.to_numeric(df['e_log(Dz/DH)'], errors='coerce')
        
        valid = cz.notna() & eta.notna() & (cz > 0)
        df_valid = df[valid].copy()
        cz = cz[valid]
        eta = eta[valid]
        
        d_baseline = (cz / self.H0_FIDUCIAL) * (10 ** -eta)
        mu_baseline = 5 * np.log10(d_baseline) + 25
        e_mu = 5 * e_eta
        
        processed = pd.DataFrame({
            'galaxy': df_valid['6dFGS'],
            'ra': df_valid['RAJ2000'],
            'dec': df_valid['DEJ2000'],
            'v_cmb': cz,
            'DM_baseline': mu_baseline,
            'e_DM': e_mu,
            'catalog': '6dFGSv',
            'indicator': 'FP'
        })
        return processed

    def fetch_sfipp(self):
        print_status("Fetching SFI++ (Tully-Fisher) from VizieR...", "PROCESS")
        out_path = self.data_raw_external / "sfipp_pv.csv"
        
        try:
            from astroquery.vizier import Vizier
            v = Vizier(row_limit=-1, columns=['AGC', 'RAJ2000', 'DEJ2000', 'cz', 'rgal', 'Vgal'])
            res = v.query_constraints(catalog='J/ApJS/172/599/table2')
            if res and len(res) > 0:
                df = res[0].to_pandas()
                df.to_csv(out_path, index=False)
                print_status(f"  Fetched {len(df)} SFI++ galaxies", "SUCCESS")
                return df
        except Exception as e:
            print_status(f"  Failed to fetch SFI++: {e}", "ERROR")
            
        return pd.DataFrame()

    def process_sfipp(self, df):
        if df.empty: return pd.DataFrame()
        cz = pd.to_numeric(df['cz'], errors='coerce')
        rgal = pd.to_numeric(df['rgal'], errors='coerce')
        
        valid = cz.notna() & rgal.notna() & (rgal > 0) & (cz > 0)
        df_valid = df[valid].copy()
        
        d_baseline = df_valid['rgal'] / 100.0
        mu_baseline = 5 * np.log10(d_baseline) + 25
        e_mu = 0.35
        
        processed = pd.DataFrame({
            'galaxy': df_valid['AGC'].astype(str),
            'ra': df_valid['RAJ2000'],
            'dec': df_valid['DEJ2000'],
            'v_cmb': df_valid['cz'],
            'DM_baseline': mu_baseline,
            'e_DM': e_mu,
            'catalog': 'SFI++',
            'indicator': 'TF'
        })
        return processed

    def run(self):
        df_6df = self.fetch_6dfgsv()
        proc_6df = self.process_6dfgsv(df_6df)
        
        df_sfi = self.fetch_sfipp()
        proc_sfi = self.process_sfipp(df_sfi)
        
        combined = pd.concat([proc_6df, proc_sfi], ignore_index=True)
        
        out_path = self.data_processed / "independent_pv_catalogs.csv"
        combined.to_csv(out_path, index=False)
        print_status(f"Saved {len(combined)} independent PV records to {out_path}", "SUCCESS")

if __name__ == "__main__":
    step = Step02dIndependentPV()
    step.run()
