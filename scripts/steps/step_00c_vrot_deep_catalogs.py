#!/usr/bin/env python3
"""
Step 00c: Deep V_rot Catalogs Cross-Match
=========================================
Expands the host potential catalog by cross-matching Pantheon+ supernovae
against deep rotation velocity catalogs directly from VizieR using astroquery.
This "de-contaminates" the low-redshift velocity field and pushes the
X_i-step testing to z > 0.03.

Catalogs (all velocity-width based unless noted):
- SPARC (Lelli+, 2016)         - J/AJ/152/157     (Vflat, resolved rotation curve)
- Kourkchi+ 2019 (CF4 TF)      - J/ApJ/884/82     (logWimx, inclination-corrected)
- Dupuy+ 2021 (CF4 HI cat.)    - J/A+A/646/A113   (W50 + incl + Vhel)
- ALFALFA a.100 (Haynes+,2018) - J/ApJ/861/49     (W50 + Vhel; incl from HyperLEDA)
- EDD All-Digital HI (Courtois+,2009) - J/AJ/138/1938 (<Wmx> + <Vh>; incl from HyperLEDA)
- 2MTF (Hong+, 2019)           - J/MNRAS/487/2061 (WHIc, inclination-corrected; cz2mrs)
- HyperLEDA (VII/237+VII/238)  - pantheon_host_vrot_vizier.csv (v_rot, corrected)

The previous WALLABY query (J/ApJ/915/70) was removed: that table carries only
the spectral channel-spacing/SNR column dV/sigC, not a galaxy linewidth, so no
rotation velocity can be derived from it.

Redshift-consistency gate: a positional match is accepted only if the catalog
systemic velocity agrees with the SN redshift, |v_CMB(cat) - c*zCMB| < DV_MAX.
This removes projected foreground/background galaxies (an HI source cannot be
the host of a z ~ 0.5 SN). For catalogs without a velocity column (Kourkchi),
a horizon gate zCMB < Z_MAX_NOCZ is applied since the source catalog only
reaches the local volume.
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

C_KMS = 299792.458
# Redshift-consistency tolerance: bound/cluster peculiar velocities plus the
# heliocentric-to-CMB dipole correction are <~1.2e3 km/s; a larger mismatch
# means the catalogued galaxy is not the SN host.
DV_MAX = 1500.0      # km/s
# Horizon gate for catalogs without a velocity column (CF4 TF sample is local)
Z_MAX_NOCZ = 0.10

# CMB dipole apex (Planck); used to convert heliocentric velocities to the
# CMB frame: v_cmb ~= v_hel + V_SUN * cos(theta_apex)
V_SUN = 369.0  # km/s
CMB_APEX_L, CMB_APEX_B = 264.021, 48.253  # deg


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

    @staticmethod
    def _v_cmb_from_helio(v_hel, ra_deg, dec_deg):
        """Approximate CMB-frame systemic velocity from a heliocentric one."""
        src = SkyCoord(ra=np.asarray(ra_deg, dtype=float)*u.deg,
                       dec=np.asarray(dec_deg, dtype=float)*u.deg, frame='icrs')
        apex = SkyCoord(l=CMB_APEX_L*u.deg, b=CMB_APEX_B*u.deg, frame='galactic').icrs
        cos_sep = np.cos(src.separation(apex).rad)
        return np.asarray(v_hel, dtype=float) + V_SUN * cos_sep

    def _redshift_gate(self, sn_idx, v_cat_helio):
        """Return boolean keep-mask for SN indices given catalog heliocentric velocity.

        Compares the catalog systemic velocity (converted to the CMB frame)
        with the SN zCMB. NaN velocities are treated as failing.
        """
        vcmb = self._v_cmb_from_helio(
            v_cat_helio,
            self.sne.iloc[sn_idx]['match_ra'].values,
            self.sne.iloc[sn_idx]['match_dec'].values,
        )
        v_sn = self.sne.iloc[sn_idx]['zCMB'].values * C_KMS
        keep = np.isfinite(v_cat_helio) & (np.abs(vcmb - v_sn) < DV_MAX)
        return keep

    # ------------------------------------------------------------------
    # Catalog queries
    # ------------------------------------------------------------------
    def query_alfalfa(self):
        print_status("Querying ALFALFA 100% (J/ApJ/861/49) from VizieR...", "PROCESS")
        v = Vizier(columns=["RAJ2000", "DEJ2000", "W50", "Vhel"], row_limit=-1)
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
            vhel = self._safe_float(cat['Vhel'][cat_indices[i]])
            if not np.isnan(width) and width > 0:
                matches.append({'CID': self.sne.iloc[sn_idx]['CID'],
                                'v_rot_alfalfa': width / 2.0,
                                'v_hel_alfalfa': vhel,
                                '_sn_idx': sn_idx})

        df = pd.DataFrame(matches)
        n_raw = len(df)
        if n_raw:
            keep = self._redshift_gate(df['_sn_idx'].values, df['v_hel_alfalfa'].values)
            df = df[keep].drop(columns=['_sn_idx', 'v_hel_alfalfa'])
            print_status(f"Matched {n_raw} SNe to ALFALFA; {len(df)} pass the redshift-consistency gate", "SUCCESS")
        else:
            print_status("Matched 0 SNe to ALFALFA", "SUCCESS")
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

    def query_kourkchi(self):
        """Kourkchi+2019 (J/ApJ/884/82): CF4 TF master catalog.

        logWimx is the log10 of the inclination-corrected HI linewidth
        (already deprojected), so V_rot = 10^logWimx / 2 with no further
        inclination handling. The catalog has no systemic velocity column;
        matches are horizon-gated (zCMB < Z_MAX_NOCZ) since the CF4 TF
        sample is local.
        """
        print_status("Querying Kourkchi+2019 (J/ApJ/884/82) from VizieR...", "PROCESS")
        v = Vizier(columns=["PGC", "_RA", "_DE", "logWimx", "Inc", "Wmx"], row_limit=-1)
        catalogs = v.get_catalogs("J/ApJ/884/82")
        if not catalogs:
            print_status("Kourkchi catalog not found.", "ERROR")
            return pd.DataFrame()

        cat = catalogs[0]
        cat_coords = SkyCoord(ra=cat['_RA'], dec=cat['_DE'], unit=(u.deg, u.deg))
        valid_mask, cat_indices = self._crossmatch(self.sne, cat_coords)

        matches = []
        valid_sne_indices = np.where(valid_mask)[0]
        for i, sn_idx in enumerate(valid_sne_indices):
            logw = self._safe_float(cat['logWimx'][cat_indices[i]])
            if np.isnan(logw) or logw <= 0:
                continue
            if self.sne.iloc[sn_idx]['zCMB'] >= Z_MAX_NOCZ:
                continue
            matches.append({'CID': self.sne.iloc[sn_idx]['CID'],
                            'v_rot_kourkchi': 0.5 * 10.0 ** logw,
                            'pgc_kourkchi': self._safe_float(cat['PGC'][cat_indices[i]])})

        df = pd.DataFrame(matches)
        print_status(f"Matched {len(df)} SNe to Kourkchi+2019 (horizon-gated)", "SUCCESS")
        return df

    def query_dupuy(self):
        """Dupuy+2021 (J/A+A/646/A113): CF4 HI data catalog, table2.

        table2 carries W50, inclination, and heliocentric velocity Vhel,
        so V_rot = W50 / (2 sin i) and the redshift-consistency gate applies
        directly.
        """
        print_status("Querying Dupuy+2021 CF4 HI (J/A+A/646/A113) from VizieR...", "PROCESS")
        v = Vizier(columns=["PGC", "_RA", "_DE", "Vhel", "W50", "incl"], row_limit=-1)
        catalogs = v.get_catalogs("J/A+A/646/A113")
        if not catalogs:
            print_status("Dupuy catalog not found.", "ERROR")
            return pd.DataFrame()

        cat = catalogs['J/A+A/646/A113/table2']
        ra = np.array(cat['_RA'].filled(np.nan), dtype=float)
        dec = np.array(cat['_DE'].filled(np.nan), dtype=float)
        ok = np.isfinite(ra) & np.isfinite(dec)
        cat_coords = SkyCoord(ra=ra[ok]*u.deg, dec=dec[ok]*u.deg)
        valid_mask, cat_indices = self._crossmatch(self.sne, cat_coords)
        okidx = np.where(ok)[0]

        matches = []
        valid_sne_indices = np.where(valid_mask)[0]
        for i, sn_idx in enumerate(valid_sne_indices):
            w50 = self._safe_float(cat['W50'][okidx[cat_indices[i]]])
            incl = self._safe_float(cat['incl'][okidx[cat_indices[i]]])
            vhel = self._safe_float(cat['Vhel'][okidx[cat_indices[i]]])
            pgc = self._safe_float(cat['PGC'][okidx[cat_indices[i]]])
            if np.isnan(w50) or w50 <= 0 or np.isnan(incl) or incl <= 0:
                continue
            sin_i = np.sin(np.radians(incl))
            if sin_i < 0.01:
                continue
            matches.append({'CID': self.sne.iloc[sn_idx]['CID'],
                            'v_rot_dupuy': w50 / (2.0 * sin_i),
                            'pgc_dupuy': pgc,
                            'v_hel_dupuy': vhel,
                            '_sn_idx': sn_idx})

        df = pd.DataFrame(matches)
        n_raw = len(df)
        if n_raw:
            keep = self._redshift_gate(df['_sn_idx'].values, df['v_hel_dupuy'].values)
            df = df[keep].drop(columns=['_sn_idx', 'v_hel_dupuy'])
            print_status(f"Matched {n_raw} SNe to Dupuy CF4-HI; {len(df)} pass the redshift-consistency gate", "SUCCESS")
        else:
            print_status("Matched 0 SNe to Dupuy CF4-HI", "SUCCESS")
        return df

    def query_edd(self):
        """EDD All-Digital HI profile catalog (Courtois+2009, J/AJ/138/1938).

        table4 carries the mean profile linewidth <Wmx> and mean heliocentric
        velocity <Vh>. The width needs an external inclination correction
        (applied at merge time from the HyperLEDA logR25-derived inclination,
        consistent with the ALFALFA channel); the velocity enables the
        redshift-consistency gate.
        """
        print_status("Querying EDD All-Digital HI (J/AJ/138/1938) from VizieR...", "PROCESS")
        v = Vizier(columns=["PGC", "_RA", "_DE", "<Vh>", "<Wmx>"], row_limit=-1)
        catalogs = v.get_catalogs("J/AJ/138/1938")
        if not catalogs:
            print_status("EDD catalog not found.", "ERROR")
            return pd.DataFrame()

        cat = catalogs['J/AJ/138/1938/table4']
        ra = np.array(cat['_RA'].filled(np.nan), dtype=float)
        dec = np.array(cat['_DE'].filled(np.nan), dtype=float)
        ok = np.isfinite(ra) & np.isfinite(dec)
        cat_coords = SkyCoord(ra=ra[ok]*u.deg, dec=dec[ok]*u.deg)
        valid_mask, cat_indices = self._crossmatch(self.sne, cat_coords)
        okidx = np.where(ok)[0]

        matches = []
        valid_sne_indices = np.where(valid_mask)[0]
        for i, sn_idx in enumerate(valid_sne_indices):
            wmx = self._safe_float(cat['<Wmx>'][okidx[cat_indices[i]]])
            vhel = self._safe_float(cat['<Vh>'][okidx[cat_indices[i]]])
            pgc = self._safe_float(cat['PGC'][okidx[cat_indices[i]]])
            if np.isnan(wmx) or wmx <= 0:
                continue
            matches.append({'CID': self.sne.iloc[sn_idx]['CID'],
                            'v_rot_edd_raw': wmx / 2.0,
                            'pgc_edd': pgc,
                            'v_hel_edd': vhel,
                            '_sn_idx': sn_idx})

        df = pd.DataFrame(matches)
        n_raw = len(df)
        if n_raw:
            keep = self._redshift_gate(df['_sn_idx'].values, df['v_hel_edd'].values)
            df = df[keep].drop(columns=['_sn_idx', 'v_hel_edd'])
            print_status(f"Matched {n_raw} SNe to EDD; {len(df)} pass the redshift-consistency gate", "SUCCESS")
        else:
            print_status("Matched 0 SNe to EDD", "SUCCESS")
        return df

    def query_2mtf(self):
        print_status("Querying 2MTF (J/MNRAS/487/2061) from VizieR...", "PROCESS")
        v = Vizier(columns=["RAJ2000", "DEJ2000", "WHIc", "cz2mrs"], row_limit=-1)
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
            vcat = self._safe_float(cat['cz2mrs'][cat_indices[i]])
            if not np.isnan(width) and width > 0:
                # WHIc is already inclination-corrected HI line width. Vrot = W/2.
                matches.append({'CID': self.sne.iloc[sn_idx]['CID'],
                                'v_rot_2mtf': width / 2.0,
                                'v_cat_2mtf': vcat,
                                '_sn_idx': sn_idx})

        df = pd.DataFrame(matches)
        n_raw = len(df)
        if n_raw:
            # cz2mrs is the 2MRS-frame velocity; within a few hundred km/s of
            # the CMB frame, so the same tolerance applies.
            keep = self._redshift_gate(df['_sn_idx'].values, df['v_cat_2mtf'].values)
            df = df[keep].drop(columns=['_sn_idx', 'v_cat_2mtf'])
            print_status(f"Matched {n_raw} SNe to 2MTF; {len(df)} pass the redshift-consistency gate", "SUCCESS")
        else:
            print_status("Matched 0 SNe to 2MTF", "SUCCESS")
        return df

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------
    def merge_deep_catalog(self, df_alfalfa, df_sparc, df_2mtf,
                           df_kourkchi, df_dupuy, df_edd):
        print_status("Merging deep V_rot sources...", "PROCESS")

        hyperleda_path = self.data_proc / "pantheon_host_vrot_vizier.csv"
        if not hyperleda_path.exists():
            print_status("pantheon_host_vrot_vizier.csv not found.", "WARN")
            df_unified = self.sne.copy()
            df_unified['v_rot'] = np.nan
        else:
            df_unified = pd.read_csv(hyperleda_path)

        # HyperLEDA redshift-consistency gate: the vizier pull carries VHI
        # (the catalog's heliocentric systemic velocity); reject rows whose
        # catalog velocity cannot be the SN host (projected interlopers).
        if 'VHI' in df_unified.columns:
            vhi = df_unified['VHI'].values
            has_vrot = df_unified['v_rot'].notna() & (df_unified['v_rot'] > 0)
            vcmb = self._v_cmb_from_helio(np.where(np.isfinite(vhi), vhi, np.nan),
                                          df_unified['sn_ra'].values,
                                          df_unified['sn_dec'].values)
            v_sn = df_unified['zCMB'].values * C_KMS
            bad = has_vrot & np.isfinite(vhi) & (np.abs(vcmb - v_sn) >= DV_MAX)
            if bad.any():
                badcids = df_unified.loc[bad, 'CID'].tolist()
                print_status(f"HyperLEDA redshift gate rejected {bad.sum()} false positional matches: {badcids}", "WARN")
                df_unified.loc[bad, 'v_rot'] = np.nan
        else:
            print_status("VHI column absent from HyperLEDA pull; redshift gate skipped for it.", "WARN")

        for df, col in [(df_alfalfa, 'v_rot_alfalfa'), (df_sparc, 'v_rot_sparc'),
                        (df_2mtf, 'v_rot_2mtf'), (df_kourkchi, 'v_rot_kourkchi'),
                        (df_dupuy, 'v_rot_dupuy'), (df_edd, 'v_rot_edd_raw')]:
            if not df.empty:
                # Merge dropping duplicate CIDs in the incoming DF to avoid fanout
                df = df.groupby('CID').first().reset_index()
                df_unified = df_unified.merge(df, on='CID', how='left')
            else:
                df_unified[col] = np.nan

        # Backfill PGC where the HyperLEDA pull left it empty, improving the
        # screening lookup downstream.
        for pgc_col in ['pgc_kourkchi', 'pgc_dupuy', 'pgc_edd']:
            if pgc_col in df_unified.columns:
                missing = df_unified['pgc'].isna() & df_unified[pgc_col].notna()
                df_unified.loc[missing, 'pgc'] = df_unified.loc[missing, pgc_col]

        def _corrected_by_incl(row, raw_col):
            """Return linewidth/(2 sin i) or NaN using the HyperLEDA logR25
            inclination. An uncorrected width is not a valid V_rot, so a
            missing inclination means the source cannot be used."""
            incl = row.get('inclination_deg', np.nan)
            if pd.notna(incl) and incl > 0:
                sin_i = np.sin(np.radians(incl))
                if sin_i > 0.01:
                    return row[raw_col] / sin_i
            return np.nan

        def resolve_vrot(row):
            """Priority-ordered source resolution; each branch either returns a
            valid inclination-corrected V_rot or cleanly falls through to the
            next source."""
            v = row.get('v_rot_sparc', np.nan)
            if pd.notna(v) and v > 0:
                return v, 'SPARC'

            v = row.get('v_rot_kourkchi', np.nan)
            if pd.notna(v) and v > 0:
                return v, 'Kourkchi'

            v = row.get('v_rot_dupuy', np.nan)
            if pd.notna(v) and v > 0:
                return v, 'Dupuy'

            v = row.get('v_rot_alfalfa', np.nan)
            if pd.notna(v) and v > 0:
                vc = _corrected_by_incl(row, 'v_rot_alfalfa')
                if pd.notna(vc) and vc > 0:
                    return vc, 'ALFALFA'

            v = row.get('v_rot_edd_raw', np.nan)
            if pd.notna(v) and v > 0:
                vc = _corrected_by_incl(row, 'v_rot_edd_raw')
                if pd.notna(vc) and vc > 0:
                    return vc, 'EDD'

            v = row.get('v_rot_2mtf', np.nan)
            if pd.notna(v) and v > 0:
                # 2MTF is already inclination corrected (WHIc)
                return v, '2MTF'

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
            "description": "Cross-match Pantheon+ against VizieR deep V_rot catalogs "
                           "(SPARC, Kourkchi+2019, Dupuy+2021 CF4-HI, ALFALFA, EDD, 2MTF) "
                           "with redshift-consistency gating",
            "redshift_gate": {"dv_max_kms": DV_MAX, "z_max_no_velocity": Z_MAX_NOCZ},
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
        df_kourkchi = self.query_kourkchi()
        df_dupuy = self.query_dupuy()
        df_edd = self.query_edd()

        self.merge_deep_catalog(df_alfalfa, df_sparc, df_2mtf,
                                df_kourkchi, df_dupuy, df_edd)
        print_status("Step 00c complete", "SUCCESS")

if __name__ == "__main__":
    step = Step00cDeepVrot()
    step.run()
