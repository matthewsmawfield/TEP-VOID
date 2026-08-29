#!/usr/bin/env python3
"""
Query Vizier HyperLEDA catalogs (VII/237 + VII/238) for rotation velocities
of Pantheon+ SN Ia host galaxies.

VII/237: Galaxy catalog (PGC, coordinates, logR25 -> inclination)
VII/238: HI data (PGC, log(2Vm) = log10(2*Vmax*sin(i)), VHI)

V_rot = 10^(log_2Vm_) / (2 * sin(i))

where inclination i is derived from logR25 (axis ratio) in VII/237.
"""

import pandas as pd
import numpy as np
import requests
import io
import time
import json
import os
import signal
import sys
from astropy.io.votable import parse_single_table
from astropy.coordinates import SkyCoord
import astropy.units as u
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──
BASE = '/Users/matthewsmawfield/www/Temporal Equivalence Principle/TEP-VOID'
PANTHEON_FILE = f'{BASE}/data/raw/Pantheon+SH0ES.dat'
OUTPUT_CSV = f'{BASE}/data/processed/pantheon_host_vrot_vizier.csv'
OUTPUT_JSON = f'{BASE}/results/outputs/step_48_vizier_vrot.json'
PARTIAL_CSV = f'{BASE}/data/processed/pantheon_host_vrot_vizier_partial.csv'

# ── Config ──
RADIUS_ARCSEC = 120  # 2 arcmin
VIZIER_URL = 'http://vizier.u-strasbg.fr/viz-bin/votable'
SLEEP_BETWEEN = 0.3  # seconds between queries
SAVE_EVERY = 50      # save partial results every N SNe


def query_vizier_catalog(ra, dec, catalog, radius_arcsec=RADIUS_ARCSEC):
    """Query a Vizier catalog by position and return parsed table or None."""
    params = {
        '-source': catalog,
        '-c': f'{ra} {dec}',
        '-c.rs': str(radius_arcsec),
        '-c._u': 's',  # radius unit = arcseconds
        '-out.all': '1',
        '-out.max': '50',
    }
    try:
        r = requests.get(VIZIER_URL, params=params, timeout=30)
        if r.status_code != 200 or len(r.content) < 100:
            return None
        table = parse_single_table(io.BytesIO(r.content))
        if len(table.array) == 0:
            return None
        return table
    except Exception:
        return None


def parse_ra_dec(ra_str, dec_str):
    """Parse sexagesimal RA/DEC from Vizier into degrees."""
    try:
        coord = SkyCoord(ra=ra_str, dec=dec_str, unit=(u.hourangle, u.deg))
        return coord.ra.deg, coord.dec.deg
    except Exception:
        return None, None


def compute_inclination(logR25, q0=0.2):
    """
    Compute inclination angle (degrees) from logR25.

    logR25 = log10(a/b) where a/b is the isophotal axis ratio at 25 mag/arcsec^2.
    q = b/a = 10^(-logR25)
    cos(i) = sqrt((q^2 - q0^2) / (1 - q0^2))
    where q0 is the intrinsic axial ratio for an edge-on disk (~0.2).

    Returns inclination in degrees, or None if not computable.
    """
    if logR25 is None or np.isnan(logR25):
        return None
    q = 10.0 ** (-logR25)
    q2 = q * q
    q0sq = q0 * q0
    if q2 <= q0sq:
        # Galaxy is close to edge-on
        return 90.0
    cos_i = np.sqrt((q2 - q0sq) / (1.0 - q0sq))
    cos_i = min(cos_i, 1.0)
    i_rad = np.arccos(cos_i)
    return np.degrees(i_rad)


def safe_float(val):
    """Safely convert a masked/numpy value to float or None."""
    try:
        if val is None:
            return None
        if hasattr(val, 'mask') and val.mask:
            return None
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    except Exception:
        return None


def main():
    # ── Load Pantheon+ ──
    print("Loading Pantheon+ data...")
    df = pd.read_csv(PANTHEON_FILE, sep=r'\s+')
    unique = df.drop_duplicates(subset='CID')[['CID', 'RA', 'DEC', 'zCMB', 'HOST_LOGMASS', 'IS_CALIBRATOR']].copy()
    unique = unique.reset_index(drop=True)
    print(f"  {len(unique)} unique SNe")

    results = []
    n_matched = 0
    n_with_vrot = 0
    n_with_incl = 0
    errors = 0

    for idx, row in unique.iterrows():
        cid = row['CID']
        ra = row['RA']
        dec = row['DEC']

        # ── Query VII/238 (HI data: log_2Vm_, VHI) ──
        tab238 = query_vizier_catalog(ra, dec, 'VII/238')

        pgc = None
        galaxy_name = None
        log_2vm = None
        e_log_2vm = None
        u_log_2vm = None
        vhi = None
        e_vhi = None
        m21 = None
        match_sep = None
        logR25 = None
        mtype = None

        if tab238 is not None:
            arr = tab238.array
            # Find closest match
            best_sep = 999.0
            best_idx = 0
            for j in range(len(arr)):
                gal_ra, gal_dec = parse_ra_dec(
                    str(arr[j]['RAJ2000']), str(arr[j]['DEJ2000'])
                )
                if gal_ra is None:
                    continue
                sep = np.sqrt((ra - gal_ra) ** 2 + (dec - gal_dec) ** 2) * 3600.0
                if sep < best_sep:
                    best_sep = sep
                    best_idx = j

            if best_sep < RADIUS_ARCSEC:
                gal = arr[best_idx]
                pgc = safe_float(gal['PGC'])
                cols238 = arr.dtype.names
                galaxy_name = str(gal['AName']).strip() if 'AName' in cols238 and str(gal['AName']).strip() else None
                log_2vm = safe_float(gal['log_2Vm_']) if 'log_2Vm_' in cols238 else None
                e_log_2vm = safe_float(gal['e_log_2Vm_']) if 'e_log_2Vm_' in cols238 else None
                u_log_2vm = safe_float(gal['u_log_2Vm_']) if 'u_log_2Vm_' in cols238 else None
                vhi = safe_float(gal['VHI']) if 'VHI' in cols238 else None
                e_vhi = safe_float(gal['e_VHI']) if 'e_VHI' in cols238 else None
                m21 = safe_float(gal['m21']) if 'm21' in cols238 else None
                match_sep = best_sep
                n_matched += 1

        # ── Query VII/237 (galaxy data: logR25, MType) ──
        # Query by position, or if we have PGC, we can match by PGC
        tab237 = query_vizier_catalog(ra, dec, 'VII/237')

        if tab237 is not None:
            arr237 = tab237.array
            # If we have PGC from VII/238, try to match by PGC
            best_sep237 = 999.0
            best_idx237 = 0
            for j in range(len(arr237)):
                gal_ra, gal_dec = parse_ra_dec(
                    str(arr237[j]['RAJ2000']), str(arr237[j]['DEJ2000'])
                )
                if gal_ra is None:
                    continue
                sep = np.sqrt((ra - gal_ra) ** 2 + (dec - gal_dec) ** 2) * 3600.0
                if sep < best_sep237:
                    best_sep237 = sep
                    best_idx237 = j

            if best_sep237 < RADIUS_ARCSEC:
                gal237 = arr237[best_idx237]
                cols237 = arr237.dtype.names
                logR25 = safe_float(gal237['logR25']) if 'logR25' in cols237 else None
                mtype = str(gal237['MType']).strip() if 'MType' in cols237 else None
                # If we didn't get PGC from VII/238, get it from VII/237
                if pgc is None:
                    pgc = safe_float(gal237['PGC'])
                    if 'ANames' in cols237:
                        anames_val = str(gal237['ANames']).strip()
                        galaxy_name = anames_val.split()[0] if anames_val else None
                    else:
                        galaxy_name = None
                    match_sep = best_sep237
                    n_matched += 1

        # ── Compute derived quantities ──
        # Vmax_apparent = 10^(log_2Vm_) / 2  (uncorrected for inclination)
        vmax_apparent = None
        if log_2vm is not None:
            vmax_apparent = (10.0 ** log_2vm) / 2.0

        # Inclination from logR25
        inclination = compute_inclination(logR25)

        # V_rot = Vmax_apparent / sin(i)
        v_rot = None
        if vmax_apparent is not None and inclination is not None and inclination > 0:
            sin_i = np.sin(np.radians(inclination))
            if sin_i > 0.01:  # Avoid division by near-zero
                v_rot = vmax_apparent / sin_i
                n_with_vrot += 1

        if inclination is not None:
            n_with_incl += 1

        # Also compute V_rot using log_2Vm_ directly:
        # log_2Vm_ = log10(2 * Vmax * sin(i))
        # V_rot = 10^(log_2Vm_) / (2 * sin(i))
        # This is the same as above but computed in log space for precision
        v_rot_log = None
        if log_2vm is not None and inclination is not None and inclination > 0:
            sin_i = np.sin(np.radians(inclination))
            if sin_i > 0.01:
                v_rot_log = (10.0 ** log_2vm) / (2.0 * sin_i)

        result = {
            'CID': cid,
            'sn_ra': ra,
            'sn_dec': dec,
            'zCMB': row['zCMB'],
            'HOST_LOGMASS': row['HOST_LOGMASS'],
            'IS_CALIBRATOR': row['IS_CALIBRATOR'],
            'pgc': int(pgc) if pgc is not None else None,
            'galaxy_name': galaxy_name,
            'match_sep_arcsec': match_sep,
            'log_2Vm': log_2vm,
            'e_log_2Vm': e_log_2vm,
            'u_log_2Vm': u_log_2vm,
            'vmax_apparent': vmax_apparent,
            'VHI': vhi,
            'e_VHI': e_vhi,
            'm21': m21,
            'logR25': logR25,
            'inclination_deg': inclination,
            'v_rot': v_rot,
            'v_rot_log': v_rot_log,
            'mtype': mtype,
        }
        results.append(result)

        # ── Progress ──
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx+1}/{len(unique)} SNe | matched={n_matched} | with_vrot={n_with_vrot}")
            # Save partial
            pd.DataFrame(results).to_csv(PARTIAL_CSV, index=False)

        time.sleep(SLEEP_BETWEEN)

    # ── Save final results ──
    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_CSV, index=False)

    # Remove partial file
    if os.path.exists(PARTIAL_CSV):
        os.remove(PARTIAL_CSV)

    # ── Summary statistics ──
    total = len(unique)
    matched = out_df['pgc'].notna().sum()
    with_log2vm = out_df['log_2Vm'].notna().sum()
    with_vrot = out_df['v_rot'].notna().sum()
    with_incl = out_df['inclination_deg'].notna().sum()
    with_both = out_df['v_rot'].notna().sum()

    # V_rot distribution
    vrot_vals = out_df['v_rot'].dropna()
    if len(vrot_vals) > 0:
        vrot_stats = {
            'n': int(len(vrot_vals)),
            'min': float(vrot_vals.min()),
            'max': float(vrot_vals.max()),
            'mean': float(vrot_vals.mean()),
            'median': float(vrot_vals.median()),
            'std': float(vrot_vals.std()),
            'p25': float(vrot_vals.quantile(0.25)),
            'p75': float(vrot_vals.quantile(0.75)),
        }
    else:
        vrot_stats = {'n': 0}

    # Vmax apparent distribution
    vmax_vals = out_df['vmax_apparent'].dropna()
    if len(vmax_vals) > 0:
        vmax_stats = {
            'n': int(len(vmax_vals)),
            'min': float(vmax_vals.min()),
            'max': float(vmax_vals.max()),
            'mean': float(vmax_vals.mean()),
            'median': float(vmax_vals.median()),
            'std': float(vmax_vals.std()),
        }
    else:
        vmax_stats = {'n': 0}

    # Inclination distribution
    incl_vals = out_df['inclination_deg'].dropna()
    if len(incl_vals) > 0:
        incl_stats = {
            'n': int(len(incl_vals)),
            'min': float(incl_vals.min()),
            'max': float(incl_vals.max()),
            'mean': float(incl_vals.mean()),
            'median': float(incl_vals.median()),
        }
    else:
        incl_stats = {'n': 0}

    # Calibrator breakdown
    calibrators = out_df[out_df['IS_CALIBRATOR'] == 1]
    cal_with_vrot = calibrators['v_rot'].notna().sum()

    summary = {
        'step': 48,
        'description': 'Vizier HyperLEDA (VII/237 + VII/238) rotation velocity query for Pantheon+ hosts',
        'catalogs_queried': ['VII/237 (HyperLEDA galaxy catalog)', 'VII/238 (HyperLEDA HI data)'],
        'query_radius_arcmin': 2.0,
        'total_unique_sne': int(total),
        'matched_to_pgc': int(matched),
        'with_log_2Vm': int(with_log2vm),
        'with_vmax_apparent': int(with_log2vm),
        'with_inclination': int(with_incl),
        'with_v_rot': int(with_vrot),
        'calibrators_total': int(len(calibrators)),
        'calibrators_with_vrot': int(cal_with_vrot),
        'match_rate': float(matched / total) if total > 0 else 0,
        'vrot_rate': float(with_vrot / total) if total > 0 else 0,
        'vrot_distribution': vrot_stats,
        'vmax_apparent_distribution': vmax_stats,
        'inclination_distribution': incl_stats,
        'velocity_correction': {
            'formula': 'V_rot = 10^(log_2Vm) / (2 * sin(i))',
            'log_2Vm_definition': 'log10(2 * Vmax * sin(i)) — apparent max rotation velocity uncorrected for inclination',
            'inclination_from': 'logR25 (axis ratio) from VII/237, using q0=0.2 intrinsic flattening',
            'note': 'V_rot is the inclination-corrected (face-on) maximum rotation velocity',
        },
        'output_csv': OUTPUT_CSV,
        'columns': list(out_df.columns),
    }

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total unique SNe:      {total}")
    print(f"Matched to PGC:        {matched} ({matched/total*100:.1f}%)")
    print(f"With log(2Vm):         {with_log2vm}")
    print(f"With inclination:      {with_incl}")
    print(f"With V_rot (corrected):{with_vrot}")
    print(f"Calibrators w/ V_rot:  {cal_with_vrot}/{len(calibrators)}")
    if len(vrot_vals) > 0:
        print(f"\nV_rot distribution:")
        print(f"  Mean:   {vrot_vals.mean():.1f} km/s")
        print(f"  Median: {vrot_vals.median():.1f} km/s")
        print(f"  Range:  {vrot_vals.min():.1f} - {vrot_vals.max():.1f} km/s")
        print(f"  Std:    {vrot_vals.std():.1f} km/s")
    print(f"\nSaved: {OUTPUT_CSV}")
    print(f"Saved: {OUTPUT_JSON}")


if __name__ == '__main__':
    main()
