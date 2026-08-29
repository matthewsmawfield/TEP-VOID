#!/usr/bin/env python3
"""
Query HyperLEDA for rotation velocities (V_rot) of Type Ia supernova host
galaxies in the Pantheon+ sample.

Approach:
  1. Load Pantheon+ data, extract unique SNe with RA/DEC and redshift.
  2. Use existing catalog data (velocity_dispersions_literature.csv,
     hosts_properties.csv) for the 42 R22 calibrator hosts.
  3. For remaining nearby SNe (z < 0.06), query HyperLEDA via SQL box
     search around each SN position to find the host galaxy.
  4. Extract V_rot (inclination-corrected maximum rotation velocity).
  5. Save results to CSV and summary JSON.

HyperLEDA query interface:
  http://atlas.obs-hp.fr/hyperleda/fG.cgi?n=meandata&c=o
      &d=pgc,objname,al2000,de2000,vrot,e_vrot
      &sql=<WHERE clause using al2000 in hours, de2000 in degrees>
      &ob=pgc

Reference: Makarov et al. 2014, A&A 570, A13.
"""

import os
import re
import json
import time
import math
import warnings
import requests
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE, "data", "raw")
DATA_PROC = os.path.join(BASE, "data", "processed")
DATA_EXT = os.path.join(BASE, "data", "raw", "external")
RESULTS = os.path.join(BASE, "results", "outputs")

PANTHEON_FILE = os.path.join(DATA_RAW, "Pantheon+SH0ES.dat")
EXISTING_VROT = os.path.join(DATA_EXT, "velocity_dispersions_literature.csv")
HOSTS_PROPS = os.path.join(DATA_EXT, "hosts_properties.csv")
CF4_VROT = os.path.join(DATA_EXT, "cf4_matched_galaxies_vrot.csv")

OUTPUT_CSV = os.path.join(DATA_PROC, "pantheon_host_vrot.csv")
OUTPUT_JSON = os.path.join(RESULTS, "step_47_hyperleda_vrot.json")

HYPERLEDA_URL = "http://atlas.obs-hp.fr/hyperleda/fG.cgi"
QUERY_FIELDS = "pgc,objname,al2000,de2000,vrot,e_vrot"
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
QUERY_DELAY = 0.5  # seconds between queries (rate limiting)
Z_MAX_FOR_QUERY = 0.06  # only query HyperLEDA for nearby SNe


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def angular_separation_deg(ra1_deg, dec1_deg, ra2_deg, dec2_deg):
    """Great-circle angular separation in degrees (Vincenty formula)."""
    ra1, dec1 = math.radians(ra1_deg), math.radians(dec1_deg)
    ra2, dec2 = math.radians(ra2_deg), math.radians(dec2_deg)
    dra = ra2 - ra1
    y = math.sqrt(
        (math.cos(dec2) * math.sin(dra)) ** 2
        + (math.cos(dec1) * math.sin(dec2)
           - math.sin(dec1) * math.cos(dec2) * math.cos(dra)) ** 2
    )
    x = math.sin(dec1) * math.sin(dec2) + math.cos(dec1) * math.cos(dec2) * math.cos(dra)
    return math.degrees(math.atan2(y, x))


def get_search_radius_arcmin(z):
    """Adaptive search radius based on redshift.

    Nearby galaxies can be very large on the sky (M101 is ~30 arcmin),
    so we need a larger search radius for very nearby SNe.
    """
    if z < 0.005:
        return 15.0
    elif z < 0.01:
        return 10.0
    elif z < 0.02:
        return 5.0
    elif z < 0.04:
        return 3.0
    else:
        return 2.0


def parse_hyperleda_response(html_text):
    """Parse the HTML response from HyperLEDA fG.cgi SQL query.

    Returns a list of dicts with keys: pgc, objname, al2000, de2000, vrot, e_vrot.
    """
    records = []

    if "no record" in html_text.lower():
        return records

    # Find all <pre> blocks
    pre_blocks = re.findall(r"<pre[^>]*>(.*?)</pre>", html_text, re.DOTALL)
    if len(pre_blocks) < 2:
        return records

    # The second <pre> block (frm1) contains the data rows
    data_block = pre_blocks[1]

    for line in data_block.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Skip header lines (contain <a> tags)
        if "<a" in line:
            continue

        # Remove HTML tags
        line_clean = re.sub(r"<[^>]+>", "", line)
        parts = line_clean.split()
        if len(parts) < 4:
            continue

        try:
            # Our query: d=pgc,objname,al2000,de2000,vrot,e_vrot
            # Format: pgc objname al2000 de2000 [vrot] [e_vrot]
            pgc = int(parts[0])
            objname = parts[1]
            al2000 = float(parts[2])
            de2000 = float(parts[3])
            vrot = None
            e_vrot = None
            if len(parts) > 4 and parts[4]:
                try:
                    vrot = float(parts[4])
                    if vrot <= 0:
                        vrot = None
                except ValueError:
                    pass
            if len(parts) > 5 and parts[5]:
                try:
                    e_vrot = float(parts[5])
                    if e_vrot <= 0:
                        e_vrot = None
                except ValueError:
                    pass

            records.append({
                "pgc": pgc,
                "objname": objname,
                "al2000": al2000,
                "de2000": de2000,
                "vrot": vrot,
                "e_vrot": e_vrot,
            })
        except (ValueError, IndexError):
            continue

    return records


def query_hyperleda_box(ra_deg, dec_deg, radius_arcmin):
    """Query HyperLEDA for galaxies in a box around (ra_deg, dec_deg).

    RA is stored in hours (0-24) in HyperLEDA, DEC in degrees.
    The RA box width is corrected for cos(DEC).
    """
    ra_hours = ra_deg / 15.0
    cos_dec = max(abs(math.cos(math.radians(dec_deg))), 0.01)
    d_ra_hours = radius_arcmin / (15.0 * 60.0 * cos_dec)
    d_dec_deg = radius_arcmin / 60.0

    ra_min = ra_hours - d_ra_hours
    ra_max = ra_hours + d_ra_hours
    dec_min = dec_deg - d_dec_deg
    dec_max = dec_deg + d_dec_deg

    # Handle RA wrapping
    conditions = []
    if ra_min < 0:
        conditions.append(f"(al2000>{ra_min + 24.0} or al2000<{ra_max})")
    elif ra_max > 24.0:
        conditions.append(f"(al2000>{ra_min} or al2000<{ra_max - 24.0})")
    else:
        conditions.append(f"al2000>{ra_min}")
        conditions.append(f"al2000<{ra_max}")

    conditions.append(f"de2000>{dec_min}")
    conditions.append(f"de2000<{dec_max}")

    sql = " and ".join(conditions)

    params = {
        "n": "meandata",
        "c": "o",
        "d": QUERY_FIELDS,
        "sql": sql,
        "ob": "pgc",
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(HYPERLEDA_URL, params=params, timeout=30)
            if resp.status_code == 200:
                return parse_hyperleda_response(resp.text)
            elif resp.status_code == 503:
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue
            else:
                time.sleep(RETRY_DELAY)
        except requests.exceptions.RequestException:
            time.sleep(RETRY_DELAY * (attempt + 1))

    return []


def is_sn_object(objname):
    """Check if an object name looks like a supernova rather than a galaxy."""
    name_upper = objname.upper()
    # SN naming patterns: SN20XX, SN20Xxxx, ASASSN-, PS1-, etc.
    sn_patterns = [
        "SN", "ASASSN", "PS1", "PTF", "CSS", "LSQ", "iPTF",
        "SN2011", "SN2012", "SN2013", "SN2014", "SN2015", "SN2016",
        "SN2017", "SN2018", "SN2019", "SN2020", "SN2021",
    ]
    # Check if it starts with SN followed by digits
    if re.match(r"^SN\d", name_upper):
        return True
    if name_upper.startswith("ASASSN"):
        return True
    if name_upper.startswith("PS1"):
        return True
    if name_upper.startswith("PTF"):
        return True
    if name_upper.startswith("iPTF"):
        return True
    if name_upper.startswith("CSS"):
        return True
    if name_upper.startswith("LSQ"):
        return True
    # Check for SN year patterns in the name
    if re.search(r"SN\d{4}", name_upper):
        return True
    return False


def find_host_galaxy(galaxies, sn_ra, sn_dec):
    """Find the best host galaxy from a list of HyperLEDA galaxies.

    Strategy:
      1. Filter out SN objects and non-galaxy entries.
      2. Among remaining galaxies, prefer those with V_rot measurements.
      3. Among galaxies with V_rot, return the nearest one.
      4. If no galaxy has V_rot, return the nearest galaxy overall.
    """
    # Filter out SN objects
    galaxy_candidates = [g for g in galaxies if not is_sn_object(g["objname"])]

    if not galaxy_candidates:
        return None, None

    # Calculate separations
    for g in galaxy_candidates:
        gal_ra_deg = g["al2000"] * 15.0
        gal_dec_deg = g["de2000"]
        g["sep_arcsec"] = angular_separation_deg(
            sn_ra, sn_dec, gal_ra_deg, gal_dec_deg
        ) * 3600.0

    # Sort by separation
    galaxy_candidates.sort(key=lambda g: g["sep_arcsec"])

    # Find nearest with V_rot
    for g in galaxy_candidates:
        if g["vrot"] is not None and g["vrot"] > 0:
            return g, g["sep_arcsec"]

    # No V_rot - return nearest galaxy
    return galaxy_candidates[0], galaxy_candidates[0]["sep_arcsec"]


def load_existing_catalog():
    """Load existing V_rot data from the literature catalog files."""
    existing = {}

    # Main catalog: velocity_dispersions_literature.csv
    if os.path.exists(EXISTING_VROT):
        df = pd.read_csv(EXISTING_VROT, comment="#")
        for _, row in df.iterrows():
            name = str(row.get("galaxy", "")).strip()
            pgc = row.get("pgc")
            vrot = row.get("vrot_kms")
            e_vrot = row.get("vrot_error_kms")
            if pd.notna(vrot) and vrot > 0:
                existing[name.lower()] = {
                    "galaxy_name": name,
                    "pgc": int(pgc) if pd.notna(pgc) else None,
                    "vrot": float(vrot),
                    "e_vrot": float(e_vrot) if pd.notna(e_vrot) else None,
                    "source": "EXISTING_CATALOG",
                }

    # CF4 catalog: cf4_matched_galaxies_vrot.csv
    if os.path.exists(CF4_VROT):
        df = pd.read_csv(CF4_VROT, comment="#")
        for _, row in df.iterrows():
            name = str(row.get("galaxy_name", "")).strip()
            pgc = row.get("pgc")
            vrot = row.get("vrot_kms")
            e_vrot = row.get("vrot_error_kms")
            if pd.notna(vrot) and vrot > 0:
                key = name.lower()
                if key not in existing:
                    existing[key] = {
                        "galaxy_name": name,
                        "pgc": int(pgc) if pd.notna(pgc) else None,
                        "vrot": float(vrot),
                        "e_vrot": float(e_vrot) if pd.notna(e_vrot) else None,
                        "source": "CF4_CATALOG",
                    }

    return existing


def load_host_properties():
    """Load host galaxy properties (RA/DEC, Pantheon+ ID mapping)."""
    if not os.path.exists(HOSTS_PROPS):
        return {}

    df = pd.read_csv(HOSTS_PROPS)
    mapping = {}
    for _, row in df.iterrows():
        pantheon_id = row.get("pantheon_id")
        if pd.notna(pantheon_id) and str(pantheon_id).strip():
            name = str(row.get("normalized_name", "")).strip()
            ra = row.get("ra")
            dec = row.get("dec")
            mapping[str(pantheon_id).strip()] = {
                "galaxy_name": name,
                "ra": float(ra) if pd.notna(ra) else None,
                "dec": float(dec) if pd.notna(dec) else None,
                "separation_arcsec": float(row.get("separation_arcsec", 0))
                    if pd.notna(row.get("separation_arcsec")) else None,
            }
    return mapping


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    print("=" * 70)
    print("HyperLEDA V_rot Query for Pantheon+ SN Host Galaxies")
    print("=" * 70)

    # 1. Load Pantheon+ data
    print("\n[1] Loading Pantheon+ data...")
    df = pd.read_csv(PANTHEON_FILE, sep=r"\s+")
    unique_sne = df.drop_duplicates(subset="CID", keep="first").copy()
    print(f"  Total rows: {len(df)}, Unique SNe: {len(unique_sne)}")

    # 2. Load existing catalogs
    print("\n[2] Loading existing V_rot catalogs...")
    existing_vrot = load_existing_catalog()
    host_props = load_host_properties()
    print(f"  Existing V_rot entries: {len(existing_vrot)}")
    print(f"  Host property mappings: {len(host_props)}")

    # 3. Prepare SN list with coordinates
    print("\n[3] Preparing SN coordinate list...")
    sne = []
    for _, row in unique_sne.iterrows():
        cid = str(row["CID"]).strip()
        ra = float(row["RA"])
        dec = float(row["DEC"])
        z = float(row["zCMB"])
        host_ra = float(row["HOST_RA"]) if row["HOST_RA"] != -999 else None
        host_dec = float(row["HOST_DEC"]) if row["HOST_DEC"] != -999 else None
        host_logmass = float(row["HOST_LOGMASS"]) if row["HOST_LOGMASS"] != -9 else None

        # Use host coordinates if available, otherwise SN coordinates
        match_ra = host_ra if host_ra is not None else ra
        match_dec = host_dec if host_dec is not None else dec

        sne.append({
            "cid": cid,
            "ra": ra,
            "dec": dec,
            "z": z,
            "host_ra": host_ra,
            "host_dec": host_dec,
            "match_ra": match_ra,
            "match_dec": match_dec,
            "host_logmass": host_logmass,
        })

    # 4. Query HyperLEDA for each SN
    print(f"\n[4] Querying HyperLEDA for SNe with z < {Z_MAX_FOR_QUERY}...")
    print(f"  Adaptive search radius: 15' (z<0.005), 10' (z<0.01), "
          f"5' (z<0.02), 3' (z<0.04), 2' (z<0.06)")
    print(f"  Rate limit: {QUERY_DELAY}s between queries")

    results = []
    queried_positions = {}  # cache: (round(ra,3), round(dec,3)) -> galaxies
    n_queried = 0
    n_matched = 0
    n_with_vrot = 0
    n_from_existing = 0

    for i, sn in enumerate(sne):
        cid = sn["cid"]
        z = sn["z"]

        # Check if this SN is already in the existing catalog via host_props
        if cid in host_props:
            hp = host_props[cid]
            gal_name = hp["galaxy_name"]
            vrot_data = existing_vrot.get(gal_name.lower())
            if vrot_data:
                results.append({
                    "CID": cid,
                    "galaxy_name": vrot_data["galaxy_name"],
                    "RA": hp["ra"] if hp["ra"] else sn["ra"],
                    "DEC": hp["dec"] if hp["dec"] else sn["dec"],
                    "V_rot": vrot_data["vrot"],
                    "V_rot_err": vrot_data["e_vrot"],
                    "logV": math.log10(vrot_data["vrot"]),
                    "logV_err": (vrot_data["e_vrot"] / (vrot_data["vrot"] * math.log(10)))
                        if vrot_data["e_vrot"] else None,
                    "match_distance_arcsec": hp["separation_arcsec"],
                    "pgc": vrot_data["pgc"],
                    "zCMB": z,
                    "source": vrot_data["source"],
                })
                n_from_existing += 1
                n_matched += 1
                n_with_vrot += 1
                continue

        # Skip distant SNe for HyperLEDA queries
        if z > Z_MAX_FOR_QUERY:
            results.append({
                "CID": cid,
                "galaxy_name": "",
                "RA": sn["ra"],
                "DEC": sn["dec"],
                "V_rot": None,
                "V_rot_err": None,
                "logV": None,
                "logV_err": None,
                "match_distance_arcsec": None,
                "pgc": None,
                "zCMB": z,
                "source": "TOO_DISTANT",
            })
            continue

        # Query HyperLEDA
        match_ra = sn["match_ra"]
        match_dec = sn["match_dec"]
        radius = get_search_radius_arcmin(z)

        # Cache by rounded position to avoid duplicate queries
        pos_key = (round(match_ra, 3), round(match_dec, 3))
        if pos_key in queried_positions:
            galaxies = queried_positions[pos_key]
        else:
            galaxies = query_hyperleda_box(match_ra, match_dec, radius)
            queried_positions[pos_key] = galaxies
            n_queried += 1
            time.sleep(QUERY_DELAY)

        if not galaxies:
            results.append({
                "CID": cid,
                "galaxy_name": "",
                "RA": sn["ra"],
                "DEC": sn["dec"],
                "V_rot": None,
                "V_rot_err": None,
                "logV": None,
                "logV_err": None,
                "match_distance_arcsec": None,
                "pgc": None,
                "zCMB": z,
                "source": "NO_MATCH",
            })
            continue

        # Find the host galaxy
        best_galaxy, best_sep_arcsec = find_host_galaxy(
            galaxies, match_ra, match_dec
        )

        if best_galaxy:
            vrot = best_galaxy["vrot"]
            e_vrot = best_galaxy["e_vrot"]
            results.append({
                "CID": cid,
                "galaxy_name": best_galaxy["objname"],
                "RA": best_galaxy["al2000"] * 15.0,
                "DEC": best_galaxy["de2000"],
                "V_rot": vrot,
                "V_rot_err": e_vrot,
                "logV": math.log10(vrot) if vrot else None,
                "logV_err": (e_vrot / (vrot * math.log(10)))
                    if vrot and e_vrot else None,
                "match_distance_arcsec": best_sep_arcsec,
                "pgc": best_galaxy["pgc"],
                "zCMB": z,
                "source": "HYPERLEDA_QUERY",
            })
            n_matched += 1
            if vrot is not None:
                n_with_vrot += 1
        else:
            results.append({
                "CID": cid,
                "galaxy_name": "",
                "RA": sn["ra"],
                "DEC": sn["dec"],
                "V_rot": None,
                "V_rot_err": None,
                "logV": None,
                "logV_err": None,
                "match_distance_arcsec": None,
                "pgc": None,
                "zCMB": z,
                "source": "NO_MATCH",
            })

        # Progress report
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(sne)} SNe | "
                  f"queries={n_queried} matched={n_matched} "
                  f"vrot={n_with_vrot} existing={n_from_existing}")

    print(f"\n  Final: {len(sne)} SNe | queries={n_queried} "
          f"matched={n_matched} vrot={n_with_vrot} "
          f"existing={n_from_existing}")

    # 5. Save CSV
    print(f"\n[5] Saving results to {OUTPUT_CSV}...")
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"  Saved {len(results_df)} rows")

    # 6. Generate summary JSON
    print(f"\n[6] Generating summary JSON...")

    vrot_values = [r["V_rot"] for r in results if r["V_rot"] is not None]
    vrot_arr = np.array(vrot_values) if vrot_values else np.array([])

    if len(vrot_arr) > 0:
        bins = [0, 50, 100, 150, 200, 250, 300, 350, 400]
        hist_counts = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            hist_counts.append(int(((vrot_arr >= lo) & (vrot_arr < hi)).sum()))
        # Add last bin (>=400)
        hist_counts.append(int((vrot_arr >= bins[-1]).sum()))
        bins.append(999)

        vrot_dist = {
            "count": int(len(vrot_arr)),
            "min_km_s": float(vrot_arr.min()),
            "max_km_s": float(vrot_arr.max()),
            "mean_km_s": float(vrot_arr.mean()),
            "median_km_s": float(np.median(vrot_arr)),
            "std_km_s": float(vrot_arr.std()),
            "percentile_25": float(np.percentile(vrot_arr, 25)),
            "percentile_75": float(np.percentile(vrot_arr, 75)),
            "histogram_bins_km_s": bins,
            "histogram_counts": hist_counts,
        }
    else:
        vrot_dist = {"count": 0}

    # Source breakdown
    source_counts = {}
    for r in results:
        s = r["source"]
        source_counts[s] = source_counts.get(s, 0) + 1

    # Redshift bins for matched SNe with V_rot
    matched = [r for r in results if r["V_rot"] is not None]
    z_bins = {"z<0.01": 0, "0.01<=z<0.02": 0, "0.02<=z<0.03": 0,
              "0.03<=z<0.05": 0, "z>=0.05": 0}
    for r in matched:
        z = r["zCMB"]
        if z < 0.01:
            z_bins["z<0.01"] += 1
        elif z < 0.02:
            z_bins["0.01<=z<0.02"] += 1
        elif z < 0.03:
            z_bins["0.02<=z<0.03"] += 1
        elif z < 0.05:
            z_bins["0.03<=z<0.05"] += 1
        else:
            z_bins["z>=0.05"] += 1

    # Unique galaxies with V_rot
    unique_galaxies = set()
    for r in matched:
        unique_galaxies.add(r["galaxy_name"])

    summary = {
        "step": 47,
        "description": "HyperLEDA rotation velocity query for Pantheon+ SN host galaxies",
        "pantheon_file": "data/raw/Pantheon+SH0ES.dat",
        "hyperleda_url": "http://atlas.obs-hp.fr/hyperleda/fG.cgi",
        "hyperleda_reference": "Makarov et al. 2014, A&A 570, A13",
        "query_date": time.strftime("%Y-%m-%d"),
        "query_method": "SQL box search (al2000 in hours, de2000 in degrees, "
                        "RA corrected for cos(DEC))",
        "search_radius": "adaptive: 15' (z<0.005), 10' (z<0.01), "
                         "5' (z<0.02), 3' (z<0.04), 2' (z<0.06)",
        "z_max_for_query": Z_MAX_FOR_QUERY,
        "n_sne_queried": len(sne),
        "n_hyperleda_queries": n_queried,
        "n_hosts_matched": n_matched,
        "n_with_vrot": n_with_vrot,
        "n_unique_galaxies_with_vrot": len(unique_galaxies),
        "n_from_existing_catalog": n_from_existing,
        "n_no_match": source_counts.get("NO_MATCH", 0),
        "n_too_distant": source_counts.get("TOO_DISTANT", 0),
        "source_breakdown": source_counts,
        "vrot_distribution": vrot_dist,
        "redshift_bins_matched": z_bins,
        "output_csv": "data/processed/pantheon_host_vrot.csv",
        "columns": ["CID", "galaxy_name", "RA", "DEC", "V_rot", "V_rot_err",
                     "logV", "logV_err", "match_distance_arcsec", "pgc",
                     "zCMB", "source"],
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved to {OUTPUT_JSON}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  SNe queried:           {len(sne)}")
    print(f"  HyperLEDA queries:     {n_queried}")
    print(f"  Hosts matched:         {n_matched}")
    print(f"  With V_rot:            {n_with_vrot}")
    print(f"  Unique galaxies:       {len(unique_galaxies)}")
    print(f"  From existing catalog: {n_from_existing}")
    print(f"  No match:              {source_counts.get('NO_MATCH', 0)}")
    print(f"  Too distant:           {source_counts.get('TOO_DISTANT', 0)}")
    if vrot_values:
        print(f"\n  V_rot distribution (km/s):")
        print(f"    Count:  {len(vrot_values)}")
        print(f"    Min:    {min(vrot_values):.1f}")
        print(f"    Max:    {max(vrot_values):.1f}")
        print(f"    Mean:   {np.mean(vrot_values):.1f}")
        print(f"    Median: {np.median(vrot_values):.1f}")
        print(f"    Std:    {np.std(vrot_values):.1f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
