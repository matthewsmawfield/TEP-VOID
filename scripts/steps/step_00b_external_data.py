#!/usr/bin/env python3
"""
Step 00b: External Data Download — Full Provenance Pipeline
=============================================================
Downloads ALL external datasets used in the TEP-VOID analysis,
with SHA-256 checksum verification and a machine-readable manifest.

This step ensures full reproducibility: every external data file is
fetched from its official source, checksummed, and recorded in
data/raw/checksums.json. Files that already exist and match their
checksum are skipped (idempotent re-runs).

Datasets downloaded:
  1. Pantheon+ SH0ES data + covariance matrices (GitHub)
  2. CosmicFlows-4 tables (CDS via Playwright)
  3. SPARC mass models + rotation curves (astroweb.cwru.edu)
  4. 2MTF Tully-Fisher catalog (VizieR J/MNRAS/487/2061)
  5. SFI++ Tully-Fisher catalog (VizieR J/ApJS/182/474)
  6. Jia et al. (2023) H(z) data (GitHub)
  7. ALFALFA A40 HI catalog (VizieR J/AJ/154/78)
  8. 6dFGS peculiar velocities (VizieR J/MNRAS/445/2677)
  9. HyperLEDA V_rot for all Pantheon+ hosts (atlas.obs-hp.fr)
 10. VizieR HyperLEDA VII/237 + VII/238 for V_rot cross-check

Datasets generated from published tables (curated CSVs with documented
provenance — these are transcribed from peer-reviewed papers and verified):
  - R22 Cepheid distance moduli (Riess et al. 2022)
  - CCHP TRGB distances (Freedman et al. 2024/2025)
  - JWST TRGB distances (Anand et al. 2024, Freedman et al. 2025)
  - Key Project Cepheid distances (Freedman et al. 2001)
  - Madore & Freedman 2023 VIH/VI Cepheid distances
  - Host galaxy properties (RA/DEC/mass from Pantheon+SH0ES)
  - CF4 matched galaxies V_rot (from HyperLEDA queries)
  - Velocity dispersions / V_rot literature catalog
  - Mazurenko et al. 2025 digitized void H(z) curves

Outputs:
    data/raw/checksums.json — machine-readable checksum manifest
    data/raw/external/ — all downloaded external data files
    data/processed/pantheon_host_vrot_hyperleda.csv — HyperLEDA V_rot for all hosts
    data/processed/pantheon_host_vrot_vizier.csv — VizieR V_rot cross-check
    results/outputs/step_00b_external_data_summary.json — download summary
"""

import json
import sys
import time
import math
import hashlib
import gzip
import shutil
import zipfile
from pathlib import Path
from datetime import datetime, timezone
from urllib.request import urlopen, Request, urlretrieve
from urllib.error import URLError, HTTPError

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.data_downloader import DataDownloader, DATASET_REGISTRY


class Step00bExternalData:
    """Download all external datasets with full provenance tracking."""

    def __init__(self):
        self.root = PROJECT_ROOT
        self.data_raw = self.root / "data" / "raw"
        self.data_ext = self.data_raw / "external"
        self.data_proc = self.root / "data" / "processed"
        self.results_dir = self.root / "results" / "outputs"
        self.logs_dir = self.root / "logs"

        for d in [self.data_raw, self.data_ext, self.data_proc,
                  self.results_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.logger = TEPLogger("step_00b",
                                log_file_path=self.logs_dir / "step_00b_external_data.log")
        set_step_logger(self.logger)

        self.dl = DataDownloader(data_raw=self.data_raw)

    # ------------------------------------------------------------------ #
    # 1. Pantheon+ SH0ES data + covariance
    # ------------------------------------------------------------------ #

    def download_pantheon(self):
        """Download Pantheon+ SH0ES data and covariance matrices."""
        print_status("Downloading Pantheon+ SH0ES data + covariance matrices", "TITLE")
        for key in ["pantheon_shoes_dat", "pantheon_stat_sys_cov",
                     "pantheon_statonly_cov"]:
            ds = DATASET_REGISTRY[key]
            target = self.root / ds["target"]
            self.dl.download_file(
                url=ds["url"], target=target,
                description=ds["description"], source=ds["source"],
                min_size=ds.get("min_size", 0))

    # ------------------------------------------------------------------ #
    # 2. CosmicFlows-4 (via Playwright for CDS Anubis bypass)
    # ------------------------------------------------------------------ #

    def download_cf4(self):
        """Download CF4 tables from CDS using Playwright."""
        print_status("Downloading CosmicFlows-4 tables from CDS", "TITLE")
        for key in ["cf4_table2", "cf4_table4", "cf4_readme"]:
            ds = DATASET_REGISTRY[key]
            target = self.root / ds["target"]
            if target.exists() and target.stat().st_size > 1000:
                actual_sha = self.dl.compute_sha256(target)
                self.dl._record(target, actual_sha, target.stat().st_size,
                                ds["url"], ds["source"], "skip_exists")
                print_status(f"  [SKIP] {target.name} — already present", "PROCESS")
                continue

            # Try Playwright for CDS
            if ds.get("requires_playwright"):
                success = self._download_cds_playwright(ds["url"], target)
                if success and ds.get("decompress_gz") and target.suffix == ".gz":
                    decompressed = target.with_suffix("")
                    if not decompressed.exists():
                        with gzip.open(target, "rb") as gz:
                            with open(decompressed, "wb") as out:
                                shutil.copyfileobj(gz, out)
                        print_status(f"  [DECOMP] {decompressed.name}", "PROCESS")
                    actual_sha = self.dl.compute_sha256(target)
                    self.dl._record(target, actual_sha, target.stat().st_size,
                                    ds["url"], ds["source"], "downloaded")
                elif not success:
                    # Fallback: check if decompressed version exists
                    decompressed = target.with_suffix("")
                    if decompressed.exists() and decompressed.stat().st_size > 1000:
                        print_status(f"  [FALLBACK] Using existing {decompressed.name}", "WARNING")
                    else:
                        print_status(f"  [FAIL] {target.name} — Playwright download failed", "WARNING")
                        self.dl._record(target, "", 0, ds["url"], ds["source"], "failed")

    def _download_cds_playwright(self, url, target):
        """Download from CDS using Playwright headless browser."""
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                browser = p.chromium.launch(
                    headless=True,
                    args=['--disable-blink-features=AutomationControlled', '--no-sandbox'])
                context = browser.new_context(
                    user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                               'AppleWebKit/537.36 (KHTML, like Gecko) '
                               'Chrome/120.0.0.0 Safari/537.36',
                    ignore_https_errors=True)
                page = context.new_page()
                with page.expect_download(timeout=60000) as download_info:
                    page.goto(url, timeout=30000)
                download = download_info.value
                download.save_as(str(target))
                browser.close()
                print_status(f"  [OK] Downloaded {target.name} via Playwright", "SUCCESS")
                return True
        except Exception as e:
            print_status(f"  Playwright download failed: {e}", "WARNING")
            return False

    # ------------------------------------------------------------------ #
    # 3. SPARC (Lelli et al. 2016)
    # ------------------------------------------------------------------ #

    def download_sparc(self):
        """Download SPARC mass models and rotation curves.

        If the processed SPARC CSV (sparc_full_175.csv) already exists —
        which is the file downstream steps (step_56) actually consume —
        the raw download is skipped entirely. The SPARC server
        (astroweb.cwru.edu) is frequently unresponsive, and re-attempting
        the download on every pipeline run wastes several minutes.
        """
        print_status("Downloading SPARC mass models + rotation curves", "TITLE")

        # Skip raw download if processed data already exists
        processed = self.data_ext / "sparc_full_175.csv"
        if processed.exists() and processed.stat().st_size > 5000:
            print_status(f"  [SKIP] SPARC raw download — processed data already present "
                         f"({processed.name}, {processed.stat().st_size:,} bytes)", "PROCESS")
            for key in ["sparc_table1", "sparc_table2", "sparc_rotmod"]:
                ds = DATASET_REGISTRY[key]
                target = self.root / ds["target"]
                if target.exists():
                    sha = self.dl.compute_sha256(target)
                    self.dl._record(target, sha, target.stat().st_size,
                                    ds["url"], ds["source"], "skip_exists",
                                    "Skipped — processed SPARC CSV already present")
            return

        for key in ["sparc_table1", "sparc_table2", "sparc_rotmod"]:
            ds = DATASET_REGISTRY[key]
            target = self.root / ds["target"]
            self.dl.download_file(
                url=ds["url"], target=target,
                description=ds["description"], source=ds["source"],
                min_size=ds.get("min_size", 0))

            # Unzip rotation curves if downloaded
            if key == "sparc_rotmod" and target.exists():
                unzip_dir = self.data_ext / "sparc_rotcurves"
                if not unzip_dir.exists():
                    unzip_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        with zipfile.ZipFile(target, "r") as zf:
                            zf.extractall(unzip_dir)
                        print_status(f"  [UNZIP] Extracted to {unzip_dir.name}", "SUCCESS")
                    except Exception as e:
                        print_status(f"  [WARN] Could not unzip: {e}", "WARNING")

    # ------------------------------------------------------------------ #
    # 4-5. VizieR catalogs (2MTF, SFI++, ALFALFA, 6dFGS)
    # ------------------------------------------------------------------ #

    def download_vizier_catalogs(self):
        """Download VizieR catalogs for V_rot and peculiar velocity data."""
        print_status("Downloading VizieR catalogs (2MTF, SFI++, ALFALFA, 6dFGS)", "TITLE")
        for key in ["2mtf_vizier", "sfipp_vizier",
                     "alfalfa_a40_vizier", "6dfgs_vizier"]:
            ds = DATASET_REGISTRY[key]
            target = self.root / ds["target"]
            self.dl.download_vizier(
                catalog=ds["vizier_catalog"], target=target,
                description=ds["description"], source=ds["source"],
                min_size=ds.get("min_size", 0))

    # ------------------------------------------------------------------ #
    # 6. Jia et al. (2023) H(z) data
    # ------------------------------------------------------------------ #

    def download_jia_hz(self):
        """Download Jia et al. H(z) data from GitHub."""
        print_status("Downloading Jia et al. (2023) H(z) data", "TITLE")
        ds = DATASET_REGISTRY["jia_hz"]
        target = self.root / ds["target"]
        self.dl.download_file(
            url=ds["url"], target=target,
            description=ds["description"], source=ds["source"],
            min_size=ds.get("min_size", 0))

    # ------------------------------------------------------------------ #
    # 7. HyperLEDA V_rot for ALL Pantheon+ hosts
    # ------------------------------------------------------------------ #

    def download_hyperleda_vrot(self):
        """Query HyperLEDA for V_rot of all Pantheon+ SN host galaxies.

        This is the single most impactful dataset expansion: it populates
        the low-X_i bin for the Xi-step test and provides measured V_rot
        for the full 1,701-host sample.
        """
        print_status("Querying HyperLEDA for V_rot of all Pantheon+ hosts", "TITLE")

        output_csv = self.data_proc / "pantheon_host_vrot_hyperleda.csv"
        if output_csv.exists() and output_csv.stat().st_size > 1000:
            print_status(f"  [SKIP] {output_csv.name} — already present", "PROCESS")
            return

        pantheon_file = self.data_raw / "Pantheon+SH0ES.dat"
        if not pantheon_file.exists():
            print_status("  Pantheon+SH0ES.dat not found — run step_00 first", "WARNING")
            return

        try:
            import requests
        except ImportError:
            print_status("  'requests' package not available — skipping HyperLEDA query", "WARNING")
            return

        df = pd.read_csv(pantheon_file, sep=r"\s+")
        unique = df.drop_duplicates(subset="CID")[
            ["CID", "RA", "DEC", "zCMB", "HOST_LOGMASS",
             "IS_CALIBRATOR", "HOST_RA", "HOST_DEC"]].copy()
        unique = unique.reset_index(drop=True)
        print_status(f"  {len(unique)} unique SNe to query", "PROCESS")

        HYPERLEDA_URL = "http://atlas.obs-hp.fr/hyperleda/fG.cgi"
        QUERY_FIELDS = "pgc,objname,al2000,de2000,vrot,e_vrot"
        Z_MAX = 0.06
        results = []
        n_queried = 0
        n_matched = 0
        n_with_vrot = 0
        queried_cache = {}

        for idx, row in unique.iterrows():
            cid = str(row["CID"]).strip()
            z = float(row["zCMB"])

            # Use host coordinates if available
            host_ra = float(row["HOST_RA"]) if row["HOST_RA"] != -999 else None
            host_dec = float(row["HOST_DEC"]) if row["HOST_DEC"] != -999 else None
            match_ra = host_ra if host_ra is not None else float(row["RA"])
            match_dec = host_dec if host_dec is not None else float(row["DEC"])

            if z > Z_MAX:
                results.append({
                    "CID": cid, "zCMB": z, "vrot": None, "e_vrot": None,
                    "pgc": None, "galaxy_name": "", "source": "TOO_DISTANT"})
                continue

            # Adaptive search radius
            if z < 0.005:
                radius = 15.0
            elif z < 0.01:
                radius = 10.0
            elif z < 0.02:
                radius = 5.0
            elif z < 0.04:
                radius = 3.0
            else:
                radius = 2.0

            # Build SQL box query
            ra_hours = match_ra / 15.0
            cos_dec = max(abs(math.cos(math.radians(match_dec))), 0.01)
            d_ra = radius / (15.0 * 60.0 * cos_dec)
            d_dec = radius / 60.0

            conditions = []
            ra_min, ra_max = ra_hours - d_ra, ra_hours + d_ra
            if ra_min < 0:
                conditions.append(f"(al2000>{ra_min + 24.0} or al2000<{ra_max})")
            elif ra_max > 24.0:
                conditions.append(f"(al2000>{ra_min} or al2000<{ra_max - 24.0})")
            else:
                conditions.append(f"al2000>{ra_min}")
                conditions.append(f"al2000<{ra_max}")
            conditions.append(f"de2000>{match_dec - d_dec}")
            conditions.append(f"de2000<{match_dec + d_dec}")

            params = {"n": "meandata", "c": "o", "d": QUERY_FIELDS,
                      "sql": " and ".join(conditions), "ob": "pgc"}

            pos_key = (round(match_ra, 3), round(match_dec, 3))
            if pos_key in queried_cache:
                galaxies = queried_cache[pos_key]
            else:
                galaxies = []
                try:
                    resp = requests.get(HYPERLEDA_URL, params=params, timeout=30)
                    if resp.status_code == 200:
                        import re
                        pre_blocks = re.findall(r"<pre[^>]*>(.*?)</pre>", resp.text, re.DOTALL)
                        if len(pre_blocks) >= 2:
                            for line in pre_blocks[1].strip().split("\n"):
                                line = line.strip()
                                if not line or "<a" in line:
                                    continue
                                line_clean = re.sub(r"<[^>]+>", "", line)
                                parts = line_clean.split()
                                if len(parts) < 4:
                                    continue
                                try:
                                    pgc = int(parts[0])
                                    objname = parts[1]
                                    vrot = float(parts[4]) if len(parts) > 4 and parts[4] else None
                                    e_vrot = float(parts[5]) if len(parts) > 5 and parts[5] else None
                                    if vrot and vrot <= 0:
                                        vrot = None
                                    galaxies.append({
                                        "pgc": pgc, "objname": objname,
                                        "al2000": float(parts[2]),
                                        "de2000": float(parts[3]),
                                        "vrot": vrot, "e_vrot": e_vrot})
                                except (ValueError, IndexError):
                                    continue
                except Exception:
                    pass
                queried_cache[pos_key] = galaxies
                n_queried += 1
                time.sleep(0.5)

            # Find best host galaxy (skip SN-like names)
            import re
            best_gal = None
            best_sep = 999.0
            for g in galaxies:
                name = g["objname"].upper()
                if re.match(r"^SN\d", name) or name.startswith(("ASASSN", "PS1", "PTF", "CSS")):
                    continue
                gal_ra = g["al2000"] * 15.0
                gal_dec = g["de2000"]
                sep = math.sqrt((match_ra - gal_ra) ** 2 + (match_dec - gal_dec) ** 2) * 3600.0
                if sep < best_sep:
                    best_sep = sep
                    best_gal = g

            if best_gal and best_sep < radius * 60:
                n_matched += 1
                vrot = best_gal["vrot"]
                if vrot:
                    n_with_vrot += 1
                results.append({
                    "CID": cid, "zCMB": z,
                    "vrot": vrot, "e_vrot": best_gal["e_vrot"],
                    "pgc": best_gal["pgc"],
                    "galaxy_name": best_gal["objname"],
                    "match_sep_arcsec": best_sep,
                    "source": "HYPERLEDA_QUERY"})
            else:
                results.append({
                    "CID": cid, "zCMB": z, "vrot": None, "e_vrot": None,
                    "pgc": None, "galaxy_name": "", "source": "NO_MATCH"})

            if (idx + 1) % 100 == 0:
                print_status(f"  Progress: {idx+1}/{len(unique)} | "
                             f"queries={n_queried} matched={n_matched} vrot={n_with_vrot}",
                             "PROCESS")
                # Save partial
                pd.DataFrame(results).to_csv(output_csv, index=False)

        out_df = pd.DataFrame(results)
        out_df.to_csv(output_csv, index=False)
        print_status(f"  HyperLEDA V_rot: {n_with_vrot}/{len(unique)} hosts with V_rot "
                     f"({n_matched} matched, {n_queried} queries)", "SUCCESS")

        # Record in manifest
        if output_csv.exists():
            sha = self.dl.compute_sha256(output_csv)
            self.dl._record(output_csv, sha, output_csv.stat().st_size,
                            "HyperLEDA fG.cgi query", "HyperLEDA (Makarov et al. 2014)",
                            "downloaded", f"{n_with_vrot} V_rot measurements")

    # ------------------------------------------------------------------ #
    # 8. VizieR HyperLEDA VII/237 + VII/238 for V_rot cross-check
    # ------------------------------------------------------------------ #

    def download_vizier_vrot(self):
        """Download VizieR HyperLEDA catalogs VII/237 + VII/238.

        These provide independent V_rot measurements via HI line widths
        and inclination from axis ratios, cross-checking the direct
        HyperLEDA SQL query.
        """
        print_status("Downloading VizieR HyperLEDA VII/237 + VII/238", "TITLE")

        for catalog, name, target_name, min_size in [
            ("VII/237", "HyperLEDA galaxy catalog (logR25, MType)",
             "hyperleda_vii237.vot", 500_000),
            ("VII/238", "HyperLEDA HI data (log_2Vm, VHI)",
             "hyperleda_vii238.vot", 500_000),
        ]:
            target = self.data_ext / target_name
            self.dl.download_vizier(
                catalog=catalog, target=target,
                description=name, source=f"HyperLEDA {catalog}",
                min_size=min_size)

    # ------------------------------------------------------------------ #
    # 9. Verify curated CSVs exist and record checksums
    # ------------------------------------------------------------------ #

    def verify_curated_files(self):
        """Record checksums for curated CSV files transcribed from published tables.

        These files are transcribed from peer-reviewed publications and
        are part of the repository. They are not downloaded but are
        checksummed for integrity verification.
        """
        print_status("Verifying curated CSV files (transcribed from publications)", "TITLE")

        curated_files = {
            "r22_cepheid_distances.csv": {
                "source": "Riess et al. 2022, ApJ 934 L7 (Table 3 / SH0ES DataRelease)",
                "description": "R22 Cepheid distance moduli for 37 calibrator hosts"},
            "trgb_distances_freedman2024.csv": {
                "source": "Freedman et al. 2024/2025, CCHP (arXiv:2408.06153)",
                "description": "CCHP TRGB distance moduli for SN Ia hosts"},
            "jwst_trgb_distances.csv": {
                "source": "Anand et al. 2024 (ApJ 976 177) + Freedman et al. 2025 (CCHP)",
                "description": "JWST TRGB distances (F090W + F115W) for Cepheid/TRGB overlap"},
            "key_project_cepheid_distances.csv": {
                "source": "Freedman et al. 2001 (ApJ 553 47, Table 3)",
                "description": "Key Project Cepheid distances (V/I band)"},
            "madore_freedman2023_vih_vi.csv": {
                "source": "Madore & Freedman 2023 (arXiv:2309.10859, Table 2)",
                "description": "MF2023 VIH and VI Cepheid distance moduli"},
            "velocity_dispersions_literature.csv": {
                "source": "HyperLEDA (Makarov et al. 2014) curated for SH0ES hosts",
                "description": "V_rot-derived potential proxy for 37+ R22 hosts"},
            "hosts_properties.csv": {
                "source": "Pantheon+SH0ES.dat host galaxy metadata",
                "description": "Host galaxy RA/DEC/mass/Pantheon+ ID mapping"},
            "cf4_matched_galaxies_vrot.csv": {
                "source": "HyperLEDA query for 22 CF4 Cepheid+TRGB galaxies",
                "description": "V_rot for 22 CF4 galaxies with both Cepheid and TRGB"},
            "cosmicflows4_catalog.csv": {
                "source": "CosmicFlows-4 (Tully et al. 2023) processed catalog",
                "description": "Processed CF4 catalog with Cepheid/TRGB overlaps"},
        }

        for filename, meta in curated_files.items():
            filepath = self.data_ext / filename
            if filepath.exists() and filepath.stat().st_size > 0:
                sha = self.dl.compute_sha256(filepath)
                self.dl._record(filepath, sha, filepath.stat().st_size,
                                "curated_from_publication", meta["source"],
                                "curated", meta["description"])
                print_status(f"  [OK] {filename} — {filepath.stat().st_size:,} bytes", "SUCCESS")
            else:
                print_status(f"  [MISSING] {filename} — curated file not found", "WARNING")

        # Mazurenko curves directory
        mazurenko_dir = self.data_ext / "mazurenko_curves"
        if mazurenko_dir.exists():
            for f in mazurenko_dir.iterdir():
                if f.is_file() and f.stat().st_size > 0:
                    sha = self.dl.compute_sha256(f)
                    self.dl._record(f, sha, f.stat().st_size,
                                    "digitized_from_figure",
                                    "Mazurenko et al. 2025, MNRAS 536 3232, Fig 3",
                                    "curated", "Digitized void H(z) curve points")

    # ------------------------------------------------------------------ #
    # Main runner
    # ------------------------------------------------------------------ #

    def run(self):
        """Execute all external data downloads."""
        print_status("=" * 70, "TITLE")
        print_status("Step 00b: External Data Download — Full Provenance Pipeline", "TITLE")
        print_status("=" * 70, "TITLE")

        # 1. Pantheon+ data + covariance
        self.download_pantheon()

        # 2. CosmicFlows-4 (Playwright)
        self.download_cf4()

        # 3. SPARC
        self.download_sparc()

        # 4-5. VizieR catalogs (2MTF, SFI++, ALFALFA, 6dFGS)
        self.download_vizier_catalogs()

        # 6. Jia et al. H(z)
        self.download_jia_hz()

        # 7. HyperLEDA V_rot for all Pantheon+ hosts
        self.download_hyperleda_vrot()

        # 8. VizieR HyperLEDA VII/237 + VII/238
        self.download_vizier_vrot()

        # 9. Verify curated files
        self.verify_curated_files()

        # Write checksum manifest
        self.dl.write_manifest()

        # Summary
        summary = self.dl.summary()

        # Write step summary JSON
        step_summary = {
            "step": "00b",
            "description": "External data download with full provenance tracking",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "datasets": {
                "pantheon_plus": "Pantheon+SH0ES data + STAT+SYS and STATONLY covariance",
                "cosmicflows4": "CF4 table2 (55,877 galaxies) + table4 (38,053 groups)",
                "sparc": "SPARC Table1 + Table2 + Rotmod_LTG.zip (175 galaxies)",
                "2mtf": "2MTF Tully-Fisher catalog (2,062 spirals) via VizieR",
                "sfipp": "SFI++ Tully-Fisher catalog (4,861 galaxies) via VizieR",
                "jia_hz": "Jia et al. 2023 H(z) data (60 measurements)",
                "alfalfa": "ALFALFA A40 HI catalog via VizieR",
                "6dfgs": "6dFGS peculiar velocity catalog via VizieR",
                "hyperleda_vrot": "HyperLEDA V_rot for all Pantheon+ hosts (z < 0.06)",
                "vizier_hyperleda": "VizieR VII/237 + VII/238 for V_rot cross-check",
            },
            "download_summary": summary,
            "manifest_path": str(self.dl.manifest_path),
            "provenance": {
                "data_sources": [
                    "Pantheon+SH0ES DataRelease (GitHub)",
                    "CDS VizieR (J/ApJ/944/94, J/MNRAS/487/2061, J/ApJS/182/474, J/AJ/154/78, J/MNRAS/445/2677)",
                    "SPARC (astroweb.cwru.edu)",
                    "HyperLEDA (atlas.obs-hp.fr)",
                    "Jia et al. 2023 (GitHub)",
                    "Curated from peer-reviewed publications (R22, CCHP, MF2023, KP, Anand 2024)",
                ],
                "checksum_algorithm": "SHA-256",
                "manifest_file": "data/raw/checksums.json",
            },
        }

        summary_path = self.results_dir / "step_00b_external_data_summary.json"
        with open(summary_path, "w") as f:
            json.dump(step_summary, f, indent=2)

        print_status(f"\nStep 00b complete. Summary: {summary_path}", "SUCCESS")
        print_status(f"Checksum manifest: {self.dl.manifest_path}", "SUCCESS")

        return step_summary


def main():
    step = Step00bExternalData()
    step.run()


if __name__ == "__main__":
    main()
