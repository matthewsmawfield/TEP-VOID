#!/usr/bin/env python3
"""
Step 03: Pantheon+ Ingestion — Supernova Ia Distance and Redshift Catalog
==========================================================================
Downloads and parses the Pantheon+ supernova Ia catalog for the
TEP-VOID analysis.

Key Tasks:
1. Locate the Pantheon+SH0ES.dat file (downloaded in Step 00) or
   download the Pantheon+ light-curve catalog from GitHub
2. Parse SN Ia distances and redshifts into a clean DataFrame
3. Output the processed catalog for subsequent void analysis

Outputs:
    data/interim/pantheon_plus_sne.csv
    results/outputs/step_03_pantheon_ingestion.json
"""

import json
import sys
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status


class Step03PantheonIngestion:
    """Step 03: Download and parse the Pantheon+ supernova Ia catalog."""

    PANTHEON_LC_URL = (
        "https://raw.githubusercontent.com/PantheonPlusSH0ES/"
        "DataRelease/main/pantheon_plus_lc.txt"
    )

    def __init__(self):
        self.root = PROJECT_ROOT
        self.data_raw = self.root / "data" / "raw"
        self.data_external = self.data_raw / "external"
        self.data_interim = self.root / "data" / "interim"
        self.results = self.root / "results" / "outputs"
        self.logs = self.root / "logs"

        for d in [
            self.data_raw,
            self.data_external,
            self.data_interim,
            self.results,
            self.logs,
        ]:
            d.mkdir(parents=True, exist_ok=True)

        self.logger = TEPLogger(
            "step_03", log_file_path=self.logs / "step_03_pantheon_ingestion.log"
        )
        set_step_logger(self.logger)

    def locate_pantheon_data(self):
        """
        Locate Pantheon+ data.

        First checks for the Pantheon+SH0ES.dat file downloaded in Step 00.
        If not found, attempts to download the Pantheon+ light-curve catalog
        from GitHub.  If that also fails, returns None — no synthetic data
        is generated.
        """
        # Check for Step 00 output first
        shoes_path = self.data_raw / "Pantheon+SH0ES.dat"
        if shoes_path.exists():
            print_status(
                f"Using Pantheon+SH0ES.dat from Step 00 "
                f"({shoes_path.stat().st_size / 1024:.1f} KB)",
                "PROCESS",
            )
            return shoes_path

        # Attempt to download pantheon_plus_lc.txt
        target = self.data_raw / "pantheon_plus_lc.txt"
        if target.exists():
            print_status(
                f"Pantheon+ light-curve file already exists "
                f"({target.stat().st_size / 1024:.1f} KB)",
                "PROCESS",
            )
            return target

        print_status(
            f"Downloading Pantheon+ light-curve data from {self.PANTHEON_LC_URL}...",
            "PROCESS",
        )
        try:
            urlretrieve(self.PANTHEON_LC_URL, target)
            print_status(
                f"Downloaded Pantheon+ light-curve data "
                f"({target.stat().st_size / 1024:.1f} KB)",
                "SUCCESS",
            )
            return target
        except (URLError, Exception) as e:
            print_status(f"Failed to download Pantheon+ data: {e}", "WARNING")
            print_status(
                "Cannot obtain Pantheon+ data from any source. "
                "No synthetic data is generated — pipeline cannot proceed without real data.",
                "ERROR",
            )
            return None

    def parse_pantheon(self, path):
        """
        Parse the Pantheon+ data file into a clean DataFrame.

        Handles the Pantheon+SH0ES.dat format (whitespace-delimited with
        columns: CID, zCMB, zHEL, m_b_corr, MU_SH0ES, HOST_LOGMASS, RA, DEC, etc.)
        Also handles the pantheon_plus_lc.txt format as a fallback.
        """
        print_status(f"Parsing Pantheon+ data from {path}...", "PROCESS")

        try:
            df = pd.read_csv(path, sep=r"\s+", comment="#")
        except Exception as e:
            print_status(f"Error reading Pantheon+ data: {e}", "ERROR")
            return pd.DataFrame()

        print_status(f"Loaded {len(df)} rows, {len(df.columns)} columns", "DEBUG")
        print_status(f"Columns: {list(df.columns)[:10]}...", "DEBUG")

        # Build output DataFrame using Pantheon+SH0ES.dat column names
        out = pd.DataFrame()

        # SN name
        if "CID" in df.columns:
            out["sn_name"] = df["CID"]
        else:
            out["sn_name"] = [f"SN_{i}" for i in range(len(df))]

        # Redshift: prefer CMB, fall back to helio
        if "zCMB" in df.columns:
            out["z"] = pd.to_numeric(df["zCMB"], errors="coerce")
            out["z_frame"] = "CMB"
        elif "zHEL" in df.columns:
            out["z"] = pd.to_numeric(df["zHEL"], errors="coerce")
            out["z_frame"] = "helio"
        elif "zHD" in df.columns:
            out["z"] = pd.to_numeric(df["zHD"], errors="coerce")
            out["z_frame"] = "HD"
        else:
            out["z"] = None
            out["z_frame"] = None

        # Distance modulus from SH0ES
        if "MU_SH0ES" in df.columns:
            out["mu"] = pd.to_numeric(df["MU_SH0ES"], errors="coerce")
            out["mu_source"] = "SH0ES"
        elif "m_b_corr" in df.columns:
            # Fallback: approximate mu from corrected m_B
            out["mu"] = pd.to_numeric(df["m_b_corr"], errors="coerce") + 19.3
            out["mu_source"] = "approx_m_b_corr+19.3"
        else:
            out["mu"] = None
            out["mu_source"] = None

        # Distance modulus error
        if "MU_SH0ES_ERR_DIAG" in df.columns:
            out["mu_err"] = pd.to_numeric(df["MU_SH0ES_ERR_DIAG"], errors="coerce")
        elif "m_b_corr_err_DIAG" in df.columns:
            out["mu_err"] = pd.to_numeric(df["m_b_corr_err_DIAG"], errors="coerce")
        else:
            out["mu_err"] = None

        # Corrected apparent magnitude
        if "m_b_corr" in df.columns:
            out["m_b_corr"] = pd.to_numeric(df["m_b_corr"], errors="coerce")
        if "m_b_corr_err_DIAG" in df.columns:
            out["m_b_corr_err"] = pd.to_numeric(df["m_b_corr_err_DIAG"], errors="coerce")

        # Host galaxy stellar mass
        if "HOST_LOGMASS" in df.columns:
            out["host_logmass"] = pd.to_numeric(df["HOST_LOGMASS"], errors="coerce")
        else:
            out["host_logmass"] = None

        if "HOST_LOGMASS_ERR" in df.columns:
            logmass_err = pd.to_numeric(df["HOST_LOGMASS_ERR"], errors="coerce")
            # Convert -9.0 missing-value placeholders to NaN (Pantheon+ convention)
            logmass_err = logmass_err.where(logmass_err > -5.0, np.nan)
            out["host_logmass_err"] = logmass_err

        # Coordinates
        if "RA" in df.columns:
            out["ra"] = pd.to_numeric(df["RA"], errors="coerce")
        if "DEC" in df.columns:
            out["dec"] = pd.to_numeric(df["DEC"], errors="coerce")

        # Peculiar velocity
        if "VPEC" in df.columns:
            out["vpec"] = pd.to_numeric(df["VPEC"], errors="coerce")

        # Calibrator flag
        if "IS_CALIBRATOR" in df.columns:
            out["is_calibrator"] = pd.to_numeric(df["IS_CALIBRATOR"], errors="coerce")

        # Drop rows without a valid redshift
        out = out.dropna(subset=["z"])
        out = out[out["z"] > 0]

        # Deduplicate by CID — some SNe appear in multiple surveys.
        # Keep the first occurrence (prefer rows with USED_IN_SH0ES_HF=1 if available).
        n_before = len(out)
        if "sn_name" in out.columns:
            # Prefer Hubble-flow SNe for duplicates
            if "USED_IN_SH0ES_HF" in df.columns:
                out["_hf_pref"] = pd.to_numeric(
                    df.loc[out.index, "USED_IN_SH0ES_HF"], errors="coerce"
                ).fillna(0)
                out = out.sort_values("_hf_pref", ascending=False).drop_duplicates(
                    subset=["sn_name"], keep="first"
                )
                out = out.drop(columns=["_hf_pref"])
            else:
                out = out.drop_duplicates(subset=["sn_name"], keep="first")
        out = out.reset_index(drop=True)
        n_dups = n_before - len(out)
        self.n_raw = n_before
        self.n_duplicates_removed = n_dups
        if n_dups > 0:
            print_status(
                f"Deduplicated {n_dups} duplicate SN rows "
                f"({n_before} → {len(out)} unique SNe by CID)",
                "PROCESS",
            )

        print_status(f"Parsed {len(out)} Pantheon+ SNe with valid redshifts.", "SUCCESS")
        if out["mu"].notna().any():
            print_status(f"  Distance moduli available for {out['mu'].notna().sum()} SNe", "DEBUG")
        if out["host_logmass"].notna().any():
            n_massive = (out["host_logmass"] > 10.0).sum()
            print_status(f"  Host masses available for {out['host_logmass'].notna().sum()} SNe ({n_massive} massive)", "DEBUG")

        return out

    def run(self):
        """Execute the full step."""
        print_status("Step 03: Pantheon+ Ingestion", "TITLE")

        print_status(
            "Scientific context: This step addresses the assembly of the Type Ia "
            "supernova distance–redshift sample that forms the Hubble-flow "
            "backbone of the TEP-VOID analysis. The Pantheon+ compilation "
            "(Scolnic et al. 2022) provides corrected apparent magnitudes, "
            "distance moduli, redshifts, and host-galaxy stellar masses for "
            "~1700 SNe Ia. When combined with the host potential catalog "
            "(Step 01) and the Cepheid–TRGB divergence (Step 00), this sample "
            "enables a stratified Hubble diagram test: if the Temporal "
            "Equivalence Principle holds, the local Hubble rate should exhibit "
            "a residual correlation with host potential depth that is "
            "distinct from the void-induced monopole invoked by the standard "
            "explanation of the Hubble tension.",
            "INFO",
        )
        print_status(
            "Data sources: Pantheon+SH0ES.dat (downloaded in Step 00 or fetched "
            "from the PantheonPlusSH0ES GitHub repository), containing the "
            "full Pantheon+ light-curve and Hubble-diagram sample. This step is "
            "part of Block 0 of the TEP-VOID pipeline and supplies the "
            "Hubble-flow SN Ia sample for all downstream distance-ladder and "
            "stratified-Hubble analyses.",
            "INFO",
        )

        # Locate / download Pantheon+ data
        pantheon_path = self.locate_pantheon_data()

        if pantheon_path is None:
            print_status("Cannot proceed without Pantheon+ data.", "ERROR")
            sne_df = pd.DataFrame()
        else:
            # Parse catalog
            print_status(
                "Methodology: The Pantheon+SH0ES.dat file is parsed as "
                "whitespace-delimited columns. Redshift is taken from zCMB "
                "(CMB frame) with fallback to zHEL or zHD. Distance moduli are "
                "taken from MU_SH0ES with fallback to m_b_corr + 19.3. "
                "Deduplication is performed by CID, preferring rows with "
                "USED_IN_SH0ES_HF=1 (Hubble-flow SNe). Host stellar masses "
                "(HOST_LOGMASS) are retained for downstream mass-stratified "
                "analyses. Rows without a valid positive redshift are dropped.",
                "PROCESS",
            )
            sne_df = self.parse_pantheon(pantheon_path)

        if sne_df.empty:
            print_status("No Pantheon+ data available.", "ERROR")
        else:
            print_status(
                f"Interpretation: {len(sne_df)} unique Pantheon+ SNe Ia with "
                f"valid redshifts span z = [{sne_df['z'].min():.5f}, "
                f"{sne_df['z'].max():.5f}], covering both the nearby calibrator "
                f"regime and the Hubble flow. This sample provides the "
                f"distance–redshift relation against which potential-stratified "
                f"residuals are measured to distinguish TEP-predicted "
                f"clock-rate modulation from void-induced Hubble-tension effects.",
                "TEST",
            )
            print_status(
                f"Pantheon+ catalog: {len(sne_df)} SNe, "
                f"z range [{sne_df['z'].min():.5f}, {sne_df['z'].max():.5f}]",
                "DEBUG",
            )

        # Write processed catalog
        output_path = self.data_interim / "pantheon_plus_sne.csv"
        sne_df.to_csv(output_path, index=False)
        print_status(f"Processed Pantheon+ catalog saved to {output_path}", "SUCCESS")

        # Summary
        summary = {
            "step": "03_pantheon_ingestion",
            "description": "Pantheon+ supernova distance-redshift catalog ingestion and deduplication",
            "input_file": str(pantheon_path),
            "n_sne_raw": int(self.n_raw) if hasattr(self, "n_raw") else None,
            "n_sne": int(len(sne_df)),
            "n_duplicates_removed": int(self.n_raw - len(sne_df)) if hasattr(self, "n_raw") else None,
            "deduplication_method": "by CID (Pantheon+ cross-ID)",
            "z_min": float(sne_df["z"].min()) if len(sne_df) > 0 else None,
            "z_max": float(sne_df["z"].max()) if len(sne_df) > 0 else None,
            "z_median": float(sne_df["z"].median()) if len(sne_df) > 0 else None,
            "z_mean": float(sne_df["z"].mean()) if len(sne_df) > 0 else None,
            "n_with_mu": int(sne_df["mu"].notna().sum()) if "mu" in sne_df.columns else 0,
            "n_with_host_mass": int(sne_df["host_logmass"].notna().sum()) if "host_logmass" in sne_df.columns else 0,
            "n_massive_hosts": int((sne_df["host_logmass"] > 10.0).sum()) if "host_logmass" in sne_df.columns else 0,
            "n_low_mass_hosts": int((sne_df["host_logmass"] <= 10.0).sum()) if "host_logmass" in sne_df.columns else 0,
            "host_mass_split_threshold": 10.0,
            "columns": list(sne_df.columns),
            "output_files": [
                str(output_path),
            ],
            "methodology": (
                "Parsing of Pantheon+SH0ES.dat with redshift from zCMB (CMB "
                "frame), distance moduli from MU_SH0ES, and host stellar masses "
                "from HOST_LOGMASS. Deduplication by CID preferring "
                "USED_IN_SH0ES_HF=1 rows. Rows without valid positive redshift "
                "are excluded."
            ),
            "provenance": {
                "data_sources": [
                    "Pantheon+SH0ES.dat (Scolnic et al. 2022, PantheonPlusSH0ES GitHub)",
                    "Pantheon+ light-curve catalog (fallback: pantheon_plus_lc.txt)",
                ],
                "software_versions": {
                    "python": sys.version.split()[0],
                    "numpy": np.__version__,
                    "pandas": pd.__version__,
                },
                "pipeline_block": "Block 0 — Data Ingestion",
            },
            "scientific_context": (
                "Assembles the Type Ia supernova distance–redshift sample that "
                "forms the Hubble-flow backbone for testing whether host "
                "gravitational potential depth induces a residual correlation "
                "in the local Hubble rate, as predicted by the Temporal "
                "Equivalence Principle, versus the void-based monopole "
                "explanation of the Hubble tension."
            ),
            "downstream_consumers": [
                "04_delta_mu_vs_potential",
                "05_stratified_hubble",
            ],
        }

        summary_path = self.results / "step_03_pantheon_ingestion.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print_status(f"Summary saved to {summary_path}", "SUCCESS")

        print_status("Step 03 complete", "SUCCESS")


if __name__ == "__main__":
    step = Step03PantheonIngestion()
    step.run()
