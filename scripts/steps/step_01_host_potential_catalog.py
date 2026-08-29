#!/usr/bin/env python3
"""
Step 01: Host Potential Catalog — Velocity Dispersion Based Gravitational
Potential Proxy
==========================================================================
Builds the host galaxy gravitational potential catalog from velocity
dispersions for the TEP-VOID analysis.

Key Tasks:
1. Locate velocity dispersions from data/raw/external/
   velocity_dispersions_literature.csv (copy from TEP-H0 if available)
2. Compute potential depth proxy: Phi_proxy = sigma_v^2
   (velocity dispersion squared, in (km/s)^2)
3. Output the host potential catalog for subsequent void analysis

Outputs:
    data/raw/external/velocity_dispersions_literature.csv
    data/processed/host_potential_catalog.csv
    results/outputs/step_01_host_potential_catalog.json
"""

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status


class Step01HostPotentialCatalog:
    """Step 01: Build host galaxy gravitational potential catalog."""

    # TEP-H0 path for velocity dispersions (copy if available)
    # Resolved relative to the project root's sibling directory
    TEP_H0_VDISP = (
        PROJECT_ROOT.parent
        / "TEP-H0"
        / "data"
        / "raw"
        / "external"
        / "velocity_dispersions_literature.csv"
    )

    def __init__(self):
        self.root = PROJECT_ROOT
        self.data_raw = self.root / "data" / "raw"
        self.data_external = self.data_raw / "external"
        self.data_processed = self.root / "data" / "processed"
        self.data_interim = self.root / "data" / "interim"
        self.results = self.root / "results" / "outputs"
        self.logs = self.root / "logs"

        for d in [
            self.data_raw,
            self.data_external,
            self.data_processed,
            self.data_interim,
            self.results,
            self.logs,
        ]:
            d.mkdir(parents=True, exist_ok=True)

        self.logger = TEPLogger(
            "step_01", log_file_path=self.logs / "step_01_host_potential_catalog.log"
        )
        set_step_logger(self.logger)

    def load_velocity_dispersions(self):
        """
        Load velocity dispersions from the external data directory.

        If the file does not exist locally, attempt to copy it from the
        TEP-H0 repository.  If that also fails, raise an error — no
        synthetic data is generated.
        """
        target = self.data_external / "velocity_dispersions_literature.csv"

        if target.exists():
            print_status(
                f"Velocity dispersions file already exists "
                f"({target.stat().st_size / 1024:.1f} KB)",
                "PROCESS",
            )
            return target

        # Attempt to copy from TEP-H0
        print_status(
            f"Local velocity dispersions not found. "
            f"Checking TEP-H0 at {self.TEP_H0_VDISP}...",
            "PROCESS",
        )
        if self.TEP_H0_VDISP.exists():
            try:
                shutil.copy2(self.TEP_H0_VDISP, target)
                print_status(
                    f"Copied velocity dispersions from TEP-H0 "
                    f"({target.stat().st_size / 1024:.1f} KB)",
                    "SUCCESS",
                )
                return target
            except Exception as e:
                print_status(f"Failed to copy from TEP-H0: {e}", "WARNING")
        else:
            print_status("TEP-H0 velocity dispersions file not available.", "WARNING")

        print_status(
            "Cannot obtain velocity dispersions from any source. "
            "No synthetic data is generated — pipeline cannot proceed without real data.",
            "ERROR",
        )
        return None

    def parse_velocity_dispersions(self, path):
        """
        Parse the velocity dispersions CSV into a clean DataFrame.

        Handles the TEP-H0 velocity dispersions format.
        """
        print_status(f"Parsing velocity dispersions from {path}...", "PROCESS")

        # Read the CSV, skipping comment lines
        try:
            df = pd.read_csv(path, comment="#")
        except Exception as e:
            print_status(f"Error reading velocity dispersions: {e}", "ERROR")
            return pd.DataFrame()

        print_status(f"Loaded {len(df)} rows, columns: {list(df.columns)}", "DEBUG")

        # Ensure required columns exist
        required = ["galaxy", "sigma_kms", "error_kms", "source_bibcode", "method"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            print_status(
                f"Missing required columns: {missing}. "
                f"Attempting to fill with defaults.",
                "WARNING",
            )
            if "galaxy" not in df.columns:
                print_status("Cannot proceed without 'galaxy' column.", "ERROR")
                return pd.DataFrame()
            for col in missing:
                if col == "sigma_kms" and "vrot_kms" in df.columns:
                    # Derive sigma from rotation velocity: sigma = vrot / sqrt(2)
                    df["sigma_kms"] = df["vrot_kms"] / (2**0.5)
                    print_status(
                        "Derived sigma_kms from vrot_kms (vrot/sqrt(2)).", "PROCESS"
                    )
                elif col == "error_kms" and "vrot_error_kms" in df.columns:
                    df["error_kms"] = df["vrot_error_kms"] / (2**0.5)
                    print_status(
                        "Derived error_kms from vrot_error_kms (vrot_err/sqrt(2)).",
                        "PROCESS",
                    )
                else:
                    df[col] = "unknown"

        # Clean up: select relevant columns, drop NaN sigma
        df = df[required].copy()
        df = df.dropna(subset=["sigma_kms"])
        df["sigma_kms"] = pd.to_numeric(df["sigma_kms"], errors="coerce")
        df["error_kms"] = pd.to_numeric(df["error_kms"], errors="coerce")
        df = df.dropna(subset=["sigma_kms"])

        print_status(f"Parsed {len(df)} galaxies with valid velocity dispersions.", "SUCCESS")
        return df

    def compute_potential_proxy(self, df):
        """
        Compute the gravitational potential depth proxy.

        Phi_proxy = sigma_v^2  (velocity dispersion squared, in (km/s)^2)

        This serves as a proxy for the depth of the host galaxy's
        gravitational potential well, which is relevant for testing
        temporal equivalence effects in different potential environments.
        """
        print_status("Computing gravitational potential depth proxy...", "PROCESS")

        df = df.copy()
        df["phi_proxy_kms2"] = df["sigma_kms"] ** 2

        # Propagate uncertainty: d(phi) = 2 * sigma * d(sigma)
        df["phi_proxy_err_kms2"] = 2.0 * df["sigma_kms"] * df["error_kms"]

        print_status(
            f"Computed Phi_proxy for {len(df)} galaxies. "
            f"Median Phi_proxy = {df['phi_proxy_kms2'].median():.1f} (km/s)^2",
            "SUCCESS",
        )
        return df

    def run(self):
        """Execute the full step."""
        print_status("Step 01: Host Potential Catalog", "TITLE")

        print_status(
            "Scientific context: This step addresses the question of how to "
            "quantify the gravitational potential depth of each SN Ia host galaxy "
            "in a manner that is physically motivated and observationally available. "
            "The velocity dispersion sigma_v is adopted as a proxy for the depth of "
            "the host potential well, since the virial theorem relates sigma_v^2 to "
            "the gravitational potential. This catalog is the independent variable "
            "against which the Cepheid–TRGB divergence (delta_mu) is regressed in "
            "subsequent blocks, enabling a direct test of whether clock-rate "
            "modulation by gravitational potential — as predicted by the Temporal "
            "Equivalence Principle — can account for the indicator divergence that "
            "is otherwise attributed to void environments.",
            "INFO",
        )
        print_status(
            "Data sources: Velocity dispersions are sourced from the "
            "velocity_dispersions_literature.csv catalog (HyperLEDA and literature "
            "compilation), copied from the TEP-H0 repository if not present locally. "
            "This step is part of Block 0 of the TEP-VOID pipeline and supplies the "
            "potential-depth stratification variable for all downstream "
            "divergence-versus-potential analyses.",
            "INFO",
        )

        # Load / locate velocity dispersions
        vdisp_path = self.load_velocity_dispersions()

        if vdisp_path is None:
            print_status("Cannot proceed without velocity dispersion data.", "ERROR")
            vdisp_df = pd.DataFrame()
        else:
            # Parse into clean DataFrame
            vdisp_df = self.parse_velocity_dispersions(vdisp_path)

        if vdisp_df.empty:
            print_status("No velocity dispersion data available.", "ERROR")
            catalog_df = pd.DataFrame()
        else:
            # Compute potential proxy
            print_status(
                "Methodology: The gravitational potential depth proxy is computed "
                "as Phi_proxy = sigma_v^2 (velocity dispersion squared, in (km/s)^2), "
                "motivated by the virial theorem (2K + U = 0 implies sigma^2 ~ |Phi|). "
                "Uncertainty is propagated via d(Phi) = 2 * sigma * d(sigma). "
                "No cosmological model is assumed; the proxy is purely "
                "observational and dimensionally consistent with a specific "
                "gravitational energy scale.",
                "PROCESS",
            )
            catalog_df = self.compute_potential_proxy(vdisp_df)

        if not catalog_df.empty:
            print_status(
                f"Interpretation: The host potential catalog contains "
                f"{len(catalog_df)} galaxies with Phi_proxy ranging over "
                f"[{catalog_df['phi_proxy_kms2'].min():.1f}, "
                f"{catalog_df['phi_proxy_kms2'].max():.1f}] (km/s)^2 "
                f"(median {catalog_df['phi_proxy_kms2'].median():.1f}). "
                f"This dynamic range in potential depth is what permits the "
                f"downstream stratified regression to detect any "
                f"potential-dependent component in the indicator divergence.",
                "TEST",
            )

        # Write catalog
        catalog_path = self.data_processed / "host_potential_catalog.csv"
        catalog_df.to_csv(catalog_path, index=False)
        print_status(f"Host potential catalog saved to {catalog_path}", "SUCCESS")

        # Summary
        summary = {
            "step": "01_host_potential_catalog",
            "description": "Host galaxy gravitational potential catalog from velocity dispersions",
            "input_file": str(vdisp_path),
            "n_galaxies": len(catalog_df),
            "median_sigma_kms": float(catalog_df["sigma_kms"].median())
            if len(catalog_df) > 0
            else None,
            "mean_sigma_kms": float(catalog_df["sigma_kms"].mean())
            if len(catalog_df) > 0
            else None,
            "std_sigma_kms": float(catalog_df["sigma_kms"].std())
            if len(catalog_df) > 0
            else None,
            "min_sigma_kms": float(catalog_df["sigma_kms"].min())
            if len(catalog_df) > 0
            else None,
            "max_sigma_kms": float(catalog_df["sigma_kms"].max())
            if len(catalog_df) > 0
            else None,
            "median_phi_proxy_kms2": float(catalog_df["phi_proxy_kms2"].median())
            if len(catalog_df) > 0
            else None,
            "mean_phi_proxy_kms2": float(catalog_df["phi_proxy_kms2"].mean())
            if len(catalog_df) > 0
            else None,
            "potential_proxy_definition": "Phi_proxy = sigma_v^2 (velocity dispersion squared, in (km/s)^2)",
            "output_files": [
                str(catalog_path),
            ],
            "methodology": (
                "Gravitational potential depth proxy computed as "
                "Phi_proxy = sigma_v^2 from velocity dispersions, with "
                "uncertainty propagated via d(Phi) = 2 * sigma * d(sigma). "
                "The proxy is motivated by the virial theorem and requires no "
                "cosmological model assumption."
            ),
            "provenance": {
                "data_sources": [
                    "velocity_dispersions_literature.csv (HyperLEDA + literature compilation)",
                    "TEP-H0 repository (fallback copy source)",
                ],
                "software_versions": {
                    "python": sys.version.split()[0],
                    "pandas": pd.__version__,
                },
                "pipeline_block": "Block 0 — Data Ingestion",
            },
            "scientific_context": (
                "Constructs the host-galaxy gravitational potential depth catalog "
                "that serves as the independent variable for testing whether "
                "potential-dependent clock-rate modulation (Temporal Equivalence "
                "Principle) can explain the Cepheid–TRGB indicator divergence "
                "alternatively attributed to local void environments."
            ),
            "downstream_consumers": [
                "04_delta_mu_vs_potential",
                "05_stratified_hubble",
            ],
        }

        summary_path = self.results / "step_01_host_potential_catalog.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print_status(f"Summary saved to {summary_path}", "SUCCESS")

        print_status("Step 01 complete", "SUCCESS")


if __name__ == "__main__":
    step = Step01HostPotentialCatalog()
    step.run()
