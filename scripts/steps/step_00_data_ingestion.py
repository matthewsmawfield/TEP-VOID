#!/usr/bin/env python3
"""
Step 00: Data Ingestion — SH0ES Cepheid + CCHP TRGB Host Samples
=================================================================
Downloads and prepares the primary distance ladder data for the
TEP-VOID analysis.

Key Tasks:
1. Download Pantheon+SH0ES.dat (Cepheid + SN Ia distance ladder)
2. Load R22 Cepheid distance moduli (Riess et al. 2022)
3. Load CCHP TRGB distance moduli (Freedman et al. 2025)
4. Load host galaxy velocity dispersions (HyperLEDA)
5. Cross-match Cepheid, TRGB, and velocity dispersion catalogs
6. Output matched host catalog for subsequent analysis

Outputs:
    data/raw/Pantheon+SH0ES.dat
    data/raw/external/r22_cepheid_distances.csv
    data/raw/external/trgb_distances_freedman2024.csv
    data/raw/external/velocity_dispersions_literature.csv
    data/raw/external/hosts_properties.csv
    data/interim/shoes_cepheid_hosts.csv
    data/interim/cchp_trgb_hosts.csv
    data/interim/matched_hosts.csv
    results/outputs/step_00_data_ingestion_summary.json
"""

import json
import sys
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status


def _norm_galaxy(name):
    """Normalize galaxy name for cross-matching."""
    s = str(name).upper().strip().replace(" ", "")
    # Normalize NGC 0691 -> NGC691
    s = s.replace("NGC0", "NGC").replace("NGC00", "NGC")
    return s


def _source_id_to_galaxy(sid):
    """Convert R22 source_id (e.g. 'mu_N1015') to galaxy name ('NGC 1015')."""
    sid = sid.replace("mu_", "")
    if sid == "M101":
        return "M 101"
    if sid == "M31":
        return "M 31"
    if sid == "LMC":
        return "LMC"
    if sid == "M1337":
        return "Mrk 1337"
    if sid.startswith("N") and not sid.startswith("NGC"):
        num = sid[1:]
        try:
            return f"NGC {int(num)}"
        except ValueError:
            return f"NGC {num}"
    if sid.startswith("U"):
        return f"UGC {sid[1:]}"
    return sid


class Step00DataIngestion:
    """Step 00: Download and prepare SH0ES + CCHP data."""

    SHOES_URL = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2BSH0ES.dat"
    TEP_H0_DATA = Path(__file__).resolve().parents[4] / "TEP-H0" / "data" / "raw" / "Pantheon+SH0ES.dat"

    def __init__(self):
        self.root = PROJECT_ROOT
        self.data_raw = self.root / "data" / "raw"
        self.data_external = self.data_raw / "external"
        self.data_interim = self.root / "data" / "interim"
        self.results = self.root / "results" / "outputs"
        self.logs = self.root / "logs"

        for d in [self.data_raw, self.data_external, self.data_interim, self.results, self.logs]:
            d.mkdir(parents=True, exist_ok=True)

        self.logger = TEPLogger("step_00", log_file_path=self.logs / "step_00_data_ingestion.log")
        set_step_logger(self.logger)

    def download_shoes_data(self):
        """Download Pantheon+SH0ES.dat from GitHub."""
        target = self.data_raw / "Pantheon+SH0ES.dat"

        if target.exists():
            print_status(f"Pantheon+SH0ES.dat already exists ({target.stat().st_size / 1024:.0f} KB)", "PROCESS")
            return target

        print_status("Downloading Pantheon+SH0ES.dat from GitHub...", "PROCESS")
        try:
            urlretrieve(self.SHOES_URL, target)
            print_status(f"Downloaded Pantheon+SH0ES.dat ({target.stat().st_size / 1024:.0f} KB)", "SUCCESS")
        except Exception as e:
            print_status(f"GitHub download failed: {e}", "WARNING")
            if self.TEP_H0_DATA.exists():
                import shutil
                shutil.copy2(self.TEP_H0_DATA, target)
                print_status(f"Copied Pantheon+SH0ES.dat from TEP-H0 ({target.stat().st_size / 1024:.0f} KB)", "SUCCESS")
            else:
                print_status(f"SH0ES data not available from TEP-H0: {self.TEP_H0_DATA}", "ERROR")
                raise
        return target

    def load_shoes_cepheid_hosts(self):
        """Load SH0ES Cepheid host galaxy sample from Pantheon+SH0ES.dat."""
        shoes_path = self.data_raw / "Pantheon+SH0ES.dat"

        print_status("Loading SH0ES Cepheid + SN Ia data...", "PROCESS")

        try:
            df = pd.read_csv(shoes_path, sep=r"\s+", comment="#")
            print_status(f"Loaded {len(df)} rows from Pantheon+SH0ES.dat", "SUCCESS")
            print_status(f"Columns: {list(df.columns)}", "DEBUG")
        except Exception as e:
            print_status(f"Error reading Pantheon+SH0ES.dat: {e}", "ERROR")
            df = pd.DataFrame()

        output = self.data_interim / "shoes_cepheid_hosts.csv"
        df.to_csv(output, index=False)
        print_status(f"Saved Cepheid hosts to {output}", "SUCCESS")
        return df

    def load_r22_cepheid_distances(self):
        """Load R22 Cepheid distance moduli (Riess et al. 2022)."""
        r22_path = self.data_external / "r22_cepheid_distances.csv"

        if r22_path.exists():
            print_status(f"Loading R22 Cepheid distances from {r22_path}", "PROCESS")
            df = pd.read_csv(r22_path)
            print_status(f"Loaded {len(df)} R22 Cepheid distance moduli", "SUCCESS")
        else:
            print_status(f"R22 Cepheid distances not found at {r22_path}", "ERROR")
            return pd.DataFrame()

        # Convert source_id to galaxy name
        df["galaxy"] = df["source_id"].apply(_source_id_to_galaxy)
        df = df.rename(columns={"value": "mu_cepheid", "error": "mu_cepheid_err"})
        df["galaxy_norm"] = df["galaxy"].apply(_norm_galaxy)

        output = self.data_interim / "r22_cepheid_hosts.csv"
        df.to_csv(output, index=False)
        print_status(f"Saved R22 Cepheid hosts to {output}", "SUCCESS")
        return df

    def load_cchp_trgb_hosts(self):
        """Load CCHP TRGB distance moduli from Freedman et al. 2025."""
        trgb_path = self.data_external / "trgb_distances_freedman2024.csv"

        if trgb_path.exists():
            print_status(f"Loading CCHP TRGB distances from {trgb_path}", "PROCESS")
            df = pd.read_csv(trgb_path)
            print_status(f"Loaded {len(df)} TRGB host galaxies", "SUCCESS")
        else:
            print_status(f"CCHP TRGB data not found at {trgb_path}. No synthetic data generated.", "ERROR")
            return pd.DataFrame()

        df["galaxy_norm"] = df["galaxy"].apply(_norm_galaxy)

        output = self.data_interim / "cchp_trgb_hosts.csv"
        df.to_csv(output, index=False)
        print_status(f"Saved TRGB hosts to {output}", "SUCCESS")
        return df

    def load_velocity_dispersions(self):
        """Load velocity dispersions from HyperLEDA catalog."""
        vdisp_path = self.data_external / "velocity_dispersions_literature.csv"

        if vdisp_path.exists():
            print_status(f"Loading velocity dispersions from {vdisp_path}", "PROCESS")
            df = pd.read_csv(vdisp_path, comment="#")
            print_status(f"Loaded {len(df)} velocity dispersion entries", "SUCCESS")
        else:
            print_status(f"Velocity dispersions not found at {vdisp_path}", "ERROR")
            return pd.DataFrame()

        df["galaxy_norm"] = df["galaxy"].apply(_norm_galaxy)
        return df

    def cross_match_hosts(self, r22_df, trgb_df, vdisp_df):
        """
        Cross-match R22 Cepheid, TRGB, and velocity dispersion catalogs.

        Produces a matched catalog with columns:
            galaxy, mu_cepheid, mu_cepheid_err, mu_trgb, mu_trgb_err,
            sigma_v, sigma_v_err, delta_mu
        """
        print_status("Cross-matching Cepheid + TRGB + velocity dispersions...", "PROCESS")

        if r22_df.empty or trgb_df.empty:
            print_status("Cannot cross-match: missing Cepheid or TRGB data", "ERROR")
            matched = pd.DataFrame()
        else:
            # Merge R22 Cepheid with TRGB on normalized galaxy name
            matched = r22_df.merge(
                trgb_df[["galaxy_norm", "galaxy", "trgb_mu", "trgb_mu_err"]],
                on="galaxy_norm",
                how="inner",
                suffixes=("", "_trgb"),
            )
            print_status(f"  Cepheid + TRGB match: {len(matched)} galaxies", "SUCCESS")

            # Merge with velocity dispersions
            if not vdisp_df.empty:
                matched = matched.merge(
                    vdisp_df[["galaxy_norm", "sigma_kms", "error_kms"]],
                    on="galaxy_norm",
                    how="left",
                )
                n_with_vdisp = matched["sigma_kms"].notna().sum()
                print_status(f"  With velocity dispersions: {n_with_vdisp}/{len(matched)}", "SUCCESS")

            # Compute delta_mu = mu_Cepheid - mu_TRGB
            matched["delta_mu"] = matched["mu_cepheid"] - matched["trgb_mu"]
            matched["delta_mu_err"] = np.sqrt(
                matched["mu_cepheid_err"] ** 2 + matched["trgb_mu_err"] ** 2
            )

            # Use the TRGB galaxy name as the primary name
            if "galaxy_trgb" in matched.columns:
                matched["galaxy"] = matched["galaxy_trgb"]

            print_status(f"  Final matched catalog: {len(matched)} galaxies", "SUCCESS")
            print_status(f"  Mean delta_mu: {matched['delta_mu'].mean():.4f} mag", "TEST")
            print_status(f"  delta_mu range: [{matched['delta_mu'].min():.4f}, {matched['delta_mu'].max():.4f}]", "TEST")

        output = self.data_interim / "matched_hosts.csv"
        matched.to_csv(output, index=False)
        print_status(f"Saved matched hosts to {output}", "SUCCESS")
        return matched

    def run(self):
        """Execute the full step."""
        print_status("Step 00: Data Ingestion — SH0ES + CCHP", "TITLE")

        print_status(
            "Scientific context: This step addresses the foundational data-assembly "
            "question of whether independent distance indicators (Cepheid variables "
            "from the SH0ES/R22 programme and TRGB stars from the CCHP/Freedman 2025 "
            "programme) yield mutually consistent distance moduli for the same set of "
            "SN Ia host galaxies. The resulting matched catalog provides the "
            "indicator-pair sample whose divergence (delta_mu = mu_Cepheid - mu_TRGB) "
            "is the primary observable for testing whether local gravitational "
            "potential depth modulates the distance ladder in a manner consistent with "
            "the Temporal Equivalence Principle versus the standard void-based "
            "explanation of the Hubble tension.",
            "INFO",
        )
        print_status(
            "Data sources: Pantheon+SH0ES.dat (GitHub, Riess et al. 2022), "
            "R22 Cepheid distance moduli, CCHP TRGB distance moduli "
            "(Freedman et al. 2025), and HyperLEDA velocity dispersions. "
            "This step constitutes Block 0 of the TEP-VOID pipeline and feeds "
            "all downstream indicator-divergence and potential-stratification "
            "analyses.",
            "INFO",
        )

        # Download SH0ES data
        shoes_path = self.download_shoes_data()

        # Load Pantheon+SH0ES data (for SN Ia analysis in later steps)
        shoes_df = self.load_shoes_cepheid_hosts()

        # Load R22 Cepheid distances (for matched-host analysis)
        r22_df = self.load_r22_cepheid_distances()

        # Load TRGB hosts
        trgb_df = self.load_cchp_trgb_hosts()

        # Load velocity dispersions
        vdisp_df = self.load_velocity_dispersions()

        # Cross-match all three catalogs
        print_status(
            "Methodology: Cross-matching is performed on normalized galaxy names "
            "(case-insensitive, whitespace-collapsed, NGC zero-padding removed). "
            "An inner join between the Cepheid and TRGB catalogs retains only "
            "galaxies with both indicators. Velocity dispersions are attached via "
            "a left join. The divergence observable delta_mu = mu_Cepheid - mu_TRGB "
            "is computed with Gaussian uncertainty propagation "
            "(delta_mu_err = sqrt(mu_cep_err^2 + mu_trgb_err^2)).",
            "PROCESS",
        )
        matched_df = self.cross_match_hosts(r22_df, trgb_df, vdisp_df)

        if not matched_df.empty:
            print_status(
                f"Interpretation: {len(matched_df)} galaxies possess both Cepheid "
                f"and TRGB distance moduli, forming the indicator-pair sample. "
                f"The mean divergence delta_mu = {matched_df['delta_mu'].mean():.4f} mag "
                f"quantifies the systematic offset between the two distance scales; "
                f"a statistically significant non-zero mean would indicate that the "
                f"two indicators are not mutually calibrated, motivating the "
                f"potential-stratified analysis in subsequent blocks.",
                "TEST",
            )

        # Summary
        summary = {
            "step": "00_data_ingestion",
            "description": (
                "Ingestion of SH0ES (R22 Cepheid) and CCHP (TRGB) distance "
                "moduli, velocity dispersions, and cross-matching to build "
                "the matched host catalog for indicator divergence testing."
            ),
            "shoes_rows": len(shoes_df),
            "r22_cepheid_hosts": len(r22_df),
            "trgb_hosts": len(trgb_df),
            "vdisp_entries": len(vdisp_df),
            "matched_hosts": len(matched_df),
            "matched_with_vdisp": int(matched_df["sigma_kms"].notna().sum()) if not matched_df.empty else 0,
            "shoes_file": str(shoes_path),
            "r22_file": str(self.data_external / "r22_cepheid_distances.csv"),
            "trgb_file": str(self.data_external / "trgb_distances_freedman2024.csv"),
            "vdisp_file": str(self.data_external / "velocity_dispersions_literature.csv"),
            "output_files": [
                str(self.data_interim / "shoes_cepheid_hosts.csv"),
                str(self.data_interim / "r22_cepheid_hosts.csv"),
                str(self.data_interim / "cchp_trgb_hosts.csv"),
                str(self.data_interim / "matched_hosts.csv"),
            ],
            "methodology": (
                "Cross-matching of R22 Cepheid and CCHP TRGB distance-modulus "
                "catalogs on normalized galaxy names (inner join), with HyperLEDA "
                "velocity dispersions attached via left join. Divergence "
                "delta_mu = mu_Cepheid - mu_TRGB computed with Gaussian "
                "uncertainty propagation."
            ),
            "provenance": {
                "data_sources": [
                    "Pantheon+SH0ES.dat (Riess et al. 2022, GitHub)",
                    "R22 Cepheid distance moduli (Riess et al. 2022)",
                    "CCHP TRGB distance moduli (Freedman et al. 2025)",
                    "HyperLEDA velocity dispersions",
                ],
                "software_versions": {
                    "python": sys.version.split()[0],
                    "numpy": np.__version__,
                    "pandas": pd.__version__,
                },
                "pipeline_block": "Block 0 — Data Ingestion",
            },
            "scientific_context": (
                "Assembles the matched Cepheid–TRGB host-galaxy sample whose "
                "indicator divergence (delta_mu) is the primary observable for "
                "testing whether local gravitational potential depth modulates the "
                "distance ladder, as predicted by the Temporal Equivalence "
                "Principle, versus the standard void-based explanation of the "
                "Hubble tension."
            ),
            "downstream_consumers": [
                "01_host_potential_catalog",
                "03_pantheon_ingestion",
            ],
        }

        summary_path = self.results / "step_00_data_ingestion_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print_status(f"Summary saved to {summary_path}", "SUCCESS")

        print_status("Step 00 complete", "SUCCESS")


if __name__ == "__main__":
    step = Step00DataIngestion()
    step.run()

