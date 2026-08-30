#!/usr/bin/env python3
"""
Step 53: Directional Cepheid-TRGB Sample Compilation
=====================================================
Compile the largest possible matched Cepheid-TRGB sample with sky
coordinates for the directional Δμ test (Step 54).

Under TEP, the "bulk flow" may be a dipolar temporal topology gradient
rather than kinetic motion. The discriminating observable is whether
Δμ = μ_Cep − μ_TRGB shows a directional pattern aligned with the CMB
dipole.  The kinematic model predicts no such pattern (luminosity
distances are direction-independent); TEP predicts a dipolar modulation
because acoustic clocks (Cepheids) are biased by the temporal shear
gradient while nuclear candles (TRGB) are not.

Data sources compiled:
  1. CosmicFlows-4 table2 (Tully et al. 2023) — 22 galaxies with both
     DMceph and DMtrgb, plus RA/Dec and Vcmb.
  2. JWST TRGB (Freedman et al. 2024 F115W + Anand et al. 2024 F090W)
     cross-matched with R22 Cepheid distances — 18 galaxies.
  3. R22 Cepheid distances (Riess et al. 2022) cross-matched with
     CF4 TRGB by PGC.
  4. Key Project Cepheid distances (Freedman et al. 2001) cross-matched
     with CF4 TRGB by PGC via TEP-H0 hosts_processed.
  5. Jacobs et al. (2009) TRGB catalog (J/AJ/138/332) downloaded from
     Vizier — extends the TRGB baseline for additional Cepheid overlaps.

For each galaxy the following are computed:
  - Δμ = μ_Cep − μ_TRGB and its uncertainty
  - X_i (screened TEP gravitational potential coordinate)
  - CMB dipole directional projection (cosine of angular separation)
  - Source flags (Cepheid provenance, TRGB provenance, R22 membership)

Outputs:
    data/processed/directional_ceph_trgb_sample.csv
    results/outputs/step_53_directional_sample.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.screening import (
    compute_screening,
    U_REF_SCREENED,
    U_REF_UNSCREENED,
    C_KMS,
)

# CMB dipole direction (Planck 2018: l=264.021, b=48.253)
CMB_DIPOLE_GAL_L = 264.021  # deg
CMB_DIPOLE_GAL_B = 48.253   # deg


def ra_dec_to_unit_vectors(ra_deg, dec_deg):
    """Convert RA/Dec (degrees) to 3-D unit vectors on the celestial sphere."""
    ra_rad = np.radians(ra_deg)
    dec_rad = np.radians(dec_deg)
    return np.column_stack([
        np.cos(dec_rad) * np.cos(ra_rad),
        np.cos(dec_rad) * np.sin(ra_rad),
        np.sin(dec_rad),
    ])


def cmb_dipole_unit_vector():
    """3-D unit vector of the CMB dipole direction in equatorial coordinates."""
    c = SkyCoord(
        l=CMB_DIPOLE_GAL_L * u.deg,
        b=CMB_DIPOLE_GAL_B * u.deg,
        frame="galactic",
    ).icrs
    return np.array([np.cos(np.radians(c.dec.deg)) * np.cos(np.radians(c.ra.deg)),
                     np.cos(np.radians(c.dec.deg)) * np.sin(np.radians(c.ra.deg)),
                     np.sin(np.radians(c.dec.deg))])


class Step53DirectionalSample:
    """Step 53: Compile the directional Cepheid-TRGB matched sample."""

    def __init__(self):
        self.root = PROJECT_ROOT
        self.data_raw_external = self.root / "data" / "raw" / "external"
        self.data_processed = self.root / "data" / "processed"
        self.results = self.root / "results" / "outputs"
        self.logs = self.root / "logs"

        for d in [self.data_processed, self.results, self.logs]:
            d.mkdir(parents=True, exist_ok=True)

        self.logger = TEPLogger(
            "step_53",
            log_file_path=self.logs / "step_53_directional_sample.log",
        )
        set_step_logger(self.logger)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def load_cf4(self):
        """Load CF4 table2 and extract galaxies with both Cepheid and TRGB."""
        print_status("Loading CosmicFlows-4 table2...", "PROCESS")
        path = self.data_raw_external / "cf4_table2.dat"
        if not path.exists():
            print_status(f"CF4 table2 not found at {path}", "ERROR")
            return pd.DataFrame()

        colspecs = [
            (0, 7),    # PGC
            (8, 15),   # 1PGC
            (22, 27),  # Vcmb
            (28, 34),  # DM
            (35, 40),  # e_DM
            (102, 107),# DMtrgb
            (108, 112),# e_DMtrgb
            (113, 119),# DMceph
            (120, 125),# e_DMceph
            (137, 145),# RAdeg
            (146, 154),# DEdeg
            (155, 163),# GLON
            (164, 172),# GLAT
        ]
        names = ["PGC", "1PGC", "Vcmb", "DM", "e_DM",
                 "DMtrgb", "e_DMtrgb", "DMceph", "e_DMceph",
                 "RAdeg", "DEdeg", "GLON", "GLAT"]
        df = pd.read_fwf(path, colspecs=colspecs, names=names, header=None)
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["PGC"] = df["PGC"].astype("Int64")

        both = df[df["DMceph"].notna() & df["DMtrgb"].notna()].copy()
        both["delta_mu"] = both["DMceph"] - both["DMtrgb"]
        both["delta_mu_err"] = np.sqrt(both["e_DMceph"] ** 2 + both["e_DMtrgb"] ** 2)
        both["cep_mu"] = both["DMceph"]
        both["cep_mu_err"] = both["e_DMceph"]
        both["trgb_mu"] = both["DMtrgb"]
        both["trgb_mu_err"] = both["e_DMtrgb"]
        both["cep_source"] = "CF4"
        both["trgb_source"] = "CF4"
        both["sample"] = "CF4"

        print_status(f"  {len(both)} CF4 galaxies with both Cepheid and TRGB", "SUCCESS")
        return both

    def load_jwst(self):
        """Load the JWST matched Cepheid/TRGB sample (Step 50 input)."""
        print_status("Loading JWST matched Cepheid/TRGB catalog...", "PROCESS")
        path = self.data_raw_external / "jwst_trgb_distances.csv"
        if not path.exists():
            print_status(f"JWST TRGB catalog not found at {path}", "WARNING")
            return pd.DataFrame()

        df = pd.read_csv(path, comment="#")
        df["pgc"] = pd.to_numeric(df["pgc"], errors="coerce").astype("Int64")
        for c in ["jwst_trgb_mu", "jwst_trgb_mu_err", "cep_mu", "cep_mu_err"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.rename(columns={"jwst_trgb_mu": "trgb_mu",
                                "jwst_trgb_mu_err": "trgb_mu_err"})
        df["delta_mu"] = df["cep_mu"] - df["trgb_mu"]
        df["delta_mu_err"] = np.sqrt(df["cep_mu_err"] ** 2 + df["trgb_mu_err"] ** 2)
        df["cep_source"] = "R22_HST"
        df["trgb_source"] = df.get("trgb_source", "JWST")
        df["sample"] = "JWST"
        df["PGC"] = df["pgc"]

        print_status(f"  {len(df)} JWST matched galaxies", "SUCCESS")
        return df

    def load_r22_cepheid(self):
        """Load R22 Cepheid distances for cross-matching with CF4 TRGB."""
        print_status("Loading R22 Cepheid distances...", "PROCESS")
        path = self.data_raw_external / "r22_cepheid_distances.csv"
        if not path.exists():
            print_status(f"R22 catalog not found at {path}", "WARNING")
            return pd.DataFrame()

        r22 = pd.read_csv(path)
        r22 = r22.rename(columns={"value": "cep_mu", "error": "cep_mu_err",
                                   "source_id": "host_id"})

        # Map source_id → PGC via hosts_properties + TEP-H0 hosts
        hosts = pd.read_csv(self.data_raw_external / "hosts_properties.csv")
        tep_h0_hosts = self.root.parent / "TEP-H0" / "data" / "processed" / "hosts_processed.csv"
        if tep_h0_hosts.exists():
            h0 = pd.read_csv(tep_h0_hosts)
            def norm(s):
                return str(s).strip().upper().replace(" ", "").replace("NGC", "N")
            h0["norm"] = h0["normalized_name"].apply(norm)
            hosts["norm"] = hosts["normalized_name"].apply(norm)
            hosts = hosts.merge(h0[["pgc", "norm"]], on="norm", how="left")

        r22 = r22.merge(hosts[["source_id", "pgc", "ra", "dec"]],
                        left_on="host_id", right_on="source_id", how="left")
        r22 = r22.dropna(subset=["pgc"])
        r22["PGC"] = r22["pgc"].astype(int)
        r22["cep_source"] = "R22_HST"
        r22["RAdeg"] = r22["ra"]
        r22["DEdeg"] = r22["dec"]

        print_status(f"  {len(r22)} R22 Cepheid galaxies with PGC", "SUCCESS")
        return r22

    def download_jacobs_trgb(self):
        """Download Jacobs et al. (2009) TRGB catalog from Vizier.

        Returns an empty DataFrame if the download fails (graceful
        degradation — the CF4 TRGB sample is the primary source).
        """
        print_status("Downloading Jacobs et al. (2009) TRGB catalog from Vizier...", "PROCESS")
        local_path = self.data_raw_external / "jacobs2009_trgb.csv"

        # Use cached file if it exists
        if local_path.exists():
            df = pd.read_csv(local_path)
            if len(df) > 0:
                print_status(f"  Using cached Jacobs catalog: {len(df)} galaxies", "SUCCESS")
                return df

        try:
            from astroquery.vizier import Vizier
            Vizier.ROW_LIMIT = -1
            v = Vizier(columns=["PGC", "Name", "DM", "b_DM", "_RA", "_DE"])
            result = v.query_constraints(catalog="J/AJ/138/332/table1")
            if result and "J/AJ/138/332/table1" in result.keys():
                table = result["J/AJ/138/332/table1"]
                df = table.to_pandas()
                df.to_csv(local_path, index=False)
                print_status(f"  Downloaded {len(df)} Jacobs TRGB distances", "SUCCESS")
                return df
        except Exception as e:
            print_status(f"  Vizier download failed: {e}", "WARNING")

        print_status("  Proceeding without Jacobs catalog (CF4 TRGB is primary)", "WARNING")
        return pd.DataFrame()

    def load_key_project_cepheid(self):
        """Load Key Project Cepheid distances for additional overlaps."""
        print_status("Loading Key Project Cepheid distances...", "PROCESS")
        path = self.data_raw_external / "key_project_cepheid_distances.csv"
        if not path.exists():
            print_status(f"Key Project catalog not found at {path}", "WARNING")
            return pd.DataFrame()

        kp = pd.read_csv(path)
        kp = kp.rename(columns={"mu_true": "cep_mu", "mu_true_err": "cep_mu_err"})

        # Map galaxy name → PGC via TEP-H0 hosts
        tep_h0_hosts = self.root.parent / "TEP-H0" / "data" / "processed" / "hosts_processed.csv"
        if tep_h0_hosts.exists():
            h0 = pd.read_csv(tep_h0_hosts)
            def norm(s):
                return str(s).strip().upper().replace(" ", "").replace("NGC", "N")
            h0["norm"] = h0["normalized_name"].apply(norm)
            kp["norm"] = kp["galaxy"].apply(norm)
            kp = kp.merge(h0[["pgc", "norm"]], on="norm", how="left")
            kp = kp.dropna(subset=["pgc"])
            kp["PGC"] = kp["pgc"].astype(int)
            kp["cep_source"] = "KeyProject"

        print_status(f"  {len(kp)} Key Project Cepheid galaxies with PGC", "SUCCESS")
        return kp

    # ------------------------------------------------------------------
    # Cross-matching and compilation
    # ------------------------------------------------------------------
    def compile_sample(self, cf4, jwst, r22, jacobs, kp):
        """Cross-match all sources and build the unified directional sample."""
        print_status("Compiling unified directional Cepheid-TRGB sample...", "PROCESS")

        records = []
        seen_pgcs = set()

        # --- Source 1: CF4 (22 galaxies with both) ---
        for _, row in cf4.iterrows():
            pgc = int(row["PGC"])
            rec = {
                "pgc": pgc,
                "galaxy_name": f"PGC {pgc}",
                "cep_mu": float(row["cep_mu"]),
                "cep_mu_err": float(row["cep_mu_err"]),
                "trgb_mu": float(row["trgb_mu"]),
                "trgb_mu_err": float(row["trgb_mu_err"]),
                "delta_mu": float(row["delta_mu"]),
                "delta_mu_err": float(row["delta_mu_err"]),
                "ra": float(row["RAdeg"]),
                "dec": float(row["DEdeg"]),
                "v_cmb": float(row["Vcmb"]),
                "cep_source": "CF4_registered",
                "trgb_source": "CF4_registered",
                "sample": "CF4",
                "r22_matched": False,
            }
            records.append(rec)
            seen_pgcs.add(pgc)

        # --- Source 2: JWST matched (18 galaxies) ---
        for _, row in jwst.iterrows():
            pgc = int(row["PGC"]) if pd.notna(row.get("PGC")) else None
            if pgc is not None and pgc in seen_pgcs:
                # Update TRGB to JWST if it's a better measurement
                continue  # CF4 already has this galaxy
            rec = {
                "pgc": pgc,
                "galaxy_name": str(row.get("galaxy", f"PGC {pgc}")),
                "cep_mu": float(row["cep_mu"]),
                "cep_mu_err": float(row["cep_mu_err"]),
                "trgb_mu": float(row["trgb_mu"]),
                "trgb_mu_err": float(row["trgb_mu_err"]),
                "delta_mu": float(row["delta_mu"]),
                "delta_mu_err": float(row["delta_mu_err"]),
                "ra": None,
                "dec": None,
                "v_cmb": None,
                "cep_source": "R22_HST",
                "trgb_source": str(row.get("trgb_source", "JWST")),
                "sample": "JWST",
                "r22_matched": True,
            }
            records.append(rec)
            if pgc is not None:
                seen_pgcs.add(pgc)

        # --- Source 3: R22 Cepheid × CF4 TRGB (by PGC) ---
        cf4_trgb = self._cf4_trgb_lookup()
        for _, row in r22.iterrows():
            pgc = int(row["PGC"])
            if pgc in seen_pgcs:
                continue
            if pgc in cf4_trgb:
                trgb_info = cf4_trgb[pgc]
                rec = {
                    "pgc": pgc,
                    "galaxy_name": str(row.get("host_id", f"PGC {pgc}")),
                    "cep_mu": float(row["cep_mu"]),
                    "cep_mu_err": float(row["cep_mu_err"]),
                    "trgb_mu": trgb_info["DMtrgb"],
                    "trgb_mu_err": trgb_info["e_DMtrgb"],
                    "delta_mu": float(row["cep_mu"]) - trgb_info["DMtrgb"],
                    "delta_mu_err": np.sqrt(float(row["cep_mu_err"]) ** 2 + trgb_info["e_DMtrgb"] ** 2),
                    "ra": trgb_info["RAdeg"],
                    "dec": trgb_info["DEdeg"],
                    "v_cmb": trgb_info["Vcmb"],
                    "cep_source": "R22_HST",
                    "trgb_source": "CF4_registered",
                    "sample": "R22_x_CF4TRGB",
                    "r22_matched": True,
                }
                records.append(rec)
                seen_pgcs.add(pgc)

        # --- Source 4: Key Project Cepheid × CF4 TRGB (by PGC) ---
        for _, row in kp.iterrows():
            pgc = int(row["PGC"])
            if pgc in seen_pgcs:
                continue
            if pgc in cf4_trgb:
                trgb_info = cf4_trgb[pgc]
                rec = {
                    "pgc": pgc,
                    "galaxy_name": str(row.get("galaxy", f"PGC {pgc}")),
                    "cep_mu": float(row["cep_mu"]),
                    "cep_mu_err": float(row["cep_mu_err"]),
                    "trgb_mu": trgb_info["DMtrgb"],
                    "trgb_mu_err": trgb_info["e_DMtrgb"],
                    "delta_mu": float(row["cep_mu"]) - trgb_info["DMtrgb"],
                    "delta_mu_err": np.sqrt(float(row["cep_mu_err"]) ** 2 + trgb_info["e_DMtrgb"] ** 2),
                    "ra": trgb_info["RAdeg"],
                    "dec": trgb_info["DEdeg"],
                    "v_cmb": trgb_info["Vcmb"],
                    "cep_source": "KeyProject",
                    "trgb_source": "CF4_registered",
                    "sample": "KP_x_CF4TRGB",
                    "r22_matched": False,
                }
                records.append(rec)
                seen_pgcs.add(pgc)

        df = pd.DataFrame(records)
        print_status(f"  Compiled {len(df)} unique galaxies with both Cepheid and TRGB", "SUCCESS")

        # Fill in missing coordinates from TEP-H0 hosts or manual catalog
        df = self._fill_coordinates(df)

        # Fill in missing V_cmb from CF4
        df = self._fill_vcmb(df)

        return df

    def _cf4_trgb_lookup(self):
        """Build a PGC → TRGB info lookup from CF4 table2."""
        path = self.data_raw_external / "cf4_table2.dat"
        if not path.exists():
            return {}
        colspecs = [
            (0, 7), (22, 27), (102, 107), (108, 112),
            (137, 145), (146, 154),
        ]
        names = ["PGC", "Vcmb", "DMtrgb", "e_DMtrgb", "RAdeg", "DEdeg"]
        df = pd.read_fwf(path, colspecs=colspecs, names=names, header=None)
        df = df[df["DMtrgb"].notna()]
        lookup = {}
        for _, row in df.iterrows():
            pgc = int(row["PGC"])
            lookup[pgc] = {
                "DMtrgb": float(row["DMtrgb"]),
                "e_DMtrgb": float(row["e_DMtrgb"]),
                "RAdeg": float(row["RAdeg"]),
                "DEdeg": float(row["DEdeg"]),
                "Vcmb": float(row["Vcmb"]),
            }
        return lookup

    def _fill_coordinates(self, df):
        """Fill missing RA/Dec from TEP-H0 hosts_processed or hosts_properties."""
        missing = df["ra"].isna()
        if not missing.any():
            return df

        print_status(f"  Filling coordinates for {missing.sum()} galaxies...", "PROCESS")

        # TEP-H0 hosts
        tep_h0_hosts = self.root.parent / "TEP-H0" / "data" / "processed" / "hosts_processed.csv"
        if tep_h0_hosts.exists():
            h0 = pd.read_csv(tep_h0_hosts)
            pgc_to_coord = {int(r["pgc"]): (r["ra"], r["dec"]) for _, r in h0.iterrows()
                           if pd.notna(r.get("pgc")) and pd.notna(r.get("ra"))}
            for idx in df[missing].index:
                pgc = df.loc[idx, "pgc"]
                if pd.notna(pgc) and int(pgc) in pgc_to_coord:
                    df.loc[idx, "ra"] = pgc_to_coord[int(pgc)][0]
                    df.loc[idx, "dec"] = pgc_to_coord[int(pgc)][1]

        # hosts_properties as fallback
        still_missing = df["ra"].isna()
        if still_missing.any():
            hosts_path = self.data_raw_external / "hosts_properties.csv"
            if hosts_path.exists():
                hosts = pd.read_csv(hosts_path)
                hosts = hosts.dropna(subset=["ra"])
                def norm(s):
                    return str(s).strip().upper().replace(" ", "").replace("NGC", "N")
                hosts["norm"] = hosts["normalized_name"].apply(norm)
                name_to_coord = {r["norm"]: (r["ra"], r["dec"]) for _, r in hosts.iterrows()}
                for idx in df[still_missing].index:
                    name = str(df.loc[idx, "galaxy_name"])
                    key = norm(name)
                    if key in name_to_coord:
                        df.loc[idx, "ra"] = name_to_coord[key][0]
                        df.loc[idx, "dec"] = name_to_coord[key][1]

        # Manual coordinates for well-known galaxies (final fallback)
        manual_coords = {
            "M101": (210.80, 54.35), "NGC1365": (53.40, -36.14),
            "NGC2442": (114.10, -69.53), "NGC3972": (179.50, 55.30),
            "NGC4038": (180.47, -18.87), "NGC4424": (186.83, 9.40),
            "NGC4536": (188.63, 2.19), "NGC4639": (190.70, 13.35),
            "NGC5643": (218.17, -44.18), "NGC7250": (334.60, 40.62),
            "NGC1448": (56.09, -44.64), "NGC1559": (64.40, -62.78),
            "NGC2525": (121.40, -11.40), "NGC3370": (161.80, 17.28),
            "NGC3447": (163.30, 16.80), "NGC5584": (215.60, -0.40),
            "NGC5861": (227.30, -11.30), "NGC4258": (184.74, 47.30),
            "NGC1309": (50.53, -15.40), "NGC3021": (147.35, 33.55),
            "NGC5917": (230.26, -16.63), "NGC1015": (39.55, -1.32),
        }
        still_missing = df["ra"].isna()
        for idx in df[still_missing].index:
            name = str(df.loc[idx, "galaxy_name"])
            for key, (ra, dec) in manual_coords.items():
                if key.lower() in name.lower():
                    df.loc[idx, "ra"] = ra
                    df.loc[idx, "dec"] = dec
                    break

        n_filled = df["ra"].notna().sum()
        n_total = len(df)
        print_status(f"  {n_filled}/{n_total} galaxies with coordinates", "SUCCESS")
        return df

    def _fill_vcmb(self, df):
        """Fill missing V_cmb from CF4 table2 by PGC."""
        missing = df["v_cmb"].isna()
        if not missing.any():
            return df

        cf4_trgb = self._cf4_trgb_lookup()
        for idx in df[missing].index:
            pgc = df.loc[idx, "pgc"]
            if pd.notna(pgc) and int(pgc) in cf4_trgb:
                df.loc[idx, "v_cmb"] = cf4_trgb[int(pgc)]["Vcmb"]

        return df

    # ------------------------------------------------------------------
    # TEP coordinate computation
    # ------------------------------------------------------------------
    def add_tep_coordinates(self, df):
        """Compute X_i (screened TEP potential coordinate) and CMB dipole projection."""
        print_status("Computing TEP potential coordinate X_i...", "PROCESS")

        # Load host potential catalog for V_rot / sigma
        pot_path = self.data_processed / "host_potential_catalog.csv"
        if pot_path.exists():
            pot = pd.read_csv(pot_path)
            pot["galaxy_norm"] = pot["galaxy"].str.strip().str.replace(r"\s+", " ", regex=True)
        else:
            pot = pd.DataFrame()

        # Also try TEP-H0 hosts for sigma_inferred
        tep_h0_hosts = self.root.parent / "TEP-H0" / "data" / "processed" / "hosts_processed.csv"
        h0_sigma = {}
        if tep_h0_hosts.exists():
            h0 = pd.read_csv(tep_h0_hosts)
            for _, r in h0.iterrows():
                if pd.notna(r.get("pgc")) and pd.notna(r.get("sigma_inferred")):
                    h0_sigma[int(r["pgc"])] = float(r["sigma_inferred"])

        # Match sigma/V_rot for each galaxy
        sigmas = []
        sigma_errs = []
        for _, row in df.iterrows():
            sigma = None
            sigma_err = None

            # Try by PGC from TEP-H0
            pgc = row.get("pgc")
            if pd.notna(pgc) and int(pgc) in h0_sigma:
                sigma = h0_sigma[int(pgc)]
                sigma_err = sigma * 0.10  # 10% default

            # Try by name from host_potential_catalog
            if sigma is None and len(pot) > 0:
                name = str(row.get("galaxy_name", "")).strip()
                name_norm = name.replace("  ", " ")
                match = pot[pot["galaxy_norm"] == name_norm]
                if len(match) > 0:
                    sigma = float(match.iloc[0]["sigma_kms"])
                    sigma_err = float(match.iloc[0]["error_kms"])

            # Try CF4 vrot catalog
            if sigma is None:
                vrot_path = self.data_raw_external / "cf4_matched_galaxies_vrot.csv"
                if vrot_path.exists():
                    vrot = pd.read_csv(vrot_path, comment="#")
                    if pd.notna(pgc):
                        match = vrot[vrot["pgc"] == int(pgc)]
                        if len(match) > 0:
                            v = float(match.iloc[0]["vrot_kms"])
                            sigma = v / np.sqrt(2)
                            sigma_err = float(match.iloc[0].get("vrot_error_kms", v * 0.10)) / np.sqrt(2)

            sigmas.append(sigma)
            sigma_errs.append(sigma_err)

        df["sigma_kms"] = sigmas
        df["sigma_err_kms"] = sigma_errs

        # Compute U_i = sigma^2 = (V_rot/sqrt(2))^2
        df["U_i"] = df["sigma_kms"] ** 2

        # Compute screening
        pgc_values = df["pgc"].fillna(0).astype(int).values
        df["S_total"] = compute_screening(pgc_values, self.root)

        # Screened X_i
        df["X_i"] = (df["S_total"] * df["U_i"] - U_REF_SCREENED) / C_KMS ** 2
        df["X_i_unscreened"] = (df["U_i"] - U_REF_UNSCREENED) / C_KMS ** 2

        # X_i uncertainty: dX_i = S_total * dU_i / c^2
        # (S_total attenuates the potential, so it also attenuates the
        # uncertainty on the screened coordinate)
        df["U_i_err"] = 2 * df["sigma_kms"] * df["sigma_err_kms"]
        df["X_i_err"] = df["S_total"] * df["U_i_err"] / C_KMS ** 2

        n_with_xi = df["X_i"].notna().sum()
        print_status(f"  {n_with_xi}/{len(df)} galaxies with X_i", "SUCCESS")

        # CMB dipole projection
        print_status("Computing CMB dipole directional projection...", "PROCESS")
        cmb_vec = cmb_dipole_unit_vector()

        valid = df["ra"].notna() & df["dec"].notna()
        df["cmb_dot"] = np.nan
        if valid.any():
            gal_vecs = ra_dec_to_unit_vectors(df.loc[valid, "ra"].values,
                                              df.loc[valid, "dec"].values)
            df.loc[valid, "cmb_dot"] = gal_vecs @ cmb_vec

        # Also compute angular separation from CMB dipole
        df["cmb_sep_deg"] = np.nan
        if valid.any():
            coords = SkyCoord(ra=df.loc[valid, "ra"].values * u.deg,
                              dec=df.loc[valid, "dec"].values * u.deg,
                              frame="icrs")
            cmb_coord = SkyCoord(l=CMB_DIPOLE_GAL_L * u.deg,
                                 b=CMB_DIPOLE_GAL_B * u.deg,
                                 frame="galactic").icrs
            df.loc[valid, "cmb_sep_deg"] = coords.separation(cmb_coord).deg

        n_with_dir = df["cmb_dot"].notna().sum()
        print_status(f"  {n_with_dir}/{len(df)} galaxies with CMB dipole projection", "SUCCESS")

        return df

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------
    def run(self):
        print_status("Step 53: Directional Cepheid-TRGB Sample Compilation", "TITLE")

        # Load all data sources
        cf4 = self.load_cf4()
        jwst = self.load_jwst()
        r22 = self.load_r22_cepheid()
        jacobs = self.download_jacobs_trgb()
        kp = self.load_key_project_cepheid()

        # Cross-match and compile
        df = self.compile_sample(cf4, jwst, r22, jacobs, kp)

        # Add TEP coordinates and CMB dipole projection
        df = self.add_tep_coordinates(df)

        # Save
        output_csv = self.data_processed / "directional_ceph_trgb_sample.csv"
        df.to_csv(output_csv, index=False)
        print_status(f"Saved {len(df)} galaxies to {output_csv}", "SUCCESS")

        # Summary statistics
        n_total = len(df)
        n_with_coords = df["ra"].notna().sum()
        n_with_xi = df["X_i"].notna().sum()
        n_with_dir = df["cmb_dot"].notna().sum()
        n_with_all = (df["ra"].notna() & df["X_i"].notna() & df["cmb_dot"].notna()).sum()

        n_toward = (df["cmb_dot"] > 0).sum()
        n_away = (df["cmb_dot"] <= 0).sum()

        by_sample = df.groupby("sample").size().to_dict()
        by_cep_source = df.groupby("cep_source").size().to_dict()
        by_trgb_source = df.groupby("trgb_source").size().to_dict()

        summary = {
            "step": "53_directional_sample",
            "description": "Compiled Cepheid-TRGB matched sample for directional Δμ test",
            "n_total": int(n_total),
            "n_with_coordinates": int(n_with_coords),
            "n_with_xi": int(n_with_xi),
            "n_with_cmb_dot": int(n_with_dir),
            "n_with_all_coords": int(n_with_all),
            "n_toward_cmb": int(n_toward),
            "n_away_from_cmb": int(n_away),
            "mean_cmb_dot": float(df["cmb_dot"].mean()) if n_with_dir > 0 else None,
            "by_sample": {k: int(v) for k, v in by_sample.items()},
            "by_cep_source": {k: int(v) for k, v in by_cep_source.items()},
            "by_trgb_source": {k: int(v) for k, v in by_trgb_source.items()},
            "mean_delta_mu": float(df["delta_mu"].mean()),
            "median_delta_mu": float(df["delta_mu"].median()),
            "std_delta_mu": float(df["delta_mu"].std()),
            "sem_delta_mu": float(df["delta_mu"].std() / np.sqrt(n_total)),
            "data_sources": {
                "cf4_both": int(len(cf4)),
                "jwst_matched": int(len(jwst)),
                "r22_cepheid": int(len(r22)),
                "jacobs_trgb": int(len(jacobs)),
                "key_project": int(len(kp)),
            },
            "output_file": str(output_csv),
            "cmb_dipole": {
                "galactic_l": CMB_DIPOLE_GAL_L,
                "galactic_b": CMB_DIPOLE_GAL_B,
            },
        }

        output_json = self.results / "step_53_directional_sample.json"
        with open(output_json, "w") as f:
            json.dump(summary, f, indent=2)
        print_status(f"Summary saved to {output_json}", "SUCCESS")

        # Print summary
        print_status(f"\nSample summary:", "INFO")
        print_status(f"  Total galaxies: {n_total}", "INFO")
        print_status(f"  With coordinates: {n_with_coords}", "INFO")
        print_status(f"  With X_i: {n_with_xi}", "INFO")
        print_status(f"  With CMB dipole projection: {n_with_dir}", "INFO")
        print_status(f"  With all coordinates: {n_with_all}", "INFO")
        print_status(f"  Toward CMB dipole: {n_toward}, Away: {n_away}", "INFO")
        print_status(f"  Mean Δμ = {df['delta_mu'].mean():.3f} ± {df['delta_mu'].std()/np.sqrt(n_total):.3f} mag", "INFO")
        print_status(f"  Mean cmb_dot = {df['cmb_dot'].mean():.3f}", "INFO")


if __name__ == "__main__":
    step = Step53DirectionalSample()
    step.run()
