"""Shared TEP screening-factor utility.

Provides a single source of truth for the TEP environmental screening factor
``S_total`` and the screened anchor reference ``U_ref_screened``, so that every
TEP-VOID step uses the same physical coordinate

    X_i = (S_total * U_i - U_ref_screened) / c^2

matching the TEP-H0 canonical definition
(``tep_correction.tep_environment_coordinate``).

``S_total`` is looked up in priority order:
1.  TEP-H0 Step 03 ``shear_suppression`` (full S_local * S_group).
2.  Tully 2015 2MRS group richness → S_group = 1 / (1 + (N_mb / N_crit)^gamma).
3.  Anchor special N_mb values (M31, NGC 4258, LMC).
4.  S_total = 1.0 for isolated / uncatalogued galaxies.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ── TEP-H0 screening parameters (must match tep_correction.py) ─────────────
N_CRIT = 10.0
GAMMA = 1.2

# Anchor PGC → N_mb mapping (from TEP-H0 tep_correction.ANCHOR_NMB)
ANCHOR_NMB: dict[int, int] = {
    2557: 11,    # M31
    39600: 65,   # NGC 4258
    17223: 2,    # LMC
}

# Screened anchor reference (from TEP-H0 compute_anchor_sigma_ref(screened=True))
SIGMA_REF_SCREENED = 30.507  # km/s
U_REF_SCREENED = SIGMA_REF_SCREENED ** 2  # ≈ 930.7 (km/s)^2

# Unscreened anchor reference (for diagnostics / sensitivity checks)
SIGMA_REF_UNSCREENED = 87.165  # km/s
U_REF_UNSCREENED = SIGMA_REF_UNSCREENED ** 2  # ≈ 7597.7 (km/s)^2

# Speed of light in km/s
C_KMS = 299792.458


def group_screening_factor(n_mb: float) -> float:
    """Group-richness screening factor S_group = 1 / (1 + (N_mb / N_crit)^gamma)."""
    return 1.0 / (1.0 + (float(n_mb) / N_CRIT) ** GAMMA)


def load_screening_maps(project_root: Path | None = None) -> tuple[dict[int, float], dict[int, float]]:
    """Load TEP-H0 Step 03 screening and Tully group richness.

    Returns ``(known_screening, tully_nmb)`` where ``known_screening`` maps
    PGC → S_total and ``tully_nmb`` maps PGC → N_mb.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]

    tep_h0_root = project_root.parent / "TEP-H0"

    # TEP-H0 Step 03 full screening (S_local * S_group)
    known_screening: dict[int, float] = {}
    strat_path = tep_h0_root / "results" / "outputs" / "step_03_stratified_h0.csv"
    if strat_path.exists():
        strat = pd.read_csv(strat_path)
        for _, row in strat.iterrows():
            pgc = row.get("pgc", None)
            S = row.get("shear_suppression", None)
            if pd.notna(pgc) and pd.notna(S):
                known_screening[int(pgc)] = float(S)

    # Tully 2015 group richness
    tully_nmb: dict[int, float] = {}
    tully_path = tep_h0_root / "data" / "raw" / "external" / "tully2015_2mrs_groups_table5.csv"
    if tully_path.exists():
        tully = pd.read_csv(tully_path)
        for _, row in tully.iterrows():
            tully_nmb[int(row["PGC"])] = float(row["Nmb"])

    return known_screening, tully_nmb


def compute_screening(
    pgc_values: np.ndarray | pd.Series | list,
    project_root: Path | None = None,
) -> np.ndarray:
    """Compute S_total for an array of PGC identifiers.

    Parameters
    ----------
    pgc_values : array-like
        PGC identifiers for each galaxy.  NaN / 0 / negative values are
        treated as uncatalogued (S_total = 1.0).
    project_root : Path, optional
        TEP-VOID project root.  Defaults to the parent of ``scripts/utils/``.

    Returns
    -------
    np.ndarray
        Screening factors S_total in [0, 1] for each input PGC.
    """
    known_screening, tully_nmb = load_screening_maps(project_root)

    screening = np.ones(len(pgc_values), dtype=float)
    for i, pgc in enumerate(pgc_values):
        if pgc is None or (isinstance(pgc, float) and np.isnan(pgc)):
            continue
        pgc_int = int(pgc)
        if pgc_int <= 0:
            continue

        # 1. TEP-H0 Step 03 full screening
        if pgc_int in known_screening:
            screening[i] = known_screening[pgc_int]
            continue

        # 2. Tully group richness
        if pgc_int in tully_nmb:
            screening[i] = group_screening_factor(tully_nmb[pgc_int])
            continue

        # 3. Anchor special values
        if pgc_int in ANCHOR_NMB:
            screening[i] = group_screening_factor(ANCHOR_NMB[pgc_int])
            continue

        # 4. Isolated / uncatalogued → S = 1.0 (already set)

    return screening


def screened_xi(U_i: np.ndarray | pd.Series, pgc_values: np.ndarray | pd.Series | list,
                project_root: Path | None = None) -> np.ndarray:
    """Compute the screened TEP potential coordinate X_i.

    X_i = (S_total * U_i - U_ref_screened) / c^2

    Parameters
    ----------
    U_i : array-like
        Host potential proxy (km/s)^2, typically u_phi^2 = (V_rot / sqrt(2))^2.
    pgc_values : array-like
        PGC identifiers for screening lookup.
    project_root : Path, optional
        TEP-VOID project root.

    Returns
    -------
    np.ndarray
        Dimensionless screened TEP coordinate X_i.
    """
    S_total = compute_screening(pgc_values, project_root)
    return (S_total * np.asarray(U_i, dtype=float) - U_REF_SCREENED) / C_KMS ** 2


def load_name_to_pgc_map(project_root: Path | None = None) -> dict[str, int]:
    """Build a canonical-name → PGC mapping from TEP-H0 hosts_processed.csv."""
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]
    tep_h0_root = project_root.parent / "TEP-H0"
    hosts_path = tep_h0_root / "data" / "processed" / "hosts_processed.csv"

    name_map: dict[str, int] = {}
    if hosts_path.exists():
        df = pd.read_csv(hosts_path)
        for _, row in df.iterrows():
            name = str(row["normalized_name"]).strip().upper().replace(" ", "")
            pgc = row.get("pgc", None)
            if pd.notna(pgc):
                name_map[name] = int(pgc)
    return name_map


def compute_screening_by_name(
    names: np.ndarray | pd.Series | list,
    project_root: Path | None = None,
) -> np.ndarray:
    """Compute S_total for an array of galaxy names.

    Looks up PGC by normalized name from TEP-H0 hosts_processed.csv,
    then applies the same priority chain as :func:`compute_screening`.
    """
    name_map = load_name_to_pgc_map(project_root)
    pgc_values = []
    for name in names:
        key = str(name).strip().upper().replace(" ", "")
        pgc_values.append(name_map.get(key, 0))
    return compute_screening(pgc_values, project_root)


__all__ = [
    "N_CRIT",
    "GAMMA",
    "ANCHOR_NMB",
    "SIGMA_REF_SCREENED",
    "U_REF_SCREENED",
    "SIGMA_REF_UNSCREENED",
    "U_REF_UNSCREENED",
    "C_KMS",
    "group_screening_factor",
    "load_screening_maps",
    "compute_screening",
    "screened_xi",
    "load_name_to_pgc_map",
    "compute_screening_by_name",
]
