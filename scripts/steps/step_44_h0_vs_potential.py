#!/usr/bin/env python3
"""
Step 44: H0 vs Calibrator-Population Potential Depth
=====================================================
Generates a figure plotting H0 values from Table 1 (distance indicators)
against the mean potential coordinate <X> for each ladder's calibrator
population. The monotonic ordering (JAGB < TRGB < Cepheids < SBF) is
predicted by the two-channel TEP model.

The potential coordinate for each ladder is estimated from the typical
host-galaxy properties of each indicator's calibrator population:
  - JAGB: low-mass hosts (dwarf ellipticals, LMC/SMC-like)
  - TRGB: mixed hosts (intermediate spirals + dwarf ellipticals)
  - Cepheids: massive spiral hosts (SH0ES calibrator sample)
  - SBF: massive early-type hosts (E/S0 galaxies in Virgo/Fornax)

The <X> values are computed from representative V_rot values for each
population, using the same U_ref = (87.165 km/s)^2 anchor as the
TEP framework.
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status


def main():
    logger = TEPLogger(
        "step_44",
        log_file_path=PROJECT_ROOT / "logs" / "step_44_h0_vs_potential.log",
    )
    set_step_logger(logger)

    print_status("Step 44: H0 vs Calibrator-Population Potential", "INFO")

    # H0 values from Table 1 (Section 2.4)
    indicators = [
        {
            "name": "JAGB",
            "H0": 67.80,
            "H0_err": 2.17,
            "H0_sys": 1.64,
            "V_rot_typical": 80.0,   # km/s, low-mass hosts
            "V_rot_spread": 20.0,    # spread in population
            "color": "#2196F3",
            "marker": "s",
        },
        {
            "name": "TRGB",
            "H0": 68.81,
            "H0_err": 1.79,
            "H0_sys": 1.32,
            "V_rot_typical": 120.0,  # km/s, mixed hosts
            "V_rot_spread": 30.0,
            "color": "#4CAF50",
            "marker": "D",
        },
        {
            "name": "Cepheids",
            "H0": 73.04,
            "H0_err": 1.04,
            "H0_sys": 1.04,
            "V_rot_typical": 200.0,  # km/s, massive spirals (SH0ES)
            "V_rot_spread": 40.0,
            "color": "#FF5722",
            "marker": "o",
        },
        {
            "name": "SBF",
            "H0": 74.60,
            "H0_err": 0.90,
            "H0_sys": 2.70,
            "V_rot_typical": 250.0,  # km/s, massive early-types
            "V_rot_spread": 50.0,
            "color": "#9C27B0",
            "marker": "^",
        },
    ]

    # Compute <X> for each population
    # X_i = (U_i - U_ref) / c^2, where U_i = (V_rot/sqrt(2))^2
    # U_ref = (87.165 km/s)^2
    C_KMS = 299792.458
    U_ref = (87.165) ** 2  # km^2/s^2

    for ind in indicators:
        V_rot = ind["V_rot_typical"]
        V_spread = ind["V_rot_spread"]
        # Mean U for the population (using typical V_rot)
        U_typ = (V_rot / np.sqrt(2)) ** 2
        # Spread: sample from a distribution to get <X> and its uncertainty
        np.random.seed(42)
        V_samples = np.random.normal(V_rot, V_spread, 10000)
        V_samples = np.clip(V_samples, 10, 400)  # physical range
        U_samples = (V_samples / np.sqrt(2)) ** 2
        X_samples = (U_samples - U_ref) / C_KMS ** 2
        ind["X_mean"] = float(np.mean(X_samples))
        ind["X_std"] = float(np.std(X_samples) / np.sqrt(10000))  # SEM
        ind["U_typ"] = U_typ

    # Print
    print_status("")
    print_status("  Indicator    H0        <X> (1e-7)    U_typ (km^2/s^2)", "TEST")
    print_status("  " + "-" * 60, "TEST")
    for ind in indicators:
        print_status(
            f"  {ind['name']:12s} {ind['H0']:6.2f}   {ind['X_mean']*1e7:8.3f}      {ind['U_typ']:10.1f}",
            "TEST",
        )

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    for ind in indicators:
        ax.errorbar(
            ind["X_mean"] * 1e7,
            ind["H0"],
            xerr=ind["X_std"] * 1e7 * 3,  # 3sigma for visibility
            yerr=ind["H0_err"],
            fmt=ind["marker"],
            color=ind["color"],
            markersize=10,
            capsize=5,
            linewidth=1.5,
            label=f"{ind['name']} ($H_0 = {ind['H0']:.1f}$)",
            zorder=5,
        )

    # Fit a line
    X_vals = np.array([ind["X_mean"] * 1e7 for ind in indicators])
    H0_vals = np.array([ind["H0"] for ind in indicators])
    H0_errs = np.array([ind["H0_err"] for ind in indicators])
    # Weighted linear fit
    from numpy.polynomial import polynomial as P
    weights = 1.0 / H0_errs ** 2
    coeffs = np.polyfit(X_vals, H0_vals, 1, w=weights)
    slope, intercept = coeffs
    print_status(f"  Linear fit: H0 = {intercept:.2f} + {slope:.2f} * <X> (1e-7)", "TEST")

    X_fit = np.linspace(-0.5, max(X_vals) * 1.2, 100)
    H0_fit = slope * X_fit + intercept
    ax.plot(
        X_fit,
        H0_fit,
        "k--",
        alpha=0.5,
        linewidth=1.5,
        label=f"Linear fit ($H_0 = {intercept:.1f} + {slope:.1f} \\langle X \\rangle$)",
    )

    # Planck reference line
    ax.axhline(67.4, color="blue", linestyle=":", alpha=0.4, linewidth=1)
    ax.text(0.95, 67.6, "Planck ($67.4$)", color="blue", fontsize=9,
            ha="right", va="bottom", alpha=0.6)

    # SH0ES reference line
    ax.axhline(73.04, color="red", linestyle=":", alpha=0.4, linewidth=1)
    ax.text(0.95, 73.2, "SH0ES ($73.0$)", color="red", fontsize=9,
            ha="right", va="bottom", alpha=0.6)

    ax.set_xlabel(r"$\langle X \rangle$ ($\times 10^{-7}$)", fontsize=12)
    ax.set_ylabel(r"$H_0$ (km s$^{-1}$ Mpc$^{-1}$)", fontsize=12)
    ax.set_title(
        r"$H_0$ vs Calibrator-Population Potential Depth",
        fontsize=13,
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(-0.5, max(X_vals) * 1.3)
    ax.set_ylim(65, 77)
    ax.grid(True, alpha=0.3)

    # Add annotation
    ax.text(
        0.05,
        0.05,
        "Monotonic ordering predicted by\nthe two-channel TEP model\n(Section 9.7)",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )

    plt.tight_layout()

    figures_dir = PROJECT_ROOT / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figures_dir / "step_44_h0_vs_potential.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print_status(f"  Figure saved to {fig_path}", "SUCCESS")

    # Also save as PDF for manuscript
    fig_path_pdf = figures_dir / "step_44_h0_vs_potential.pdf"
    fig.savefig(fig_path_pdf, bbox_inches="tight")
    print_status(f"  PDF saved to {fig_path_pdf}", "SUCCESS")

    # Save summary
    summary = {
        "step": "44_h0_vs_potential",
        "description": (
            "H0 vs calibrator-population potential depth. The monotonic "
            "ordering JAGB < TRGB < Cepheids < SBF is predicted by the "
            "two-channel TEP model."
        ),
        "indicators": [
            {
                "name": ind["name"],
                "H0": ind["H0"],
                "H0_err": ind["H0_err"],
                "X_mean": ind["X_mean"],
                "X_std": ind["X_std"],
                "V_rot_typical": ind["V_rot_typical"],
            }
            for ind in indicators
        ],
        "linear_fit": {
            "slope": float(slope),
            "intercept": float(intercept),
            "description": "H0 = intercept + slope * <X> (in units of 1e-7)",
        },
        "interpretation": (
            "The monotonic ordering of H0 by calibrator-population "
            "potential depth is predicted by the two-channel TEP model: "
            "deeper-potential calibrator populations carry larger clock "
            "biases through both the Cepheid period-transport channel "
            "and the SN light-curve stretch channel."
        ),
    }

    summary_path = PROJECT_ROOT / "results" / "outputs" / "step_44_h0_vs_potential.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print_status(f"  Summary saved to {summary_path}", "SUCCESS")
    print_status("Step 44 complete", "SUCCESS")


if __name__ == "__main__":
    main()
