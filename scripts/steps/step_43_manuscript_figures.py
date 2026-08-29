#!/usr/bin/env python3
"""
Step 43: Manuscript Figures — Void vs TEP Composite
=====================================================
Generates the composite manuscript figure showing all four discriminating
observables that distinguish the KBC void model from the TEP framework.

Panels:
  (a) Indicator-Specific Distance Divergence: Cepheid vs TRGB DM — step_30
  (b) Peculiar Velocity Calibration Sensitivity — step_31
  (c) H0(z) Decay: void sharp boundary vs TEP smooth decay — step_32
  (d) Void Boundary Test: H0 in massive hosts at z > 0.3 — step_34

Outputs:
    results/outputs/step_43_manuscript_figures.json
    results/figures/step_43_manuscript_composite.png
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status


class Step43ManuscriptFigures:
    """Step 43: Generate manuscript figures from void-falsification results."""

    STEP_FILES = [
        "step_30_bulk_flow_recalculation.json",
        "step_31_peculiar_velocity_artifact.json",
        "step_32_redshift_decay_profile.json",
        "step_34_void_boundary_test.json",
        "step_40_redshift_shear_reconstruction.json",
        "step_42_falsification_summary.json",
    ]

    H0_CMB = 67.4
    H0_SH0ES = 73.0
    C_KMS = 299792.458

    def __init__(self):
        self.root = PROJECT_ROOT
        self.results = self.root / "results" / "outputs"
        self.figures = self.root / "results" / "figures"
        self.logs = self.root / "logs"

        for d in [self.results, self.figures, self.logs]:
            d.mkdir(parents=True, exist_ok=True)

        self.logger = TEPLogger(
            "step_43", log_file_path=self.logs / "step_43_manuscript_figures.log"
        )
        set_step_logger(self.logger)

    def load_step_results(self):
        """Load all previous step results."""
        print_status("Loading step results for figure generation...", "PROCESS")

        step_data = {}
        for step_file in self.STEP_FILES:
            path = self.results / step_file
            if path.exists():
                try:
                    with open(path, "r") as f:
                        data = json.load(f)
                    step_data[step_file.replace(".json", "")] = data
                    print_status(f"  Loaded {step_file}", "SUCCESS")
                except Exception as e:
                    print_status(f"  Error reading {step_file}: {e}", "WARNING")
            else:
                print_status(f"  {step_file} not found", "WARNING")

        return step_data

    def plot_panel_indicator_divergence(self, ax, step_data):
        """Panel (a): Indicator-specific distance divergence (Cepheid vs TRGB DM)."""
        ax.set_title("(a) Indicator Divergence: Cepheid vs TRGB", fontsize=12, fontweight="bold")

        s30 = step_data.get("step_30_bulk_flow_recalculation", {})
        test_a = s30.get("test_a_indicator_comparison", {})
        n_overlap = test_a.get("n_overlap", 0)

        if n_overlap >= 5:
            # Load the actual data for plotting
            df2_path = self.root / "data" / "interim" / "cf4_galaxies.csv"
            if df2_path.exists():
                df2 = pd.read_csv(df2_path)
                if "DMceph" in df2.columns and "DMtrgb" in df2.columns:
                    both = df2[df2["DMceph"].notna() & df2["DMtrgb"].notna()]
                    ax.scatter(both["DMtrgb"], both["DMceph"], s=60, alpha=0.7,
                               color="#2ca02c", edgecolors="black", linewidth=0.5)
                    lims = [min(both["DMtrgb"].min(), both["DMceph"].min()) - 0.5,
                            max(both["DMtrgb"].max(), both["DMceph"].max()) + 0.5]
                    ax.plot(lims, lims, "k--", alpha=0.5, label="1:1 (void)")
                    mean_delta = test_a.get("mean_delta_mu", 0)
                    ax.plot(lims, [l + mean_delta for l in lims], "r--", alpha=0.7,
                            label=f"TEP: $\\Delta\\mu$={mean_delta:.3f}")
                    ax.set_xlabel("$\\mu_{TRGB}$ (mag)", fontsize=11)
                    ax.set_ylabel("$\\mu_{Cep}$ (mag)", fontsize=11)
                    ax.set_aspect("equal")
                    sigma = test_a.get("significance_sigma", 0)
                    ax.text(0.05, 0.95, f"N={n_overlap}, {sigma:.2f}$\\sigma$",
                            transform=ax.transAxes, fontsize=9, va="top",
                            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
        else:
            ax.text(0.5, 0.5, "Data unavailable", transform=ax.transAxes,
                    ha="center", fontsize=11, color="gray")

        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.3)

    def plot_panel_calibration_sensitivity(self, ax, step_data):
        """Panel (b): Peculiar velocity calibration sensitivity."""
        ax.set_title("(b) Calibration Sensitivity", fontsize=12, fontweight="bold")

        s30 = step_data.get("step_30_bulk_flow_recalculation", {})
        s31 = step_data.get("step_31_peculiar_velocity_artifact", {})

        test_b = s30.get("test_b_bulk_flow_sensitivity", {})
        bf_cep = test_b.get("bulk_flow_cepheid", {})
        bf_trgb = test_b.get("bulk_flow_trgb", {})

        bins = sorted([int(b) for b in bf_cep.keys() if bf_cep.get(str(b), {}).get("v_bf") is not None])

        if bins:
            cep_v = [bf_cep.get(str(b), {}).get("v_bf", 0) for b in bins]
            cep_e = [bf_cep.get(str(b), {}).get("v_bf_err", 0) for b in bins]
            trgb_v = [bf_trgb.get(str(b), {}).get("v_bf", 0) for b in bins]
            trgb_e = [bf_trgb.get(str(b), {}).get("v_bf_err", 0) for b in bins]
            ax.errorbar(bins, cep_v, yerr=cep_e, fmt="o-", color="#d62728",
                        label=f"Cepheid ($H_0$=73.0)", capsize=4, markersize=7)
            ax.errorbar(bins, trgb_v, yerr=trgb_e, fmt="s-", color="#2ca02c",
                        label=f"TRGB ($H_0$=69.8)", capsize=4, markersize=7)

            mean_red = test_b.get("mean_reduction_pct", 0)
            ax.text(0.05, 0.95, f"{mean_red:.1f}% reduction",
                    transform=ax.transAxes, fontsize=9, va="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
        else:
            ax.text(0.5, 0.5, "Data unavailable", transform=ax.transAxes,
                    ha="center", fontsize=11, color="gray")

        ax.set_xlabel("$R_{max}$ (Mpc)", fontsize=11)
        ax.set_ylabel("$|v_{bf}|$ (km/s)", fontsize=11)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3)

    def plot_panel_h0_decay(self, ax, step_data):
        """Panel (c): H0(z) decay profile."""
        ax.set_title("(c) $H_0(z)$ Decay: Void vs TEP", fontsize=12, fontweight="bold")

        s32 = step_data.get("step_32_redshift_decay_profile", {})
        h0_z = s32.get("h0_z_data", {})
        mc = s32.get("model_comparison", {})

        if h0_z:
            z_vals = sorted([float(k) for k in h0_z.keys()])
            h0_obs = [h0_z[str(z)]["h0"] for z in z_vals]
            h0_err = [h0_z[str(z)]["h0_err"] for z in z_vals]
            ax.errorbar(z_vals, h0_obs, yerr=h0_err, fmt="ko", capsize=4,
                        markersize=7, label="Pantheon+ $H_0(z)$", zorder=5)
        else:
            ax.text(0.5, 0.5, "Data unavailable", transform=ax.transAxes,
                    ha="center", fontsize=11, color="gray")

        z_fine = np.linspace(0.005, 1.5, 500)
        z_wall = 300.0 * self.H0_SH0ES / self.C_KMS
        delta_h0 = self.H0_SH0ES - self.H0_CMB
        h0_void = self.H0_CMB + delta_h0 * 0.5 * (1 - np.tanh(5.0 * (z_fine - z_wall)))
        ax.plot(z_fine, h0_void, "r--", linewidth=2, label="Void model")

        h0_tep = self.H0_CMB + delta_h0 / (1.0 + z_fine) ** 0.3
        ax.plot(z_fine, h0_tep, "b-", linewidth=2, label="TEP model")

        ax.axhline(self.H0_CMB, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Redshift $z$", fontsize=11)
        ax.set_ylabel("$H_0(z)$ (km/s/Mpc)", fontsize=11)
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1.5)
        ax.set_ylim(65, 75)

        if mc:
            textstr = f"$\\Delta$AIC = {mc.get('delta_aic', 0):.1f}"
            ax.text(0.60, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    def plot_panel_boundary_test(self, ax, step_data):
        """Panel (d): Void boundary test at z > 0.3."""
        ax.set_title("(d) Void Boundary Test ($z > 0.3$)", fontsize=12, fontweight="bold")

        s34 = step_data.get("step_34_void_boundary_test", {})
        h0_z = s34.get("h0_z_by_host_mass", {})
        bins = h0_z.get("bins", [])

        z_fine = np.linspace(0.005, 2.3, 500)
        z_wall = 300.0 * self.H0_SH0ES / self.C_KMS
        delta_h0 = self.H0_SH0ES - self.H0_CMB
        h0_void = self.H0_CMB + delta_h0 * 0.5 * (1 - np.tanh(5.0 * (z_fine - z_wall)))
        ax.plot(z_fine, h0_void, "r--", linewidth=2, label="Void model")

        h0_tep = self.H0_CMB + delta_h0 / (1.0 + z_fine) ** 0.3
        ax.plot(z_fine, h0_tep, "b-", linewidth=2, label="TEP model")

        for host_type, color, marker, label in [
            ("massive", "#d62728", "^", "Massive hosts"),
            ("low_mass", "#1f77b4", "s", "Low-mass hosts"),
        ]:
            z_pts, h0_pts, err_pts = [], [], []
            for b in bins:
                cat = b.get(host_type, {})
                if cat.get("n_sne", 0) >= 5 and not np.isnan(cat.get("h0_median", np.nan)):
                    z_pts.append(b["z_mid"])
                    h0_pts.append(cat["h0_median"])
                    err_pts.append(cat.get("h0_sem", 0))
            if z_pts:
                ax.errorbar(z_pts, h0_pts, yerr=err_pts, fmt=marker, color=color,
                            capsize=4, markersize=7, label=label, zorder=5)

        ax.axvline(z_wall, color="red", linestyle=":", alpha=0.3)
        ax.axhline(self.H0_CMB, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Redshift $z$", fontsize=11)
        ax.set_ylabel("$H_0(z)$ (km/s/Mpc)", fontsize=11)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 2.3)
        ax.set_ylim(60, 80)

    def generate_composite_figure(self, step_data):
        """Generate the 4-panel composite manuscript figure."""
        print_status("Generating composite manuscript figure...", "PROCESS")

        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

        ax_a = fig.add_subplot(gs[0, 0])
        self.plot_panel_indicator_divergence(ax_a, step_data)

        ax_b = fig.add_subplot(gs[0, 1])
        self.plot_panel_calibration_sensitivity(ax_b, step_data)

        ax_c = fig.add_subplot(gs[1, 0])
        self.plot_panel_h0_decay(ax_c, step_data)

        ax_d = fig.add_subplot(gs[1, 1])
        self.plot_panel_boundary_test(ax_d, step_data)

        fig.suptitle("TEP-VOID: Four Discriminating Observables — Void vs TEP",
                     fontsize=15, fontweight="bold", y=1.02)
        fig.subplots_adjust(hspace=0.35, wspace=0.3)

        fig_path = self.figures / "step_43_manuscript_composite.png"
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print_status(f"Composite figure saved to {fig_path}", "SUCCESS")
        return fig_path

    def run(self):
        """Execute the full step."""
        print_status("Step 43: Manuscript Figures — Void vs TEP Composite", "TITLE")

        print_status(
            "This step generates the composite manuscript figure displaying all "
            "four discriminating observables that distinguish the KBC void model "
            "from the TEP framework. The figure serves as the visual synthesis of "
            "the Block III falsification analysis, consolidating the indicator "
            "divergence, calibration sensitivity, H0(z) decay profile, and void "
            "boundary test into a single publication-ready panel.",
            "INFO",
        )

        step_data = self.load_step_results()

        print_status(
            "Methodology: a four-panel composite figure is assembled from upstream "
            "step results (steps 30, 31, 32, 34, 40, 42). Panel (a) plots Cepheid "
            "vs TRGB distance moduli; panel (b) shows bulk-flow calibration "
            "sensitivity; panel (c) displays the H0(z) decay profile with void and "
            "TEP model overlays; panel (d) presents the void boundary test in "
            "massive and low-mass hosts. Model curves use H0_CMB = 67.4 and "
            "H0_SH0ES = 73.0 km/s/Mpc.",
            "PROCESS",
        )
        composite_path = self.generate_composite_figure(step_data)

        print_status(
            f"Interpretation: the composite figure consolidates the four "
            f"discriminating observables into a single visual synthesis. "
            f"{len(step_data)} upstream step results were loaded for panel "
            f"generation. The figure is saved as a publication-ready PNG at "
            f"200 DPI.",
            "SUCCESS",
        )

        source_figures = [
            "step_30_bulk_flow_comparison.png",
            "step_31_peculiar_velocity_artifact.png",
            "step_32_h0_vs_redshift.png",
            "step_34_void_boundary_test.png",
            "step_40_pantheon_tep_correction.png",
        ]

        existing = []
        for fig_name in source_figures:
            fig_path = self.figures / fig_name
            if fig_path.exists():
                existing.append(fig_name)

        summary = {
            "step": "43_manuscript_figures",
            "description": (
                "Generate 4-panel composite manuscript figure showing all "
                "discriminating observables for void vs TEP comparison"
            ),
            "n_steps_loaded": len(step_data),
            "figures_generated": [
                {"name": "Composite (4 panels)", "path": str(composite_path)},
            ],
            "source_figures_available": existing,
            "composite_figure_panels": [
                "(a) Indicator-Specific Distance Divergence: Cepheid vs TRGB DM",
                "(b) Peculiar Velocity Calibration Sensitivity: bulk-flow H0 dependence",
                "(c) H0(z) Decay: Void sharp boundary vs TEP smooth decay",
                "(d) Void Boundary Test: H0 in massive hosts at z > 0.3",
            ],
            "tep_prediction": (
                "The composite figure should visually demonstrate that the TEP "
                "model curves are consistent with the observed data across all "
                "four panels, while the void model curves are not."
            ),
            "void_prediction": (
                "The KBC void model predicts indicator-independent distances, "
                "calibration-independent bulk flow, and a sharp H0(z) boundary "
                "at the void wall redshift."
            ),
            "methodology": (
                "Four-panel composite figure assembled from upstream step results. "
                "Panel (a): Cepheid vs TRGB distance modulus scatter with 1:1 "
                "reference. Panel (b): bulk-flow amplitude vs R_max for Cepheid "
                "and TRGB calibrations. Panel (c): H0(z) with void (tanh boundary) "
                "and TEP ((1+z)^{-0.3}) model overlays. Panel (d): H0(z) in "
                "massive and low-mass hosts with void and TEP model overlays. "
                "Model parameters: H0_CMB = 67.4, H0_SH0ES = 73.0 km/s/Mpc."
            ),
            "provenance": {
                "data_sources": [
                    "step_30_bulk_flow_recalculation.json",
                    "step_31_peculiar_velocity_artifact.json",
                    "step_32_redshift_decay_profile.json",
                    "step_34_void_boundary_test.json",
                    "step_40_redshift_shear_reconstruction.json",
                    "step_42_falsification_summary.json",
                ],
                "pipeline_block": "Block III — TEP reconstruction and synthesis",
            },
            "scientific_context": (
                "Visual synthesis of the Block III falsification analysis. The "
                "composite figure consolidates the four discriminating observables "
                "into a single publication-ready panel for the manuscript."
            ),
            "downstream_consumers": "none — terminal step",
            "output_files": [
                str(self.results / "step_43_manuscript_figures.json"),
                str(composite_path),
            ],
        }

        summary_path = self.results / "step_43_manuscript_figures.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print_status(f"Summary saved to {summary_path}", "SUCCESS")

        print_status("Step 43 complete", "SUCCESS")


if __name__ == "__main__":
    step = Step43ManuscriptFigures()
    step.run()
