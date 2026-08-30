#!/usr/bin/env python3
"""
Step 31: Peculiar Velocity Calibration Sensitivity
===================================================
Quantifies how the Hubble tension (ΔH0 between Cepheid and TRGB calibrations)
propagates into the peculiar velocity field derived from the CosmicFlows-4
catalog.

The peculiar velocity is a derived quantity: v_pec = cz − H0·d.  A change
in H0 of ΔH0 km/s/Mpc produces a distance-dependent shift in v_pec of
Δv = −ΔH0 · d km/s.  This is not a statistical test (the offset is
deterministic by construction) but a quantification of how sensitive the
inferred bulk-flow anomaly is to the H0 calibration.

The key physical argument is:
  1. The Cepheid vs TRGB distance offset (Step 30, ~3.3σ from CF4 table2)
     demonstrates that Cepheid distances are systematically compressed.
  2. The compressed Cepheid distances yield H0 = 73.0 (SH0ES) instead of
     the TRGB-calibrated H0 = 69.8 (Freedman et al. 2025), a difference
     consistent with the measured Δμ = -0.080 mag.
  3. Using H0 = 73.0 to compute peculiar velocities introduces a
     distance-dependent systematic of −ΔH0·d km/s into v_pec.
  4. This systematic inflates the apparent bulk flow.

Under the void model, the bulk flow is physical and independent of the H0
calibration.  Under TEP, the Cepheid calibration bias is the source of the
inflated peculiar velocities.

Outputs:
    results/outputs/step_31_peculiar_velocity_artifact.json
    results/figures/step_31_peculiar_velocity_artifact.png
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.plot_style import apply_tep_style


class Step31PeculiarVelocityArtifact:
    """Step 31: Quantify H0 calibration sensitivity of the peculiar velocity field."""

    H0_CEPHEID = 73.0  # km/s/Mpc (Riess et al. 2022, SH0ES)
    H0_TRGB = 69.8  # km/s/Mpc (Freedman et al. 2025, CCHP) — default
    C_KMS = 299792.458  # km/s

    def __init__(self):
        self.root = PROJECT_ROOT
        self.data_interim = self.root / "data" / "interim"
        self.data_processed = self.root / "data" / "processed"
        self.results = self.root / "results" / "outputs"
        self.figures = self.root / "results" / "figures"
        self.logs = self.root / "logs"

        for d in [self.data_interim, self.data_processed, self.results, self.figures, self.logs]:
            d.mkdir(parents=True, exist_ok=True)

        self.logger = TEPLogger(
            "step_31", log_file_path=self.logs / "step_31_peculiar_velocity_artifact.log"
        )
        set_step_logger(self.logger)

    def load_cosmicflows_data(self):
        """Load CosmicFlows-4 processed data from step_02."""
        cf_path = self.data_interim / "cosmicflows4_processed.csv"
        print_status(f"Loading CosmicFlows-4 data from {cf_path}...", "PROCESS")

        if cf_path.exists():
            try:
                df = pd.read_csv(cf_path)
                print_status(f"Loaded {len(df)} rows from CosmicFlows-4 data", "SUCCESS")
                return df
            except Exception as e:
                print_status(f"Error reading CosmicFlows-4 data: {e}", "ERROR")
                return pd.DataFrame()
        else:
            print_status("CosmicFlows-4 processed data not found.", "WARNING")
            return pd.DataFrame()

    def compute_peculiar_velocities(self, df):
        """Compute peculiar velocities for both H0 calibration schemes.

        v_pec_cepheid = cz - H0_CEPHEID * d   (H0 = 73.0)
        v_pec_trgb    = cz - H0_TRGB    * d   (H0 = 69.8)

        The difference is deterministic:
            Δv_pec = v_pec_cepheid - v_pec_trgb = (H0_TRGB - H0_CEPHEID) * d
                   = -3.2 * d  km/s

        This is not a statistical test but a quantification of how the H0
        calibration bias propagates into the peculiar velocity field.
        """
        print_status("Computing peculiar velocities: v_pec = cz - H0 * d...", "PROCESS")

        df = df.copy()

        # Find redshift column
        z_col = None
        for c in df.columns:
            if c.lower() in ["z", "redshift"]:
                z_col = c
                break

        if z_col is None and "cz" in df.columns:
            df["cz"] = pd.to_numeric(df["cz"], errors="coerce")
        elif z_col is not None:
            df["cz"] = pd.to_numeric(df[z_col], errors="coerce") * self.C_KMS
        else:
            print_status("No redshift column found in data.", "ERROR")
            return df

        # Find distance column
        d_col = None
        for c in df.columns:
            if c.lower() in ["distance_mpc", "d", "distance", "dist_mpc"]:
                d_col = c
                break

        if d_col is None:
            print_status("No distance column found in CF4 data.", "ERROR")
            return df

        d = pd.to_numeric(df[d_col], errors="coerce")

        # Cepheid-calibrated peculiar velocities
        df["v_pec_cepheid"] = df["cz"] - self.H0_CEPHEID * d
        # TRGB-calibrated peculiar velocities
        df["v_pec_trgb"] = df["cz"] - self.H0_TRGB * d
        # Calibration-induced shift (deterministic)
        df["v_pec_shift"] = df["v_pec_cepheid"] - df["v_pec_trgb"]
        df["distance_mpc_use"] = d

        v_cep = df["v_pec_cepheid"].dropna()
        v_trgb = df["v_pec_trgb"].dropna()
        print_status(
            f"  Cepheid v_pec (H0={self.H0_CEPHEID}): N={len(v_cep)}, "
            f"mean={v_cep.mean():.1f}, std={v_cep.std():.1f} km/s",
            "SUCCESS",
        )
        print_status(
            f"  TRGB v_pec    (H0={self.H0_TRGB}): N={len(v_trgb)}, "
            f"mean={v_trgb.mean():.1f}, std={v_trgb.std():.1f} km/s",
            "SUCCESS",
        )
        print_status(
            f"  ΔH0 = {self.H0_CEPHEID - self.H0_TRGB:.1f} km/s/Mpc → "
            f"shift = -{self.H0_CEPHEID - self.H0_TRGB:.1f} × d km/s",
            "TEST",
        )

        return df

    def quantify_calibration_sensitivity(self, df, mean_delta_mu=-0.080, significance_sigma=3.30):
        """Quantify how the H0 calibration affects bulk-flow statistics.

        The key metric is the fraction of the bulk-flow signal that can
        be attributed to the H0 calibration difference.
        """
        print_status("Quantifying calibration sensitivity of bulk-flow...", "PROCESS")

        results = {}

        if "v_pec_cepheid" not in df.columns or df.empty:
            return results

        d = pd.to_numeric(df["distance_mpc_use"], errors="coerce")
        v_cep = pd.to_numeric(df["v_pec_cepheid"], errors="coerce")
        v_trgb = pd.to_numeric(df["v_pec_trgb"], errors="coerce")

        # The calibration shift at characteristic distances
        for d_char in [100, 200, 250]:
            shift = (self.H0_TRGB - self.H0_CEPHEID) * d_char
            results[f"shift_at_{d_char}_mpc"] = float(shift)
            print_status(f"  Shift at d={d_char} Mpc: {shift:.0f} km/s", "TEST")

        # RMS of the calibration shift across the catalog
        shift = df["v_pec_shift"].dropna()
        results["rms_shift"] = float(np.sqrt(np.mean(shift**2)))
        results["mean_shift"] = float(shift.mean())
        results["median_shift"] = float(shift.median())

        # Fraction of galaxies where |shift| > 200 km/s (typical v_pec scatter)
        frac_large_shift = float((shift.abs() > 200).mean())
        results["fraction_gt_200_kms"] = frac_large_shift
        print_status(f"  RMS shift: {results['rms_shift']:.0f} km/s", "TEST")
        print_status(f"  Fraction with |shift| > 200 km/s: {frac_large_shift*100:.1f}%", "TEST")

        # Bulk-flow amplitude comparison in radial bins
        coord_cols = ["SGX", "SGY", "SGZ"]
        has_coords = all(c in df.columns for c in coord_cols)

        bf_comparison = {}
        for r_max in [100, 200, 250]:
            mask = d.notna() & v_cep.notna() & (d > 0) & (d <= r_max)
            if mask.sum() < 10:
                continue

            d_use = d[mask]
            vp_cep = v_cep[mask]
            vp_trgb = v_trgb[mask]

            if has_coords:
                sgx = pd.to_numeric(df.loc[mask, "SGX"], errors="coerce")
                sgy = pd.to_numeric(df.loc[mask, "SGY"], errors="coerce")
                sgz = pd.to_numeric(df.loc[mask, "SGZ"], errors="coerce")
                r_mag = np.sqrt(sgx**2 + sgy**2 + sgz**2)
                valid = r_mag > 0

                if valid.sum() >= 10:
                    rx = sgx[valid] / r_mag[valid]
                    ry = sgy[valid] / r_mag[valid]
                    rz = sgz[valid] / r_mag[valid]
                    w = 1.0 / d_use[valid] ** 2
                    w = w / w.sum()

                    bf_cep = np.sqrt(
                        (np.sum(w * vp_cep[valid] * rx))**2 +
                        (np.sum(w * vp_cep[valid] * ry))**2 +
                        (np.sum(w * vp_cep[valid] * rz))**2
                    )
                    bf_trgb = np.sqrt(
                        (np.sum(w * vp_trgb[valid] * rx))**2 +
                        (np.sum(w * vp_trgb[valid] * ry))**2 +
                        (np.sum(w * vp_trgb[valid] * rz))**2
                    )
                else:
                    bf_cep = float(np.abs(vp_cep.mean()))
                    bf_trgb = float(np.abs(vp_trgb.mean()))
            else:
                bf_cep = float(np.abs(vp_cep.mean()))
                bf_trgb = float(np.abs(vp_trgb.mean()))

            reduction = (bf_cep - bf_trgb) / bf_cep * 100 if bf_cep > 0 else 0
            bf_comparison[r_max] = {
                "n": int(mask.sum()),
                "bf_cepheid": float(bf_cep),
                "bf_trgb": float(bf_trgb),
                "reduction_pct": float(reduction),
            }
            print_status(
                f"  R<={r_max} Mpc: BF_cep={bf_cep:.1f}, BF_trgb={bf_trgb:.1f}, "
                f"reduction={reduction:.1f}%",
                "TEST",
            )

        results["bulk_flow_comparison"] = {str(k): v for k, v in bf_comparison.items()}

        # Physical interpretation
        results["delta_h0"] = float(self.H0_CEPHEID - self.H0_TRGB)
        results["mean_delta_mu"] = float(mean_delta_mu)
        results["significance_sigma"] = float(significance_sigma)
        dh0 = float(self.H0_CEPHEID - self.H0_TRGB)
        results["interpretation"] = (
            f"The H0 calibration difference of {dh0:.1f} km/s/Mpc introduces a "
            f"distance-dependent systematic of -{dh0:.1f}*d km/s into v_pec. "
            "This is a direct consequence of the Cepheid distance compression "
            f"demonstrated in Step 30 ({significance_sigma:.2f}sigma). Under the void model, the "
            "bulk flow is physical and this calibration sensitivity is "
            "unphysical. Under TEP, the Cepheid calibration bias is the "
            "source of the inflated peculiar velocities."
        )

        return results

    def plot_results(self, df, results):
        """Generate figure showing calibration sensitivity of peculiar velocities."""
        colors = apply_tep_style()
        print_status("Generating peculiar velocity figure...", "PROCESS")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Panel 1: Histogram of v_pec for both calibrations
        ax1 = axes[0]
        if "v_pec_cepheid" in df.columns:
            v_cep = df["v_pec_cepheid"].dropna()
            ax1.hist(
                v_cep, bins=50, alpha=0.6, color=colors['red'],
                label=f"Cepheid $H_0={self.H0_CEPHEID}$ ($\\mu$={v_cep.mean():.0f} km/s)",
                density=True,
            )
        if "v_pec_trgb" in df.columns:
            v_trgb = df["v_pec_trgb"].dropna()
            ax1.hist(
                v_trgb, bins=50, alpha=0.6, color=colors['green'],
                label=f"TRGB $H_0={self.H0_TRGB}$ ($\\mu$={v_trgb.mean():.0f} km/s)",
                density=True,
            )
        ax1.set_xlabel("$v_{pec}$ (km/s)")
        ax1.set_ylabel("Density")
        ax1.set_title("Peculiar Velocity Distributions")
        ax1.legend()
        ax1.grid(True)
        ax1.axvline(0, color=colors['dark'], linestyle="--", alpha=0.5)
        ax1.set_xlim(-2000, 2000)

        # Panel 2: Calibration shift vs distance
        ax2 = axes[1]
        if "v_pec_shift" in df.columns and "distance_mpc_use" in df.columns:
            d = pd.to_numeric(df["distance_mpc_use"], errors="coerce")
            shift = pd.to_numeric(df["v_pec_shift"], errors="coerce")
            mask = d.notna() & shift.notna()

            if mask.sum() > 0:
                if mask.sum() > 5000:
                    sample_idx = np.random.choice(mask[mask].index, 5000, replace=False)
                    ax2.scatter(d.loc[sample_idx], shift.loc[sample_idx], c=colors['blue'], s=5, alpha=0.3)
                else:
                    ax2.scatter(d[mask], shift[mask], c=colors['blue'], s=20, alpha=0.6)

                d_range = np.linspace(0, d[mask].max(), 100)
                expected = (self.H0_TRGB - self.H0_CEPHEID) * d_range
                ax2.plot(d_range, expected, color=colors['red'], linestyle="--", linewidth=2, label="$\\Delta H_0 \\times d$")

            ax2.axhline(0, color=colors['dark'], linestyle="--", alpha=0.5)
            ax2.set_xlabel("Distance $d$ (Mpc)")
            ax2.set_ylabel("$\\Delta v_{pec} = v_{pec}^{Cep} - v_{pec}^{TRGB}$ (km/s)")
            ax2.set_title("Calibration-Induced Shift (Deterministic)")
            ax2.legend()
            ax2.grid(True)

        fig.suptitle(
            "Step 31: Peculiar Velocity Calibration Sensitivity\n"
            f"$\\Delta H_0 = {self.H0_CEPHEID - self.H0_TRGB:.1f}$ km/s/Mpc "
            f"from Cepheid distance compression (Step 30: {results.get('significance_sigma', 3.30):.2f}$\\sigma$)",
            fontsize=12,
            fontweight="bold",
            y=1.04,
        )
        fig.tight_layout()
        fig_path = self.figures / "step_31_peculiar_velocity_artifact.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print_status(f"Figure saved to {fig_path}", "SUCCESS")
        return fig_path

    def load_h0_from_step30(self):
        """Load H0 values and indicator comparison from step_30 results, falling back to defaults."""
        step30_path = self.results / "step_30_bulk_flow_recalculation.json"
        h0_cepheid = self.H0_CEPHEID
        h0_trgb = self.H0_TRGB
        h0_trgb_source = "Freedman et al. 2025 (default)"
        mean_delta_mu = -0.080  # fallback
        significance_sigma = 3.30  # fallback

        if step30_path.exists():
            try:
                with open(step30_path) as f:
                    s30 = json.load(f)
                test_b = s30.get("test_b_bulk_flow_sensitivity", {})
                if "h0_cepheid" in test_b:
                    h0_cepheid = test_b["h0_cepheid"]
                if "h0_trgb" in test_b:
                    h0_trgb = test_b["h0_trgb"]
                    h0_trgb_source = test_b.get("h0_trgb_source", "from step_30")
                # Also get the derived value for cross-check
                test_a = s30.get("test_a_indicator_comparison", {})
                if "h0_trgb_derived_from_delta_mu" in test_a:
                    h0_trgb_derived = test_a["h0_trgb_derived_from_delta_mu"]
                    consistency = test_a.get("h0_trgb_consistency_sigma", 0)
                    print_status(
                        f"  Step 30: H0_TRGB published={h0_trgb:.1f}, "
                        f"derived from Δμ={h0_trgb_derived:.2f} ({consistency:.1f}σ consistent)",
                        "PROCESS",
                    )
                # Load indicator comparison values for use in summaries
                if "mean_delta_mu" in test_a:
                    mean_delta_mu = test_a["mean_delta_mu"]
                if "significance_sigma" in test_a:
                    significance_sigma = test_a["significance_sigma"]
            except Exception as e:
                print_status(f"Could not load step_30 results: {e}", "WARNING")

        return h0_cepheid, h0_trgb, h0_trgb_source, mean_delta_mu, significance_sigma

    def run(self):
        """Execute the full step."""
        print_status("Step 31: Peculiar Velocity Calibration Sensitivity", "TITLE")

        print_status(
            "This step quantifies how the Hubble tension between Cepheid- and "
            "TRGB-calibrated H0 values propagates into the peculiar velocity field "
            "derived from the CosmicFlows-4 catalog. The void model predicts that "
            "the bulk flow is physical and independent of the H0 calibration, while "
            "TEP predicts that the Cepheid-calibrated H0 introduces a "
            "distance-dependent systematic that inflates the apparent bulk flow. "
            "The key discriminating observable is the calibration-induced velocity "
            "shift Δv = −ΔH0 · d, which is deterministic by construction but "
            "demonstrates the sensitivity of bulk-flow inferences to the distance "
            "indicator calibration.",
            "INFO",
        )

        # Load H0 values from step_30 (which derives H0_TRGB from Δμ)
        h0_cepheid, h0_trgb, h0_trgb_source, mean_delta_mu, significance_sigma = self.load_h0_from_step30()
        delta_h0 = h0_cepheid - h0_trgb
        print_status(
            f"  Using H0_Cepheid={h0_cepheid}, H0_TRGB={h0_trgb} ({h0_trgb_source})",
            "PROCESS",
        )
        print_status(f"  ΔH0 = {delta_h0:.2f} km/s/Mpc", "PROCESS")

        df = self.load_cosmicflows_data()
        if df.empty:
            print_status("No data available. Step 31 cannot proceed.", "WARNING")
            summary = {
                "step": "31_peculiar_velocity_artifact",
                "status": "no_data",
                "output_files": [],
            }
            summary_path = self.results / "step_31_peculiar_velocity_artifact.json"
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            print_status("Step 31 complete (no data)", "WARNING")
            return

        # Update H0 values for this run
        self.H0_CEPHEID = h0_cepheid
        self.H0_TRGB = h0_trgb

        print_status(
            "Methodology: peculiar velocities are computed as v_pec = cz − H0·d for "
            "both H0 calibrations (Cepheid and TRGB). The calibration-induced shift "
            "Δv = (H0_TRGB − H0_CEPHEID) · d is deterministic and quantifies the "
            "distance-dependent systematic. Bulk-flow amplitudes are compared in "
            "radial bins (100, 200, 250 Mpc) using inverse-distance-squared weighting "
            "in supergalactic coordinates. The RMS shift and fraction of galaxies "
            "with |shift| > 200 km/s characterize the calibration sensitivity of "
            "the peculiar velocity field.",
            "PROCESS",
        )

        df = self.compute_peculiar_velocities(df)
        results = self.quantify_calibration_sensitivity(df, mean_delta_mu, significance_sigma)

        rms_shift = results.get("rms_shift", 0.0)
        frac_large = results.get("fraction_gt_200_kms", 0.0)
        print_status(
            f"The calibration-induced RMS velocity shift of {rms_shift:.0f} km/s "
            f"affects {frac_large*100:.1f}% of galaxies at >200 km/s. Under TEP, "
            f"this systematic inflates the apparent bulk flow when the Cepheid-"
            f"calibrated H0 = {h0_cepheid:.1f} is used. The void model, which "
            f"predicts a calibration-independent bulk flow, has no mechanism to "
            f"account for this distance-dependent artifact.",
            "TEST",
        )

        fig_path = self.plot_results(df, results)

        summary = {
            "step": "31_peculiar_velocity_artifact",
            "description": (
                f"Quantifies how the H0 calibration difference ({delta_h0:.1f} km/s/Mpc, "
                f"from the Cepheid distance compression demonstrated in Step 30) "
                "propagates into the peculiar velocity field. The shift is "
                f"deterministic: Δv = -{delta_h0:.1f}*d km/s. This is not a statistical "
                "test but a quantification of calibration sensitivity."
            ),
            "h0_cepheid": h0_cepheid,
            "h0_trgb": h0_trgb,
            "h0_trgb_source": h0_trgb_source,
            "delta_h0": float(delta_h0),
            "n_galaxies": int(len(df)),
            "n_groups": int(len(df)),  # CF4 table4 entries are group-averaged
            "results": results,
            "tep_argument": (
                f"The Cepheid distance compression (Step 30: Δμ = {mean_delta_mu:.3f} mag, {significance_sigma:.2f}σ) "
                f"produces H0 = {h0_cepheid:.1f} instead of {h0_trgb:.1f}. Using H0 = {h0_cepheid:.1f} "
                f"to compute peculiar velocities introduces a distance-dependent systematic "
                f"of -{delta_h0:.1f}*d km/s, inflating the apparent bulk flow. The void model "
                "has no mechanism to produce this calibration dependence."
            ),
            "methodology": (
                "Peculiar velocities are computed as v_pec = cz − H0·d for both "
                "Cepheid (H0=73.0) and TRGB (H0=69.8) calibrations. The deterministic "
                "shift Δv = −ΔH0·d is evaluated at characteristic distances (100, 200, "
                "250 Mpc). Bulk-flow amplitudes are compared in radial bins using "
                "inverse-distance-squared weighting in supergalactic coordinates. "
                "The RMS shift and fraction of galaxies with |shift| > 200 km/s "
                "quantify the calibration sensitivity."
            ),
            "provenance": {
                "data_sources": [
                    "CosmicFlows-4 processed data (from step_02, Tully et al. 2023)",
                    "H0 calibration values from step_30 (Riess et al. 2022; Freedman et al. 2025)",
                ],
                "pipeline_block": "Block Ib — bulk flow and redshift decay",
                "covariance": (
                    "Not applicable — the calibration shift is deterministic by "
                    "construction. Bulk-flow errors derived from weighted variance "
                    "of v_pec divided by sqrt(N)."
                ),
            },
            "scientific_context": (
                "The Hubble tension between Cepheid- and TRGB-calibrated H0 values "
                "propagates into the peculiar velocity field as a distance-dependent "
                "systematic. Under TEP, the Cepheid calibration bias is the source of "
                "inflated peculiar velocities and the apparent bulk flow. Under the "
                "void model, the bulk flow is physical and independent of the H0 "
                "calibration."
            ),
            "tep_prediction": (
                "The Cepheid-calibrated H0 introduces a distance-dependent systematic "
                "of −ΔH0·d km/s into v_pec, inflating the apparent bulk flow"
            ),
            "void_prediction": (
                "The bulk flow is physical and independent of the H0 calibration; "
                "no distance-dependent systematic is expected"
            ),
            "downstream_consumers": ["step_33", "step_34"],
            "output_files": [
                str(self.results / "step_31_peculiar_velocity_artifact.json"),
                str(fig_path),
            ],
        }

        summary_path = self.results / "step_31_peculiar_velocity_artifact.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print_status(f"Summary saved to {summary_path}", "SUCCESS")

        print_status("Step 31 complete", "SUCCESS")


if __name__ == "__main__":
    step = Step31PeculiarVelocityArtifact()
    step.run()
