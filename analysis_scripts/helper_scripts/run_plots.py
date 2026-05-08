#!/usr/bin/env python3
"""
Generate plots (Rg, KE, interactive HTML) for all trajectories
in a sweep directory.

Usage:
    python run_plots.py <sweep_dir>

Reads trajectories from:
    BASE_DIR/<sweep_dir>/<param_folder>/trajectory_*.gsd
and writes:
    BASE_DIR/<sweep_dir>/<param_folder>/plots/trajectory_*/<plots>.png/html
"""

import sys
from pathlib import Path
import gsd.hoomd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

analysis_functions_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(analysis_functions_dir))

# import your plotting functions

from analysis_scripts.analyze_leona import (
    rg_calc,
    compute_ke_over_time,
    plot_rg_time,
    plot_rg_autocorr,
    plot_rg_hist_last,
    plot_ke_time,
    angles_by_type_over_time,
    plot_angle_heatmap,
    plot_replicate_variability_pct,
)


BASE_DIR = analysis_functions_dir / "bulk" / "output_dir"

PLOT_ANGLES = [
    "Phosphate-Phosphate-Phosphate",
    "Phosphate-Phosphate-Base",
    "Base-Phosphate-Base",
]

# Metrics for plotting replicate variability
METRICS = ["rg_0_6_mean", "ke_mean_0.2",
           "inversion_score", "paired_frac_last_0_2_mean"]

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_plots.py <sweep_dir>")
        sys.exit(1)

    sweep_path = BASE_DIR / sys.argv[1]
    if not sweep_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {sweep_path}")
    
    per_rep_csv = sweep_path / "analysis_per_replicate.csv"


    if not per_rep_csv.exists():
        raise FileNotFoundError(f"Missing {per_rep_csv} (run analysis first)")

    df_per_rep = pd.read_csv(per_rep_csv)

    # Walk through all parameter folders
    for folder in sweep_path.rglob("*"):
        if not folder.is_dir():
            continue

        traj_files = sorted(folder.glob("trajectory_*.gsd"))
        if not traj_files:
            continue

        print(f"\nFound trajectory folder: {folder}")

        # Create /plots/<trajectory_*>/ directories
        plots_root = folder / "plots"
        plots_root.mkdir(exist_ok=True)
        params_path = folder / "params.json"

        #rg_all = np.array([])

        for traj_file in traj_files:
            name = traj_file.stem  # e.g. trajectory_0
            traj_plot_dir = plots_root / name
            traj_plot_dir.mkdir(exist_ok=True)

            print(f"  Processing {traj_file.name}")

            traj = gsd.hoomd.open(traj_file, "r")

            # === Calculate quantities ===
            rg_values = rg_calc(traj, scale = 0.6)
            # We concatenate all Rg values to plot a histogram across replicates
            #rg_all = np.concatenate((rg_all, rg_values))
            ke_values = compute_ke_over_time(traj)
            angles = angles_by_type_over_time(traj, stride=10, bpp = False)

            # === Rg over time ===
            fig, ax = plt.subplots()
            plot_rg_time(rg_values, ax=ax, smooth_sigma=5)
            fig.savefig(traj_plot_dir / "rg_time.png", dpi=150)
            plt.close(fig)

            # === Rg autocorrelation ===
            fig, ax = plt.subplots()
            plot_rg_autocorr(rg_values, ax=ax, max_lag=500)
            fig.savefig(traj_plot_dir / "rg_autocorr.png", dpi=150)
            plt.close(fig)

            # === Rg histogram (last 20%) ===
            fig, ax = plt.subplots()
            plot_rg_hist_last(rg_values, ax=ax, fraction_last=0.2)
            fig.savefig(traj_plot_dir / "rg_hist_last.png", dpi=150)
            plt.close(fig)

            # === KE over time ===
            fig, ax = plt.subplots()
            plot_ke_time(ke_values, ax=ax, smooth_sigma=5)
            fig.savefig(traj_plot_dir / "ke_time.png", dpi=150)
            plt.close(fig)

            # === Angle heatmaps ===
            for angle_name in PLOT_ANGLES:
                if angle_name not in angles:
                    continue

                fig, ax = plt.subplots()
                plot_angle_heatmap(angles[angle_name], angle_name,
                                params_path=params_path, ax=ax)

                outname = f"angle_heatmap_{angle_name.replace('-', '_')}.png"
                fig.savefig(traj_plot_dir / outname, dpi=150)
                plt.close(fig)
            
            # === Replicate variablity plot ===
            # matches df_per_rep["folder_name"] 
            folder_key = folder.name
            try:
                ax = plot_replicate_variability_pct(
                    df_per_rep, folder_key, metrics=METRICS, clip_pct=50)
                fig = ax.figure
                fig.savefig(plots_root / "replicate_variability.png", dpi=150)
                plt.close(fig)
            except Exception as e:
                print(f"  Skipping replicate variability for {folder_key}: {e}")

        print(f"  Saved plots to {plots_root}")

    print("\nDone generating all plots.")


if __name__ == "__main__":
    main()
