#!/usr/bin/env python3

"""
Run animate for ALL trajectories in a sweep directory.

Usage:
    python run_animate_traj.py <sweep_dir>
Note: change SaveInteractiveTrajectory(trajectory=traj[::int] to frames to be saved

Reads data from BASE_DIR/<sweep_dir> and writes results back into
the same directory in the respective param combination folder.
"""


import sys
from pathlib import Path
import gsd.hoomd

from pathlib import Path
import sys
analysis_functions_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(analysis_functions_dir))

from analysis_scripts.visualisation import SaveInteractiveTrajectory, AnimateTrajectory
from analysis_scripts.analyze_leona import analyze_trajectories, save_analysis_csvs


BASE_DIR = analysis_functions_dir/ "bulk" / "output_dir"

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_animate_traj.py <sweep_dir>")
        sys.exit(1)

    sweep_path = BASE_DIR / sys.argv[1]

    if not sweep_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {sweep_path}")

    # Walk through all directories
    for folder in sweep_path.rglob("*"):
        if not folder.is_dir():
            continue

        # detect if this folder contains any .gsd files
        gsd_files = sorted(folder.glob("*.gsd"))
        if not gsd_files:
            continue  # skip folders with no trajectories

        print(f"\nFound trajectory folder: {folder}")

        # create target directory
        out_dir = folder / "animated_traj"
        out_dir.mkdir(exist_ok=True)

        # process all gsd files
        for i, gsd_file in enumerate(gsd_files):
            out_file = out_dir / f"{gsd_file.stem}.html"
            print(f"  {gsd_file.name} -> {out_file.name}")

            traj = gsd.hoomd.open(gsd_file, "r")
            SaveInteractiveTrajectory(
                trajectory=traj[::100],
                structure_type="directional-branch",
                axlim=12,
                filename=str(out_file)
            )

    print("\nDone.")

if __name__ == "__main__":
    main()
