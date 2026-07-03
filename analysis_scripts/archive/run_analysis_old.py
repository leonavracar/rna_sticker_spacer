#!/usr/bin/env python3

"""
Run analysis for trajectories in a sweep directory.
 
Usage:
    python run_analysis.py <sweep_dir>
 
Reads data from BASE_DIR/<sweep_dir> and writes results back into
the same directory.
"""

import argparse
from pathlib import Path
import sys
analysis_functions_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(analysis_functions_dir))

from analysis_scripts.analyze_leona import analyze_trajectories, save_analysis_csvs

BASE_DIR = analysis_functions_dir / "bulk" / "output_dir"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "sweep_dir",
        help="Name of the sweep directory inside BASE_DIR"
    )
    parser.add_argument(
        "--pairing-force-recompute",
        action="store_true",
        help="Recompute saved pairing arrays even if cached .npz files already exist."
    )
    parser.add_argument(
        "--simple_model",
        action="store_true",
        help="Flag for simplified 2-particles per nucleotide model"
    )
    parser.add_argument(
        "--energy-from-csv",
        action="store_true",
        help=(
            "Read framewise CSVs from framewise_csvs/ and add last-5%% mean "
            "energy columns (bond_total, bond_bp, bond_pp, angle_*, dihedral_*) "
            "to the per-replicate and averaged output CSVs."
        )
    )
    args = parser.parse_args()

    sweep_path = BASE_DIR / args.sweep_dir

    if not sweep_path.is_dir():
        raise FileNotFoundError(f"Sweep directory not found: {sweep_path}")
    print(f"Reading from: {sweep_path}")

    df_per_rep, df_avg = analyze_trajectories(
        BASE_DIR,
        args.sweep_dir,
        pairing_force_recompute=args.pairing_force_recompute,
        simple_model=args.simple_model,
        energy_from_csv=args.energy_from_csv,
    )
    print(f"Writing results to: {sweep_path}")

    save_analysis_csvs(df_per_rep, df_avg, sweep_path)

    print(f"Analysis completed. Results saved in: {sweep_path}")


if __name__ == "__main__":
    main()
