#!/usr/bin/env python3

"""
Run framewise analysis for trajectories in a sweep directory.

Usage:
    python run_time_df.py <sweep_dir>

Reads trajectories from:
    BASE_DIR/<sweep_dir>/*/trajectory_*.gsd

Writes per-frame CSVs into:
    BASE_DIR/<sweep_dir>/<run_folder>/framewise_csvs/
"""

import argparse
from pathlib import Path
import sys

analysis_functions_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(analysis_functions_dir))

from analysis_scripts.analyze_leona import make_framewise_metrics_dataframe

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
        "--stride",
        type=int,
        default=1,
        help="Frame stride for framewise metrics."
    )
    parser.add_argument(
        "--rg-scale",
        type=float,
        default=0.6,
        help="Scale factor passed to rg_calc."
    )
    parser.add_argument(
        "--exclude-chain",
        type=int,
        default=2,
        help="Sequence-distance exclusion for closest-partner search."
    )
    args = parser.parse_args()

    sweep_path = BASE_DIR / args.sweep_dir

    if not sweep_path.is_dir():
        raise FileNotFoundError(f"Sweep directory not found: {sweep_path}")

    print(f"Reading from: {sweep_path}")

    traj_paths = sorted(sweep_path.glob("*/trajectory_*.gsd"))
    if not traj_paths:
        raise FileNotFoundError(
            f"No trajectory_*.gsd files found under: {sweep_path}")

    n_done = 0
    n_failed = 0

    for traj_path in traj_paths:
        folder = traj_path.parent
        params_path = folder / "params.json"
        cache_dir = folder / "pairing_cache"
        cache_dir.mkdir(exist_ok=True)
        cache_path = cache_dir / f"{traj_path.stem}_pairing_cache.npz"
        
        # create per-run output folder
        out_dir = folder / "framewise_csvs"
        out_dir.mkdir(exist_ok=True)

        # save inside that folder
        out_csv = out_dir / f"{traj_path.stem}_framewise.csv"

        if not params_path.exists():
            print(f"Skipping {traj_path}: missing params.json")
            n_failed += 1
            continue

        try:
            print(f"Processing: {traj_path}")
            print(f"Writing to:   {out_csv}")

            df = make_framewise_metrics_dataframe(
                traj_file=str(traj_path),
                params_path=str(params_path),
                cache_path=str(cache_path),
                exclude_chain=args.exclude_chain,
                stride=args.stride,
                rg_scale=args.rg_scale,
                pairing_force_recompute=args.pairing_force_recompute,
            )

            df.to_csv(out_csv, index=False)
            n_done += 1

        except Exception as e:
            print(f"Failed on {traj_path}: {e}")
            n_failed += 1

    print(f"Finished. Saved {n_done} framewise CSVs. Failed: {n_failed}")


if __name__ == "__main__":
    main()
