#!/usr/bin/env python3

"""
Save .npz framewise pairing data and molecule_id (cutoff-based nlist, filtered) for single-bead multi-molecule systems.

Usage:
    python run_compute_ps.py <sweep_dir> <stride> <exclude_chain>

"""
from __future__ import annotations
from pathlib import Path
import sys
import argparse
import numpy as np
import freud
import networkx as nx  # do we need it here?

analysis_functions_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(analysis_functions_dir))

from analysis_scripts.helpers import (
    as_traj,
    read_param,
)


BASE_DIR = analysis_functions_dir / "bulk" / "output_dir"

from analysis_scripts.phase_separation import (
    compute_filtered_nlist_base_pairs_traj,
    make_molecule_lookup_table,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and save .npz file per replicate from freud.NeighborList object containing 3 arrays."
    )

    parser.add_argument(
        "sweep_dir",
        help="Name of the sweep directory inside BASE_DIR.",
    )

    parser.add_argument(
        "--stride",
        default=1,
        type=int,
        help="Stride determines every n-th frame to be processed.",
    )

    parser.add_argument(
        "--exclude_chain",
        type=int,
        default=3,
        help="How many consecutive base indexes are excluded from pairing? e.g. 3 -> 1-2, 1-3, 1-4 excluded",
    )

    parser.add_argument(
    "--run_ids",
    nargs="+",
    default=None,
    help="Optional numeric folder prefixes to process, e.g. 251 252 253.",
)
    parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Recompute outputs even if valid .npz files already exist.",
)

    args = parser.parse_args()
    # Confirm sweep path
    sweep_path = BASE_DIR / args.sweep_dir
    if not sweep_path.is_dir():
        raise FileNotFoundError(f"Sweep directory not found: {sweep_path}")

    print(f"Reading from: {sweep_path}")
    if args.run_ids is None:
        traj_paths = sorted(sweep_path.glob("*/trajectory_*.gsd")) # get list of paths for each traj file
    else: # in case we run script only on specific run folders
        traj_paths = []
        for run_id in args.run_ids:
            matches = sorted(sweep_path.glob(f"{run_id}_*/trajectory_*.gsd"))
            traj_paths.extend(matches)


    # Folder structure + calculation of pairing for each traj file
    for traj_path in traj_paths:
        run_dir = traj_path.parent  # run folder
        output_dir = run_dir / "computed_metrics" / "pairing"
        output_dir.mkdir(parents=True, exist_ok=True)  # create the output folder
        output_file = output_dir / f"{traj_path.stem}_pairing.npz"
        
        if output_file.exists() and not args.overwrite:
            print(f"Skipping {traj_path}: output already exists at {output_file}")
            continue

        if output_file.exists() and args.overwrite:
            print(f"Overwriting existing output: {output_file}")
            output_file.unlink()
        r_att = read_param(
            params_path=run_dir / "params.json", key="r_att_base"
        )  # get attraction radius
        r_rep = read_param(
            params_path=run_dir / "params.json", key="r_rep_base"
        )  # get rep radius
        r_opt = (r_att + r_rep) / 2
        print(f"Processing {traj_path}. Cutoff radius = {r_att}, r_opt = {r_opt}")

        with as_traj(
            traj_file=traj_path
        ) as traj:  # open each traj, calculate, save npz
            # Save molecule lookup once per run_dir
            mol_lookup_file = output_dir / "molecule_lookup.npy"
            if not mol_lookup_file.exists():
                molecule_id = make_molecule_lookup_table(traj[0])
                np.save(mol_lookup_file, molecule_id)
            nlist_dict = compute_filtered_nlist_base_pairs_traj(
                traj=traj,
                r_att=r_att,
                exclude_chain=args.exclude_chain,
                stride=args.stride,
                r_opt=r_opt,
            )
            np.savez(output_file, **nlist_dict)


if __name__ == "__main__":
    main()
