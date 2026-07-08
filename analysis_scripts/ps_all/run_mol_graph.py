#!/usr/bin/env python3

"""
Build molecule-level contact edge-list .npz files from saved particle-level pairing .npz files.

Input per run:
    computed_metrics/pairing/*_pairing.npz
    computed_metrics/pairing/molecule_lookup.npy

Output per trajectory:
    computed_metrics/mol_graph/*_molecule_contact_edges.npz

Output NPZ format:
    frame          shape (n_edges_total,)
    mol_a          shape (n_edges_total,)
    mol_b          shape (n_edges_total,)
    n_contacts     shape (n_edges_total,)
    frame_indices  shape (n_frames,)
    n_molecules    shape (1,)

Usage:
    python run_mol_graph.py <sweep_dir>
    python run_mol_graph.py <sweep_dir> --run_ids 251 252 253
    python run_mol_graph.py <sweep_dir> --overwrite
"""

from __future__ import annotations

from pathlib import Path
import sys
import argparse

import numpy as np


analysis_functions_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(analysis_functions_dir))


BASE_DIR = analysis_functions_dir / "bulk" / "output_dir"


from analysis_scripts.phase_separation import (
    build_molecule_contact_edge_dict_from_pairing_npz,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build molecule-level contact edge-list .npz files from saved pairing .npz files."
    )

    parser.add_argument(
        "sweep_dir",
        help="Name of the sweep directory inside BASE_DIR.",
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
        help="Recompute outputs even if valid molecule contact edge files already exist.",
    )
    parser.add_argument(
    "--folder_name_suffix",
        type=str,
        default="",
        help="Suffix for both pairing input dir and mol_graph output dir, e.g. '_wide' -> pairing_wide/ and mol_graph_wide/.",
    )

    args = parser.parse_args()

    sweep_path = BASE_DIR / args.sweep_dir

    if not sweep_path.is_dir():
        raise FileNotFoundError(f"Sweep directory not found: {sweep_path}")

    print(f"Reading from: {sweep_path}")

    if args.run_ids is None:
        run_dirs = sorted(path for path in sweep_path.iterdir() if path.is_dir())
    else:
        run_dirs = []
        for run_id in args.run_ids:
            matches = sorted(sweep_path.glob(f"{run_id}_*"))
            run_dirs.extend(path for path in matches if path.is_dir())

    for run_dir in run_dirs:
        suffix = args.folder_name_suffix
        pairing_dir = run_dir / f"computed_metrics{suffix}" / "pairing"
        output_dir = run_dir / f"computed_metrics{suffix}" / "mol_graph"
        output_dir.mkdir(parents=True, exist_ok=True)

        mol_lookup_file = pairing_dir / "molecule_lookup.npy"

        if not mol_lookup_file.exists():
            print(f"Skipping {run_dir}: molecule lookup not found at {mol_lookup_file}")
            continue

        pairing_files = sorted(pairing_dir.glob("*_pairing.npz"))

        if len(pairing_files) == 0:
            print(f"Skipping {run_dir}: no pairing NPZ files found in {pairing_dir}")
            continue

        molecule_id = np.load(mol_lookup_file)

        for pairing_file in pairing_files:
            output_file = output_dir / pairing_file.name.replace(
                "_pairing.npz",
                "_molecule_contact_edges.npz",
            )

            if output_file.exists() and not args.overwrite:
                print(f"Skipping {pairing_file}: output already exists at {output_file}")
                continue

            if output_file.exists() and args.overwrite:
                print(f"Overwriting existing output: {output_file}")
                output_file.unlink()

            print(f"Processing {pairing_file}")

            with np.load(pairing_file) as pairing_npz:
                edge_dict = build_molecule_contact_edge_dict_from_pairing_npz(
                    pairing_npz=pairing_npz,
                    molecule_id=molecule_id,
                )

            np.savez(output_file, **edge_dict)

            print(f"Saved molecule contact edges: {output_file}")


if __name__ == "__main__":
    main()