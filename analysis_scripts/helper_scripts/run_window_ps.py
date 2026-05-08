#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

analysis_functions_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(analysis_functions_dir))

from analysis_scripts.helpers import load_npz
from analysis_scripts.phase_separation import (
    count_molecule_contacts_from_all_arrays,
    window_ps_rows_from_pair_counts_all_frames,
)

BASE_DIR = analysis_functions_dir / "bulk" / "output_dir"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build per-trajectory persistent-window phase-separation CSVs from pairing .npz files."
    )
    parser.add_argument("sweep_dir", help="Name of the sweep directory inside BASE_DIR")
    parser.add_argument("--min-intermol-bonds", type=int, default=2)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--min-num-frames", type=int, default=2)
    args = parser.parse_args()

    sweep_path = BASE_DIR / args.sweep_dir
    if not sweep_path.is_dir():
        raise FileNotFoundError(f"Sweep directory not found: {sweep_path}")

    npz_paths = sorted(sweep_path.glob("*/computed_metrics/pairing/*_pairing.npz"))
    if not npz_paths:
        raise FileNotFoundError(f"No pairing .npz files found under: {sweep_path}")

    n_done = 0
    n_failed = 0

    for npz_path in npz_paths:
        run_dir = npz_path.parent.parent.parent
        mol_id_path = npz_path.parent / "molecule_lookup.npy"

        out_dir = run_dir / "window"
        out_dir.mkdir(exist_ok=True)
        out_csv = out_dir / f"{npz_path.stem}_window_ps.csv"

        if not mol_id_path.exists():
            print(f"Skipping {npz_path}: missing molecule_lookup.npy")
            n_failed += 1
            continue

        try:
            data = load_npz(npz_path)
            molecule_id = np.load(mol_id_path)
            n_molecules = int(molecule_id[-1]) + 1

            pair_counts_all_frames = count_molecule_contacts_from_all_arrays(
                data, molecule_id
            )
            rows = window_ps_rows_from_pair_counts_all_frames(
                pair_counts_all_frames=pair_counts_all_frames,
                n_molecules=n_molecules,
                window=args.window,
                min_num_frames=args.min_num_frames,
                min_intermol_bonds=args.min_intermol_bonds,
            )
            df = pd.DataFrame(rows)
            df.to_csv(out_csv, index=False)

            print(f"Saved: {out_csv}")
            n_done += 1

        except Exception as e:
            print(f"Failed on {npz_path}: {e}")
            n_failed += 1

    print(f"Finished. Saved {n_done} window CSVs. Failed: {n_failed}")


if __name__ == "__main__":
    main()