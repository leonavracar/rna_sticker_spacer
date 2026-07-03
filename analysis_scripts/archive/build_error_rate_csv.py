#!/usr/bin/env python3
"""
Build a standalone error-rate CSV from pairing .npz files.

Columns: frame, error_rate, run_id, folder_name, traj_replicate,
         molecule_file, E_att, angle_k_ppp, density
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

analysis_functions_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(analysis_functions_dir))

from analysis_scripts.helpers import read_param

BASE_DIR = analysis_functions_dir / "bulk" / "output_dir"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a standalone error-rate CSV from pairing .npz files."
    )
    parser.add_argument("sweep_dir", help="Name of the sweep directory inside BASE_DIR.")
    parser.add_argument(
        "--run_ids",
        nargs="+",
        default=None,
        help="Optional numeric folder prefixes to process, e.g. 0 1 2.",
    )
    args = parser.parse_args()

    sweep_path = BASE_DIR / args.sweep_dir
    if not sweep_path.is_dir():
        raise FileNotFoundError(f"Sweep directory not found: {sweep_path}")

    if args.run_ids is None:
        npz_paths = sorted(sweep_path.glob("*/computed_metrics/pairing/*_pairing.npz"))
    else:
        npz_paths = []
        for run_id in args.run_ids:
            npz_paths.extend(
                sorted(sweep_path.glob(f"{run_id}_*/computed_metrics/pairing/*_pairing.npz"))
            )

    if not npz_paths:
        raise FileNotFoundError(f"No pairing .npz files under: {sweep_path}")

    all_rows = []
    n_done, n_skip = 0, 0

    for npz_path in npz_paths:
        run_dir = npz_path.parent.parent.parent
        traj_stem = npz_path.stem.removesuffix("_pairing")

        try:
            with np.load(npz_path) as data:
                if "error_rates" not in data.files:
                    print(f"Skipping {npz_path}: no 'error_rates' in npz")
                    n_skip += 1
                    continue
                frame_indices = data["frame_indices"]
                error_rates   = data["error_rates"]

            df = pd.DataFrame({
                "frame": frame_indices,
                "error_rate": error_rates,
            })

            run_number = run_dir.name.split("_", 1)[0]
            traj_id = traj_stem.replace("trajectory_", "traj_")
            df["run_id"] = f"{run_number}__{traj_id}"
            df["folder_name"] = run_dir.name
            df["traj_replicate"] = traj_stem

            params_json = run_dir / "params.json"
            df["molecule_file"] = read_param(params_json, "molecule_file")
            df["E_att"] = read_param(params_json, "E_att")
            df["angle_k_ppp"] = read_param(params_json, "angle_k_ppp")
            df["density"] = read_param(params_json, "density")

            all_rows.append(df)
            n_done += 1

        except Exception as e:
            print(f"Failed on {npz_path}: {type(e).__name__}: {e!r}")
            n_skip += 1

    if not all_rows:
        print("No data collected.")
        return

    combined = pd.concat(all_rows, ignore_index=True)
    out_csv = sweep_path / "analysis_error_rate.csv"
    combined.to_csv(out_csv, index=False)
    print(
        f"Saved: {out_csv}  "
        f"({n_done} runs, {len(combined)} rows, skipped {n_skip})"
    )


if __name__ == "__main__":
    main()