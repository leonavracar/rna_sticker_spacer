#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback
import numpy as np
import pandas as pd

analysis_functions_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(analysis_functions_dir))

from analysis_scripts.helpers import load_npz, as_traj, read_param
from analysis_scripts.phase_separation import (
    framewise_ps_rows_from_nlist_dict,
    framewise_molecule_recurrence_rows_from_nlist_dict,
    get_log_KE_over_time,
    get_energy_terms,
)
BASE_DIR = analysis_functions_dir / "bulk" / "output_dir"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build per-trajectory framewise phase-separation CSVs from pairing .npz files."
    )
    parser.add_argument("sweep_dir", help="Name of the sweep directory inside BASE_DIR")
    parser.add_argument("--min-intermol-bonds", type=int, default=2)
    parser.add_argument("--recurrence-window", type=int, default=25)
    parser.add_argument(
    "--run_ids",
    nargs="+",
    default=None,
    help="Optional numeric folder prefixes to process, e.g. 18 19 20.",)
    args = parser.parse_args()

    sweep_path = BASE_DIR / args.sweep_dir
    if not sweep_path.is_dir():
        raise FileNotFoundError(f"Sweep directory not found: {sweep_path}")
    # Only running run_ids if provided
    if args.run_ids is None:
        npz_paths = sorted(sweep_path.glob("*/computed_metrics/pairing/*_pairing.npz"))
    else:
        npz_paths = []
        for run_id in args.run_ids:
            npz_paths.extend(
                sorted(sweep_path.glob(f"{run_id}_*/computed_metrics/pairing/*_pairing.npz"))
            )
    if not npz_paths:
        raise FileNotFoundError(f"No pairing .npz files found under: {sweep_path}")

    n_done = 0
    n_failed = 0


    for npz_path in npz_paths:
        run_dir = npz_path.parent.parent.parent
        mol_id_path = npz_path.parent / "molecule_lookup.npy"

        out_dir = run_dir / "framewise"
        out_dir.mkdir(exist_ok=True)
        out_csv = out_dir / f"{npz_path.stem}_framewise_ps.csv"

        if not mol_id_path.exists():
            print(f"Skipping {npz_path}: missing molecule_lookup.npy")
            n_failed += 1
            continue

        try:
            data = load_npz(npz_path)
            molecule_id = np.load(mol_id_path)

            rows = framewise_ps_rows_from_nlist_dict(
                nlist_dict=data,
                molecule_id=molecule_id,
                min_intermol_bonds=args.min_intermol_bonds,
            )
            df = pd.DataFrame(rows)
            df_err = pd.DataFrame({
                "frame": data["frame_indices"],
                "error_rate": data["error_rates"],
            })
            df = df.merge(df_err, on="frame", how="left")
            # Recurrence metrics
            rec_rows = framewise_molecule_recurrence_rows_from_nlist_dict(
                nlist_dict=data,
                molecule_id=molecule_id,
                recurrence_window=args.recurrence_window,
                min_intermol_bonds=args.min_intermol_bonds,
            )
            df_rec = pd.DataFrame(rec_rows)
            df = df.join(df_rec.set_index("frame"), on="frame") # joining with rest of df

            traj_stem = npz_path.stem.removesuffix("_pairing")
            traj_path = run_dir / f"{traj_stem}.gsd"

            with as_traj(traj_path) as traj:
                frame_idx = df["frame"].to_numpy(dtype=int) # get frame indices
                steps = np.array([traj[int(i)].configuration.step for i in frame_idx], dtype = np.int64) # convert to steps
                try: # KE only if logged
                    ke = get_log_KE_over_time(traj)
                except KeyError:
                    ke = None
                # Energy terms only if logged
                ENERGY_KEYS = [
                    "md/compute/ThermodynamicQuantities/potential_energy",
                    "sqd_plugin/sqd_potential/SquareDensity/energy",
                    "md/bond/Harmonic/energy",
                    "md/angle/Harmonic/energy",
                ]
                if all(k in traj[0].log for k in ENERGY_KEYS):
                    energy_data = get_energy_terms(traj)
                    energy_df = pd.DataFrame(energy_data).set_index("frame")
                    df = df.join(energy_df, on="frame")
                # KE over time
            
            dt = float(read_param(run_dir / "params.json", "dt"))
            df["step"] = steps
            df["time"] = df["step"] * dt
            df["ke_per_particle"] = ke[frame_idx] if ke is not None else np.nan
            # Metadata
            run_number = run_dir.name.split("_", 1)[0]
            traj_id = traj_stem.replace("trajectory_", "traj_")

            df["run_id"] = f"{run_number}__{traj_id}"
            df["folder_name"] = run_dir.name
            df["traj_replicate"] = traj_stem
            df["molecule_file"] = read_param(run_dir / "params.json", "molecule_file")
            df["dt"] = dt
            df["E_att"] = read_param(run_dir / "params.json", "E_att")
            df["angle_k_ppp"] = read_param(run_dir / "params.json", "angle_k_ppp")
            df["density"] = read_param(run_dir / "params.json", "density")
            # Make csv
            df.to_csv(out_csv, index=False)


            print(f"Saved: {out_csv}")
            n_done += 1

        except Exception as e:
            print(f"Failed on {npz_path}: {type(e).__name__}: {e!r}")
            traceback.print_exc(limit=2)
            n_failed += 1
    # Building the combined framewise
    framewise_csvs = sorted(sweep_path.glob("*/framewise/*_framewise_ps.csv"))

    if framewise_csvs:
        combined_df = pd.concat(
            [pd.read_csv(csv_path) for csv_path in framewise_csvs],
            ignore_index=True
        )
        combined_csv = sweep_path / "analysis_framewise.csv"
        combined_df.to_csv(combined_csv, index=False)
        print(f"Saved combined framewise CSV: {combined_csv}")
        print(f"Included {len(framewise_csvs)} per-run framewise CSVs")
        
    print(f"Finished. Saved {n_done} framewise CSVs. Failed: {n_failed}")


if __name__ == "__main__":
    main()