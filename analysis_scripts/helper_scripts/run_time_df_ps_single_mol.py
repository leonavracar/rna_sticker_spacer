#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

analysis_functions_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(analysis_functions_dir))
from analysis_scripts.helper_scripts.utils import unwrap_frame
from analysis_scripts.helpers import load_npz, as_traj, read_param
from analysis_scripts.phase_separation import (
    framewise_single_chain_rows_from_nlist_dict,
    calc_rg_from_positions,
    calc_end_to_end_from_positions,
    get_log_KE_over_time,
    get_energy_terms,
)


BASE_DIR = analysis_functions_dir / "bulk" / "output_dir"


ENERGY_KEYS = [
    "md/compute/ThermodynamicQuantities/potential_energy",
    "sqd_plugin/sqd_potential/SquareDensity/energy",
    "md/bond/Harmonic/energy",
    "md/angle/Harmonic/energy",
]


def build_contact_recurrence_df(
    nlist_dict: dict[str, np.ndarray],
    recurrence_window: int = 25,
    fixed_bin_size: int = 5,
    repeat_size: int = 3,
    repeat_window: int = 3,
) -> pd.DataFrame:
    """
    Build single-chain contact recurrence dataframe for all binning modes.
    """
    modes = [
        (
            "particle",
            {
                "bin_mode": "particle",
                "bin_size": fixed_bin_size,
                "repeat_size": repeat_size,
                "repeat_window": repeat_window,
            },
        ),
        (
            "fixed",
            {
                "bin_mode": "fixed",
                "bin_size": fixed_bin_size,
                "repeat_size": repeat_size,
                "repeat_window": repeat_window,
            },
        ),
        (
            "repeat",
            {
                "bin_mode": "repeat",
                "bin_size": fixed_bin_size,
                "repeat_size": repeat_size,
                "repeat_window": repeat_window,
            },
        ),
        (
            "repeat_window",
            {
                "bin_mode": "repeat_window",
                "bin_size": fixed_bin_size,
                "repeat_size": repeat_size,
                "repeat_window": repeat_window,
            },
        ),
    ]

    df_out: pd.DataFrame | None = None

    for suffix, kwargs in modes:
        rows = framewise_single_chain_rows_from_nlist_dict(
            nlist_dict=nlist_dict,
            recurrence_window=recurrence_window,
            **kwargs,
        )

        df_mode = pd.DataFrame(rows)

        df_mode = df_mode.rename(
            columns={
                "contact_regions": f"contact_regions_{suffix}",
                "contact_region_recurrence_window": (
                    f"contact_region_recurrence_{suffix}"
                ),
            }
        )

        keep_cols = [
            "frame",
            f"contact_regions_{suffix}",
            f"contact_region_recurrence_{suffix}",
        ]

        if df_out is None:
            df_out = df_mode[
                ["frame", "particle_contacts", *keep_cols[1:]]
            ].copy()
        else:
            df_out = df_out.join(
                df_mode[keep_cols].set_index("frame"),
                on="frame",
            )

    if df_out is None:
        raise ValueError("No contact recurrence rows were generated.")

    return df_out


def build_traj_metric_df(
    traj,
    frame_indices: np.ndarray,
    scale: float = 1.0,
) -> pd.DataFrame:
    """
    Build framewise trajectory metrics for selected frames.
    """
    rows = []

    for frame_idx in frame_indices:
        frame_idx = int(frame_idx)
        frame = traj[frame_idx]

        positions = unwrap_frame(frame)

        rows.append(
            {
                "frame": frame_idx,
                "step": int(frame.configuration.step),
                "rg": calc_rg_from_positions(positions, scale=scale),
                "end_to_end_distance": calc_end_to_end_from_positions(
                    positions,
                    scale=scale,
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build per-trajectory framewise single-molecule CSVs from pairing .npz files."
    )

    parser.add_argument(
        "sweep_dir",
        help="Name of the sweep directory inside BASE_DIR.",
    )

    parser.add_argument(
        "--recurrence-window",
        type=int,
        default=25,
        help="Rolling window for contact recurrence.",
    )

    parser.add_argument(
        "--fixed-bin-size",
        type=int,
        default=5,
        help="Bin size for fixed contact-region binning.",
    )

    parser.add_argument(
        "--repeat-size",
        type=int,
        default=3,
        help="Repeat unit size. For CAG, use 3.",
    )

    parser.add_argument(
        "--repeat-window",
        type=int,
        default=3,
        help="Number of repeats per repeat-window region.",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=0.6,
        help="Length scaling factor for Rg and end-to-end distance.",
    )

    parser.add_argument(
        "--run_ids",
        nargs="+",
        default=None,
        help="Optional numeric folder prefixes to process, e.g. 251 252 253.",
    )

    args = parser.parse_args()

    sweep_path = BASE_DIR / args.sweep_dir

    if not sweep_path.is_dir():
        raise FileNotFoundError(f"Sweep directory not found: {sweep_path}")

    if args.run_ids is None:
        npz_paths = sorted(
            sweep_path.glob("*/computed_metrics/pairing/*_pairing.npz")
        )
    else:
        npz_paths = []
        for run_id in args.run_ids:
            npz_paths.extend(
                sorted(
                    sweep_path.glob(
                        f"{run_id}_*/computed_metrics/pairing/*_pairing.npz"
                    )
                )
            )

    if not npz_paths:
        raise FileNotFoundError(f"No pairing .npz files found under: {sweep_path}")

    n_done = 0
    n_failed = 0
    all_dfs = []

    for npz_path in npz_paths:
        run_dir = npz_path.parent.parent.parent

        out_dir = run_dir / "framewise"
        out_dir.mkdir(exist_ok=True)

        out_csv = out_dir / f"{npz_path.stem}_framewise_ps.csv"

        try:
            data = load_npz(npz_path)

            df_contacts = build_contact_recurrence_df(
                nlist_dict=data,
                recurrence_window=args.recurrence_window,
                fixed_bin_size=args.fixed_bin_size,
                repeat_size=args.repeat_size,
                repeat_window=args.repeat_window,
            )
            # add error rate
            df_err = pd.DataFrame({
                "frame": data["frame_indices"],
                "error_rate": data["error_rates"],
            })
            df_contacts = df_contacts.merge(df_err, on="frame", how="left")
            
            traj_stem = npz_path.stem.removesuffix("_pairing")
            traj_path = run_dir / f"{traj_stem}.gsd"

            with as_traj(traj_path) as traj:
                frame_idx = df_contacts["frame"].to_numpy(dtype=int)

                df_traj = build_traj_metric_df(
                    traj=traj,
                    frame_indices=frame_idx,
                    scale=args.scale,
                )

                try:
                    ke = get_log_KE_over_time(traj)
                except KeyError:
                    ke = None

                df = df_contacts.join(
                    df_traj.set_index("frame"),
                    on="frame",
                )

                if all(key in traj[0].log for key in ENERGY_KEYS):
                    energy_data = get_energy_terms(traj)
                    energy_df = pd.DataFrame(energy_data).set_index("frame")
                    df = df.join(energy_df, on="frame")

            dt = float(read_param(run_dir / "params.json", "dt"))

            df["time"] = df["step"] * dt
            df["ke_per_particle"] = (
                ke[frame_idx] if ke is not None else np.nan
            )

            run_number = run_dir.name.split("_", 1)[0]
            traj_id = traj_stem.replace("trajectory_", "traj_")

            df["run_id"] = f"{run_number}__{traj_id}"
            df["folder_name"] = run_dir.name
            df["traj_replicate"] = traj_stem
            df["molecule_file"] = read_param(
                run_dir / "params.json",
                "molecule_file",
            )
            df["dt"] = dt
            df["E_att"] = read_param(run_dir / "params.json", "E_att")
            df["E_rep"] = read_param(run_dir / "params.json", "E_rep")
            df["angle_k_ppp"] = read_param(
                run_dir / "params.json",
                "angle_k_ppp",
            )
            df["density"] = read_param(run_dir / "params.json", "density")

            df.to_csv(out_csv, index=False)
            all_dfs.append(df)

            print(f"Saved: {out_csv}")
            n_done += 1

        except Exception as error:
            print(f"Failed on {npz_path}: {type(error).__name__}: {error!r}")
            traceback.print_exc(limit=2)
            n_failed += 1

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_csv = sweep_path / "analysis_framewise.csv"
        combined_df.to_csv(combined_csv, index=False)
        print(f"Saved combined framewise CSV: {combined_csv}")

    print(f"Finished. Saved {n_done} framewise CSVs. Failed: {n_failed}")


if __name__ == "__main__":
    main()