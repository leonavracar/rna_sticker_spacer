#!/usr/bin/env python3

from __future__ import annotations
import pandas as pd
import traceback

import argparse
import json
import sys
from pathlib import Path

analysis_functions_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(analysis_functions_dir))
from analysis_scripts.framewise import (
    compute_energy_terms,
    compute_observables,
    compute_framewise_metrics,
)
from analysis_scripts.pairing import compute_pairing_arrays
from analysis_scripts.geometry import compute_geometry
from analysis_scripts.helpers import as_traj, load_npz, save_npz, read_param


BASE_DIR = analysis_functions_dir / "bulk" / "output_dir"


# def read_param(params_path: Path, key: str) -> float | int | str | None:
#     """Read one parameter from a params.json file."""
#     with open(params_path, "r", encoding="utf-8") as f:
#         data = json.load(f)
#     return data.get(key)


def validate_requested_flags(
    args: argparse.Namespace,
) -> tuple[bool, bool, bool, bool, bool]:
    """Resolve compute flags from CLI arguments."""
    do_pairing = args.all or args.pairing
    do_geometry = args.all or args.geometry
    do_observables = args.all or args.observables
    do_energy = args.all or args.energy
    do_framewise = args.all or args.framewise

    if not any([do_pairing, do_geometry, do_observables, do_energy, do_framewise]):
        raise ValueError(
            "Nothing selected. Use --all or at least one of "
            "--pairing, --geometry, --observables, --energy, --framewise."
        )

    return do_pairing, do_geometry, do_observables, do_energy, do_framewise


def ensure_metric_dirs(run_dir: Path) -> dict[str, Path]:
    """Create and return output directories for all metric types."""
    metrics_root = run_dir / "computed_metrics"
    dirs = {
        "root": metrics_root,
        "pairing": metrics_root / "pairing",
        "geometry": metrics_root / "geometry",
        "observables": metrics_root / "observables",
        "energy": metrics_root / "energy",
        "framewise": metrics_root / "framewise",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def load_or_compute_needed(
    traj,
    traj_path: Path,
    params_path: Path,
    metric_dirs: dict[str, Path],
    do_pairing: bool,
    do_geometry: bool,
    do_observables: bool,
    do_energy: bool,
    do_framewise: bool,
    overwrite: bool,
    stride: int,
    rg_scale: float,
    exclude_chain: int,
    cutoff: float | None,
):
    """Load cached data or compute and save what is needed."""
    stem = traj_path.stem

    pairing_path = metric_dirs["pairing"] / f"{stem}_pairing.npz"
    geometry_path = metric_dirs["geometry"] / f"{stem}_geometry.npz"
    observables_path = metric_dirs["observables"] / f"{stem}_observables.npz"
    energy_path = metric_dirs["energy"] / f"{stem}_energy.npz"

    need_pairing = do_pairing or do_framewise
    need_geometry = do_geometry or do_framewise
    need_observables = do_observables or do_framewise
    need_energy = do_energy or do_framewise

    pairing = None
    geometry = None
    observables = None
    energy = None

    resolved_cutoff = cutoff
    if need_pairing and resolved_cutoff is None:
        if not params_path.exists():
            raise FileNotFoundError(
                f"Missing params.json for pairing cutoff: {params_path}"
            )
        resolved_cutoff = read_param(params_path, "r_att_base")
        if resolved_cutoff is None:
            raise ValueError(f"Could not read 'r_att_base' from {params_path}")

    if need_pairing:
        if pairing_path.exists() and not overwrite:
            print(f"  loading pairing     -> {pairing_path.name}")
            pairing = load_npz(pairing_path)
        else:
            print(f"  computing pairing   -> {pairing_path.name}")
            pairing = compute_pairing_arrays(
                traj,
                cutoff=float(resolved_cutoff),
                exclude_chain=exclude_chain,
                stride=stride,
            )
            save_npz(pairing_path, pairing)

    if need_geometry:
        if geometry_path.exists() and not overwrite:
            print(f"  loading geometry    -> {geometry_path.name}")
            geometry = load_npz(geometry_path)
        else:
            print(f"  computing geometry  -> {geometry_path.name}")
            geometry = compute_geometry(traj, stride=stride)
            save_npz(geometry_path, geometry)

    if need_observables:
        if observables_path.exists() and not overwrite:
            print(f"  loading observables -> {observables_path.name}")
            observables = load_npz(observables_path)
        else:
            print(f"  computing observables -> {observables_path.name}")
            observables = compute_observables(
                traj,
                rg_scale=rg_scale,
                ke_mass=1.0,
                ke_per_particle=True,
                stride=stride,
            )
            save_npz(observables_path, observables)

    if need_energy:
        if energy_path.exists() and not overwrite:
            print(f"  loading energy      -> {energy_path.name}")
            energy = load_npz(energy_path)
        else:
            print(f"  computing energy    -> {energy_path.name}")
            energy = compute_energy_terms(traj, stride=stride)
            save_npz(energy_path, energy)

    return pairing, geometry, observables, energy


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute pairing, geometry, observables, energy, and framewise CSVs for trajectories in a sweep."
    )
    parser.add_argument(
        "sweep_dir",
        help="Name of the sweep directory inside BASE_DIR.",
    )
    parser.add_argument("--pairing", action="store_true", help="Compute pairing cache.")
    parser.add_argument(
        "--geometry", action="store_true", help="Compute geometry cache."
    )
    parser.add_argument(
        "--observables", action="store_true", help="Compute observables cache."
    )
    parser.add_argument("--energy", action="store_true", help="Compute energy cache.")
    parser.add_argument("--framewise", action="store_true", help="Build framewise CSV.")
    parser.add_argument("--all", action="store_true", help="Compute everything.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute outputs even if cached files already exist.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Frame stride for all computations.",
    )
    parser.add_argument(
        "--rg-scale",
        type=float,
        default=0.6,
        help="Scale factor for radius of gyration.",
    )
    parser.add_argument(
        "--exclude-chain",
        type=int,
        default=2,
        help="Sequence-distance exclusion for pairing.",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=None,
        help="Pairing cutoff. If not provided, read r_att_base from params.json.",
    )
    args = parser.parse_args()

    if args.stride < 1:
        raise ValueError("--stride must be >= 1")

    do_pairing, do_geometry, do_observables, do_energy, do_framewise = (
        validate_requested_flags(args)
    )

    sweep_path = BASE_DIR / args.sweep_dir
    if not sweep_path.is_dir():
        raise FileNotFoundError(f"Sweep directory not found: {sweep_path}")

    print(f"Reading from: {sweep_path}")

    traj_paths = sorted(sweep_path.glob("*/trajectory_*.gsd"))
    if not traj_paths:
        raise FileNotFoundError(f"No trajectory_*.gsd files found under: {sweep_path}")

    n_done = 0
    n_failed = 0

    for traj_path in traj_paths:
        run_dir = traj_path.parent
        params_path = run_dir / "params.json"
        metric_dirs = ensure_metric_dirs(run_dir)
        framewise_csv = metric_dirs["framewise"] / f"{traj_path.stem}_framewise.csv"

        print(f"\nProcessing: {traj_path}")

        try:
            with as_traj(traj_path) as traj:
                pairing, geometry, observables, energy = load_or_compute_needed(
                    traj=traj,
                    traj_path=traj_path,
                    params_path=params_path,
                    metric_dirs=metric_dirs,
                    do_pairing=do_pairing,
                    do_geometry=do_geometry,
                    do_observables=do_observables,
                    do_energy=do_energy,
                    do_framewise=do_framewise,
                    overwrite=args.overwrite,
                    stride=args.stride,
                    rg_scale=args.rg_scale,
                    exclude_chain=args.exclude_chain,
                    cutoff=args.cutoff,
                )

            if do_framewise:
                print(f"  building framewise  -> {framewise_csv.name}")

                if pairing is None:
                    pairing = load_npz(
                        metric_dirs["pairing"] / f"{traj_path.stem}_pairing.npz"
                    )
                if geometry is None:
                    geometry = load_npz(
                        metric_dirs["geometry"] / f"{traj_path.stem}_geometry.npz"
                    )
                if observables is None:
                    observables = load_npz(
                        metric_dirs["observables"] / f"{traj_path.stem}_observables.npz"
                    )
                if energy is None:
                    energy = load_npz(
                        metric_dirs["energy"] / f"{traj_path.stem}_energy.npz"
                    )

                df = compute_framewise_metrics(
                    energy=energy,
                    geometry=geometry,
                    pairing=pairing,
                    observables=observables,
                    expected=None,  # set this if you want Q against a fixed expected map
                )
                df.to_csv(framewise_csv, index=False)

            n_done += 1
            print("  done")

        except Exception as e:
            print(f"  failed: {type(e).__name__}: {e!r}")
            traceback.print_exc()
            n_failed += 1

    print(f"\nFinished. Successful: {n_done}. Failed: {n_failed}.")


if __name__ == "__main__":
    main()
