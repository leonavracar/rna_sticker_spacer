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

from analysis_scripts.helpers import as_traj, read_param
from analysis_scripts.phase_separation import (
    component_size_distribution_rows_from_edge_npz,
)

BASE_DIR = analysis_functions_dir / "bulk" / "output_dir"

def add_homogeneous_material_columns(
    df: pd.DataFrame,
    n_molecules_total: int,
    n_particles_total: int,
) -> pd.DataFrame:
    """Add material-normalized cluster columns for homogeneous molecule-length systems.

    Assumes all molecules in this run have the same number of particles.
    """
    if n_molecules_total <= 0:
        raise ValueError("n_molecules_total must be > 0.")

    if n_particles_total <= 0:
        raise ValueError("n_particles_total must be > 0.")

    if n_particles_total % n_molecules_total != 0:
        raise ValueError(
            "n_particles_total is not divisible by n_molecules_total. "
            "This script only supports homogeneous molecule-length systems."
        )

    df = df.copy()

    mol_len = n_particles_total // n_molecules_total

    df["mol_len"] = int(mol_len)
    df["n_molecules_total"] = int(n_molecules_total)
    df["n_particles_total"] = int(n_particles_total)

    df["one_cluster_mol_frac"] = (
        df["cluster_size"] / df["n_molecules_total"]
    )
    

    df["one_cluster_particle_count"] = (
        df["cluster_size"] * df["mol_len"]
    )

    df["one_cluster_particle_frac"] = (
        df["one_cluster_particle_count"] / df["n_particles_total"]
    )

    df["n_particles_in_clusters"] = (
        df["n_mols_in_clusters"] * df["mol_len"]
    )

    df["frac_particles"] = (
        df["n_particles_in_clusters"] / df["n_particles_total"]
    )

    df["mass_weighted_cluster_score_contrib"] = (
        df["n_clusters"] * df["one_cluster_particle_frac"] ** 2
    )

    df["mass_weighted_size_contrib_mol"] = (
        df["frac_particles"] * df["cluster_size"]
    )

    df["mass_weighted_size_contrib_particle"] = (
        df["frac_particles"] * df["one_cluster_particle_count"]
    )

    return df

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build one long-format framewise cluster-size distribution CSV from molecule-edge NPZ files."
    )

    parser.add_argument(
        "sweep_dir",
        help="Name of the sweep directory inside BASE_DIR.",
    )

    parser.add_argument(
        "--min_contacts",
        type=int,
        default=2,
        help="Minimum molecule-pair contacts required for an edge.",
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
        help="Overwrite analysis_clusters.csv if it already exists.",
    )
    parser.add_argument(
        "--folder_name_suffix",
        type=str,
        default="",
        help="Suffix for computed_metrics dir and output CSV, e.g. '_rmax07'.",
    )
    args = parser.parse_args()

    sweep_path = BASE_DIR / args.sweep_dir

    if not sweep_path.is_dir():
        raise FileNotFoundError(f"Sweep directory not found: {sweep_path}")

    suffix = args.folder_name_suffix
    out_csv = sweep_path / f"analysis_clusters_{suffix}.csv"

    if out_csv.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {out_csv}. Use --overwrite to replace it."
        )

    print(f"Reading from: {sweep_path}", flush=True)

    if args.run_ids is None:
        edge_npz_paths = sorted(
            sweep_path.glob(
                f"*/computed_metrics_{suffix}/mol_graph/*_molecule_contact_edges.npz"
            )
        )
    else:
        edge_npz_paths = []
        for run_id in args.run_ids:
            edge_npz_paths.extend(
                sorted(
                    sweep_path.glob(
                        f"{run_id}_*/computed_metrics_{suffix}/mol_graph/*_molecule_contact_edges.npz"
                    )
                )
            )

    if not edge_npz_paths:
        raise FileNotFoundError(
            f"No molecule contact edge NPZ files found under: {sweep_path}"
        )

    all_dfs = []
    n_done = 0
    n_failed = 0

    for edge_npz_path in edge_npz_paths:
        run_dir = edge_npz_path.parent.parent.parent

        try:
            print(f"Processing {edge_npz_path}")

            with np.load(edge_npz_path) as edge_npz:
                n_molecules_total = int(edge_npz["n_molecules"][0])

                rows = component_size_distribution_rows_from_edge_npz(
                    edge_npz=edge_npz,
                    min_contacts=args.min_contacts,
                )

            df = pd.DataFrame(rows)

            if df.empty:
                print(f"Skipping {edge_npz_path}: no rows generated")
                n_failed += 1
                continue

            traj_stem = edge_npz_path.stem.removesuffix(
                "_molecule_contact_edges"
            )
            traj_path = run_dir / f"{traj_stem}.gsd"

            with as_traj(traj_path) as traj:
                n_particles_total = int(traj[0].particles.N)

                steps_by_frame = {
                    int(frame): int(traj[int(frame)].configuration.step)
                    for frame in sorted(df["frame"].unique())
                }

            df["step"] = df["frame"].map(steps_by_frame)

            df = add_homogeneous_material_columns(
                df=df,
                n_molecules_total=n_molecules_total,
                n_particles_total=n_particles_total,
            )

            run_number = run_dir.name.split("_", 1)[0]
            traj_id = traj_stem.replace("trajectory_", "traj_")

            df["run_id"] = f"{run_number}__{traj_id}"
            df["folder_name"] = run_dir.name
            df["traj_replicate"] = traj_stem

            params_path = run_dir / "params.json"

            df["E_att"] = read_param(params_path, "E_att")
            df["angle_k_ppp"] = read_param(params_path, "angle_k_ppp")
            df["density"] = read_param(params_path, "density")
            df["molecule_file"] = read_param(params_path, "molecule_file")

            # Keep the most useful columns first.
            first_cols = [
                "run_id",
                "folder_name",
                "traj_replicate",
                "frame",
                "step",
                "E_att",
                "angle_k_ppp",
                "density",
                "molecule_file",
                "min_contacts",

                "mol_len",
                "n_molecules_total",
                "n_particles_total",

                "cluster_size",
                "n_clusters",
                "n_mols_in_clusters",
                "frac_mols",

                "one_cluster_mol_frac",
                "one_cluster_particle_count",
                "one_cluster_particle_frac",
                "n_particles_in_clusters",
                "frac_particles",

                "mass_weighted_cluster_score_contrib",
                "mass_weighted_size_contrib_mol",
                "mass_weighted_size_contrib_particle",
            ]

            other_cols = [col for col in df.columns if col not in first_cols]
            df = df[first_cols + other_cols]

            all_dfs.append(df)
            n_done += 1

        except Exception as exc:
            print(f"Failed on {edge_npz_path}: {type(exc).__name__}: {exc!r}")
            traceback.print_exc(limit=2)
            n_failed += 1

    if not all_dfs:
        raise RuntimeError("No cluster-distribution dataframes were generated.")

    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df.to_csv(out_csv, index=False)

    print(f"Saved combined cluster analysis CSV: {out_csv}")
    print(f"Finished. Saved {n_done} trajectories. Failed: {n_failed}")


if __name__ == "__main__":
    main()