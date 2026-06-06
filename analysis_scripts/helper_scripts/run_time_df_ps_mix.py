#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
import traceback

import networkx as nx
import numpy as np
import pandas as pd

analysis_functions_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(analysis_functions_dir))

from analysis_scripts.helpers import load_npz, as_traj, read_param
from analysis_scripts.phase_separation import (
    framewise_ps_rows_from_nlist_dict,
    get_log_KE_over_time,
    get_energy_terms,
)

BASE_DIR = analysis_functions_dir / "bulk" / "output_dir"


def make_molecule_type_lookup(
    molecule_id: np.ndarray,
    type_lengths: list[int],
    type_labels: list[str],
) -> np.ndarray:
    """
    Infer molecule type from molecule bead count.

    Example:
        CAG10 -> 30 beads
        CAG47 -> 141 beads
    """
    if len(type_lengths) != len(type_labels):
        raise ValueError("--type-lengths and --type-labels must have same length")

    size_to_label = dict(zip(type_lengths, type_labels))

    n_molecules = int(molecule_id.max()) + 1
    molecule_sizes = np.bincount(molecule_id, minlength=n_molecules)

    molecule_type = np.empty(n_molecules, dtype=object)

    for mol_idx, size in enumerate(molecule_sizes):
        if int(size) not in size_to_label:
            raise ValueError(
                f"Molecule {mol_idx} has size {size}, which is not in "
                f"type_lengths={type_lengths}. Unique molecule sizes: "
                f"{sorted(set(molecule_sizes.tolist()))}"
            )
        molecule_type[mol_idx] = size_to_label[int(size)]

    return molecule_type


def safe_frac(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return np.nan
    return numerator / denominator


def pair_label(type_a: str, type_b: str) -> str:
    ordered = sorted([type_a, type_b])
    return f"{ordered[0]}_{ordered[1]}"


def expected_molecule_pair_fractions(
    molecule_type: np.ndarray,
    type_labels: list[str],
) -> dict[str, float]:
    """
    Expected pair fractions from molecule counts only.
    """
    n_by_type = {label: int(np.sum(molecule_type == label)) for label in type_labels}
    n_total = len(molecule_type)
    total_possible = n_total * (n_total - 1) / 2

    expected = {}

    for i, label_a in enumerate(type_labels):
        for label_b in type_labels[i:]:
            key = pair_label(label_a, label_b)

            if label_a == label_b:
                possible = n_by_type[label_a] * (n_by_type[label_a] - 1) / 2
            else:
                possible = n_by_type[label_a] * n_by_type[label_b]

            expected[key] = safe_frac(possible, total_possible)

    return expected


def mixture_metrics_for_frame(
    qi: np.ndarray,
    pi: np.ndarray,
    molecule_id: np.ndarray,
    molecule_type: np.ndarray,
    type_labels: list[str],
    min_intermol_bonds: int,
    drop_threshold: int,
) -> dict[str, object]:
    """
    Compute framewise mixture metrics from particle contact arrays.
    """
    row: dict[str, object] = {}

    n_molecules = len(molecule_type)
    type_labels = list(type_labels)

    n_by_type = {
        label: int(np.sum(molecule_type == label))
        for label in type_labels
    }

    for label in type_labels:
        row[f"n_mol_{label}"] = n_by_type[label]
        row[f"frac_mol_{label}"] = safe_frac(n_by_type[label], n_molecules)

    mol_i = molecule_id[qi]
    mol_j = molecule_id[pi]

    inter_mask = mol_i != mol_j
    inter_mol_i = mol_i[inter_mask]
    inter_mol_j = mol_j[inter_mask]

    # ------------------------------------------------------------
    # Particle-level intermolecular contact type counts
    # ------------------------------------------------------------
    particle_type_counts = Counter()

    for a, b in zip(inter_mol_i, inter_mol_j):
        label = pair_label(molecule_type[a], molecule_type[b])
        particle_type_counts[label] += 1

    total_inter_particle_contacts = sum(particle_type_counts.values())
    row["inter_particle_contacts_total"] = total_inter_particle_contacts

    expected_pairs = expected_molecule_pair_fractions(molecule_type, type_labels)

    pair_keys = []
    for i, label_a in enumerate(type_labels):
        for label_b in type_labels[i:]:
            pair_keys.append(pair_label(label_a, label_b))

    for key in pair_keys:
        count = particle_type_counts[key]
        frac = safe_frac(count, total_inter_particle_contacts)

        row[f"inter_particle_contacts_{key}"] = count
        row[f"frac_inter_particle_contacts_{key}"] = frac

    # ------------------------------------------------------------
    # Molecule-pair counts and qualifying graph edges
    # ------------------------------------------------------------
    molecule_pair_counts = Counter()

    for a, b in zip(inter_mol_i, inter_mol_j):
        pair = tuple(sorted((int(a), int(b))))
        molecule_pair_counts[pair] += 1

    qualifying_edges = [
        pair for pair, count in molecule_pair_counts.items()
        if count >= min_intermol_bonds
    ]

    graph = nx.Graph()
    graph.add_nodes_from(range(n_molecules))
    graph.add_edges_from(qualifying_edges)

    components = [set(c) for c in nx.connected_components(graph)]
    cluster_sizes = [len(c) for c in components]

    # ------------------------------------------------------------
    # Qualifying molecule-contact type counts
    # ------------------------------------------------------------
    mol_contact_type_counts = Counter()

    for a, b in qualifying_edges:
        key = pair_label(molecule_type[a], molecule_type[b])
        mol_contact_type_counts[key] += 1

    total_mol_contacts = sum(mol_contact_type_counts.values())
    row["mol_contacts_total"] = total_mol_contacts

    for key in pair_keys:
        count = mol_contact_type_counts[key]
        frac = safe_frac(count, total_mol_contacts)
        expected = expected_pairs[key]
        enrichment = safe_frac(frac, expected)

        row[f"mol_contacts_{key}"] = count
        row[f"frac_mol_contacts_{key}"] = frac
        row[f"expected_mol_contacts_{key}"] = expected
        row[f"enrichment_mol_contacts_{key}"] = enrichment

    # ------------------------------------------------------------
    # Per-type mono / oligo / drop fractions
    # ------------------------------------------------------------
    molecule_cluster_size = np.zeros(n_molecules, dtype=int)

    for comp in components:
        size = len(comp)
        for mol in comp:
            molecule_cluster_size[mol] = size

    for label in type_labels:
        type_mask = molecule_type == label
        denom = int(np.sum(type_mask))

        mono = int(np.sum(type_mask & (molecule_cluster_size == 1)))
        oligo = int(
            np.sum(
                type_mask
                & (molecule_cluster_size > 1)
                & (molecule_cluster_size < drop_threshold)
            )
        )
        drop = int(np.sum(type_mask & (molecule_cluster_size >= drop_threshold)))

        row[f"mono_frac_{label}"] = safe_frac(mono, denom)
        row[f"oligo_frac_{label}"] = safe_frac(oligo, denom)
        row[f"drop_frac_{label}"] = safe_frac(drop, denom)

    # ------------------------------------------------------------
    # Largest-cluster composition
    # ------------------------------------------------------------
    largest_comp = max(components, key=len) if components else set()
    largest_size = len(largest_comp)

    row["largest_cluster_size_mixture"] = largest_size

    for label in type_labels:
        n_label = sum(molecule_type[mol] == label for mol in largest_comp)
        row[f"largest_cluster_n_{label}"] = n_label
        row[f"largest_cluster_frac_{label}"] = safe_frac(n_label, largest_size)

    if largest_size > 0:
        row["largest_cluster_purity"] = max(
            row[f"largest_cluster_n_{label}"] for label in type_labels
        ) / largest_size
    else:
        row["largest_cluster_purity"] = np.nan

    # ------------------------------------------------------------
    # Mixed-cluster metrics
    # ------------------------------------------------------------
    mixed_clusters = []
    pure_cluster_counts = {label: 0 for label in type_labels}

    molecules_in_mixed_clusters = set()

    for comp in components:
        present_types = set(molecule_type[list(comp)])

        if len(present_types) > 1:
            mixed_clusters.append(comp)
            molecules_in_mixed_clusters.update(comp)
        else:
            only_type = next(iter(present_types))
            pure_cluster_counts[only_type] += 1

    row["n_clusters_mixed"] = len(mixed_clusters)
    row["frac_clusters_mixed"] = safe_frac(len(mixed_clusters), len(components))
    row["largest_mixed_cluster"] = max((len(c) for c in mixed_clusters), default=0)

    for label in type_labels:
        row[f"n_clusters_pure_{label}"] = pure_cluster_counts[label]

    row["frac_molecules_in_mixed_clusters"] = safe_frac(
        len(molecules_in_mixed_clusters),
        n_molecules,
    )

    for label in type_labels:
        type_mols = set(np.where(molecule_type == label)[0])
        n_type_in_mixed = len(type_mols & molecules_in_mixed_clusters)

        row[f"frac_{label}_in_mixed_clusters"] = safe_frac(
            n_type_in_mixed,
            len(type_mols),
        )

    # ------------------------------------------------------------
    # Degree by type
    # ------------------------------------------------------------
    degrees = dict(graph.degree())

    for label in type_labels:
        type_nodes = np.where(molecule_type == label)[0]
        row[f"mean_degree_{label}"] = (
            np.mean([degrees[int(node)] for node in type_nodes])
            if len(type_nodes) > 0 else np.nan
        )

        homotypic_degrees = []
        heterotypic_degrees = []

        for node in type_nodes:
            node = int(node)
            neighbors = list(graph.neighbors(node))

            homotypic_degrees.append(
                sum(molecule_type[n] == label for n in neighbors)
            )
            heterotypic_degrees.append(
                sum(molecule_type[n] != label for n in neighbors)
            )

        row[f"mean_homotypic_degree_{label}"] = (
            np.mean(homotypic_degrees) if homotypic_degrees else np.nan
        )
        row[f"mean_heterotypic_degree_{label}"] = (
            np.mean(heterotypic_degrees) if heterotypic_degrees else np.nan
        )

    return row


def framewise_mixture_rows_from_nlist_dict(
    nlist_dict: dict[str, np.ndarray],
    molecule_id: np.ndarray,
    molecule_type: np.ndarray,
    type_labels: list[str],
    min_intermol_bonds: int,
    drop_threshold: int,
) -> list[dict[str, object]]:
    rows = []

    frame_indices = nlist_dict["frame_indices"]

    for frame in frame_indices:
        frame = int(frame)

        qi = nlist_dict[f"qi_{frame}"]
        pi = nlist_dict[f"pi_{frame}"]

        row = {"frame": frame}
        row.update(
            mixture_metrics_for_frame(
                qi=qi,
                pi=pi,
                molecule_id=molecule_id,
                molecule_type=molecule_type,
                type_labels=type_labels,
                min_intermol_bonds=min_intermol_bonds,
                drop_threshold=drop_threshold,
            )
        )
        rows.append(row)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build framewise mixture phase-separation CSVs from pairing .npz files."
    )
    parser.add_argument("sweep_dir", help="Name of the sweep directory inside BASE_DIR")
    parser.add_argument("--min-intermol-bonds", type=int, default=2)
    parser.add_argument("--drop-threshold", type=int, default=5)
    parser.add_argument(
        "--type-lengths",
        nargs="+",
        type=int,
        default=[30, 141],
        help="Molecule bead lengths used to infer molecule type. Default: 30 141.",
    )
    parser.add_argument(
        "--type-labels",
        nargs="+",
        default=["cag10", "cag47"],
        help="Labels corresponding to --type-lengths. Default: cag10 cag47.",
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
        npz_paths = sorted(sweep_path.glob("*/computed_metrics/pairing/*_pairing.npz"))
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

            molecule_type = make_molecule_type_lookup(
                molecule_id=molecule_id,
                type_lengths=args.type_lengths,
                type_labels=args.type_labels,
            )

            # Original homogeneous metrics
            base_rows = framewise_ps_rows_from_nlist_dict(
                nlist_dict=data,
                molecule_id=molecule_id,
                min_intermol_bonds=args.min_intermol_bonds,
            )
            base_df = pd.DataFrame(base_rows)

            # New mixture metrics
            mixture_rows = framewise_mixture_rows_from_nlist_dict(
                nlist_dict=data,
                molecule_id=molecule_id,
                molecule_type=molecule_type,
                type_labels=args.type_labels,
                min_intermol_bonds=args.min_intermol_bonds,
                drop_threshold=args.drop_threshold,
            )
            mixture_df = pd.DataFrame(mixture_rows)

            df = base_df.merge(mixture_df, on="frame", how="left")

            traj_stem = npz_path.stem.removesuffix("_pairing")
            traj_path = run_dir / f"{traj_stem}.gsd"

            with as_traj(traj_path) as traj:
                frame_idx = df["frame"].to_numpy(dtype=int)
                steps = np.array(
                    [traj[int(i)].configuration.step for i in frame_idx],
                    dtype=np.int64,
                )

                try:
                    ke = get_log_KE_over_time(traj)
                except KeyError:
                    ke = None

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

            dt = float(read_param(run_dir / "params.json", "dt"))

            df["step"] = steps
            df["time"] = df["step"] * dt
            df["ke_per_particle"] = ke[frame_idx] if ke is not None else np.nan

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
            df["min_intermol_bonds"] = args.min_intermol_bonds
            df["drop_threshold"] = args.drop_threshold

            df.to_csv(out_csv, index=False)
            all_dfs.append(df)

            print(f"Saved: {out_csv}")
            n_done += 1

        except Exception as exc:
            print(f"Failed on {npz_path}: {type(exc).__name__}: {exc!r}")
            traceback.print_exc(limit=2)
            n_failed += 1

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_csv = sweep_path / "analysis_mixture_framewise.csv"
        combined_df.to_csv(combined_csv, index=False)
        print(f"Saved combined mixture framewise CSV: {combined_csv}")

    print(f"Finished. Saved {n_done} mixture framewise CSVs. Failed: {n_failed}")


if __name__ == "__main__":
    main()