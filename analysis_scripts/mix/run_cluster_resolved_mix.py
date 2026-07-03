#!/usr/bin/env python3

"""
Build framewise cluster-resolved CSVs from molecule-level inter- and intramolecular contact files.

Input per run directory:
    computed_metrics/mol_graph/*_molecule_contact_edges.npz
    computed_metrics/intramol_graph/*_intra_molecule_contacts.npz
    computed_metrics/pairing/molecule_lookup.npy
    params.json
    trajectory_*.gsd

Output per trajectory:
    framewise_cluster/*_framewise_cluster.csv

Combined output:
    <sweep_dir>/analysis_cluster_resolved.csv

Usage:
    python run_analysis_cluster_resolved.py 52b_mixed_repeat_length_full
    python run_analysis_cluster_resolved.py 52b_mixed_repeat_length_full --run_ids 0 1 2
    python run_analysis_cluster_resolved.py 52b_mixed_repeat_length_full --min_contacts 2 --overwrite
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


analysis_functions_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(analysis_functions_dir))

BASE_DIR = analysis_functions_dir / "bulk" / "output_dir"


# ============================================================================
# Small helpers
# ============================================================================


def read_param(params_path: Path, key: str) -> float | int | str | None:
    """Read one parameter from a params.json file."""
    with open(params_path, "r", encoding="utf-8") as f:
        content = f.read()

    decoder = json.JSONDecoder()
    data, _ = decoder.raw_decode(content)

    return data.get(key)


def read_required_param(params_path: Path, key: str) -> float | int | str:
    """Read one required parameter from params.json."""
    value = read_param(params_path, key)

    if value is None:
        raise KeyError(f"Missing required parameter {key!r} in {params_path}")

    return value


def safe_divide(num: float | int, den: float | int) -> float:
    """Divide safely and return NaN when the denominator is zero."""
    if den == 0:
        return float("nan")

    return float(num) / float(den)


def get_trajectory_stem_from_mol_graph_file(mol_graph_file: Path) -> str:
    """
    Convert:
        trajectory_0_molecule_contact_edges.npz
    into:
        trajectory_0
    """
    suffix = "_molecule_contact_edges.npz"

    if not mol_graph_file.name.endswith(suffix):
        raise ValueError(f"Unexpected mol_graph filename: {mol_graph_file.name}")

    return mol_graph_file.name.removesuffix(suffix)


def get_traj_replicate_from_gsd(run_dir: Path, trajectory_stem: str) -> str:
    """
    Get traj_replicate from the matching .gsd filename stem.
    """
    gsd_file = run_dir / f"{trajectory_stem}.gsd"

    if not gsd_file.exists():
        raise FileNotFoundError(f"Matching GSD file not found: {gsd_file}")

    return gsd_file.stem


def find_run_dirs(sweep_path: Path, run_ids: list[str] | None) -> list[Path]:
    """Find run directories to process."""
    if run_ids is None:
        return sorted(path for path in sweep_path.iterdir() if path.is_dir())

    run_dirs = []

    for run_id in run_ids:
        matches = sorted(sweep_path.glob(f"{run_id}_*"))
        run_dirs.extend(path for path in matches if path.is_dir())

    return run_dirs


# ============================================================================
# Union-find
# ============================================================================


class UnionFind:
    """Small union-find for connected components."""

    def __init__(self, n_nodes: int) -> None:
        self.parent = np.arange(n_nodes, dtype=np.int64)
        self.size = np.ones(n_nodes, dtype=np.int64)

    def find(self, node: int) -> int:
        """Find root with path compression."""
        node = int(node)
        root = node

        while self.parent[root] != root:
            root = int(self.parent[root])

        while self.parent[node] != node:
            parent = int(self.parent[node])
            self.parent[node] = root
            node = parent

        return root

    def union(self, a: int, b: int) -> None:
        """Join two components by size."""
        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return

        if self.size[root_a] < self.size[root_b]:
            root_a, root_b = root_b, root_a

        self.parent[root_b] = root_a
        self.size[root_a] += self.size[root_b]


# ============================================================================
# Molecule metadata
# ============================================================================


def build_molecule_metadata(
    molecule_id: np.ndarray,
    cag10_len: int,
    cag47_len: int,
) -> dict[str, np.ndarray | int]:
    """
    Build molecule lengths and CAG10/CAG47 species codes from molecule_lookup.npy.
    """
    molecule_id = np.asarray(molecule_id, dtype=np.int64)

    n_molecules = int(molecule_id.max()) + 1
    molecule_lengths = np.bincount(
        molecule_id,
        minlength=n_molecules,
    ).astype(np.int64)

    species_code = np.full(n_molecules, -1, dtype=np.int64)
    species_code[molecule_lengths == cag10_len] = 10
    species_code[molecule_lengths == cag47_len] = 47

    unknown_lengths = sorted(set(molecule_lengths[species_code == -1].tolist()))

    if unknown_lengths:
        raise ValueError(
            f"Found molecule lengths that are neither {cag10_len} nor "
            f"{cag47_len}: {unknown_lengths}"
        )

    total_mols_cag10 = int(np.sum(species_code == 10))
    total_mols_cag47 = int(np.sum(species_code == 47))

    total_particles_cag10 = int(np.sum(molecule_lengths[species_code == 10]))
    total_particles_cag47 = int(np.sum(molecule_lengths[species_code == 47]))
    total_particles_system = int(np.sum(molecule_lengths))

    return {
        "molecule_lengths": molecule_lengths,
        "species_code": species_code,
        "n_molecules": n_molecules,
        "total_mols_cag10": total_mols_cag10,
        "total_mols_cag47": total_mols_cag47,
        "total_particles_cag10": total_particles_cag10,
        "total_particles_cag47": total_particles_cag47,
        "total_particles_system": total_particles_system,
    }


# ============================================================================
# Fast frame slicing
# ============================================================================


def make_frame_slices(frame: np.ndarray) -> dict[int, slice]:
    """
    Build frame -> slice lookup for arrays already sorted by frame.
    """
    if len(frame) == 0:
        return {}

    unique_frames, starts, counts = np.unique(
        frame,
        return_index=True,
        return_counts=True,
    )

    return {
        int(f): slice(int(start), int(start + count))
        for f, start, count in zip(unique_frames, starts, counts)
    }


def sort_edge_arrays_by_frame(
    frame: np.ndarray,
    mol_a: np.ndarray,
    mol_b: np.ndarray,
    n_contacts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[int, slice]]:
    """
    Sort molecule-edge arrays by frame and return frame slices.
    """
    order = np.argsort(frame)

    frame = np.asarray(frame[order], dtype=np.int64)
    mol_a = np.asarray(mol_a[order], dtype=np.int64)
    mol_b = np.asarray(mol_b[order], dtype=np.int64)
    n_contacts = np.asarray(n_contacts[order], dtype=np.int64)

    frame_slices = make_frame_slices(frame)

    return frame, mol_a, mol_b, n_contacts, frame_slices


def sort_intra_arrays_by_frame(
    frame: np.ndarray,
    mol_id: np.ndarray,
    n_intra_contacts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, slice]]:
    """
    Sort intramolecular contact arrays by frame and return frame slices.
    """
    order = np.argsort(frame)

    frame = np.asarray(frame[order], dtype=np.int64)
    mol_id = np.asarray(mol_id[order], dtype=np.int64)
    n_intra_contacts = np.asarray(n_intra_contacts[order], dtype=np.int64)

    frame_slices = make_frame_slices(frame)

    return frame, mol_id, n_intra_contacts, frame_slices


def load_intramol_arrays(
    intramol_file: Path | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, slice]]:
    """
    Load sparse intramolecular contact arrays.

    Missing intramol file returns empty arrays.
    """
    if intramol_file is None or not intramol_file.exists():
        empty = np.array([], dtype=np.int64)
        return empty, empty, empty, {}

    with np.load(intramol_file) as data:
        frame = np.asarray(data["frame"], dtype=np.int64)
        mol_id = np.asarray(data["mol_id"], dtype=np.int64)
        n_intra_contacts = np.asarray(data["n_intra_contacts"], dtype=np.int64)

    return sort_intra_arrays_by_frame(
        frame=frame,
        mol_id=mol_id,
        n_intra_contacts=n_intra_contacts,
    )


# ============================================================================
# Row builders
# ============================================================================


def make_cluster_row(
    base_metadata: dict[str, object],
    frame_idx: int,
    step: int,
    min_contacts: int,
    cluster_kind: str,
    cluster_rank_particles: int,
    n_clusters_aggregated: int,
    mol_ids: np.ndarray,
    molecule_lengths: np.ndarray,
    species_code: np.ndarray,
    total_particles_system: int,
    total_mols_cag10: int,
    total_mols_cag47: int,
    total_particles_cag10: int,
    total_particles_cag47: int,
    pairable_sites_cag10: int,
    pairable_sites_cag47: int,
    mol_graph_edges: int,
    inter_10_10: int,
    inter_10_47: int,
    inter_47_47: int,
    intra_10: int,
    intra_47: int,
    cag10_len: int,
    cag47_len: int,
) -> dict[str, object]:
    """
    Build one output row for either one real cluster or the aggregated monomer row.
    """
    cluster_species = species_code[mol_ids]
    cluster_lengths = molecule_lengths[mol_ids]

    n_mols_cag10 = int(np.sum(cluster_species == 10))
    n_mols_cag47 = int(np.sum(cluster_species == 47))

    n_particles_cag10 = int(np.sum(cluster_lengths[cluster_species == 10]))
    n_particles_cag47 = int(np.sum(cluster_lengths[cluster_species == 47]))

    cluster_size_mols = int(len(mol_ids))
    cluster_size_particles = int(np.sum(cluster_lengths))

    available_sites_10 = int(n_mols_cag10 * pairable_sites_cag10)
    available_sites_47 = int(n_mols_cag47 * pairable_sites_cag47)

    inter_total = int(inter_10_10 + inter_10_47 + inter_47_47)
    intra_total = int(intra_10 + intra_47)

    row = {
        **base_metadata,
        "frame": int(frame_idx),
        "step": int(step),
        "min_contacts": int(min_contacts),
        "cluster_kind": cluster_kind,
        "cluster_rank_particles": int(cluster_rank_particles),
        "n_clusters_aggregated": int(n_clusters_aggregated),
        "cluster_size_mols": cluster_size_mols,
        "cluster_size_particles": cluster_size_particles,
        "n_mols_cag10": n_mols_cag10,
        "n_mols_cag47": n_mols_cag47,
        "n_particles_cag10": n_particles_cag10,
        "n_particles_cag47": n_particles_cag47,
        "frac_cluster_particles_cag10": safe_divide(
            n_particles_cag10,
            cluster_size_particles,
        ),
        "frac_cluster_particles_cag47": safe_divide(
            n_particles_cag47,
            cluster_size_particles,
        ),
        "frac_total_particles_in_cluster": safe_divide(
            cluster_size_particles,
            total_particles_system,
        ),
        "frac_total_particles_cag10_in_cluster": safe_divide(
            n_particles_cag10,
            total_particles_system,
        ),
        "frac_total_particles_cag47_in_cluster": safe_divide(
            n_particles_cag47,
            total_particles_system,
        ),
        "is_mixed": bool(n_mols_cag10 > 0 and n_mols_cag47 > 0),
        "mol_graph_edges": int(mol_graph_edges),
        "mean_mol_graph_degree": safe_divide(
            2 * mol_graph_edges,
            cluster_size_mols,
        ),
        "inter_particle_contacts_total": inter_total,
        "inter_particle_contacts_cag10_cag10": int(inter_10_10),
        "inter_particle_contacts_cag10_cag47": int(inter_10_47),
        "inter_particle_contacts_cag47_cag47": int(inter_47_47),
        "available_pairable_sites_cag10": available_sites_10,
        "available_pairable_sites_cag47": available_sites_47,
        "inter_contact_saturation_cag10": safe_divide(
            2 * inter_10_10 + inter_10_47,
            available_sites_10,
        ),
        "inter_contact_saturation_cag47": safe_divide(
            2 * inter_47_47 + inter_10_47,
            available_sites_47,
        ),
        "homotypic_saturation_cag10": safe_divide(
            2 * inter_10_10,
            available_sites_10,
        ),
        "heterotypic_saturation_cag10": safe_divide(
            inter_10_47,
            available_sites_10,
        ),
        "homotypic_saturation_cag47": safe_divide(
            2 * inter_47_47,
            available_sites_47,
        ),
        "heterotypic_saturation_cag47": safe_divide(
            inter_10_47,
            available_sites_47,
        ),
        "intra_particle_contacts_total": intra_total,
        "intra_particle_contacts_cag10": int(intra_10),
        "intra_particle_contacts_cag47": int(intra_47),
        "intra_contact_saturation_cag10": safe_divide(
            2 * intra_10,
            available_sites_10,
        ),
        "intra_contact_saturation_cag47": safe_divide(
            2 * intra_47,
            available_sites_47,
        ),
        "molecule_length_cag10": int(cag10_len),
        "molecule_length_cag47": int(cag47_len),
        "pairable_sites_per_mol_cag10": int(pairable_sites_cag10),
        "pairable_sites_per_mol_cag47": int(pairable_sites_cag47),
        "total_mols_cag10": int(total_mols_cag10),
        "total_mols_cag47": int(total_mols_cag47),
        "total_particles_cag10": int(total_particles_cag10),
        "total_particles_cag47": int(total_particles_cag47),
        "total_particles_system": int(total_particles_system),
    }

    return row


def build_cluster_resolved_df(
    mol_graph_file: Path,
    intramol_file: Path | None,
    molecule_id: np.ndarray,
    base_metadata: dict[str, object],
    n_save: int,
    min_contacts: int,
    cag10_len: int,
    cag47_len: int,
    pairable_sites_cag10: int,
    pairable_sites_cag47: int,
) -> pd.DataFrame:
    """
    Build one framewise cluster-resolved dataframe for one trajectory.
    """
    mol_meta = build_molecule_metadata(
        molecule_id=molecule_id,
        cag10_len=cag10_len,
        cag47_len=cag47_len,
    )

    molecule_lengths = mol_meta["molecule_lengths"]
    species_code = mol_meta["species_code"]

    n_molecules = int(mol_meta["n_molecules"])
    total_mols_cag10 = int(mol_meta["total_mols_cag10"])
    total_mols_cag47 = int(mol_meta["total_mols_cag47"])
    total_particles_cag10 = int(mol_meta["total_particles_cag10"])
    total_particles_cag47 = int(mol_meta["total_particles_cag47"])
    total_particles_system = int(mol_meta["total_particles_system"])

    with np.load(mol_graph_file) as graph_npz:
        frame_indices = np.asarray(graph_npz["frame_indices"], dtype=np.int64)

        frame_all = np.asarray(graph_npz["frame"], dtype=np.int64)
        mol_a_all = np.asarray(graph_npz["mol_a"], dtype=np.int64)
        mol_b_all = np.asarray(graph_npz["mol_b"], dtype=np.int64)
        n_contacts_all = np.asarray(graph_npz["n_contacts"], dtype=np.int64)

    (
        frame_all,
        mol_a_all,
        mol_b_all,
        n_contacts_all,
        all_edge_slices,
    ) = sort_edge_arrays_by_frame(
        frame=frame_all,
        mol_a=mol_a_all,
        mol_b=mol_b_all,
        n_contacts=n_contacts_all,
    )

    keep = n_contacts_all >= min_contacts

    (
        frame_qual,
        mol_a_qual,
        mol_b_qual,
        n_contacts_qual,
        qual_edge_slices,
    ) = sort_edge_arrays_by_frame(
        frame=frame_all[keep],
        mol_a=mol_a_all[keep],
        mol_b=mol_b_all[keep],
        n_contacts=n_contacts_all[keep],
    )

    (
        intra_frame,
        intra_mol_id,
        intra_contacts,
        intra_slices,
    ) = load_intramol_arrays(intramol_file)

    rows = []

    for frame_idx in frame_indices:
        frame_idx = int(frame_idx)
        step = int(frame_idx * n_save)

        union_find = UnionFind(n_molecules)

        if frame_idx in qual_edge_slices:
            sl = qual_edge_slices[frame_idx]

            for a, b in zip(mol_a_qual[sl], mol_b_qual[sl]):
                union_find.union(int(a), int(b))

        root_by_mol = np.asarray(
            [union_find.find(mol_id) for mol_id in range(n_molecules)],
            dtype=np.int64,
        )

        components: dict[int, list[int]] = {}

        for mol_id, root in enumerate(root_by_mol):
            components.setdefault(int(root), []).append(int(mol_id))

        mol_graph_edges_by_root = {root: 0 for root in components}

        if frame_idx in qual_edge_slices:
            sl = qual_edge_slices[frame_idx]

            for a in mol_a_qual[sl]:
                root = int(root_by_mol[int(a)])
                mol_graph_edges_by_root[root] += 1

        inter_10_10_by_root = {root: 0 for root in components}
        inter_10_47_by_root = {root: 0 for root in components}
        inter_47_47_by_root = {root: 0 for root in components}

        if frame_idx in all_edge_slices:
            sl = all_edge_slices[frame_idx]

            for a, b, count in zip(
                mol_a_all[sl],
                mol_b_all[sl],
                n_contacts_all[sl],
            ):
                a = int(a)
                b = int(b)
                count = int(count)

                root_a = int(root_by_mol[a])
                root_b = int(root_by_mol[b])

                if root_a != root_b:
                    continue

                species_a = int(species_code[a])
                species_b = int(species_code[b])
                pair_species = tuple(sorted((species_a, species_b)))

                if pair_species == (10, 10):
                    inter_10_10_by_root[root_a] += count
                elif pair_species == (10, 47):
                    inter_10_47_by_root[root_a] += count
                elif pair_species == (47, 47):
                    inter_47_47_by_root[root_a] += count

        intra_10_by_root = {root: 0 for root in components}
        intra_47_by_root = {root: 0 for root in components}

        if frame_idx in intra_slices:
            sl = intra_slices[frame_idx]

            for mol_id, count in zip(intra_mol_id[sl], intra_contacts[sl]):
                mol_id = int(mol_id)
                count = int(count)

                root = int(root_by_mol[mol_id])
                species = int(species_code[mol_id])

                if species == 10:
                    intra_10_by_root[root] += count
                elif species == 47:
                    intra_47_by_root[root] += count

        component_stats = []

        for root, mol_ids in components.items():
            mol_array = np.asarray(mol_ids, dtype=np.int64)
            cluster_lengths = molecule_lengths[mol_array]

            component_stats.append(
                {
                    "root": int(root),
                    "mol_ids": mol_array,
                    "cluster_size_mols": int(len(mol_array)),
                    "cluster_size_particles": int(np.sum(cluster_lengths)),
                    "min_mol_id": int(np.min(mol_array)),
                }
            )

        multimers = [
            comp
            for comp in component_stats
            if comp["cluster_size_mols"] > 1
        ]

        monomers = [
            comp
            for comp in component_stats
            if comp["cluster_size_mols"] == 1
        ]

        multimers = sorted(
            multimers,
            key=lambda row: (
                -row["cluster_size_particles"],
                -row["cluster_size_mols"],
                row["min_mol_id"],
            ),
        )

        for rank, comp in enumerate(multimers, start=1):
            root = int(comp["root"])
            mol_ids = comp["mol_ids"]

            row = make_cluster_row(
                base_metadata=base_metadata,
                frame_idx=frame_idx,
                step=step,
                min_contacts=min_contacts,
                cluster_kind="cluster",
                cluster_rank_particles=rank,
                n_clusters_aggregated=1,
                mol_ids=mol_ids,
                molecule_lengths=molecule_lengths,
                species_code=species_code,
                total_particles_system=total_particles_system,
                total_mols_cag10=total_mols_cag10,
                total_mols_cag47=total_mols_cag47,
                total_particles_cag10=total_particles_cag10,
                total_particles_cag47=total_particles_cag47,
                pairable_sites_cag10=pairable_sites_cag10,
                pairable_sites_cag47=pairable_sites_cag47,
                mol_graph_edges=mol_graph_edges_by_root[root],
                inter_10_10=inter_10_10_by_root[root],
                inter_10_47=inter_10_47_by_root[root],
                inter_47_47=inter_47_47_by_root[root],
                intra_10=intra_10_by_root[root],
                intra_47=intra_47_by_root[root],
                cag10_len=cag10_len,
                cag47_len=cag47_len,
            )

            rows.append(row)

        if monomers:
            monomer_mol_ids = np.concatenate(
                [comp["mol_ids"] for comp in monomers]
            )

            monomer_roots = [int(comp["root"]) for comp in monomers]

            intra_10 = sum(intra_10_by_root[root] for root in monomer_roots)
            intra_47 = sum(intra_47_by_root[root] for root in monomer_roots)

            row = make_cluster_row(
                base_metadata=base_metadata,
                frame_idx=frame_idx,
                step=step,
                min_contacts=min_contacts,
                cluster_kind="monomers",
                cluster_rank_particles=0,
                n_clusters_aggregated=len(monomers),
                mol_ids=monomer_mol_ids,
                molecule_lengths=molecule_lengths,
                species_code=species_code,
                total_particles_system=total_particles_system,
                total_mols_cag10=total_mols_cag10,
                total_mols_cag47=total_mols_cag47,
                total_particles_cag10=total_particles_cag10,
                total_particles_cag47=total_particles_cag47,
                pairable_sites_cag10=pairable_sites_cag10,
                pairable_sites_cag47=pairable_sites_cag47,
                mol_graph_edges=0,
                inter_10_10=0,
                inter_10_47=0,
                inter_47_47=0,
                intra_10=intra_10,
                intra_47=intra_47,
                cag10_len=cag10_len,
                cag47_len=cag47_len,
            )

            rows.append(row)

    return pd.DataFrame(rows)


# ============================================================================
# File matching and writing
# ============================================================================


def matching_intramol_file(
    mol_graph_file: Path,
    intramol_dir: Path,
) -> Path:
    """Get expected intramolecular contact file for one mol_graph file."""
    trajectory_stem = get_trajectory_stem_from_mol_graph_file(mol_graph_file)

    return intramol_dir / f"{trajectory_stem}_intra_molecule_contacts.npz"


def output_csv_file(
    mol_graph_file: Path,
    output_dir: Path,
) -> Path:
    """Get output CSV path for one mol_graph file."""
    trajectory_stem = get_trajectory_stem_from_mol_graph_file(mol_graph_file)

    return output_dir / f"{trajectory_stem}_framewise_cluster.csv"


def append_csv_to_combined(
    csv_file: Path,
    combined_file: Path,
    write_header: bool,
) -> None:
    """Append one per-trajectory CSV to the combined sweep-level CSV."""
    df = pd.read_csv(csv_file)

    df.to_csv(
        combined_file,
        mode="w" if write_header else "a",
        header=write_header,
        index=False,
    )


# ============================================================================
# Runner
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build framewise cluster-resolved CSVs from mol_graph and "
            "intramol_graph files."
        )
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
        "--min_contacts",
        "--min-contacts",
        type=int,
        default=2,
        help="Minimum inter-molecular particle contacts required for a molecule edge.",
    )

    parser.add_argument(
        "--cag10_len",
        type=int,
        default=30,
        help="Particle length of a CAG10 molecule.",
    )

    parser.add_argument(
        "--cag47_len",
        type=int,
        default=141,
        help="Particle length of a CAG47 molecule.",
    )

    parser.add_argument(
        "--pairable_sites_cag10",
        type=int,
        default=20,
        help="Number of pairable C/G sites per CAG10 molecule.",
    )

    parser.add_argument(
        "--pairable_sites_cag47",
        type=int,
        default=94,
        help="Number of pairable C/G sites per CAG47 molecule.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute per-trajectory CSVs even if they already exist.",
    )

    args = parser.parse_args()

    sweep_path = BASE_DIR / args.sweep_dir

    if not sweep_path.is_dir():
        raise FileNotFoundError(f"Sweep directory not found: {sweep_path}")

    combined_file = sweep_path / "analysis_cluster_resolved.csv"

    if combined_file.exists():
        print(f"Removing existing combined CSV: {combined_file}")
        combined_file.unlink()

    print(f"Reading from: {sweep_path}")
    print(f"Combined output: {combined_file}")

    run_dirs = find_run_dirs(
        sweep_path=sweep_path,
        run_ids=args.run_ids,
    )

    write_combined_header = True
    n_written = 0
    n_failed = 0

    for run_dir in run_dirs:
        params_path = run_dir / "params.json"

        if not params_path.exists():
            print(f"Skipping {run_dir}: params.json not found")
            n_failed += 1
            continue

        pairing_dir = run_dir / "computed_metrics" / "pairing"
        mol_graph_dir = run_dir / "computed_metrics" / "mol_graph"
        intramol_dir = run_dir / "computed_metrics" / "intramol_graph"
        output_dir = run_dir / "framewise_cluster"
        output_dir.mkdir(parents=True, exist_ok=True)

        mol_lookup_file = pairing_dir / "molecule_lookup.npy"

        if not mol_lookup_file.exists():
            print(f"Skipping {run_dir}: molecule lookup not found at {mol_lookup_file}")
            n_failed += 1
            continue

        if not mol_graph_dir.is_dir():
            print(f"Skipping {run_dir}: mol_graph directory not found at {mol_graph_dir}")
            n_failed += 1
            continue

        mol_graph_files = sorted(mol_graph_dir.glob("*_molecule_contact_edges.npz"))

        if len(mol_graph_files) == 0:
            print(f"Skipping {run_dir}: no mol_graph NPZ files found in {mol_graph_dir}")
            n_failed += 1
            continue

        molecule_id = np.load(mol_lookup_file)

        dt = float(read_required_param(params_path, "dt"))
        n_save = int(read_required_param(params_path, "N_write"))

        molecule_file = read_param(params_path, "molecule_file")
        e_att = read_param(params_path, "E_att")
        angle_k_ppp = read_param(params_path, "angle_k_ppp")
        density = read_param(params_path, "density")

        for mol_graph_file in mol_graph_files:
            try:
                trajectory_stem = get_trajectory_stem_from_mol_graph_file(
                    mol_graph_file
                )

                traj_replicate = get_traj_replicate_from_gsd(
                    run_dir=run_dir,
                    trajectory_stem=trajectory_stem,
                )

                intramol_file = matching_intramol_file(
                    mol_graph_file=mol_graph_file,
                    intramol_dir=intramol_dir,
                )

                output_file = output_csv_file(
                    mol_graph_file=mol_graph_file,
                    output_dir=output_dir,
                )

                if output_file.exists() and not args.overwrite:
                    print(f"Skipping {mol_graph_file}: output already exists at {output_file}")

                    append_csv_to_combined(
                        csv_file=output_file,
                        combined_file=combined_file,
                        write_header=write_combined_header,
                    )

                    write_combined_header = False
                    n_written += 1
                    continue

                if output_file.exists() and args.overwrite:
                    print(f"Overwriting existing output: {output_file}")
                    output_file.unlink()

                if not intramol_file.exists():
                    print(
                        f"Warning: intramol file not found for {mol_graph_file}. "
                        f"Intra-contact columns will be zero: {intramol_file}"
                    )
                    intramol_file_for_build = None
                else:
                    intramol_file_for_build = intramol_file

                base_metadata = {
                    "run_id": f"{run_dir.name}__{traj_replicate}",
                    "folder_name": run_dir.name,
                    "traj_replicate": traj_replicate,
                    "molecule_file": molecule_file,
                    "dt": dt,
                    "N_save": n_save,
                    "E_att": e_att,
                    "angle_k_ppp": angle_k_ppp,
                    "density": density,
                }

                print(f"Processing {mol_graph_file}")

                df = build_cluster_resolved_df(
                    mol_graph_file=mol_graph_file,
                    intramol_file=intramol_file_for_build,
                    molecule_id=molecule_id,
                    base_metadata=base_metadata,
                    n_save=n_save,
                    min_contacts=args.min_contacts,
                    cag10_len=args.cag10_len,
                    cag47_len=args.cag47_len,
                    pairable_sites_cag10=args.pairable_sites_cag10,
                    pairable_sites_cag47=args.pairable_sites_cag47,
                )

                df.to_csv(output_file, index=False)
                print(f"Saved framewise cluster CSV: {output_file}")

                df.to_csv(
                    combined_file,
                    mode="w" if write_combined_header else "a",
                    header=write_combined_header,
                    index=False,
                )

                write_combined_header = False
                n_written += 1

            except Exception as exc:
                print(f"Failed on {mol_graph_file}: {type(exc).__name__}: {exc}")
                n_failed += 1

    if n_written == 0:
        print("No per-trajectory CSVs were written or appended.")
    else:
        print(f"Saved combined CSV: {combined_file}")
        print(f"Number of trajectory CSVs included: {n_written}")

    print(f"Failed: {n_failed}")


if __name__ == "__main__":
    main()