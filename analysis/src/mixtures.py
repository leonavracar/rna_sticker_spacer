"""Mixture composition, molecular contacts, recruitment and saturation from NPZs."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .cluster_distribution import connected_components, ranked_components
from .framewise import edge_frames


def identify_species(lengths: np.ndarray, labels_by_length: dict[int, str]) -> np.ndarray:
    """Assign molecule species from known RNA lengths."""
    lengths = np.asarray(lengths)
    missing = set(int(n) for n in lengths) - set(labels_by_length)
    if missing:
        raise ValueError(f"Unrecognized chain lengths: {sorted(missing)}")
    return np.array([labels_by_length[int(n)] for n in lengths])


def intramolecular_frames(intra_npz) -> dict[int, dict[int, int]]:
    """Index per-molecule intramolecular contacts by saved frame."""
    frames = {int(f): {} for f in intra_npz["frame_indices"]}
    for frame, mol, count in zip(
        intra_npz["frame"], intra_npz["mol_id"], intra_npz["n_intra_contacts"]
    ):
        frames[int(frame)][int(mol)] = int(count)
    return frames


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _pair_key(a: str, b: str) -> str:
    return "_".join(sorted((a, b)))


def mixture_cluster_rows(
    edge_npz,
    lengths: np.ndarray,
    species: np.ndarray,
    min_contacts: int = 2,
    intra_npz=None,
    pairable_sites_by_species: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return ranked, largest-cluster, contact and system-wide framewise metrics.

    Particle contacts include all accepted intermolecular pairs; molecule edges
    and connected components require at least `min_contacts` per molecule pair.
    Enrichment follows the original molecule-pair-count null model.
    """
    lengths = np.asarray(lengths, dtype=int)
    species = np.asarray(species, dtype=str)
    if len(lengths) != len(species):
        raise ValueError("Species and lengths must describe the same molecules")

    total_particles = int(lengths.sum())
    n_molecules = len(lengths)
    names = sorted(set(species))
    pairable_sites_by_species = pairable_sites_by_species or {}
    if set(pairable_sites_by_species) != set(names):
        raise ValueError("Pairable-site counts must be specified for every mixture species")

    n_by_type = {sp: int(np.count_nonzero(species == sp)) for sp in names}
    possible_pairs = n_molecules * (n_molecules - 1) // 2
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i:]]
    expected = {
        _pair_key(a, b): _ratio(
            n_by_type[a] * (n_by_type[a] - 1) // 2
            if a == b else n_by_type[a] * n_by_type[b],
            possible_pairs,
        )
        for a, b in pairs
    }

    intra = intramolecular_frames(intra_npz) if intra_npz is not None else {}
    ranked_rows: list[dict] = []
    contact_rows: list[dict] = []
    largest_rows: list[dict] = []
    mixture_rows: list[dict] = []

    for frame, (a, b, weights) in edge_frames(edge_npz).items():
        qualifies = weights >= min_contacts
        qa, qb = a[qualifies], b[qualifies]
        comps = connected_components(
            n_molecules, qa, qb, weights[qualifies], min_contacts=1
        )
        row: dict[str, int | float] = {"frame": frame}
        for sp in names:
            n_sp = n_by_type[sp]
            row[f"n_mol_{sp}"] = n_sp
            row[f"frac_mol_{sp}"] = _ratio(n_sp, n_molecules)

        particle_counts: Counter[str] = Counter()
        for x, y, n in zip(a, b, weights):
            particle_counts[_pair_key(str(species[int(x)]), str(species[int(y)]))] += int(n)
        qualifying_counts: Counter[str] = Counter(
            _pair_key(str(species[int(x)]), str(species[int(y)]))
            for x, y in zip(qa, qb)
        )
        total_contacts = sum(particle_counts.values())
        n_edges = len(qa)
        row["inter_particle_contacts_total"] = total_contacts
        row["mol_contacts_total"] = n_edges

        for pair_a, pair_b in pairs:
            key = _pair_key(pair_a, pair_b)
            row[f"inter_particle_contacts_{key}"] = particle_counts[key]
            row[f"frac_inter_particle_contacts_{key}"] = _ratio(
                particle_counts[key], total_contacts
            )
            row[f"mol_contacts_{key}"] = qualifying_counts[key]
            fraction = _ratio(qualifying_counts[key], n_edges)
            row[f"frac_mol_contacts_{key}"] = fraction
            row[f"expected_mol_contacts_{key}"] = expected[key]
            row[f"enrichment_mol_contacts_{key}"] = _ratio(
                fraction, expected[key]
            )

        degrees = np.zeros(n_molecules, dtype=int)
        homo = np.zeros(n_molecules, dtype=int)
        hetero = np.zeros(n_molecules, dtype=int)
        for x, y in zip(qa, qb):
            x, y = int(x), int(y)
            degrees[x] += 1
            degrees[y] += 1
            if species[x] == species[y]:
                homo[x] += 1
                homo[y] += 1
            else:
                hetero[x] += 1
                hetero[y] += 1
        for sp in names:
            nodes = species == sp
            row[f"mean_degree_{sp}"] = float(degrees[nodes].mean())
            row[f"mean_homotypic_degree_{sp}"] = float(homo[nodes].mean())
            row[f"mean_heterotypic_degree_{sp}"] = float(hetero[nodes].mean())

        mixed = [comp for comp in comps if len(set(species[comp])) > 1]
        mixed_ids = set(int(mol) for comp in mixed for mol in comp)
        row["n_clusters_mixed"] = len(mixed)
        row["frac_clusters_mixed"] = _ratio(len(mixed), len(comps))
        row["largest_mixed_cluster"] = max((len(comp) for comp in mixed), default=0)
        row["frac_molecules_in_mixed_clusters"] = _ratio(
            len(mixed_ids), n_molecules
        )
        for sp in names:
            row[f"n_clusters_pure_{sp}"] = sum(
                1 for comp in comps if np.all(species[comp] == sp)
            )
            row[f"frac_{sp}_in_mixed_clusters"] = _ratio(
                sum(species[mol] == sp for mol in mixed_ids), n_by_type[sp]
            )

        # The old mixture-wide largest cluster was ranked by molecule count.
        largest_by_molecules = max(comps, key=len)
        row["largest_cluster_size_mixture"] = len(largest_by_molecules)
        for sp in names:
            count = int(np.count_nonzero(species[largest_by_molecules] == sp))
            row[f"largest_cluster_n_{sp}"] = count
            row[f"largest_cluster_frac_{sp}"] = _ratio(
                count, len(largest_by_molecules)
            )
        row["largest_cluster_purity"] = _ratio(
            max(int(row[f"largest_cluster_n_{sp}"]) for sp in names),
            len(largest_by_molecules),
        )

        intra_counts = intra.get(frame, {})
        for sp in names:
            intra_count = sum(
                intra_counts.get(int(mol), 0)
                for mol in np.flatnonzero(species == sp)
            )
            homo_key = _pair_key(sp, sp)
            hetero_count = sum(
                particle_counts[_pair_key(sp, other)]
                for other in names if other != sp
            )
            sites = n_by_type[sp] * pairable_sites_by_species[sp]
            row[f"intra_particle_contacts_{sp}"] = intra_count
            row[f"inter_contact_saturation_{sp}"] = _ratio(
                2 * particle_counts[homo_key] + hetero_count, sites
            )
            row[f"homotypic_saturation_{sp}"] = _ratio(
                2 * particle_counts[homo_key], sites
            )
            row[f"heterotypic_saturation_{sp}"] = _ratio(
                hetero_count, sites
            )
            row[f"intra_contact_saturation_{sp}"] = _ratio(
                2 * intra_count, sites
            )
        mixture_rows.append(row)

        ranks = ranked_components(comps, lengths)
        for rank, ids in ranks:
            for sp in names:
                members = ids[species[ids] == sp]
                n_particles = int(lengths[members].sum())
                ranked_rows.append({
                    "frame": frame, "rank": rank, "species": sp,
                    "n_molecules": len(members), "n_particles": n_particles,
                    "fraction_total_particles": n_particles / total_particles,
                })

        # This table retains the original particle-count ranking.
        largest = ranks[0][1]
        member_mask = np.zeros(n_molecules, dtype=bool)
        member_mask[largest] = True
        internal = member_mask[a] & member_mask[b]
        qual_internal = internal & qualifies
        n_internal_edges = int(np.count_nonzero(qual_internal))
        largest_row: dict[str, int | float] = {
            "frame": frame, "n_molecules": len(largest),
            "n_particles": int(lengths[largest].sum()),
            "fraction_total_particles": float(lengths[largest].sum() / total_particles),
            "n_edges": n_internal_edges,
            "mean_degree": _ratio(2 * n_internal_edges, len(largest)),
        }
        internal_counts: Counter[str] = Counter()
        for x, y, n in zip(a[internal], b[internal], weights[internal]):
            internal_counts[_pair_key(str(species[int(x)]), str(species[int(y)]))] += int(n)
        internal_total = sum(internal_counts.values())
        for pair_a, pair_b in pairs:
            key = _pair_key(pair_a, pair_b)
            contact_rows.append({
                "frame": frame, "type": f"{pair_a}__{pair_b}",
                "n_inter_contacts": internal_counts[key],
                "fraction_inter_contacts": _ratio(internal_counts[key], internal_total),
            })
        for sp in names:
            sp_members = largest[species[largest] == sp]
            n_sp = len(sp_members)
            n_intra = sum(intra_counts.get(int(mol), 0) for mol in sp_members)
            homo_count = internal_counts[_pair_key(sp, sp)]
            hetero_count = sum(
                internal_counts[_pair_key(sp, other)]
                for other in names if other != sp
            )
            sites = n_sp * pairable_sites_by_species[sp]
            largest_row[f"n_molecules_{sp}"] = n_sp
            largest_row[f"available_pairable_sites_{sp}"] = sites
            largest_row[f"inter_contact_saturation_{sp}"] = _ratio(
                2 * homo_count + hetero_count, sites
            )
            largest_row[f"homotypic_saturation_{sp}"] = _ratio(
                2 * homo_count, sites
            )
            largest_row[f"heterotypic_saturation_{sp}"] = _ratio(
                hetero_count, sites
            )
            largest_row[f"intra_contact_saturation_{sp}"] = _ratio(
                2 * n_intra, sites
            )
            contact_rows.append({
                "frame": frame, "type": f"intra__{sp}",
                "n_inter_contacts": np.nan, "n_intra_contacts": n_intra,
                "fraction_inter_contacts": np.nan,
            })
        largest_rows.append(largest_row)

    return (
        pd.DataFrame(ranked_rows),
        pd.DataFrame(contact_rows),
        pd.DataFrame(largest_rows),
        pd.DataFrame(mixture_rows),
    )
