"""Connected molecular components, cluster-size bins and ranked clusters."""
from __future__ import annotations

from collections.abc import Iterable
import numpy as np

MOLECULE_BINS = ("mono", "di", "tri", "4-9", "10-19", "20-49", "50+")
PARTICLE_BINS = ("mono", "<2%", "2-<3%", "3-<10%", "10-<20%", "20-<50%", ">=50%")


def connected_components(
    n_molecules: int, mol_a: np.ndarray, mol_b: np.ndarray,
    weights: np.ndarray, min_contacts: int = 2,
) -> list[np.ndarray]:
    """Components of the qualifying *undirected* molecular graph, including singletons."""
    if n_molecules < 1 or min_contacts < 1:
        raise ValueError("n_molecules and min_contacts must be positive")
    parent = np.arange(n_molecules, dtype=int)
    sizes = np.ones(n_molecules, dtype=int)

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    for a, b, w in zip(mol_a, mol_b, weights):
        if w < min_contacts:
            continue
        a, b = int(a), int(b)
        if not (0 <= a < n_molecules and 0 <= b < n_molecules):
            raise ValueError("Molecule edge references an unknown molecule")
        ra, rb = root(a), root(b)
        if ra != rb:
            if sizes[ra] < sizes[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            sizes[ra] += sizes[rb]

    groups: dict[int, list[int]] = {}
    for i in range(n_molecules):
        groups.setdefault(root(i), []).append(i)
    return [np.asarray(ids, dtype=int) for ids in groups.values()]


def component_records(components: Iterable[np.ndarray], lengths: np.ndarray):
    """Minimal per-component information. Particle fraction uses N_particles."""
    lengths = np.asarray(lengths, dtype=int)
    total = int(lengths.sum())
    if total <= 0:
        raise ValueError("Total particle count must be positive")
    for component in components:
        ids = np.asarray(component, dtype=int)
        particles = int(lengths[ids].sum())
        yield {
            "molecule_ids": ids,
            "cluster_size_molecules": len(ids),
            "cluster_size_particles": particles,
            "cluster_particle_fraction": particles / total,
        }


def molecule_size_bin(size: int) -> str:
    if size == 1:
        return "mono"
    if size == 2:
        return "di"
    if size == 3:
        return "tri"
    if size < 10:
        return "4-9"
    if size < 20:
        return "10-19"
    if size < 50:
        return "20-49"
    return "50+"


def particle_size_bin(size: int, particle_fraction: float) -> str:
    """Assign a particle-fraction category, reserving a bin for monomers."""
    if size == 1:
        return "mono"
    if particle_fraction < .02:
        return "<2%"
    if particle_fraction < .03:
        return "2-<3%"
    if particle_fraction < .10:
        return "3-<10%"
    if particle_fraction < .20:
        return "10-<20%"
    if particle_fraction < .50:
        return "20-<50%"
    return ">=50%"


def cluster_bin_fractions(components: Iterable[np.ndarray], lengths: np.ndarray):
    """Fractions of *particles* in each cluster-size category for both schemes."""
    by_molecules = dict.fromkeys(MOLECULE_BINS, 0.0)
    by_particles = dict.fromkeys(PARTICLE_BINS, 0.0)
    for c in component_records(components, lengths):
        fraction = c["cluster_particle_fraction"]
        size = c["cluster_size_molecules"]
        by_molecules[molecule_size_bin(size)] += fraction
        by_particles[particle_size_bin(size, fraction)] += fraction
    return {"molecule_count": by_molecules, "particle_percentage": by_particles}


def ranked_components(components: Iterable[np.ndarray], lengths: np.ndarray):
    """Rank multimers by particle count; retain all monomer material as one class.

    Return (rank_label, molecule_ids), including empty ranks for absent components.
    """
    lengths = np.asarray(lengths)
    groups = [np.asarray(c, dtype=int) for c in components]
    multimers = [c for c in groups if len(c) > 1]
    multimers.sort(key=lambda c: (-int(lengths[c].sum()), int(c.min())))
    labels = ("largest", "second", "third")
    result = [(label, multimers[i] if i < len(multimers) else np.empty(0, dtype=int))
              for i, label in enumerate(labels)]
    result.append(("other_multimers",
                   np.concatenate(multimers[3:]) if len(multimers) > 3
                   else np.empty(0, dtype=int)))
    monomers = [c for c in groups if len(c) == 1]
    result.append(("monomers", np.concatenate(monomers) if monomers
                   else np.empty(0, dtype=int)))
    return result


def ranked_cluster_rows(components: Iterable[np.ndarray], lengths: np.ndarray,
                        species: np.ndarray | None = None):
    """Particle-weighted ranked cluster material; species columns if supplied."""
    lengths = np.asarray(lengths, dtype=int)
    total = int(lengths.sum())
    if species is not None:
        species = np.asarray(species, dtype=str)
        if len(species) != len(lengths):
            raise ValueError("species must provide one label per molecule")
    rows = []
    for label, ids in ranked_components(components, lengths):
        n_particles = int(lengths[ids].sum())
        row = {"rank": label, "n_molecules": len(ids),
               "n_particles": n_particles, "fraction_particles": n_particles / total}
        if species is not None:
            for name in np.unique(species):
                count = int(lengths[ids[species[ids] == name]].sum())
                row[f"fraction_particles_{name}"] = count / total
                row[f"n_molecules_{name}"] = int(np.count_nonzero(species[ids] == name))
        rows.append(row)
    return rows
