"""Per-frame RNA species composition and largest-component contact analysis."""
from __future__ import annotations

from collections import Counter
import numpy as np
import pandas as pd
from .cluster_distribution import connected_components, ranked_components
from .framewise import edge_frames


def identify_species(lengths: np.ndarray, labels_by_length: dict[int, str]) -> np.ndarray:
    """Assign one RNA species to each molecule using known chain lengths."""
    lengths = np.asarray(lengths)
    missing = set(int(x) for x in lengths) - set(labels_by_length)
    if missing:
        raise ValueError(f"Unrecognized chain lengths: {sorted(missing)}")
    return np.array([labels_by_length[int(n)] for n in lengths])


def intramolecular_frames(intra_npz) -> dict[int, dict[int, int]]:
    """Read the existing sparse per-molecule intramolecular contact NPZ."""
    frames = {int(f): {} for f in intra_npz["frame_indices"]}
    for f, mol, n in zip(intra_npz["frame"], intra_npz["mol_id"],
                         intra_npz["n_intra_contacts"]):
        frames[int(f)][int(mol)] = int(n)
    return frames


def mixture_cluster_rows(edge_npz, lengths: np.ndarray, species: np.ndarray,
                         min_contacts: int = 2, intra_npz=None):
    """Compact long-format ranked material and largest-component contact data.

    Contact type counts include accepted inter-particle contacts between nodes
    in the largest component, including weights below graph edge threshold,
    consistent with the distinction between bonding and graph connectivity.
    """
    lengths = np.asarray(lengths, dtype=int)
    species = np.asarray(species, dtype=str)
    if len(lengths) != len(species):
        raise ValueError("species and molecule lengths must align")
    total = int(lengths.sum())
    frame_edges = edge_frames(edge_npz)
    intra = intramolecular_frames(intra_npz) if intra_npz is not None else {}
    ranked_rows, contact_rows, largest_rows = [], [], []
    names = sorted(set(species))
    pair_types = [f"{a}__{b}" for i, a in enumerate(names) for b in names[i:]]
    for frame, (a, b, w) in frame_edges.items():
        qualifying = w >= min_contacts
        comps = connected_components(len(lengths), a[qualifying], b[qualifying],
                                     w[qualifying], min_contacts=1)
        ranks = ranked_components(comps, lengths)
        for rank, ids in ranks:
            for sp in names:
                members = ids[species[ids] == sp]
                ranked_rows.append({"frame": frame, "rank": rank, "species": sp,
                                    "n_molecules": len(members),
                                    "n_particles": int(lengths[members].sum()),
                                    "fraction_total_particles": float(lengths[members].sum() / total)})
        largest = ranks[0][1]
        member_set = set(int(x) for x in largest)
        in_largest = np.array([(int(x) in member_set and int(y) in member_set)
                               for x,y in zip(a,b)], dtype=bool)
        qualifying_internal = in_largest & qualifying
        n_edges = int(qualifying_internal.sum())
        largest_rows.append({"frame": frame, "n_molecules": len(largest),
                             "n_particles": int(lengths[largest].sum()),
                             "fraction_total_particles": float(lengths[largest].sum() / total),
                             "n_edges": n_edges,
                             "mean_degree": 2*n_edges/len(largest) if len(largest) else np.nan})
        counts = Counter()
        for u,v,n in zip(a[in_largest],b[in_largest],w[in_largest]):
            pair = "__".join(sorted((str(species[u]), str(species[v]))))
            counts[pair] += int(n)
        total_inter = sum(counts.values())
        for pair in pair_types:
            contact_rows.append({"frame": frame, "type": pair, "n_inter_contacts": counts[pair],
                                 "fraction_inter_contacts": counts[pair]/total_inter
                                 if total_inter else np.nan})
        if intra:
            counts_intra = intra.get(frame, {})
            for sp in names:
                ids = largest[species[largest] == sp]
                contact_rows.append({"frame": frame, "type": f"intra__{sp}",
                                     "n_inter_contacts": np.nan,
                                     "n_intra_contacts": sum(counts_intra.get(int(m),0) for m in ids),
                                     "fraction_inter_contacts": np.nan})
    return (pd.DataFrame(ranked_rows), pd.DataFrame(contact_rows),
            pd.DataFrame(largest_rows))
