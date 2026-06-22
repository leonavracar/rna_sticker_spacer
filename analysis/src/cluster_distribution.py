import numpy as np
from collections import Counter
import networkx as nx

def get_cluster_sizes_count(cluster_sizes: list[int]) -> dict:
    return Counter(cluster_sizes)


def get_cluster_sizes_from_molecule_pairs(
    persistent_pairs: Counter[tuple[int, int]] | set[tuple[int, int]],
    n_molecules: int,
) -> list[int]:
    """Build molecular clusters from persistent molecule-pair edges.

    Parameters
    ----------
    persistent_pairs : Counter[tuple[int, int]] or set[tuple[int, int]]
        Persistent intermolecular edges. If a Counter is provided, only the keys
        are used as graph edges; the counts are ignored for cluster membership.
    n_molecules : int
        Total number of molecules in the system.

    Returns
    -------
    list[int]
        Sizes of all connected components (clusters), including monomers.
    """
    G = nx.Graph()
    G.add_nodes_from(range(n_molecules))

    if isinstance(persistent_pairs, Counter):
        edge_iter = persistent_pairs.keys()
    else:
        edge_iter = persistent_pairs

    for mol_a, mol_b in edge_iter:
        G.add_edge(mol_a, mol_b)

    return [len(component) for component in nx.connected_components(G)]



def component_size_distribution_from_edge_npz_frame(
    edge_npz: np.lib.npyio.NpzFile,
    frame_idx: int,
    min_contacts: int = 1,
) -> list[dict[str, float | int]]:
    """
    Compute exact component-size distribution for one frame from molecule-edge NPZ.

    Returns one row per observed cluster/component size.

    Output columns:
        cluster_size
        n_clusters
        n_mols_in_clusters
        frac_mols

    Notes
    -----
    Monomers are included because all molecules are added as graph nodes before edges.
    """
    n_molecules = int(edge_npz["n_molecules"][0])

    frame = edge_npz["frame"]
    mol_a = edge_npz["mol_a"]
    mol_b = edge_npz["mol_b"]
    n_contacts = edge_npz["n_contacts"]

    keep = (frame == frame_idx) & (n_contacts >= min_contacts)

    G = nx.Graph()
    G.add_nodes_from(range(n_molecules))

    for a, b, count in zip(mol_a[keep], mol_b[keep], n_contacts[keep]):
        G.add_edge(int(a), int(b), weight=int(count))

    cluster_sizes = [len(component) for component in nx.connected_components(G)]
    cluster_counts = Counter(cluster_sizes)

    rows = []

    for cluster_size in sorted(cluster_counts):
        n_clusters = int(cluster_counts[cluster_size])
        n_mols_in_clusters = int(cluster_size * n_clusters)

        rows.append(
            {
                "cluster_size": int(cluster_size),
                "n_clusters": n_clusters,
                "n_mols_in_clusters": n_mols_in_clusters,
                "frac_mols": (
                    n_mols_in_clusters / n_molecules
                    if n_molecules > 0
                    else np.nan
                ),
            }
        )

    return rows

def _find(parent: np.ndarray, x: int) -> int:
    """Find root with path compression."""
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = int(parent[x])
    return int(x)


def _union(parent: np.ndarray, size: np.ndarray, a: int, b: int) -> None:
    """Union two components by size."""
    root_a = _find(parent, a)
    root_b = _find(parent, b)

    if root_a == root_b:
        return

    if size[root_a] < size[root_b]:
        root_a, root_b = root_b, root_a

    parent[root_b] = root_a
    size[root_a] += size[root_b]

def component_size_distribution_rows_from_edge_npz(
    edge_npz: np.lib.npyio.NpzFile,
    min_contacts: int = 1,
) -> list[dict[str, float | int]]:
    """
    Fast exact component-size distribution from molecule-edge NPZ.

    Output rows:
        frame
        min_contacts
        cluster_size
        n_clusters
        n_mols_in_clusters
        frac_mols

    This avoids scanning the full edge table once per frame.
    """
    n_molecules = int(edge_npz["n_molecules"][0])

    frame_indices = edge_npz["frame_indices"].astype(np.int64)

    frame = edge_npz["frame"].astype(np.int64)
    mol_a = edge_npz["mol_a"].astype(np.int64)
    mol_b = edge_npz["mol_b"].astype(np.int64)
    n_contacts = edge_npz["n_contacts"].astype(np.int64)

    keep = n_contacts >= min_contacts

    frame = frame[keep]
    mol_a = mol_a[keep]
    mol_b = mol_b[keep]

    order = np.argsort(frame)
    frame = frame[order]
    mol_a = mol_a[order]
    mol_b = mol_b[order]

    unique_frames, starts, counts = np.unique(
        frame,
        return_index=True,
        return_counts=True,
    )

    frame_to_slice = {
        int(f): slice(int(start), int(start + count))
        for f, start, count in zip(unique_frames, starts, counts)
    }

    rows = []

    for frame_idx in frame_indices:
        frame_idx = int(frame_idx)

        parent = np.arange(n_molecules, dtype=np.int64)
        size = np.ones(n_molecules, dtype=np.int64)

        if frame_idx in frame_to_slice:
            sl = frame_to_slice[frame_idx]

            for a, b in zip(mol_a[sl], mol_b[sl]):
                _union(parent, size, int(a), int(b))

        roots = np.array([_find(parent, i) for i in range(n_molecules)])
        root_counts = np.bincount(roots, minlength=n_molecules)
        component_sizes = root_counts[root_counts > 0]

        cluster_counts = Counter(component_sizes.tolist())

        for cluster_size in sorted(cluster_counts):
            n_clusters = int(cluster_counts[cluster_size])
            n_mols_in_clusters = int(cluster_size * n_clusters)

            rows.append(
                {
                    "frame": frame_idx,
                    "min_contacts": min_contacts,
                    "cluster_size": int(cluster_size),
                    "n_clusters": n_clusters,
                    "n_mols_in_clusters": n_mols_in_clusters,
                    "frac_mols": n_mols_in_clusters / n_molecules,
                }
            )

    return rows

def mono_oligo_drop_dict(cluster_counts: Counter) -> dict:
    mono_oligo_drop = {"mono": 0, "oligo": 0, "drop": 0}
    for mer in cluster_counts:
        n_molecules = cluster_counts[mer] * mer  # clusters × size = molecules
        if mer == 1:
            mono_oligo_drop["mono"] += n_molecules
        elif 1 < mer <= 4:
            mono_oligo_drop["oligo"] += n_molecules
        else:
            mono_oligo_drop["drop"] += n_molecules
    return mono_oligo_drop
