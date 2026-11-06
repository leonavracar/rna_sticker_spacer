"""Detect complementary nucleotide contacts in GSD trajectories."""

from __future__ import annotations


import numpy as np
import freud
import gsd.hoomd

from .single_chain import radius_of_gyration

def get_nlist(
    frame: gsd.hoomd.Frame, r_max: float = 0.7
) -> freud.locality.NeighborList:
    """Queries the neighbor list for a given frame using a cutoff radius."""
    aq = freud.locality.AABBQuery(
        box=frame.configuration.box, points=frame.particles.position
    )
    query_result = aq.query(
        frame.particles.position, dict(r_max=r_max, exclude_ii=True)
    )
    return query_result.toNeighborList()


def find_molecule_ends(frame: gsd.hoomd.Frame) -> np.ndarray:
    """Return particle indices ending each chain in an ordered-bond trajectory."""
    bonds = frame.bonds.group
    if len(bonds) == 0:
        return np.array([], dtype=int)
    break_mask = bonds[:-1, 1] != bonds[1:, 0]
    return bonds[:-1][break_mask, 1]


def make_molecule_lookup_table(frame: gsd.hoomd.Frame) -> np.ndarray:
    """Map each particle to a molecule ID; requires chain-ordered particles and bonds."""
    mol_ends = find_molecule_ends(frame)
    fence_posts = np.concatenate([[-1], mol_ends, [frame.particles.N - 1]])

    mol_lengths = np.diff(fence_posts)

    mol_ids = np.arange(len(mol_lengths))
    return np.repeat(mol_ids, mol_lengths)


def make_complementary_indexes(frame: gsd.hoomd.Frame) -> dict[int, int]:
    """Build a lookup table mapping each base type index to its complement type index."""
    types = frame.particles.types

    complementary = {
        "A": "U",
        "U": "A",
        "G": "C",
        "C": "G",
    }
    base_indexes = {}
    for idx in range(len(types)):
        base_indexes[types[idx]] = idx
    complementary_idxs = {}
    for key, value in complementary.items():
        complementary_idxs[base_indexes[key]] = base_indexes[value]

    return complementary_idxs


def unique_mask(nlist: freud.NeighborList) -> np.ndarray:
    """Returns a boolean mask that removes duplicate bonds from a nlist object containing particle bond indices."""
    return nlist.query_point_indices < nlist.point_indices


def complementary_mask(
    nlist: freud.locality.NeighborList,
    typeids: np.ndarray,
    comp_map: dict[int, int],
) -> np.ndarray:
    """Returns a boolean mask for pairs that are valid A-U or G-C complements..

    Converts the comp_map dict into a lookup array for vectorized speed:
    map_arr[type_id] returns the type_id of the required complement partner.
    Then checks for each bond (i, j) whether type_j == expected complement of type_i.

    Parameters
    ----------
    nlist : The neighbor list containing bond pairs to check.
    typeids : Array of shape (N_particles,) mapping particle index to its type id.
    comp_map :Dict mapping type_id -> complement type_id, e.g. {0: 1, 1: 0, 2: 3, 3: 2} for A->U, U->A, G->C, C->G..

    Returns
    -------
    np.ndarray of bool, shape (N_bonds,)
        True where bond (i, j) is a valid complementary pair."""
    max_type = max(comp_map.keys())
    map_arr = np.zeros(max_type + 1, dtype=int)
    for base, comp in comp_map.items():
        map_arr[base] = comp

    type_i = typeids[nlist.query_point_indices]
    type_j = typeids[nlist.point_indices]

    return type_j == map_arr[type_i]


def neighbor_exclusion_mask(
    nlist: freud.NeighborList,
    molecule_lookup_table: np.ndarray,
    exclude_chain: int = 3,
) -> np.ndarray:
    """Return a boolean mask for bonds that should be kept.

    Bonds are excluded only if they connect particles from the same molecule
    and the particles are within `exclude_chain` indices of each other along
    the chain. If two particles are in different molecules, they are kept
    even if their indices are close.
    """
    qi = nlist.query_point_indices.astype(np.int64)  # cast to signed!
    pi = nlist.point_indices.astype(np.int64)

    n_exc_mask = (
        np.abs(qi - pi) <= exclude_chain
    ) & (
        molecule_lookup_table[qi] == molecule_lookup_table[pi]
    )
    return ~n_exc_mask


def valency_mask(qi: np.ndarray, pi: np.ndarray, distances: np.ndarray, r_opt: float = 0.5) -> np.ndarray:
    """Retain a pair only when both particles prefer each other by distance to r_opt."""
    if len(qi) == 0:
        return np.zeros(0, dtype=bool)

    scores = np.abs(distances - r_opt)
    idx_arr = np.arange(len(qi))
    parts  = np.concatenate([qi, pi])
    scs    = np.concatenate([scores, scores])
    pidx   = np.concatenate([idx_arr, idx_arr])
    order = np.lexsort((scs, parts))
    parts_s, scs_s, pidx_s = parts[order], scs[order], pidx[order]
    first = np.empty_like(parts_s, dtype=bool)
    first[0] = True
    first[1:] = parts_s[1:] != parts_s[:-1]

    n_particles = int(parts.max()) + 1
    best_pair_idx = np.full(n_particles, -1, dtype=np.int64)
    best_pair_idx[parts_s[first]] = pidx_s[first]

    keep = (best_pair_idx[qi] == idx_arr) & (best_pair_idx[pi] == idx_arr)
    return keep


def get_filtered_nlist_base_pairs(
    frame: gsd.hoomd.Frame,
    r_min: float | None = 0.0,
    r_max: float | None = 0.7,
    exclude_chain: int = 3,
    r_opt: float = 0.5,
    molecule_lookup_table: np.ndarray | None = None,
) -> tuple[freud.locality.NeighborList, float, int, int]:
    """Apply complementarity, chain-distance, spatial and mutual-partner filters."""
    nlist = get_nlist(frame=frame, r_max=r_max)

    if molecule_lookup_table is None:
        molecule_lookup_table = make_molecule_lookup_table(frame=frame)

    uniq_mask = unique_mask(nlist=nlist)

    comp_map = make_complementary_indexes(frame=frame)
    typeids = frame.particles.typeid

    comp_mask = complementary_mask(
        nlist=nlist,
        typeids=typeids,
        comp_map=comp_map,
    )

    n_exc_mask = neighbor_exclusion_mask(
        nlist=nlist,
        molecule_lookup_table=molecule_lookup_table,
        exclude_chain=exclude_chain,
    )

    dist_mask = np.ones(len(nlist.distances), dtype=bool)

    if r_min is not None:
        dist_mask &= nlist.distances >= r_min

    if r_max is not None:
        dist_mask &= nlist.distances <= r_max

    mask_all = uniq_mask & comp_mask & n_exc_mask & dist_mask
    nlist.filter(mask_all)

    n_pre = len(nlist.query_point_indices)

    val_mask = valency_mask(
        nlist.query_point_indices,
        nlist.point_indices,
        nlist.distances,
        r_opt=r_opt,
    )

    nlist.filter(val_mask)

    n_post = len(nlist.query_point_indices)
    error_rate = (n_pre - n_post) / n_pre if n_pre > 0 else 0.0

    return nlist, error_rate, n_pre, n_post

def compute_filtered_nlist_base_pairs_traj(
    traj: gsd.hoomd.HOOMDTrajectory,
    r_min: float | None = 0.0,
    r_max: float | None = 0.7,
    exclude_chain: int = 3,
    stride: int = 1,
    r_opt: float = 0.5,
    rg_nm_per_sim_length_unit: float | None = None,
) -> dict[str, np.ndarray]:
    """Collect contacts and optional single-chain Rg from the *same* GSD frame read."""
    nlist_dict = {}
    n_frames = len(traj)
    sampled_indices = range(0, n_frames, stride)
    nlist_dict["frame_indices"] = np.array(sampled_indices)
    ambiguity_rates = []
    n_pre_values = []
    n_post_values = []
    steps = []
    rg_nm = []
    molecule_lookup_table = make_molecule_lookup_table(traj[0])
    if rg_nm_per_sim_length_unit is not None and len(np.unique(molecule_lookup_table)) != 1:
        raise ValueError("Single-chain radius of gyration requires exactly one RNA molecule")
    for frame_idx in sampled_indices:
        frame = traj[frame_idx]
        steps.append(int(frame.configuration.step))
        if rg_nm_per_sim_length_unit is not None:
            rg_nm.append(radius_of_gyration(frame, rg_nm_per_sim_length_unit))
        nlist_filt, error_rate, n_pre, n_post = get_filtered_nlist_base_pairs(
            frame=frame,
            r_min=r_min,
            r_max=r_max,
            exclude_chain=exclude_chain,
            r_opt=r_opt,
            molecule_lookup_table=molecule_lookup_table,
        )
        nlist_dict[f"qi_{frame_idx}"] = nlist_filt.query_point_indices
        nlist_dict[f"pi_{frame_idx}"] = nlist_filt.point_indices
        nlist_dict[f"distances_{frame_idx}"] = nlist_filt.distances
        ambiguity_rates.append(error_rate)
        n_pre_values.append(n_pre)
        n_post_values.append(n_post)

    nlist_dict["steps"] = np.asarray(steps, dtype=np.int64)
    if rg_nm_per_sim_length_unit is not None:
        nlist_dict["Rg_nm"] = np.asarray(rg_nm, dtype=np.float64)
    nlist_dict["error_rates"] = np.asarray(ambiguity_rates)
    nlist_dict["n_pre"] = np.asarray(n_pre_values)
    nlist_dict["n_post"] = np.asarray(n_post_values)

    return nlist_dict
