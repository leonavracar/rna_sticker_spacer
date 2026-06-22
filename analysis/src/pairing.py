import numpy as np
import freud
import gsd.hoomd

"""Particle-level contact detection.

This module identifies valid particle-particle contacts from GSD trajectories.
Contacts are filtered by distance, base-complementarity rules, local
chain-neighbor exclusion, and valency matching.

Main output:
    pairing NPZ with frame-wise qi/pi/distances arrays and molecule id mapped to particle
    index.
"""

def get_nlist(
    frame: gsd.hoomd.Frame, r_max: float = 0.75
) -> freud.locality.NeighborList:
    """Queries the neighbor list for a given frame using a cutoff radius."""
    aq = freud.locality.AABBQuery(
        box=frame.configuration.box, points=frame.particles.position
    )
    # exclude_ii=True removes self-interactions (particle i with particle i)
    query_result = aq.query(
        frame.particles.position, dict(r_max=r_max, exclude_ii=True)
    )
    return query_result.toNeighborList()


def find_molecule_ends(frame: gsd.hoomd.Frame) -> np.ndarray:
    """Identifies the last particle index of each molecule chain."""
    bonds = frame.bonds.group

    # If there are no bonds, we can't find molecule ends this way
    if len(bonds) == 0:
        return np.array([], dtype=int)

    # We compare the 'end' of bond N to the 'start' of bond N+1
    # This mask will have length (len(bonds) - 1)
    break_mask = bonds[:-1, 1] != bonds[1:, 0]

    # We apply the mask to 'bonds[:-1]' so the sizes match
    # This gives us the particle ID at the end of every molecule except the last one
    return bonds[:-1][break_mask, 1]


def make_molecule_lookup_table(frame: gsd.hoomd.Frame) -> np.ndarray:
    """Creates an array mapping each particle index to a Molecule ID.
    Assumes particles and bonds are ordered by molecule/chain."""
    mol_ends = find_molecule_ends(frame)
    # The 'Fence Post' method: start at -1, mark ends, end at N-1
    fence_posts = np.concatenate([[-1], mol_ends, [frame.particles.N - 1]])

    mol_lengths = np.diff(fence_posts)

    mol_ids = np.arange(len(mol_lengths))
    return np.repeat(mol_ids, mol_lengths)


def make_complementary_indexes(frame: gsd.hoomd.Frame) -> dict:
    """Build a lookup table mapping each base type index to its complement type index."""
    types = frame.particles.types

    complementary = {
        "A": "U",
        "U": "A",
        "G": "C",
        "C": "G",
    }

    # Map base names like "A" to their numeric type indices
    base_indexes = {}
    for idx in range(len(types)):
        base_indexes[types[idx]] = idx

    # Convert base-name complement rules into type-index complement rules
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
    comp_map: dict,
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


def valency_mask(qi, pi, distances, r_opt=0.5):
    """For every candidate particle-pair contact, score it by abs(distance - r_opt).
        A contact is kept only if both particles select each other as their best
        available partner. This enforces one contact partner per particle."""
    if len(qi) == 0:
        return np.zeros(0, dtype=bool)

    scores = np.abs(distances - r_opt)
    idx_arr = np.arange(len(qi))

    # Stack: each pair contributes two (particle, score, pair_idx) rows
    parts  = np.concatenate([qi, pi])
    scs    = np.concatenate([scores, scores])
    pidx   = np.concatenate([idx_arr, idx_arr])

    # Sort by particle, then by score ascending (first row per particle is its best)
    order = np.lexsort((scs, parts))
    parts_s, scs_s, pidx_s = parts[order], scs[order], pidx[order]

    # First occurrence of each particle = its best pair
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
    r_min: float | None = 0.42,
    r_max: float | None = 0.58,
    exclude_chain: int = 3,
    r_opt: float = 0.5,
    molecule_lookup_table: np.ndarray | None = None,
) -> tuple[freud.locality.NeighborList, float]:
    """Generate a NeighborList of complementary base pairs with exclusions applied."""
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

    return nlist, error_rate

def compute_filtered_nlist_base_pairs_traj(
    traj: gsd.hoomd.HOOMDTrajectory,
    r_min: float | None = 0.42, # stricter lower boundary for deciding pairs
    r_max: float | None = 0.58, # stricter upper boundary for deciding pairs
    exclude_chain: int = 3,
    stride: int = 1,
    r_opt: float = 0.5,
) -> dict[str, np.ndarray]:
    """Runs get_filtered_nlist_base_pairs over entire trajectory.
    Returns a dictionary with 4 np.arrays(frame_idx, base0, base1, distances) with columns frames.
    """
    nlist_dict = {}

    # Stride handling
    n_frames = len(traj)
    sampled_indices = range(0, n_frames, stride)
    # Add frame indices to nlist_dict
    nlist_dict["frame_indices"] = np.array(sampled_indices)
    ambiguity_rates = []
    molecule_lookup_table = make_molecule_lookup_table(traj[0])
    # loop and compute nlist over frames
    for frame_idx in sampled_indices:
        frame = traj[frame_idx]
        nlist_filt, error_rate = get_filtered_nlist_base_pairs(
            frame=frame,
            r_min=r_min,
            r_max=r_max,
            exclude_chain=exclude_chain,
            r_opt=r_opt,
            molecule_lookup_table=molecule_lookup_table,
        )
        # three attributes from nlist we are saving
        nlist_dict[f"qi_{frame_idx}"] = nlist_filt.query_point_indices
        nlist_dict[f"pi_{frame_idx}"] = nlist_filt.point_indices
        nlist_dict[f"distances_{frame_idx}"] = nlist_filt.distances
        ambiguity_rates.append(error_rate)

    nlist_dict["error_rates"] = np.array(ambiguity_rates)  # shape (n_frames,)
    nlist_dict["r_min"] = np.array([np.nan if r_min is None else r_min], dtype=float)
    nlist_dict["r_max"] = np.array([np.nan if r_max is None else r_max], dtype=float)
    nlist_dict["r_opt"] = np.array([r_opt], dtype=float)
    nlist_dict["exclude_chain"] = np.array([exclude_chain], dtype=np.int64)
    nlist_dict["stride"] = np.array([stride], dtype=np.int64)
    
    return nlist_dict

