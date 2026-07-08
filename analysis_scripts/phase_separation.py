import numpy as np
import freud
import gsd.hoomd
from collections import Counter
import networkx as nx
import scipy.constants as const
import pandas as pd
from numpy.typing import NDArray
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.signal import find_peaks
from collections import deque
from collections.abc import Iterable
from pathlib import Path

# ============================================================================
# PAIRING ANALYSIS: Single nucleotide model
# ============================================================================


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
    """Creates an array mapping each particle index to a Molecule ID."""
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
    """Mutual best-partner matching: keep pair iff each endpoint's
    closest-to-r_opt candidate is the other endpoint."""
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
    r_min: float | None = 0.0, # stricter lower boundary for deciding pairs
    r_max: float | None = 0.7, # stricter upper boundary for deciding pairs
    exclude_chain: int = 3,
    r_opt: float = 0.5,
) -> freud.locality.NeighborList:
    """Generate a NeighborList of complementary base pairs with exclusions applied.

    Filters the raw neighbor list to keep only:
    1. Unique bonds (i < j)
    2. Complementary base pairs (A-U, G-C)
    3. Non-excluded pairs (not same molecule and too close along chain)

    Parameters
    ----------
    frame : Simulation frame containing particle positions and types.
    r_att : Cutoff distance for considering two particles as neighbors.
    exclude_chain : Exclude bonds between particles within this many chain indices of each other if they belong to the same molecule.

    Returns
    -------
    freud.locality.NeighborList: Filtered neighbor list containing only valid base pairing bonds.
    """
    nlist = get_nlist(frame=frame, r_max=r_max)  # make nlist
    molecule_lookup_table = make_molecule_lookup_table(frame=frame)
    uniq_mask = unique_mask(nlist=nlist)

    comp_map = make_complementary_indexes(frame=frame)
    typeids = frame.particles.typeid

    comp_mask = complementary_mask(nlist=nlist, typeids=typeids, comp_map=comp_map)
    n_exc_mask = neighbor_exclusion_mask(
        nlist=nlist,
        molecule_lookup_table=molecule_lookup_table,
        exclude_chain=exclude_chain,
    )
    # Distance filtering with r_max and r_min
    dist_mask = np.ones(len(nlist.distances), dtype=bool)

    if r_min is not None:
        dist_mask &= nlist.distances >= r_min

    if r_max is not None:
        dist_mask &= nlist.distances <= r_max
    
    mask_all = uniq_mask & comp_mask & n_exc_mask & dist_mask
    nlist.filter(mask_all)
    
    # error rate (particles bonded to >1 partner)
    n_pre = len(nlist.query_point_indices)
    # valency filter
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
    r_min: float | None = 0, # stricter lower boundary for deciding pairs
    r_max: float | None = 0.7, # stricter upper boundary for deciding pairs
    exclude_chain: int = 3,
    stride: int = 1,
    r_opt: float = 0.5,
) -> dict[str, np.ndarray]:
    """Runs get_filtered_nlist_base_pairs over entire trajectory.
    Returns a dictionary with 4 np.arrays(frame_idx, base0, base1, distances) with columns frames.
    """
    nlist_dict = {}
    n_pre_list = []
    n_post_list = []
    # Stride handling
    n_frames = len(traj)
    sampled_indices = range(0, n_frames, stride)
    # Add frame indices to nlist_dict
    nlist_dict["frame_indices"] = np.array(sampled_indices)
    ambiguity_rates = []
    # loop and compute nlist over frames
    for frame_idx in sampled_indices:
        frame = traj[frame_idx]
        nlist_filt, error_rate, n_pre, n_post = get_filtered_nlist_base_pairs(
            frame=frame,
            r_min=r_min,
            r_max=r_max,
            exclude_chain=exclude_chain,
            r_opt=r_opt,
        )
        # three attributes from nlist we are saving
        nlist_dict[f"qi_{frame_idx}"] = nlist_filt.query_point_indices
        nlist_dict[f"pi_{frame_idx}"] = nlist_filt.point_indices
        nlist_dict[f"distances_{frame_idx}"] = nlist_filt.distances
        ambiguity_rates.append(error_rate)
        n_pre_list.append(n_pre)
        n_post_list.append(n_post)
    nlist_dict["error_rates"] = np.array(ambiguity_rates)  # shape (n_frames,)
    nlist_dict["n_pre"] = np.array(n_pre_list)              # candidates before valency filter
    nlist_dict["n_post"] = np.array(n_post_list)            # kept pairs after valency filter
    return nlist_dict

def transform_and_filter_counter_to_dict(molecule_edges_counter: Counter, min_contacts: int = 1) -> dict:
    """
    Filters a molecule contact counter by removing pairs with < min_contacts and transforms counter to a dict.
    *to be used on a single frame Counter
    """
    edge_dict = {}

    for pair, count in molecule_edges_counter.items():
        if count >= min_contacts:
            edge_dict[(int(pair[0]), int(pair[1]))] = count

    return edge_dict

# ============================================================================
# Mono/Oligo/Droplet-mer analysis
# ============================================================================

def get_cluster_sizes_from_array(
    qi: np.ndarray,
    pi: np.ndarray,
    molecule_id: np.ndarray,
    min_intermol_bonds: int = 2,
) -> list[int]:
    "Find cluster sizes from three arrays using connected graph. This version is used when npz pairing file exists."
    G = nx.Graph()
    n_molecules = molecule_id[-1] + 1
    G.add_nodes_from(range(n_molecules))

    pair_counts = Counter()
    for i, j in zip(qi, pi):
        mol_i = molecule_id[i]
        mol_j = molecule_id[j]
        # to avoid duplicates e.g. (1,2) and (2, 1) appearing as separate keys
        mol_a = min(mol_i, mol_j)
        mol_b = max(mol_i, mol_j)
        if mol_a != mol_b:
            pair_counts[(mol_a, mol_b)] += 1

    for (mol_a, mol_b), n_contacts in pair_counts.items():
        if n_contacts >= min_intermol_bonds: # only add edge if molecules have >= min_intermol_bonds
            G.add_edge(mol_a, mol_b)
    return [len(c) for c in nx.connected_components(G)]


def get_cluster_sizes_count(cluster_sizes: list[int]) -> dict:
    return Counter(cluster_sizes)


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


# ============================================================================
# Persistent clusters analysis
# ============================================================================

def count_molecule_contacts_from_array(qi: np.ndarray,
    pi: np.ndarray, molecule_id: np.ndarray) -> Counter[tuple[int, int]]:
    """For each molecule pair in a frame, count the number of contacts between them."""
    pair_counts = Counter()
    for i, j in zip(qi, pi):
        mol_i = molecule_id[i]
        mol_j = molecule_id[j]
        if mol_i == mol_j:
            continue
        # to avoid duplicates e.g. (1,2) and (2, 1) appearing as separate keys
        mol_a = min(mol_i, mol_j)
        mol_b = max(mol_i, mol_j)

        pair_counts[(mol_a, mol_b)] += 1
    return pair_counts

def count_molecule_contacts_from_all_arrays(nlist_dict:dict[str,np.ndarray], molecule_id:np.ndarray ) -> dict[int, Counter[tuple[int,int]]]:
    """Count intermolecular contacts for all saved frames in an array-based contact dictionary.

    For each frame index stored in ``nlist_dict["frame_indices"]``, read
    the particle-level contact arrays ``qi_<frame_idx>`` and ``pi_<frame_idx>``, maps
    particle indices to molecule indices using ``molecule_id``, and counts how many
    contacts occur between each distinct molecule pair in that frame.

    Parameters
    ----------
    nlist_dict : dict[str, np.ndarray]
        Dictionary containing saved per-frame particle contact arrays. Expected keys are:
        ``"frame_indices"`` and, for each frame index ``f``, the arrays ``f"qi_{f}"``
        and ``f"pi_{f}"``.
    molecule_id : np.ndarray
        Array mapping each particle index to its molecule index. Shape should be
        ``(N_particles,)``.

    Returns
    -------
    dict[int, Counter[tuple[int, int]]]
        Dictionary mapping each frame index to a ``Counter`` of intermolecular contact
        counts for that frame. Each ``Counter`` maps a molecule pair
        ``(mol_a, mol_b)`` to the number of particle-level contacts observed between
        those two molecules in that frame.
        Example output:
        {
            0: Counter({(0, 4): 3, (2, 7): 1}),
            10: Counter({(0, 4): 2}),
        }

    Notes
    -----
    This function does not apply any threshold on the number of contacts required for
    a molecule pair to count as connected. 
    
    """
    pair_counts_all_mols = {}
    for frame_idx in nlist_dict["frame_indices"]:
        frame_idx = int(frame_idx) # convert to python int from np int64
        qi = nlist_dict[f"qi_{frame_idx}"] # get left-side pair particles
        pi = nlist_dict[f"pi_{frame_idx}"] # get right-side pair particles
        pair_counts = count_molecule_contacts_from_array(qi, pi, molecule_id)
        pair_counts_all_mols[frame_idx] = pair_counts
    return pair_counts_all_mols

def filt_molecule_contacts_in_frame(pair_counts:Counter[tuple[int,int]], min_intermol_bonds:int = 2) -> set[tuple[int, int]]:
    """Takes the pair_counts e.g. (0,4):1, (1,6): 3 and filters out the pairs that are < min_intermol_bonds"""
    pairs_filtered = set()
    for pair, count in pair_counts.items():
        if count >= min_intermol_bonds:
            pairs_filtered.add(pair)
    return pairs_filtered

def cluster_persistence_per_window(
    pair_counts_for_n_frames: dict[int, Counter[tuple[int, int]]],
    min_num_frames: int = 2,
    min_intermol_bonds: int = 2,
) -> Counter[tuple[int, int]]:
    """Determine which molecule pairs persist across a window of frames.

    A molecule pair is counted as present in a frame if it has at least
    `min_intermol_bonds` contacts in that frame. A molecule pair is counted as
    persistent in the window if it is present in at least `min_num_frames`
    frames of the window.

    Parameters
    ----------
    pair_counts_for_n_frames : dict[int, Counter[tuple[int, int]]]
        Dictionary mapping frame index to a Counter of molecule-pair contact
        counts for that frame.
    min_num_frames : int, optional
        Minimum number of frames in the window in which a molecule pair must be
        present to count as persistent.
    min_intermol_bonds : int, optional
        Minimum number of contacts required for a molecule pair to count as
        present in a single frame.

    Returns
    -------
    Counter[tuple[int, int]]
        Counter mapping each persistent molecule pair to the number of frames in
        the window in which it was present.
        Example output: Counter({(0,4): 2, (5,9): 2,}) -> pairs in 2 frames of window
    """
    frame_occurrence = Counter()

    for frame_idx, pair_counts in pair_counts_for_n_frames.items():
        pairs_filtered = filt_molecule_contacts_in_frame(
            pair_counts,
            min_intermol_bonds=min_intermol_bonds,
        )

        for pair in pairs_filtered:
            frame_occurrence[pair] += 1

    persistent_pairs = Counter()
    for pair, n_frames in frame_occurrence.items():
        if n_frames >= min_num_frames:
            persistent_pairs[pair] = n_frames

    return persistent_pairs

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

def persistent_cluster_sizes_all_windows(
    pair_counts_all_frames: dict[int, Counter[tuple[int, int]]],
    n_molecules: int,
    window: int = 3,
    min_num_frames: int = 2,
    min_intermol_bonds: int = 2,
) -> dict[int, list[int]]:
    """Compute persistent cluster sizes for all sliding windows.

    For each sliding window of saved frames, this function:
    1. identifies molecule pairs that are present in at least `min_num_frames`
       frames of the window, where "present in a frame" means having at least
       `min_intermol_bonds` contacts in that frame
    2. builds a molecular graph from those persistent pairs
    3. returns the connected-component sizes for that window

    The output is keyed by the center frame index of each window.

    Parameters
    ----------
    pair_counts_all_frames : dict[int, Counter[tuple[int, int]]]
        Dictionary mapping frame index to a Counter of molecule-pair contact
        counts for that frame.
    n_molecules : int
        Total number of molecules in the system.
    window : int, optional
        Number of saved frames in each sliding window.
    min_num_frames : int, optional
        Minimum number of frames in a window in which a molecule pair must be
        present to count as persistent.
    min_intermol_bonds : int, optional
        Minimum number of contacts required for a molecule pair to count as
        present in a single frame.

    Returns
    -------
    dict[int, list[int]]
        Dictionary mapping the center frame index of each window to the list of
        persistent cluster sizes in that window.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    if min_num_frames < 1:
        raise ValueError("min_num_frames must be >= 1")
    if min_num_frames > window:
        raise ValueError("min_num_frames cannot be larger than window")

    frame_indices = sorted(pair_counts_all_frames.keys())
    cluster_sizes_by_window = {}

    for start in range(len(frame_indices) - window + 1):
        window_frames = frame_indices[start : start + window]

        pair_counts_in_window = {
            frame_idx: pair_counts_all_frames[frame_idx]
            for frame_idx in window_frames
        }

        persistent_pairs = cluster_persistence_per_window(
            pair_counts_for_n_frames=pair_counts_in_window,
            min_num_frames=min_num_frames,
            min_intermol_bonds=min_intermol_bonds,
        )

        cluster_sizes = get_cluster_sizes_from_molecule_pairs(
            persistent_pairs=persistent_pairs,
            n_molecules=n_molecules,
        )

        center_frame = window_frames[window // 2]
        cluster_sizes_by_window[center_frame] = cluster_sizes

    return cluster_sizes_by_window


# ============================================================================
# CONCENTRATION
# ============================================================================

# Access using 'Avogadro'
avogadro_constant = const.Avogadro


def calc_volume(density, N_particles):
    return N_particles / density


def calc_box_length(volume):
    return volume ** (1 / 3)


def calc_molar_conc(N_molecules, V):
    return N_molecules / V


def calc_nuc_conc(length, N_molecules, V):
    # assumes same length for all molecules
    return (length * N_molecules) / V


def calc_volume_in_liters(v):
    # assumes 0.6 is one simulation unit
    return v * 0.216 * 10 ** (-24)


def calc_conc_in_mol_per_liter(N_molecules, v_in_liters):
    return (N_molecules / const.Avogadro) / v_in_liters


def calc_density_from_desired_conc(conc_in_mol_per_liter, length):
    "input concentration in mol/liter and length"
    return conc_in_mol_per_liter * const.Avogadro * length * 0.216 * 10 ** (-24)


def calc_all_density_stats(density, length, N_molecules):
    N_particles = length * N_molecules
    v = calc_volume(density, N_particles)
    v_in_liters = calc_volume_in_liters(v)
    print("volume", calc_volume(density, N_particles))
    print("box length", calc_box_length(v))
    print("molar conc", calc_molar_conc(N_molecules, v))
    print("nuc conc", calc_nuc_conc(length, N_molecules, v))
    print("volume in liters", v_in_liters)
    print("conc_in_mol_per_liter", calc_conc_in_mol_per_liter(N_molecules, v_in_liters))

### fix this so function names make sense

# AVOGADRO = const.Avogadro
# SIM_UNIT_NM = 0.6
# SIM_VOLUME_TO_L = SIM_UNIT_NM**3 * 1e-24  # 0.216e-24 L


# def calc_volume_sim_units(density, n_particles):
#     """Simulation volume from reduced particle density."""
#     return n_particles / density


# def calc_box_length_sim_units(volume):
#     """Cubic box length in simulation units."""
#     return volume ** (1 / 3)


# def calc_volume_liters(volume_sim):
#     """Convert simulation volume to liters."""
#     return volume_sim * SIM_VOLUME_TO_L


# def calc_chain_conc_mol_per_liter(n_molecules, volume_liters):
#     """RNA molecule concentration in mol/L."""
#     return (n_molecules / AVOGADRO) / volume_liters


# def calc_nuc_conc_mol_per_liter(n_particles, volume_liters):
#     """Nucleotide/bead concentration in mol/L."""
#     return (n_particles / AVOGADRO) / volume_liters


# def calc_density_from_chain_conc(chain_conc_mol_per_liter, length):
#     """
#     Convert desired RNA molecule concentration to reduced particle density.
#     length = number of nucleotides/beads per molecule.
#     """
#     return chain_conc_mol_per_liter * AVOGADRO * length * SIM_VOLUME_TO_L


# def calc_density_from_nuc_conc(nuc_conc_mol_per_liter):
#     """
#     Convert desired nucleotide concentration to reduced particle density.
#     """
#     return nuc_conc_mol_per_liter * AVOGADRO * SIM_VOLUME_TO_L


# def calc_all_density_stats(density, length, n_molecules):
#     n_particles = length * n_molecules

#     volume_sim = calc_volume_sim_units(density, n_particles)
#     box_length_sim = calc_box_length_sim_units(volume_sim)
#     box_length_nm = box_length_sim * SIM_UNIT_NM
#     volume_liters = calc_volume_liters(volume_sim)

#     chain_conc_M = calc_chain_conc_mol_per_liter(n_molecules, volume_liters)
#     nuc_conc_M = calc_nuc_conc_mol_per_liter(n_particles, volume_liters)

#     print(f"n_particles: {n_particles}")
#     print(f"volume_sim: {volume_sim}")
#     print(f"box_length_sim: {box_length_sim}")
#     print(f"box_length_nm: {box_length_nm}")
#     print(f"volume_liters: {volume_liters:.4e}")
#     print(f"reduced_chain_density: {n_molecules / volume_sim}")
#     print(f"reduced_nucleotide_density: {n_particles / volume_sim}")
#     print(f"chain_conc: {chain_conc_M:.6f} M = {chain_conc_M * 1e3:.3f} mM = {chain_conc_M * 1e6:.1f} µM")
#     print(f"nuc_conc: {nuc_conc_M:.6f} M = {nuc_conc_M * 1e3:.3f} mM")

# ============================================================================
# KINETIC ENERGY
# ============================================================================

def get_log_KE_per_particle(frame:gsd.hoomd.Frame) -> np.ndarray:
    "Used to retrieve per-particle kinetic energy when it is logged during sim"
    return frame.log["md/compute/ThermodynamicQuantities/kinetic_energy"] / frame.particles.N

def get_log_KE_over_time(traj: gsd.hoomd.HOOMDTrajectory) -> np.ndarray:
    """Return logged kinetic energy per particle for each frame in a trajectory."""
    return np.array([
        get_log_KE_per_particle(frame)
        for frame in traj
    ])

def get_log_KE_stats(traj: gsd.hoomd.HOOMDTrajectory, fraction_last:float = 0.3):
    if not 0 < fraction_last <= 1:
        raise ValueError("fraction_last must be in the interval (0, 1].")
    
    ke_time = get_log_KE_over_time(traj)
    ke_suffix = str(fraction_last * 100)
    
    n_last = max(1, int(len(ke_time) * fraction_last))
    ke_last = ke_time[-n_last:]

    ke_suffix = int(fraction_last * 100)

    return {
        f"ke_mean_last_{ke_suffix}": float(np.mean(ke_last)),
        f"ke_std_last_{ke_suffix}": float(np.std(ke_last)),
        f"ke_max_last_{ke_suffix}": float(np.max(ke_last)),
    }   

def ke_stats_from_series(
    ke: np.ndarray | pd.Series,
    fraction_last: float,
) -> dict[str, float]:
    """Compute mean/std/max of KE over the last `fraction_last` of frames."""
    ke = np.asarray(ke, dtype=float)
    ke = ke[~np.isnan(ke)]
    if len(ke) == 0:
        return {}
    n_last = max(1, int(len(ke) * fraction_last))
    ke_last = ke[-n_last:]
    suffix = int(fraction_last * 100)
    return {
        f"ke_mean_last_{suffix}": float(np.mean(ke_last)),
        f"ke_std_last_{suffix}": float(np.std(ke_last)),
        f"ke_max_last_{suffix}": float(np.max(ke_last)),
    }

# ============================================================================
# MOLECULE GRAPH HELPERS
# ============================================================================

def molecule_contact_counter_to_edge_arrays(
    pair_counts: Counter[tuple[int, int]],
    frame_idx: int,
) -> dict[str, np.ndarray]:
    """
    Convert molecule-pair contact counts for one frame into aligned edge-list arrays.

    Output arrays all have shape (n_edges_in_frame,).

    Returns
    -------
    dict[str, np.ndarray]
        {
            "frame": repeated frame index for each edge,
            "mol_a": first molecule ID,
            "mol_b": second molecule ID,
            "n_contacts": number of particle-level contacts between mol_a and mol_b,
        }
    """
    if len(pair_counts) == 0:
        return {
            "frame": np.array([], dtype=np.int64),
            "mol_a": np.array([], dtype=np.int64),
            "mol_b": np.array([], dtype=np.int64),
            "n_contacts": np.array([], dtype=np.int64),
        }

    pairs = sorted(pair_counts.items())

    mol_a = np.array([pair[0][0] for pair in pairs], dtype=np.int64)
    mol_b = np.array([pair[0][1] for pair in pairs], dtype=np.int64)
    n_contacts = np.array([count for _, count in pairs], dtype=np.int64)
    frame = np.full(len(pairs), frame_idx, dtype=np.int64)

    return {
        "frame": frame,
        "mol_a": mol_a,
        "mol_b": mol_b,
        "n_contacts": n_contacts,
    }


def build_molecule_contact_edge_dict_from_pairing_npz(
    pairing_npz: np.lib.npyio.NpzFile,
    molecule_id: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Build one molecule-level contact edge-list dictionary from a saved pairing NPZ.

    Output format:
        frame          shape (n_edges_total,)
        mol_a          shape (n_edges_total,)
        mol_b          shape (n_edges_total,)
        n_contacts     shape (n_edges_total,)
        frame_indices  shape (n_frames,)
        n_molecules    shape (1,)

    Each row k describes one molecule-molecule edge:
        frame[k], mol_a[k], mol_b[k], n_contacts[k]
    """
    frame_indices = pairing_npz["frame_indices"].astype(np.int64)
    n_molecules = int(molecule_id.max()) + 1

    frame_chunks = []
    mol_a_chunks = []
    mol_b_chunks = []
    n_contacts_chunks = []

    for frame_idx in frame_indices:
        frame_idx = int(frame_idx)

        qi = pairing_npz[f"qi_{frame_idx}"]
        pi = pairing_npz[f"pi_{frame_idx}"]

        pair_counts = count_molecule_contacts_from_array(
            qi=qi,
            pi=pi,
            molecule_id=molecule_id,
        )

        edge_arrays = molecule_contact_counter_to_edge_arrays(
            pair_counts=pair_counts,
            frame_idx=frame_idx,
        )

        frame_chunks.append(edge_arrays["frame"])
        mol_a_chunks.append(edge_arrays["mol_a"])
        mol_b_chunks.append(edge_arrays["mol_b"])
        n_contacts_chunks.append(edge_arrays["n_contacts"])

    if len(frame_chunks) == 0:
        frame = np.array([], dtype=np.int64)
        mol_a = np.array([], dtype=np.int64)
        mol_b = np.array([], dtype=np.int64)
        n_contacts = np.array([], dtype=np.int64)
    else:
        frame = np.concatenate(frame_chunks)
        mol_a = np.concatenate(mol_a_chunks)
        mol_b = np.concatenate(mol_b_chunks)
        n_contacts = np.concatenate(n_contacts_chunks)

    return {
        "frame": frame,
        "mol_a": mol_a,
        "mol_b": mol_b,
        "n_contacts": n_contacts,
        "frame_indices": frame_indices,
        "n_molecules": np.array([n_molecules], dtype=np.int64),
    }

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

# ============================================================================
# Functions for making a csv/df
# ============================================================================


def summarize_cluster_sizes(cluster_sizes: list[int]) -> dict[str, float | int]:
    """Summarize cluster sizes into simple fractions and counts."""
    if len(cluster_sizes) == 0:
        return {
            "mono_frac": 0.0,
            "oligo_frac": 0.0,
            "drop_frac": 0.0,
            "largest_cluster": 0,
            "n_clusters": 0,
        }

    cluster_counts = get_cluster_sizes_count(cluster_sizes)
    mod = mono_oligo_drop_dict(cluster_counts)
    n_molecules = sum(size * count for size, count in cluster_counts.items())

    return {
        "mono_frac": mod["mono"] / n_molecules if n_molecules > 0 else 0.0,
        "oligo_frac": mod["oligo"] / n_molecules if n_molecules > 0 else 0.0,
        "drop_frac": mod["drop"] / n_molecules if n_molecules > 0 else 0.0,
        "largest_cluster": max(cluster_sizes),
        "n_clusters": len(cluster_sizes),
    }


def instantaneous_frame_metrics_from_array(
    qi: np.ndarray,
    pi: np.ndarray,
    molecule_id: np.ndarray,
    min_intermol_bonds: int = 2,
) -> dict[str, float | int]:
    """Compute instantaneous clustering/contact metrics for one frame."""
    n_molecules = int(np.max(molecule_id)) + 1
    n_particles = int(len(molecule_id))

    pair_counts = count_molecule_contacts_from_array(qi, pi, molecule_id)
    qualifying_pairs = filt_molecule_contacts_in_frame(
        pair_counts, min_intermol_bonds=min_intermol_bonds
    )
    cluster_sizes = get_cluster_sizes_from_molecule_pairs(
        qualifying_pairs, n_molecules=n_molecules
    )
    summary = summarize_cluster_sizes(cluster_sizes)

    total_contacts = int(len(qi))
    mol_i = molecule_id[qi]
    mol_j = molecule_id[pi]
    inter = int(np.sum(mol_i != mol_j))
    intra = int(np.sum(mol_i == mol_j))

    summary.update(
        {
            "n_particles_total": n_particles,
            "n_molecules_total": n_molecules,
            "mol_len": n_particles / n_molecules,
            "particle_contacts": total_contacts,
            "inter_contacts": inter,
            "intra_contacts": intra,

            "per_particle_contacts_total": (
                total_contacts / n_particles
                if n_particles > 0 else np.nan
            ),
            "per_particle_contacts_inter": (
                inter / n_particles
                if n_particles > 0 else np.nan
            ),
            "per_particle_contacts_intra": (
                intra / n_particles
                if n_particles > 0 else np.nan
            ),
            # raw and qualifying (n_pairs >=2 to count as an edge)
            "n_molecule_pairs_raw": len(pair_counts),
            "n_molecule_pairs_qualifying": len(qualifying_pairs),
            # distinct and qualifying only kept for compatibility with old notebooks
            "distinct_molecule_pairs": len(pair_counts),
            "qualifying_molecule_pairs": len(qualifying_pairs),
            # fractions
            "inter_frac": inter / total_contacts if total_contacts > 0 else 0.0,
            "intra_frac": intra / total_contacts if total_contacts > 0 else 0.0,
        }
    )
    return summary


def framewise_ps_rows_from_nlist_dict(
    nlist_dict: dict[str, np.ndarray],
    molecule_id: np.ndarray,
    min_intermol_bonds: int = 2,
) -> list[dict[str, float | int]]:
    """Build framewise instantaneous phase-separation rows from saved pairing arrays."""
    rows = []
    for frame_idx in nlist_dict["frame_indices"]:
        frame_idx = int(frame_idx)
        qi = nlist_dict[f"qi_{frame_idx}"]
        pi = nlist_dict[f"pi_{frame_idx}"]

        row = {"frame": frame_idx}
        row.update(
            instantaneous_frame_metrics_from_array(
                qi=qi,
                pi=pi,
                molecule_id=molecule_id,
                min_intermol_bonds=min_intermol_bonds,
            )
        )
        rows.append(row)
    return rows


def persistent_window_metrics(
    pair_counts_in_window: dict[int, Counter[tuple[int, int]]],
    n_molecules: int,
    min_num_frames: int = 2,
    min_intermol_bonds: int = 2,
) -> dict[str, float | int]:
    """Compute persistent clustering metrics for one window."""
    persistent_pairs = cluster_persistence_per_window(
        pair_counts_for_n_frames=pair_counts_in_window,
        min_num_frames=min_num_frames,
        min_intermol_bonds=min_intermol_bonds,
    )
    cluster_sizes = get_cluster_sizes_from_molecule_pairs(
        persistent_pairs=persistent_pairs,
        n_molecules=n_molecules,
    )
    summary = summarize_cluster_sizes(cluster_sizes)
    summary["molecule_pairs"] = len(persistent_pairs)
    return summary


def window_ps_rows_from_pair_counts_all_frames(
    pair_counts_all_frames: dict[int, Counter[tuple[int, int]]],
    n_molecules: int,
    window: int = 3,
    min_num_frames: int = 2,
    min_intermol_bonds: int = 2,
) -> list[dict[str, float | int]]:
    """Build per-window persistent phase-separation rows."""
    if window < 1:
        raise ValueError("window must be >= 1")
    if min_num_frames < 1:
        raise ValueError("min_num_frames must be >= 1")
    if min_num_frames > window:
        raise ValueError("min_num_frames cannot be larger than window")

    frame_indices = sorted(pair_counts_all_frames.keys())
    rows = []

    for start in range(len(frame_indices) - window + 1):
        window_frames = frame_indices[start : start + window]
        pair_counts_in_window = {
            frame_idx: pair_counts_all_frames[frame_idx]
            for frame_idx in window_frames
        }

        row = {
            "center_frame": window_frames[window // 2],
            "window_start": window_frames[0],
            "window_end": window_frames[-1],
        }
        row.update(
            persistent_window_metrics(
                pair_counts_in_window=pair_counts_in_window,
                n_molecules=n_molecules,
                min_num_frames=min_num_frames,
                min_intermol_bonds=min_intermol_bonds,
            )
        )
        rows.append(row)

    return rows

# ============================================================================
# SINGLE MOLECULE: Radius of gyration, end-end distance, persistence length
# ============================================================================

def calc_rg_for_frame(
    frame: gsd.hoomd.Frame,
    scale: float = 1.0,
) -> np.float64:
    """
    Compute the radius of gyration for one frame. Returns the Rg value scaled by scale.
    """
    box = freud.box.Box.from_box(frame.configuration.box)
    positions = box.unwrap(frame.particles.position, frame.particles.image)

    com = np.mean(positions, axis=0)
    diff = positions - com
    sq_dist = np.sum(diff**2, axis=1)

    return np.float64(np.sqrt(np.mean(sq_dist)) * scale)


def compute_rg_over_time(
    traj: gsd.hoomd.HOOMDTrajectory,
    scale: float = 1.0,
    stride: int = 1,
) -> NDArray[np.float64]:
    """
    Compute the radius of gyration for each sampled frame.
    Returns an array of Rg values with shape (n_sampled_frames,).
    """
    frame_indices = np.arange(0, len(traj), stride, dtype=np.int64)
    rg_values = np.empty(len(frame_indices), dtype=np.float64)

    for i, frame_idx in enumerate(frame_indices):
        rg_values[i] = calc_rg_for_frame(traj[int(frame_idx)], scale=scale)

    return rg_values

def calc_rg_from_positions(
    positions: NDArray[np.float64],
    scale: float = 1.0,
) -> np.float64:
    """
    Calculates rg from unwrapped positions.
    """
    com = np.mean(positions, axis=0)
    diff = positions - com
    sq_dist = np.sum(diff**2, axis=1)

    return np.float64(np.sqrt(np.mean(sq_dist)) * scale)

# ============================================================================
# END-TO-END DISTANCE
# ============================================================================

def calc_end_to_end_for_frame(
    frame: gsd.hoomd.Frame,
    scale: float = 1.0,
) -> np.float64:
    """
    Compute the end-to-end distance for one frame. Returns the distance scaled by scale.

    Assumes that particle order follows chain order, so the first and last particles
    are the chain ends.
    """
    box = freud.box.Box.from_box(frame.configuration.box)
    positions = box.unwrap(frame.particles.position, frame.particles.image)

    end_to_end = np.linalg.norm(positions[-1] - positions[0])

    return np.float64(end_to_end * scale)


def compute_end_to_end_over_time(
    traj: gsd.hoomd.HOOMDTrajectory,
    scale: float = 1.0,
    stride: int = 1,
) -> NDArray[np.float64]:
    """
    Compute the end-to-end distance for each sampled frame.

    Returns an array of end-to-end distances with shape (n_sampled_frames,).
    """
    frame_indices = np.arange(0, len(traj), stride, dtype=np.int64)
    end_to_end_values = np.empty(len(frame_indices), dtype=np.float64)

    for i, frame_idx in enumerate(frame_indices):
        end_to_end_values[i] = calc_end_to_end_for_frame(
            traj[int(frame_idx)],
            scale=scale,
        )

    return end_to_end_values

def calc_end_to_end_from_positions(
    positions: NDArray[np.float64],
    scale: float = 1.0,
) -> np.float64:
    """
    Takes unwrapped positions and calculates end-end distance.
    """
    end_to_end = np.linalg.norm(positions[-1] - positions[0])

    return np.float64(end_to_end * scale)


# ============================================================================
# Recurrence/contact metrics
# ============================================================================

def recurrence_from_contact_sets(
    contact_sets: Iterable[set[tuple[int, int]]],
) -> np.float64:
    """
    Calculate recurrence from contact sets in a rolling window.
    """
    all_contacts = [
        contact
        for contact_set in contact_sets
        for contact in contact_set
    ]

    total_observations = len(all_contacts)

    if total_observations == 0:
        return np.float64(np.nan)

    unique_contacts = len(set(all_contacts))
    recurrence = 1.0 - unique_contacts / total_observations

    return np.float64(recurrence)

# ============================================================================
# SINGLE MOLECULE: Recurrence/contact metrics
# ============================================================================

def contact_regions_from_arrays(
    qi: np.ndarray,
    pi: np.ndarray,
    bin_mode: str = "repeat_window",
    bin_size: int = 5,
    repeat_size: int = 3,
    repeat_window: int = 3,
) -> set[tuple[int, int]]:
    """
    Convert particle contacts into chain-region contacts.
    """
    if bin_mode == "particle":
        region_size = 1
    elif bin_mode == "fixed":
        region_size = bin_size
    elif bin_mode == "repeat":
        region_size = repeat_size
    elif bin_mode == "repeat_window":
        region_size = repeat_size * repeat_window
    else:
        raise ValueError(
            "bin_mode must be one of: 'particle', 'fixed', 'repeat', 'repeat_window'."
        )

    regions = set()

    for i, j in zip(qi, pi):
        region_i = int(i) // region_size
        region_j = int(j) // region_size

        if region_i == region_j:
            continue

        regions.add((min(region_i, region_j), max(region_i, region_j)))

    return regions

def single_chain_frame_metrics_from_array(
    qi: np.ndarray,
    pi: np.ndarray,
    bin_mode: str = "repeat_window",
    bin_size: int = 5,
    repeat_size: int = 3,
    repeat_window: int = 3,

) -> tuple[dict[str, float | int], set[tuple[int, int]]]:
    """
    Compute single-chain contact metrics for one frame.
    """
    contact_regions = contact_regions_from_arrays(
        qi=qi,
        pi=pi,
        bin_mode=bin_mode,
        bin_size=bin_size,
        repeat_size=repeat_size,
        repeat_window=repeat_window,
    )

    row = {
        "particle_contacts": int(len(qi)),
        "contact_regions": int(len(contact_regions)),

    }

    return row, contact_regions

def framewise_single_chain_rows_from_nlist_dict(
    nlist_dict: dict[str, np.ndarray],
    recurrence_window: int = 25,
    bin_mode: str = "repeat_window",
    bin_size: int = 5,
    repeat_size: int = 3,
    repeat_window: int = 3,
) -> list[dict[str, float | int]]:
    """
    Build framewise single-chain contact recurrence rows.
    """
    rows = []
    recent_contact_sets: deque[set[tuple[int, int]]] = deque(
        maxlen=recurrence_window
    )

    for frame_idx in nlist_dict["frame_indices"]:
        frame_idx = int(frame_idx)

        qi = nlist_dict[f"qi_{frame_idx}"]
        pi = nlist_dict[f"pi_{frame_idx}"]

        row = {"frame": frame_idx}

        frame_metrics, contact_regions = single_chain_frame_metrics_from_array(
            qi=qi,
            pi=pi,
            bin_mode=bin_mode,
            bin_size=bin_size,
            repeat_size=repeat_size,
            repeat_window=repeat_window,
        )

        row.update(frame_metrics)

        recent_contact_sets.append(contact_regions)

        row["contact_region_recurrence_window"] = recurrence_from_contact_sets(
            recent_contact_sets
        )

        rows.append(row)

    return rows

# MFE (ViennaRNA comparisons)

def load_mfe_pairs_csv(mfe_csv: Path) -> set[tuple[int, int]]:
    """
    Load ViennaRNA MFE base pairs from CSV.

    Expected columns:i, j
    """
    mfe_df = pd.read_csv(mfe_csv)

    if {"i", "j"}.issubset(mfe_df.columns):
        i_col, j_col = "i", "j"
    else:
        raise ValueError(
            f"MFE CSV must contain i/j columns. "
            f"Found columns: {list(mfe_df.columns)}"
        )

    return {
        tuple(sorted((int(row[i_col]), int(row[j_col]))))
        for _, row in mfe_df.iterrows()
    }

def calc_mfe_matches(
    sim_pairs: set[tuple[int, int]],
    mfe_pairs: set[tuple[int, int]],
) -> int:
    """
    Count how many simulated base pairs match the ViennaRNA MFE base pairs.

    Parameters
    ----------
    sim_pairs
        Set of simulated base pairs for one frame.
    mfe_pairs
        Set of ViennaRNA MFE base pairs.

    Returns
    -------
    int
        Number of simulated pairs that are also present in the MFE structure.
    """
    return len(mfe_pairs.intersection(sim_pairs))


def calc_mfe_recall(
    mfe_matches: int,
    mfe_pairs_len: int,
) -> float:
    """
    Calculate the fraction of ViennaRNA MFE pairs recovered in one frame.

    Returns 0.0 if the MFE pair set is empty.
    """
    return mfe_matches / mfe_pairs_len if mfe_pairs_len > 0 else 0.0


def calc_precision(
    mfe_matches: int,
    sim_pairs_len: int,
) -> float:
    """
    Calculate the fraction of simulated pairs that match ViennaRNA MFE pairs.

    Returns 0.0 if the simulated frame has no base pairs.
    """
    return mfe_matches / sim_pairs_len if sim_pairs_len > 0 else 0.0


def calc_f1_score(
    precision: float,
    mfe_recall: float,
) -> float:
    """
    Calculate the F1 score from precision and MFE recall.

    Returns 0.0 if both precision and recall are zero.
    """
    denom = precision + mfe_recall
    return (2 * precision * mfe_recall) / denom if denom > 0 else 0.0


def calc_bp_distance(
    sim_pairs: set[tuple[int, int]],
    mfe_pairs: set[tuple[int, int]],
) -> int:
    """
    Calculate base-pair distance between simulated and MFE pair sets.

    This is the number of missing MFE pairs plus the number of extra simulated
    pairs relative to the MFE structure.
    """
    return len(mfe_pairs - sim_pairs) + len(sim_pairs - mfe_pairs)


def convert_nlist_to_idx_one_set(
    qi: np.ndarray,
    pi: np.ndarray,
    offset: int = 1,
) -> set[tuple[int, int]]:
    """
    Convert paired-index arrays from one frame into a normalized pair set.

    Parameters
    ----------
    qi, pi
        Arrays containing paired particle/base indices.
    offset
        Value added to each index. Use offset=1 to convert 0-indexed
        simulation output to ViennaRNA's 1-indexed nucleotide numbering.
        Use offset=0 if the simulation output is already 1-indexed.

    Returns
    -------
    set[tuple[int, int]]
        Set of normalized base pairs with lower index first.
    """
    return {
        tuple(sorted((int(i) + offset, int(j) + offset)))
        for i, j in zip(qi, pi)
        if int(i) != int(j)
    }


def calc_framewise_mfe_metrics(
    sim_pairs: set[tuple[int, int]],
    mfe_pairs: set[tuple[int, int]],
) -> dict[str, float | int]:
    """
    Calculate framewise agreement metrics against the ViennaRNA MFE structure.

    Parameters
    ----------
    sim_pairs
        Simulated base-pair set for one frame.
    mfe_pairs
        ViennaRNA MFE base-pair set.

    Returns
    -------
    dict[str, float | int]
        Frame-level MFE comparison metrics.
    """
    sim_pairs_len = len(sim_pairs)
    mfe_pairs_len = len(mfe_pairs)

    mfe_matches = calc_mfe_matches(sim_pairs, mfe_pairs)
    mfe_recall = calc_mfe_recall(mfe_matches, mfe_pairs_len)
    precision = calc_precision(mfe_matches, sim_pairs_len)
    f1_score = calc_f1_score(precision, mfe_recall)
    bp_distance = calc_bp_distance(sim_pairs, mfe_pairs)

    return {
        "n_sim_pairs": sim_pairs_len,
        "n_mfe_pairs": mfe_pairs_len,
        "mfe_matches": mfe_matches,
        "mfe_recall": mfe_recall,
        "mfe_precision": precision,
        "mfe_f1_score": f1_score,
        "mfe_bp_distance": bp_distance,
    }


def build_framewise_mfe_df(
    nlist_dict: dict[str, np.ndarray],
    mfe_pairs: set[tuple[int, int]],
    offset: int = 1,
) -> pd.DataFrame:
    """
    Build a framewise MFE comparison dataframe from pairing arrays.

    Parameters
    ----------
    nlist_dict
        Dictionary loaded from a pairing .npz file. Expected keys include
        frame_indices and per-frame qi_<frame>, pi_<frame> arrays.
    mfe_pairs
        ViennaRNA MFE base-pair set.
    offset
        Index offset applied to simulated indices before comparison.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per frame and MFE comparison metrics.
    """
    rows = []

    for frame_idx in nlist_dict["frame_indices"]:
        frame_idx = int(frame_idx)

        qi = nlist_dict[f"qi_{frame_idx}"]
        pi = nlist_dict[f"pi_{frame_idx}"]

        sim_pairs = convert_nlist_to_idx_one_set(
            qi=qi,
            pi=pi,
            offset=offset,
        )

        row = {"frame": frame_idx}
        row.update(
            calc_framewise_mfe_metrics(
                sim_pairs=sim_pairs,
                mfe_pairs=mfe_pairs,
            )
        )

        rows.append(row)

    return pd.DataFrame(rows)

# ============================================================================
# MULTIPLE MOLECULES: Recurrence/contact metrics
# ============================================================================

def molecule_contact_set_from_pair_counts(
    pair_counts: Counter[tuple[int, int]],
    min_intermol_bonds: int = 1,
) -> set[tuple[int, int]]:
    """
    Convert molecule-pair contact counts into a molecule-pair contact set.
    """
    return {
        pair
        for pair, count in pair_counts.items()
        if count >= min_intermol_bonds
    }

def framewise_molecule_recurrence_rows_from_nlist_dict(
    nlist_dict: dict[str, np.ndarray],
    molecule_id: np.ndarray,
    recurrence_window: int = 25,
    min_intermol_bonds: int = 1,
) -> list[dict[str, float | int]]:
    """
    Build framewise molecule-contact and particle-contact recurrence rows.

    Molecule recurrence asks whether the same molecule pairs recur.
    Particle recurrence asks whether the same particle-particle contacts recur.
    """
    rows = []

    recent_molecule_contact_sets: deque[set[tuple[int, int]]] = deque(
        maxlen=recurrence_window
    )
    recent_particle_contact_sets: deque[set[tuple[int, int]]] = deque(
        maxlen=recurrence_window
    )

    for frame_idx in nlist_dict["frame_indices"]:
        frame_idx = int(frame_idx)

        qi = nlist_dict[f"qi_{frame_idx}"]
        pi = nlist_dict[f"pi_{frame_idx}"]

        pair_counts = count_molecule_contacts_from_array(
            qi=qi,
            pi=pi,
            molecule_id=molecule_id,
        )

        molecule_contacts = molecule_contact_set_from_pair_counts(
            pair_counts=pair_counts,
            min_intermol_bonds=min_intermol_bonds,
        )

        particle_contacts = set()

        for q, p in zip(qi, pi):
            q = int(q)
            p = int(p)

            mol_q = int(molecule_id[q])
            mol_p = int(molecule_id[p])

            if mol_q == mol_p:
                continue

            mol_pair = (min(mol_q, mol_p), max(mol_q, mol_p))

            if pair_counts[mol_pair] < min_intermol_bonds:
                continue

            particle_contacts.add((min(q, p), max(q, p)))

        recent_molecule_contact_sets.append(molecule_contacts)
        recent_particle_contact_sets.append(particle_contacts)

        rows.append(
            {
                "frame": frame_idx,
                "molecule_contact_recurrence_window": recurrence_from_contact_sets(
                    recent_molecule_contact_sets
                ),
                "particle_contact_recurrence_window": recurrence_from_contact_sets(
                    recent_particle_contact_sets
                ),
            }
        )

    return rows

# ============================================================================
# MIXTURES CLUSTER ANALYSIS
# ============================================================================




# ============================================================================
# Functions for computing EDA statistics 
# ============================================================================

def add_metric_summary_stats(row: dict[str, object],
    df: pd.DataFrame,
    cols: list[str],
    prefix: str = "",
    fraction_last: float = 0.2,
) -> None:
    """
    Add summary statistics for fraction-like time series columns to a per-replicate row.

    Computes full-trajectory last, mean, std, min, and max, plus mean and std
    over the final fraction of rows.
    """
    
    last_row = df.iloc[-1]
    late_start_idx = int(len(df) * (1 - fraction_last))
    df_last = df.iloc[late_start_idx:]
    for col in cols:
        row[f"{prefix}{col}_last"] = last_row[col]
        row[f"{prefix}{col}_mean"] = df[col].mean()
        row[f"{prefix}{col}_std"] = df[col].std()
        row[f"{prefix}{col}_min"] = df[col].min()
        row[f"{prefix}{col}_max"] = df[col].max()
        row[f"{prefix}{col}_late_mean"] = df_last[col].mean()
        row[f"{prefix}{col}_late_std"] = df_last[col].std()

# ============================================================================
# Energy terms
# ============================================================================

def get_energy_terms(
    traj: gsd.hoomd.HOOMDTrajectory,
) -> dict[str, NDArray]:
    """Compute framewise energy terms (E_pot, E_sqd, E_bond, E_angle) from trajectory logs."""
    frame_indices = np.arange(0, len(traj), dtype=np.int64)

    keys = {
        "E_pot":   "md/compute/ThermodynamicQuantities/potential_energy",
        "E_sqd":   "sqd_plugin/sqd_potential/SquareDensity/energy",
        "E_bond":  "md/bond/Harmonic/energy",
        "E_angle": "md/angle/Harmonic/energy",
    }

    data = {"frame": frame_indices} | {k: np.empty(len(frame_indices), dtype=np.float64) for k in keys}

    for i, frame_idx in enumerate(frame_indices):
        log = traj[int(frame_idx)].log
        for k, log_key in keys.items():
            data[k][i] = np.asarray(log[log_key]).item()

    return data

# ============================================================================
# Plots for energy
# ============================================================================

def compute_transition_score(
    df: pd.DataFrame,
    smooth_window: int = 25,
    score_smooth_window: int = 100,
    features: list[str] | None = None,
) -> tuple[pd.Series, pd.Series]:
    """
    Compute a per-frame transition score as Euclidean speed in normalised
    feature space.

    Parameters
    ----------
    df : pd.DataFrame
        Framewise dataframe containing simulation observables.
    smooth_window : int
        Rolling window used to smooth raw features before differencing.
    score_smooth_window : int
        Rolling window applied to the raw score for peak detection.
    features : list[str] | None
        Columns to include. Defaults to
        ["mono_frac", "drop_frac", "E_pot", "E_sqd"].

    Returns
    -------
    transition_score : pd.Series  (raw)
    score_smooth     : pd.Series  (smoothed, used for peak detection)
    """
    if features is None:
        features = ["mono_frac", "drop_frac", "E_pot", "E_sqd"]

    smoothed = {
        col: df[col].rolling(window=smooth_window, center=True, min_periods=1).mean()
        for col in features if col in df.columns
    }
    X = pd.DataFrame(smoothed)
    X = (X - X.mean()) / X.std(ddof=0)
    transition_score = X.diff().pow(2).sum(axis=1).pow(0.5)
    score_smooth = transition_score.rolling(
        window=score_smooth_window, center=True, min_periods=1
    ).mean()
    return transition_score, score_smooth


def detect_transitions(
    steps: np.ndarray,
    score_smooth: pd.Series,
    height_quantile: float = 0.87,
    distance_fraction: float = 1 / 30,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect transition events from the smoothed transition score.

    Parameters
    ----------
    steps : np.ndarray
        Step values (x-axis).
    score_smooth : pd.Series
        Smoothed transition score.
    height_quantile : float
        Minimum peak height as a quantile of the score distribution.
    distance_fraction : float
        Minimum distance between peaks as a fraction of total frames.

    Returns
    -------
    transition_steps : np.ndarray
        Step values at detected transitions.
    peak_indices : np.ndarray
        Frame indices of detected peaks.
    """
    peaks, _ = find_peaks(
        score_smooth,
        height=np.quantile(score_smooth.dropna(), height_quantile),
        distance=max(1, int(len(score_smooth) * distance_fraction)),
    )
    return steps[peaks], peaks


def make_summary_figure(
    df: pd.DataFrame,
    smooth_window: int = 25,
    score_smooth_window: int = 100,
    height_quantile: float = 0.87,
    title: str | None = None,
) -> go.Figure:
    """
    Build an interactive Plotly summary figure for a single framewise CSV.

    Subplots (6 rows, shared x-axis):
        1 Potential energy (E_pot)
        2 SQD energy (E_sqd)
        3 Bond energy (E_bond)
        4 Angle energy (E_angle)
        5 Structure (e.g. mono_frac, drop_frac)
        6 Transition score (raw + smoothed + detected peaks)

    Vertical dashed lines mark detected transitions across all subplots.
    Open-circle markers at the last frame show end values.
    Works for any system size — fractions (mono_frac, drop_frac) are
    already normalised by molecule count upstream.
    Energy columns (E_pot, E_sqd, E_bond, E_angle) are optional;
    rows are silently skipped if the column is absent.

    Parameters
    ----------
    df : pd.DataFrame
        Framewise dataframe (output of framewise_ps_rows_from_nlist_dict
        or loaded from CSV), must contain a "step" column.
    smooth_window : int
        Rolling window for smoothing energy / structure signals.
    score_smooth_window : int
        Rolling window for smoothing the transition score.
    height_quantile : float
        Quantile threshold for transition peak detection.
    title : str | None
        Optional figure title.

    Returns
    -------
    fig : go.Figure
    """
    df = df.copy().reset_index(drop=True)
    steps = df["step"].values

    cols_to_smooth = ["E_pot", "E_sqd", "E_bond", "E_angle", "mono_frac","oligo_frac", "drop_frac"]
    for col in cols_to_smooth:
        if col in df.columns:
            df[f"{col}_smooth"] = df[col].rolling(window=smooth_window, center=True, min_periods=1).mean()

    transition_score, score_smooth = compute_transition_score(
        df, smooth_window=smooth_window, score_smooth_window=score_smooth_window
    )
    transition_steps, peak_indices = detect_transitions(steps, score_smooth, height_quantile=height_quantile)

    vline_shapes = [
        dict(type="line", x0=s, x1=s, y0=0, y1=1,
             xref="x", yref="paper", line=dict(color="gray", width=1, dash="dot"))
        for s in transition_steps
    ]

    fig = make_subplots(
        rows=6, cols=1, shared_xaxes=True,
        subplot_titles=["Potential Energy", "SQD", "Bond", "Angle", "Structure", "Transition Score"],
        vertical_spacing=0.03,
        row_heights=[1, 1, 0.8, 0.8, 1, 0.8],
    )

    def _hover(label):
        return f"{label}: %{{y:.4g}}<extra></extra>"

    def _raw(ax, col, color="lightblue"):
        if col not in df.columns:
            return
        fig.add_trace(go.Scatter(
            x=steps, y=df[col], mode="lines",
            line=dict(color=color, width=0.5), opacity=0.35,
            showlegend=False, hovertemplate=_hover(col),
        ), row=ax, col=1)

    def _smooth(ax, col, label, color, width=2):
        src = df.get(f"{col}_smooth", df.get(col))
        if src is None:
            return
        fig.add_trace(go.Scatter(
            x=steps, y=src, mode="lines",
            line=dict(color=color, width=width),
            name=label, hovertemplate=_hover(label),
        ), row=ax, col=1)

    def _end_marker(ax, col, label, color):
        """Open circle + value label at the last valid frame."""
        src = df.get(f"{col}_smooth", df.get(col))
        if src is None:
            return
        valid = src.dropna()
        if valid.empty:
            return
        last_x, last_y = steps[valid.index[-1]], valid.iloc[-1]
        fig.add_trace(go.Scatter(
            x=[last_x], y=[last_y], mode="markers+text",
            marker=dict(color=color, size=8, symbol="circle-open", line=dict(width=2)),
            text=[f"{last_y:.3g}"], textposition="middle right",
            textfont=dict(size=9, color=color), showlegend=False,
            hovertemplate=f"last — {label}: %{{y:.4g}}<extra></extra>",
        ), row=ax, col=1)

    # row 1: E_pot
    _raw(1, "E_pot")
    _smooth(1, "E_pot", "E_pot", "darkorange")
    skip = max(1, len(df) // 50)
    pe = df.get("E_pot_smooth", df.get("E_pot"))
    if pe is not None:
        pe = pe.iloc[skip:]
        pad = (pe.max() - pe.min()) * 0.25
        fig.update_yaxes(range=[pe.min() - pad, pe.max() + pad], row=1, col=1)
    _end_marker(1, "E_pot", "E_pot", "darkorange")

    # row 2: E_sqd
    _raw(2, "E_sqd")
    _smooth(2, "E_sqd", "E_sqd", "darkorange")
    _end_marker(2, "E_sqd", "E_sqd", "darkorange")

    # row 3: E_bond
    _smooth(3, "E_bond", "E_bond", "steelblue", width=1.0)
    _end_marker(3, "E_bond", "E_bond", "steelblue")

    # row 4: E_angle
    _smooth(4, "E_angle", "E_angle", "steelblue", width=1.0)
    _end_marker(4, "E_angle", "E_angle", "steelblue")

    # row 5: structure
    for col, label, color in [("mono_frac", "mono_frac", "steelblue"), ("drop_frac", "drop_frac", "darkorange")]:
        _smooth(5, col, label, color)
    _end_marker(5, "mono_frac", "mono_frac", "steelblue")

    # row 6: transition score
    fig.add_trace(go.Scatter(
        x=steps, y=transition_score, mode="lines",
        line=dict(color="steelblue", width=0.4), opacity=0.4,
        name="raw score", hovertemplate=_hover("raw"),
    ), row=6, col=1)
    fig.add_trace(go.Scatter(
        x=steps, y=score_smooth, mode="lines",
        line=dict(color="black", width=1.0),
        name="score (smoothed)", hovertemplate=_hover("score"),
    ), row=6, col=1)
    fig.add_trace(go.Scatter(
        x=transition_steps, y=score_smooth.values[peak_indices],
        mode="markers", marker=dict(color="red", size=8),
        name="transitions", hovertemplate="score: %{y:.4g}<extra></extra>",
    ), row=6, col=1)

    # x-axis ticks
    x_max = steps.max()
    major = np.arange(0, x_max + 1e5, 1e6)
    minor = np.arange(0.5e6, x_max, 1e6)
    all_ticks = np.sort(np.concatenate([major, minor]))
    fig.update_xaxes(
        tickvals=all_ticks, ticktext=[f"{t/1e6:.1f}" for t in all_ticks],
        tickfont=dict(size=9), row=6, col=1,
    )
    fig.update_xaxes(title_text="step (x10⁶)", row=6, col=1)

    fig.update_layout(
        height=1400, width=1200, shapes=vline_shapes,
        hovermode="x unified", template="plotly_white",
        legend=dict(orientation="v", x=1.01, y=1),
        margin=dict(r=130),
        title=dict(text=title, x=0.5) if title else None,
    )
    fig.update_annotations(font_size=11)
    return fig

# Molecule graph (new)

def _edge_strength_summary(
    contacts: np.ndarray,
    suffix: str,
) -> dict[str, float | int]:
    """Summarize molecule-edge strengths for one frame."""
    if len(contacts) == 0:
        return {
            f"edge_strength_mean_{suffix}": 0.0,
            f"edge_strength_q90_{suffix}": 0.0,
            f"edge_strength_max_{suffix}": 0.0,
            f"total_particle_contacts_{suffix}": 0,
        }

    return {
        f"edge_strength_mean_{suffix}": float(np.mean(contacts)),
        f"edge_strength_q90_{suffix}": float(np.percentile(contacts, 90)),
        f"edge_strength_max_{suffix}": int(np.max(contacts)),
        f"total_particle_contacts_{suffix}": int(np.sum(contacts)),
    }


def graph_framewise_summary_rows_from_edge_npz(
    edge_npz: np.lib.npyio.NpzFile,
    n_particles_total: int,
    min_contacts: int = 1,
) -> list[dict[str, float | int]]:
    """Build framewise molecule-graph summary rows from molecule-edge NPZ.

    Row meaning:
        one row = one trajectory frame

    Raw edges:
        molecule pairs with at least one particle-level contact

    Qualifying edges:
        molecule pairs with n_contacts >= min_contacts
    """
    n_molecules_total = int(edge_npz["n_molecules"][0])
    frame_indices = edge_npz["frame_indices"].astype(np.int64)

    frame = edge_npz["frame"].astype(np.int64)
    n_contacts = edge_npz["n_contacts"].astype(np.int64)

    possible_molecule_pairs = (
        n_molecules_total * (n_molecules_total - 1) / 2
    )

    order = np.argsort(frame)
    frame_sorted = frame[order]
    contacts_sorted = n_contacts[order]

    unique_frames, starts, counts = np.unique(
        frame_sorted,
        return_index=True,
        return_counts=True,
    )

    contacts_by_frame = {
        int(frame_idx): contacts_sorted[int(start): int(start + count)]
        for frame_idx, start, count in zip(unique_frames, starts, counts)
    }

    rows = []

    for frame_idx in frame_indices:
        frame_idx = int(frame_idx)

        raw_contacts = contacts_by_frame.get(
            frame_idx,
            np.array([], dtype=np.int64),
        )
        qualifying_contacts = raw_contacts[raw_contacts >= min_contacts]

        n_molecule_pairs_raw = int(len(raw_contacts))
        n_molecule_pairs_qualifying = int(len(qualifying_contacts))

        total_particle_contacts_raw = int(np.sum(raw_contacts))
        total_particle_contacts_qualifying = int(np.sum(qualifying_contacts))

        row = {
            "frame": frame_idx,
            "n_molecules_total": n_molecules_total,
            "n_particles_total": int(n_particles_total),
            "min_contacts": int(min_contacts),
            "possible_molecule_pairs": possible_molecule_pairs,
            "n_molecule_pairs_raw": n_molecule_pairs_raw,
            "n_molecule_pairs_qualifying": n_molecule_pairs_qualifying,
            "edge_density_raw": (
                n_molecule_pairs_raw / possible_molecule_pairs
                if possible_molecule_pairs > 0
                else np.nan
            ),
            "edge_density_qualifying": (
                n_molecule_pairs_qualifying / possible_molecule_pairs
                if possible_molecule_pairs > 0
                else np.nan
            ),
            "mean_degree_raw": (
                2 * n_molecule_pairs_raw / n_molecules_total
                if n_molecules_total > 0
                else np.nan
            ),
            "mean_degree_qualifying": (
                2 * n_molecule_pairs_qualifying / n_molecules_total
                if n_molecules_total > 0
                else np.nan
            ),
        }

        row.update(_edge_strength_summary(raw_contacts, "raw"))
        row.update(_edge_strength_summary(qualifying_contacts, "qualifying"))

        row["particle_contacts_per_particle_raw"] = (
            total_particle_contacts_raw / n_particles_total
            if n_particles_total > 0
            else np.nan
        )
        row["particle_contacts_per_particle_qualifying"] = (
            total_particle_contacts_qualifying / n_particles_total
            if n_particles_total > 0
            else np.nan
        )

        rows.append(row)

    return rows

### Intramol contact graph
from collections import Counter
import numpy as np


def build_molecule_intra_contact_dict_from_pairing_npz(
    pairing_npz,
    molecule_id: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Build sparse molecule-level intramolecular contact counts from pairing data.

    One output row represents one molecule in one frame with at least one
    intramolecular particle contact.

    Parameters
    ----------
    pairing_npz
        Loaded pairing .npz file. Expected keys include frame_indices and
        per-frame qi_<frame>, pi_<frame> arrays.
    molecule_id
        Particle-to-molecule lookup array. molecule_id[p] gives the molecule
        index of particle p.

    Returns
    -------
    dict[str, np.ndarray]
        Sparse intramolecular contact table with arrays:
        frame, mol_id, n_intra_contacts, frame_indices, n_molecules.
    """
    frame_out = []
    mol_out = []
    contact_out = []

    molecule_id = np.asarray(molecule_id, dtype=int)
    n_molecules = int(molecule_id.max()) + 1

    frame_indices = np.asarray(pairing_npz["frame_indices"], dtype=int)

    for frame_idx in frame_indices:
        frame_idx = int(frame_idx)

        qi = np.asarray(pairing_npz[f"qi_{frame_idx}"], dtype=int)
        pi = np.asarray(pairing_npz[f"pi_{frame_idx}"], dtype=int)

        counts = Counter()
        seen_particle_pairs = set()

        for q, p in zip(qi, pi):
            q = int(q)
            p = int(p)

            if q == p:
                continue

            mol_q = int(molecule_id[q])
            mol_p = int(molecule_id[p])

            if mol_q != mol_p:
                continue

            particle_pair = tuple(sorted((q, p)))

            if particle_pair in seen_particle_pairs:
                continue

            seen_particle_pairs.add(particle_pair)
            counts[mol_q] += 1

        for mol, n_contacts in counts.items():
            frame_out.append(frame_idx)
            mol_out.append(mol)
            contact_out.append(n_contacts)

    return {
        "frame": np.asarray(frame_out, dtype=np.int64),
        "mol_id": np.asarray(mol_out, dtype=np.int64),
        "n_intra_contacts": np.asarray(contact_out, dtype=np.int64),
        "frame_indices": frame_indices,
        "n_molecules": np.asarray([n_molecules], dtype=np.int64),
    }