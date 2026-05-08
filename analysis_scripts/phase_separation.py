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
# ============================================================================
# PAIRING ANALYSIS: Single nucleotide model
# ============================================================================


def get_nlist(
    frame: gsd.hoomd.Frame, r_att: float = 0.75
) -> freud.locality.NeighborList:
    """Queries the neighbor list for a given frame using a cutoff radius."""
    aq = freud.locality.AABBQuery(
        box=frame.configuration.box, points=frame.particles.position
    )
    # exclude_ii=True removes self-interactions (particle i with particle i)
    query_result = aq.query(
        frame.particles.position, dict(r_max=r_att, exclude_ii=True)
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


def valency_mask(
    qi: np.ndarray, pi: np.ndarray, distances: np.ndarray, r_opt: float = 0.5
) -> np.ndarray:
    """Mask to filter out particles with multiple possible bonded partners.
    Only the partner closest to r_opt is kept.
    """
    scores = np.abs(distances - r_opt)
    order = np.argsort(scores)  # best score (closest to r_opt) first
    bonded = set()
    keep = np.zeros(len(qi), dtype=bool)

    for idx in order:
        i, j = qi[idx], pi[idx]
        if i not in bonded and j not in bonded:
            keep[idx] = True
            bonded.add(i)
            bonded.add(j)

    return keep


def get_filtered_nlist_base_pairs(
    frame: gsd.hoomd.Frame,
    r_att: float = 0.75,
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
    nlist = get_nlist(frame=frame, r_att=r_att)  # make nlist
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

    mask_all = uniq_mask & comp_mask & n_exc_mask
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
    return nlist, error_rate

def compute_filtered_nlist_base_pairs_traj(
    traj: gsd.hoomd.HOOMDTrajectory,
    r_att: float = 0.75,
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
    # loop and compute nlist over frames
    for frame_idx in sampled_indices:
        frame = traj[frame_idx]
        nlist_filt, error_rate = get_filtered_nlist_base_pairs(
            frame=frame,
            r_att=r_att,
            exclude_chain=exclude_chain,
            r_opt=r_opt,
        )
        # three attributes from nlist we are saving
        nlist_dict[f"qi_{frame_idx}"] = nlist_filt.query_point_indices
        nlist_dict[f"pi_{frame_idx}"] = nlist_filt.point_indices
        nlist_dict[f"distances_{frame_idx}"] = nlist_filt.distances
        ambiguity_rates.append(error_rate)

    nlist_dict["error_rates"] = np.array(ambiguity_rates)  # shape (n_frames,)

    return nlist_dict



# ============================================================================
# Mono/Oligo/Droplet-mer analysis
# ============================================================================


# def get_cluster_sizes_from_nlist(
#     nlist_filt: freud.NeighborList,
#     molecule_id: np.ndarray,
#     min_intermol_bonds: int = 2,
# ) -> list[int]:
#     """Find cluster sizes from nlist using connected graph."""
#     G = nx.Graph()
#     n_molecules = molecule_id[-1] + 1
#     G.add_nodes_from(range(n_molecules))

#     pair_counts = Counter()
#     # add edge for molecule connections if they exceed min_intermol_bonds
#     for i, j in zip(nlist_filt.query_point_indices, nlist_filt.point_indices):
#         mol_i = molecule_id[i]
#         mol_j = molecule_id[j]
#         # to avoid duplicates e.g. (1,2) and (2, 1) appearing as separate keys
#         mol_a = min(mol_i, mol_j)
#         mol_b = max(mol_i, mol_j)
#         if mol_a != mol_b:
#             pair_counts[(mol_a, mol_b)] += 1
#             G.add_edge(mol_a, mol_b)
#     cluster_sizes = [len(c) for c in nx.connected_components(G)]
#     return cluster_sizes


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

# ============================================================================
# STRUCTURE FACTOR
# ============================================================================




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
    n_molecules = int(molecule_id[-1]) + 1
    pair_counts = count_molecule_contacts_from_array(qi, pi, molecule_id)
    qualifying_pairs = filt_molecule_contacts_in_frame(
        pair_counts, min_intermol_bonds=min_intermol_bonds
    )
    cluster_sizes = get_cluster_sizes_from_molecule_pairs(
        qualifying_pairs, n_molecules=n_molecules
    )
    summary = summarize_cluster_sizes(cluster_sizes)

    total_contacts = len(qi)
    mol_i = molecule_id[qi]
    mol_j = molecule_id[pi]
    inter = int(np.sum(mol_i != mol_j))
    intra = int(np.sum(mol_i == mol_j))

    summary.update(
        {
            "particle_contacts": total_contacts,
            "distinct_molecule_pairs": len(pair_counts),
            "qualifying_molecule_pairs": len(qualifying_pairs),
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

    cols_to_smooth = ["E_pot", "E_sqd", "E_bond", "E_angle", "mono_frac","oligo_frac" "drop_frac"]
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