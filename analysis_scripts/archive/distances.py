import json
from pathlib import Path

import numpy as np
import freud
import gsd.hoomd


EXPECTED_PAIRS_AGUC30 = np.array([
    29, 28, 27, 26, 25, 24, 23, 22, 21, 20,
    19, 18, 17, 16, 15, 14, 13, 12, 11, 10,
    9,  8,  7,  6,  5,  4,  3,  2,  1,  0
], dtype=int)

# ============================================================================
# HELPERS
# ============================================================================

def pbc_distances_freud(box, ref_pos, other_pos, sqrt=True):
    """
    Compute minimum-image distances from one reference position to many
    candidate positions under periodic boundary conditions.

    Args:
        box: freud.box.Box
        ref_pos: np.ndarray of shape (3,)
        other_pos: np.ndarray of shape (M, 3)
        sqrt: if True, euclidean distance (sqrt) is returned, if False, squared distance

    Returns:
        np.ndarray of shape (M,) with PBC-aware distances.
    """
    dr = other_pos - ref_pos
    dr = box.wrap(dr)
    if sqrt:
        return np.linalg.norm(dr, axis=1)
    else:
        return np.sum(dr * dr, axis=1)


def pairs_from_pair_array(pair_array):
    """
    Convert a per-residue partner array into a set of unique unordered pairs.

    Args:
        pair_array: np.ndarray of shape (N,), where pair_array[i] is the
            partner index for residue i, or -1 if unpaired.

    Returns:
        Set of tuples (i, j) with i < j, representing unique pairs.
    """
    pairs = set()
    for i, j in enumerate(pair_array):
        if j >= 0:
            j = int(j)
            if i < j:
                pairs.add((i, j))
    return pairs


# ============================================================================
# PAIRING CALCULATIONS
# ============================================================================


def get_pair_distances(
    pair_array: np.ndarray,
    positions: np.ndarray,
    box: freud.box.Box,
    particles_per_residue: int = 3,
    atom_offset: int = 1
) -> np.ndarray:
    """
    Given a completed pair_array, returns an array of the same length
    where entry i is the distance between base i and its partner,
    or np.nan if unpaired.
    
    Args:
        pair_array: output of get_pair_array, shape (N,)
        positions: particle positions, shape (N*particles_per_residue, 3)
        box: freud box for PBC distance calculation
        
    Returns:
        distances: shape (N,), np.nan for unpaired bases
    """
    distances = np.full(len(pair_array), np.nan)
    for i, partner in enumerate(pair_array):
        if partner >= 0:
            ref = positions[atom_offset + particles_per_residue * i]
            other = positions[atom_offset + particles_per_residue * partner]
            distances[i] = pbc_distances_freud(
                box, ref, other[np.newaxis, :])[0]
    return distances

# ============================================================================
# STRICT WC BOTH BASES PAIRING
# ============================================================================


def get_pair_array(snapshot, cutoff=1.5, exclude_chain=2):

    complimentary = {
        "A": "U",
        "U": "A",
        "G": "C",
        "C": "G"}
    positions = snapshot.particles.position
    box = freud.box.Box.from_box(snapshot.configuration.box)

    # sequence
    sequence = np.array([snapshot.particles.types[x][0]
                        # types[identity][first element e.g. A1 -> A] from typeid[1::3] -> start w base (idx 1), increments of 3 (base-base-phosphate)
                         for x in snapshot.particles.typeid[1::3]])
    # sequence = simplified sequence to be just the bases and just A, G etc. not A1, G1 etc.

    # Pairing status (initially all -2, changes to -1 if no partner found)
    paired = np.full((len(sequence)), -2)
    for base_type in ["A", "G"]:
        self_mask = sequence == base_type  # boolean mask to map positions of A, G
        # boolean mask to map positions of U A
        complimentary_mask = sequence == complimentary[base_type]

        # positions to check for A and G (the iterations of outer for loop)
        check_idxs = np.where(self_mask)[0]
        for idx in check_idxs:
            pair = find_mutual_pair(
                positions, box, idx, paired, self_mask, complimentary_mask, base=0, cutoff=cutoff, exclude_chain=exclude_chain)
            if pair:
                # reset indices so that these points can be checked again!
                paired[pair[0]], paired[pair[1]] = -2, -2
                brothers = find_mutual_pair(
                    positions, box, idx, paired, self_mask, complimentary_mask, base=1, cutoff=cutoff, exclude_chain=exclude_chain)
                if brothers == (idx, pair[1]):
                    paired[pair[0]], paired[pair[1]] = pair[1], pair[0]
    paired[paired == -2] = -1
    return paired


def find_single_partner(positions, box, idx, paired, type_mask, base=0, cutoff=1.5, curr_idx=None, exclude_chain=2):
    if paired[idx] != -2:
        return np.array([])

    paired[idx] = -1
    # Pass curr_idx here
    mask = get_mask(positions, idx, paired, type_mask, exclude_chain=exclude_chain, curr_idx=curr_idx)

    if not np.any(mask):
        return np.array([])

    filtered_positions = positions[base+1::3][mask]
    ref = positions[base+1+3*idx]
    distances = pbc_distances_freud(box, ref, filtered_positions)

    below_cutoff = np.nonzero(distances < cutoff)[0]
    original_idxs = np.where(mask)[0][below_cutoff]

    if len(original_idxs) > 1:
        # To fix the "less pairs at higher cutoff" issue,
        # pick the CLOSEST one instead of returning None/empty
        closest_local_idx = np.argmin(distances[below_cutoff])
        return np.array([original_idxs[closest_local_idx]])

    elif len(original_idxs) == 1:
        return np.array([original_idxs[0]])
    else:
        return np.array([curr_idx]) if curr_idx is not None else np.array([])


def get_mask(positions, idx, paired, type_mask, exclude_chain=2, curr_idx=None):
    neighbor_mask = np.abs(np.arange(len(positions)//3) - idx) > exclude_chain
    checked_mask = (paired == -2)

    # If this is a handshake/confirmation, the partner MUST be visible
    if curr_idx is not None:
        checked_mask[curr_idx] = True

    return type_mask & checked_mask & neighbor_mask


def find_mutual_pair(positions, box, idx, paired, self_mask, complimentary_mask, base=0, cutoff=1.5, exclude_chain=2):

    partner_arr = find_single_partner(
        positions, box, idx, paired, complimentary_mask, base=base, cutoff=cutoff, exclude_chain=exclude_chain)

    if partner_arr.size > 0:
        partner = partner_arr[0]

        partner_pairing = find_single_partner(
            positions, box, partner, paired, self_mask, curr_idx=idx, base=base, cutoff=cutoff, exclude_chain=exclude_chain)

        if partner_pairing.size > 0:
            return (idx, partner)

    return None


def num_paired(traj_file, cutoff=1.5):
    """
    Returns the average number of base-pairs per frame in the trajectory.
    """
    traj = gsd.hoomd.open(traj_file, "r") if isinstance(
        traj_file, str) else traj_file

    total_pairs = 0.0
    n_frames = 0

    for snapshot in traj:
        ar = get_pair_array(snapshot, cutoff=cutoff)
        total_pairs += np.count_nonzero(ar >= 0) / 2.0
        n_frames += 1

    return total_pairs / n_frames if n_frames else 0.0


def read_param(params_path: str, key: str):
    """
    Reads a parameter from a JSON file

    params_path: path to params.json
    key: name of parameter to extract
    """
    with open(params_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get(key)


def paired_fraction_last_mean_from_pair_arrays(pair_arrays, fraction_last=0.2):
    """
    Compute the mean fraction of paired residues over the last fraction
    of cached frames.

    Args:
        pair_arrays: np.ndarray of shape (T, N)
        fraction_last: Fraction of trailing frames to include.

    Returns:
        Mean paired fraction across the selected trailing frames.
    """
    if len(pair_arrays) == 0:
        return float("nan")

    start = int((1.0 - fraction_last) * len(pair_arrays))
    tail = pair_arrays[start:]

    per_frame = np.mean(tail >= 0, axis=1)
    return float(np.mean(per_frame))


def get_mask_no_mutual(N, idx, type_mask, exclude_chain=2):
    neighbor_mask = np.abs(np.arange(N) - idx) > exclude_chain
    return type_mask & neighbor_mask


def find_closest_partner_no_mutual(positions, box, idx, target_mask, base=0, cutoff=1.5, exclude_chain=2):
    """
    Returns (partner_index, distance) for the closest eligible partner to idx,
    or (-1, np.nan) if none within cutoff.
    """
    N = len(positions) // 3  # number of nucleotides
    mask = get_mask_no_mutual(N, idx, target_mask, exclude_chain=exclude_chain)

    if not np.any(mask):
        return -1, np.nan

    candidates = positions[base + 1::3][mask]          # (M,3)
    ref = positions[base + 1 + 3*idx]                  # (3,)
    d = pbc_distances_freud(box, ref, candidates)      # (M,)

    within = np.nonzero(d < cutoff)[0]
    if within.size == 0:
        return -1, np.nan

    # pick closest within cutoff
    local_best = within[np.argmin(d[within])]
    partner = np.where(mask)[0][local_best]
    return int(partner), float(d[local_best])

# ============================================================================
# STRICT BASE PAIRING FOR SIMPLIFIED MODEL
# ============================================================================

def get_pair_array_simple(snapshot, cutoff=1.5, exclude_chain=2):
    """
    Simple pairing for 2-bead model.
    Assumes per nucleotide: (backbone, base), so bases are at 1::2.

    Returns:
        np.ndarray of shape (N_nt,), where entry i is the partner index of nt i,
        or -1 if unpaired.
    """
    complimentary = {
        "A": "U",
        "U": "A",
        "G": "C",
        "C": "G",
    }

    positions = snapshot.particles.position
    box = freud.box.Box.from_box(snapshot.configuration.box)

    # base identity from base bead types
    sequence = np.array([
        snapshot.particles.types[x][0]
        for x in snapshot.particles.typeid[1::2]
    ])

    N = len(sequence)
    paired = np.full(N, -1, dtype=int)

    for i in range(N):
        if paired[i] != -1:
            continue

        target_base = complimentary.get(sequence[i], None)
        if target_base is None:
            continue

        # eligible complementary partners
        mask = (sequence == target_base)
        mask &= (np.abs(np.arange(N) - i) > exclude_chain)
        mask &= (paired == -1)

        if not np.any(mask):
            continue

        ref = positions[1 + 2*i]
        candidates = positions[1::2][mask]
        d = pbc_distances_freud(box, ref, candidates)

        within = np.where(d < cutoff)[0]
        if len(within) == 0:
            continue

        # closest candidate for i
        local_j = within[np.argmin(d[within])]
        j = np.where(mask)[0][local_j]

        # mutual check: is i also the closest valid partner for j?
        back_target = complimentary.get(sequence[j], None)
        back_mask = (sequence == back_target)
        back_mask &= (np.abs(np.arange(N) - j) > exclude_chain)
        back_mask &= ((paired == -1) | (np.arange(N) == i))

        ref_j = positions[1 + 2*j]
        candidates_j = positions[1::2][back_mask]
        d_j = pbc_distances_freud(box, ref_j, candidates_j)

        within_j = np.where(d_j < cutoff)[0]
        if len(within_j) == 0:
            continue

        local_i = within_j[np.argmin(d_j[within_j])]
        i_back = np.where(back_mask)[0][local_i]

        if i_back == i:
            paired[i] = j
            paired[j] = i

    return paired


# ============================================================================
# WC EACH BASE PAIRING
# ============================================================================


def get_closest_partner_arrays(snapshot, cutoff=1.5, exclude_chain=2):
    """
    For each residue in a frame, find the closest complementary partner for
    brother/base site 0 and brother/base site 1 independently, without the
    mutual-handshake requirement.

    This function returns only partner indices, not distances.

    Args:
        snapshot: GSD snapshot for one frame.
        cutoff: Maximum distance threshold for accepting a partner.
        exclude_chain: Exclude residues within this sequence distance.

    Returns:
        partner0: np.ndarray of shape (N,), closest partner indices for base=0,
            or -1 where none exists within cutoff.
        partner1: np.ndarray of shape (N,), closest partner indices for base=1,
            or -1 where none exists within cutoff.
    """
    complimentary = {"A": "U", "U": "A", "G": "C", "C": "G"}

    positions = snapshot.particles.position
    box = freud.box.Box.from_box(snapshot.configuration.box)

    sequence = np.array([snapshot.particles.types[x][0]
                         for x in snapshot.particles.typeid[1::3]])
    N = len(sequence)

    partner0 = np.full(N, -1, dtype=int)
    partner1 = np.full(N, -1, dtype=int)

    masks = {b: (sequence == b) for b in ["A", "U", "G", "C"]}

    for base_type in ["A", "U", "G", "C"]:
        src_idxs = np.where(masks[base_type])[0]
        target_mask = masks[complimentary[base_type]]

        for idx in src_idxs:
            p, _ = find_closest_partner_no_mutual(
                positions, box, idx, target_mask, base=0,
                cutoff=cutoff, exclude_chain=exclude_chain
            )
            partner0[idx] = p

            p, _ = find_closest_partner_no_mutual(
                positions, box, idx, target_mask, base=1,
                cutoff=cutoff, exclude_chain=exclude_chain
            )
            partner1[idx] = p

    return partner0, partner1


def get_closest_partner_distances(snapshot, cutoff=1.5, exclude_chain=2):
    """
    For each residue in a frame, find the closest complementary partner for
    brother/base site 0 and brother/base site 1 independently, without the
    mutual-handshake requirement.

    This function returns both partner indices and distances.

    Args:
        snapshot: GSD snapshot for one frame.
        cutoff: Maximum distance threshold for accepting a partner.
        exclude_chain: Exclude residues within this sequence distance.

    Returns:
        (partner0, dist0), (partner1, dist1):
            partner0: np.ndarray of shape (N,), partner indices for base=0
            dist0: np.ndarray of shape (N,), corresponding distances
            partner1: np.ndarray of shape (N,), partner indices for base=1
            dist1: np.ndarray of shape (N,), corresponding distances

        Entries with no valid partner are -1 for partner arrays and np.nan
        for distance arrays.
    """
    complimentary = {"A": "U", "U": "A", "G": "C", "C": "G"}

    positions = snapshot.particles.position
    box = freud.box.Box.from_box(snapshot.configuration.box)

    sequence = np.array([snapshot.particles.types[x][0]
                         for x in snapshot.particles.typeid[1::3]])
    N = len(sequence)

    partner0 = np.full(N, -1, dtype=int)
    dist0 = np.full(N, np.nan, dtype=float)
    partner1 = np.full(N, -1, dtype=int)
    dist1 = np.full(N, np.nan, dtype=float)

    masks = {b: (sequence == b) for b in ["A", "U", "G", "C"]}

    for base_type in ["A", "U", "G", "C"]:
        src_idxs = np.where(masks[base_type])[0]
        target_mask = masks[complimentary[base_type]]

        for idx in src_idxs:
            p, d = find_closest_partner_no_mutual(
                positions, box, idx, target_mask, base=0,
                cutoff=cutoff, exclude_chain=exclude_chain
            )
            partner0[idx], dist0[idx] = p, d

            p, d = find_closest_partner_no_mutual(
                positions, box, idx, target_mask, base=1,
                cutoff=cutoff, exclude_chain=exclude_chain
            )
            partner1[idx], dist1[idx] = p, d

    return (partner0, dist0), (partner1, dist1)

########### PAIRING TIME SERIES HELPERS


def paired_fraction_over_time_from_pair_arrays(pair_arrays):
    """
    Compute per-frame paired fraction from cached pair arrays.

    Args:
        pair_arrays: np.ndarray of shape (T, N)

    Returns:
        np.ndarray of shape (T,) with fraction paired in each frame.
    """
    if len(pair_arrays) == 0:
        return np.array([], dtype=float)

    return np.mean(pair_arrays >= 0, axis=1)


def canonical_Q_over_time_from_pair_arrays(pair_arrays, expected):
    """
    Compute per-frame canonical pairing fraction Q from cached pair arrays.

    Args:
        pair_arrays: np.ndarray of shape (T, N)
        expected: Expected canonical pairing specification.

    Returns:
        np.ndarray of shape (T,) with Q for each frame.
    """
    if len(pair_arrays) == 0:
        return np.array([], dtype=float)

    exp_pairs = canonical_pairs_from_expected(expected)
    if not exp_pairs:
        return np.zeros(len(pair_arrays), dtype=float)

    denom = len(exp_pairs)
    out = np.zeros(len(pair_arrays), dtype=float)

    for t, row in enumerate(pair_arrays):
        observed = pairs_from_pair_array(row)
        hits = sum((p in observed) for p in exp_pairs)
        out[t] = hits / denom

    return out


def disagree_fraction_over_time_from_closest_arrays(closest_partner0, closest_partner1):
    """
    Compute per-frame disagreement fraction from cached closest-partner arrays.

    A residue contributes if both brother assignments are valid and differ.

    Args:
        closest_partner0: np.ndarray of shape (T, N)
        closest_partner1: np.ndarray of shape (T, N)

    Returns:
        np.ndarray of shape (T,) with disagreement fraction per frame.
    """
    if len(closest_partner0) == 0:
        return np.array([], dtype=float)

    disagree = (
        (closest_partner0 >= 0) &
        (closest_partner1 >= 0) &
        (closest_partner0 != closest_partner1)
    )
    return np.mean(disagree, axis=1)


def nonstrict_paired_fraction_over_time_from_closest_arrays(closest_partner0, closest_partner1):
    """
    Compute per-frame non-strict paired fraction from closest-partner arrays.

    A residue counts as paired if either brother/base site has any valid
    complementary partner assignment.

    Args:
        closest_partner0: np.ndarray of shape (T, N)
        closest_partner1: np.ndarray of shape (T, N)

    Returns:
        np.ndarray of shape (T,) with non-strict paired fraction per frame.
    """
    if len(closest_partner0) == 0:
        return np.array([], dtype=float)

    paired_any = (closest_partner0 >= 0) | (closest_partner1 >= 0)
    return np.mean(paired_any, axis=1)


def compute_pairing_frame_metrics(
    traj_file,
    cache_path,
    cutoff=1.5,
    exclude_chain=2,
    expected=None,
    start=None,
    stop=None,
    stride=1,
    force_recompute=False,
    compute_closest=True,
    pair_array_func=None,
):
    """
    Load or compute cached pairing arrays and return framewise pairing metrics.

    Returns:
        dict with keys:
            "frame_indices"
            "paired_strict"
            "paired_nonstrict"
            "disagree"
            "Q"
    """
    cached = load_or_compute_pairing_arrays(
        traj_file=traj_file,
        cache_path=cache_path,
        cutoff=cutoff,
        exclude_chain=exclude_chain,
        start=start,
        stop=stop,
        stride=stride,
        force_recompute=force_recompute,
        compute_closest=compute_closest,
        pair_array_func=pair_array_func,
    )

    pair_arrays = cached["pair_arrays"]
    frame_indices = cached["frame_indices"]

    out = {
        "frame_indices": frame_indices,
        "paired_strict": paired_fraction_over_time_from_pair_arrays(pair_arrays),
        "Q": (
            canonical_Q_over_time_from_pair_arrays(pair_arrays, expected)
            if expected is not None else np.full(len(pair_arrays), np.nan)
        ),
    }

    closest_partner0 = cached.get("closest_partner0")
    closest_partner1 = cached.get("closest_partner1")

    if closest_partner0 is not None and closest_partner1 is not None:
        out["paired_nonstrict"] = nonstrict_paired_fraction_over_time_from_closest_arrays(
            closest_partner0, closest_partner1
        )
        out["disagree"] = disagree_fraction_over_time_from_closest_arrays(
            closest_partner0, closest_partner1
        )
    else:
        out["paired_nonstrict"] = np.full(len(pair_arrays), np.nan)
        out["disagree"] = np.full(len(pair_arrays), np.nan)

    return out



########### PAIR STATES FOR 2x2 nucleotides (2 molecules)

def pair_states_from_closest_result(closest_result, intended_pairs, cutoff=0.75):
    """
    Convert one get_closest_partner_distances(...) result into per-pair states.

    Args:
        closest_result:
            output of get_closest_partner_distances(snapshot, ...)
            i.e. ((partner0, dist0), (partner1, dist1))
        intended_pairs:
            list of intended nt index pairs, e.g. [(0, 3), (1, 2)]
        cutoff:
            distance cutoff for considering a brother paired

    Returns:
        np.ndarray of shape (len(intended_pairs),), values in {0,1,2}
        0 = none paired
        1 = one brother paired
        2 = both brothers paired
    """
    (partner0, dist0), (partner1, dist1) = closest_result

    states = np.zeros(len(intended_pairs), dtype=int)

    for k, (i, j) in enumerate(intended_pairs):
        b0 = (partner0[i] == j) and np.isfinite(
            dist0[i]) and (dist0[i] < cutoff)
        b1 = (partner1[i] == j) and np.isfinite(
            dist1[i]) and (dist1[i] < cutoff)

        if b0 and b1:
            states[k] = 2
        elif b0 or b1:
            states[k] = 1
        else:
            states[k] = 0

    return states


state_labels = {
    (2, 2): 5,  # both proper
    (2, 1): 4,  # one proper, one one-base
    (1, 2): 4,
    (2, 0): 3,  # one proper, one none
    (0, 2): 3,
    (1, 1): 2,  # both one-base
    (1, 0): 1,  # one one-base, one none
    (0, 1): 1,
    (0, 0): 0,  # none
}


def frame_state_from_closest_result(closest_result, intended_pairs, state_labels, cutoff=0.75):
    """
    Convert one closest_result into a single frame state label.
    """
    pair_states = pair_states_from_closest_result(
        closest_result, intended_pairs=intended_pairs, cutoff=cutoff
    )
    return state_labels[tuple(pair_states)]


def frame_states_over_traj(traj, intended_pairs, state_labels, cutoff=0.75, exclude_chain=0):
    """
    Iterate over traj and return one state label per frame.
    USAGE: A-G + U-C (2 molecules setup) ONLY

    Returns:
        np.ndarray of shape (n_frames,), containing values like 0..5
    """
    out = np.zeros(len(traj), dtype=int)

    for f, snapshot in enumerate(traj):
        closest_result = get_closest_partner_distances(
            snapshot, cutoff=cutoff, exclude_chain=exclude_chain
        )
        out[f] = frame_state_from_closest_result(
            closest_result,
            intended_pairs=intended_pairs,
            state_labels=state_labels,
            cutoff=cutoff
        )

    return out

# 

# COMPUTING AND LOADING PAIRING ARRAYS


def compute_and_save_pairing_arrays(
    traj_file,
    out_path,
    cutoff=1.5,
    exclude_chain=2,
    start=None,
    stop=None,
    stride=1,
    compute_closest=True,
    pair_array_func=None,
):
    """
    Traverse a trajectory once, compute frame-wise pairing arrays, and save
    the results to a compressed .npz file for reuse.

        For each selected frame, this computes:
    - mutual pair assignments from get_pair_array
    - closest partner assignments for brother/base site 0
    - closest partner assignments for brother/base site 1

    Args:
        traj_file: Path to trajectory .gsd file or an already opened trajectory.
        out_path: Output path for the .npz cache file.
        cutoff: Pairing cutoff passed to pair-finding functions.
        exclude_chain: Sequence-distance exclusion for closest-partner search.
        start: Optional start frame index.
        stop: Optional stop frame index (exclusive).
        stride: Frame stride.
        compute_closest: True for 3-beads model, false for 2-beads (no disagreement calc)
        pair_array_func: None for 3-beads, get_pair_array_simple for 2-beads.

    Returns:
        A dict with keys:
            "pair_arrays"
            "closest_partner0"
            "closest_partner1"
            "frame_indices"
            "cache_path"
    """
    opened_here = False
    if isinstance(traj_file, str):
        traj = gsd.hoomd.open(traj_file, "r")
        opened_here = True
    else:
        traj = traj_file

    if pair_array_func is None:
        pair_array_func = get_pair_array

    try:
        n_total = len(traj)
        frame_indices = np.arange(n_total)[slice(start, stop, stride)]

        if frame_indices.size == 0:
            raise ValueError(
                "No frames selected for pairing cache computation.")

        first_pair = pair_array_func(
            traj[int(frame_indices[0])], cutoff=cutoff)
        N = len(first_pair)
        T = len(frame_indices)

        pair_arrays = np.full((T, N), -1, dtype=np.int32)
        pair_arrays[0] = first_pair

        if compute_closest:
            closest_partner0 = np.full((T, N), -1, dtype=np.int32)
            closest_partner1 = np.full((T, N), -1, dtype=np.int32)

            p0, p1 = get_closest_partner_arrays(
                traj[int(frame_indices[0])],
                cutoff=cutoff,
                exclude_chain=exclude_chain
            )
            closest_partner0[0] = p0
            closest_partner1[0] = p1

        for k, frame_idx in enumerate(frame_indices[1:], start=1):
            snap = traj[int(frame_idx)]
            pair_arrays[k] = pair_array_func(snap, cutoff=cutoff)

            if compute_closest:
                p0, p1 = get_closest_partner_arrays(
                    snap, cutoff=cutoff, exclude_chain=exclude_chain
                )
                closest_partner0[k] = p0
                closest_partner1[k] = p1

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "pair_arrays": pair_arrays,
            "frame_indices": frame_indices.astype(np.int32),
            "cutoff": np.array(cutoff, dtype=float),
            "exclude_chain": np.array(exclude_chain, dtype=np.int32),
            "start": np.array(-1 if start is None else start, dtype=np.int32),
            "stop": np.array(-1 if stop is None else stop, dtype=np.int32),
            "stride": np.array(stride, dtype=np.int32),
        }

        if compute_closest:
            save_dict["closest_partner0"] = closest_partner0
            save_dict["closest_partner1"] = closest_partner1

        np.savez_compressed(out_path, **save_dict)

        out = {
            "pair_arrays": pair_arrays,
            "frame_indices": frame_indices,
            "cache_path": str(out_path),
        }

        if compute_closest:
            out["closest_partner0"] = closest_partner0
            out["closest_partner1"] = closest_partner1

        return out

    finally:
        if opened_here:
            traj.close()

def load_pairing_arrays(cache_path):
    """
    Load cached frame-wise pairing arrays from a .npz file.

    Args:
        cache_path: Path to the .npz cache file.

    Returns:
        dict containing:
            "pair_arrays"
            "closest_partner0"
            "closest_partner1"
            "frame_indices"
            plus any saved metadata if needed later.
    """
    data = np.load(cache_path)

    out = {
        "pair_arrays": data["pair_arrays"],
        "frame_indices": data["frame_indices"],
    }

    if "closest_partner0" in data:
        out["closest_partner0"] = data["closest_partner0"]
    if "closest_partner1" in data:
        out["closest_partner1"] = data["closest_partner1"]

    return out


def load_or_compute_pairing_arrays(
    traj_file,
    cache_path,
    cutoff=1.5,
    exclude_chain=2,
    start=None,
    stop=None,
    stride=1,
    force_recompute=False,
    compute_closest=True,
    pair_array_func=None,
):
    """
    Load pairing arrays from cache if available, otherwise compute and save them.

    Args:
        traj_file: Path to trajectory .gsd file or an opened trajectory.
        cache_path: Path to .npz cache file.
        cutoff: Pairing cutoff.
        exclude_chain: Sequence-distance exclusion for closest-partner search.
        start: Optional start frame index.
        stop: Optional stop frame index (exclusive).
        stride: Frame stride.
        force_recompute: If True, recompute even if cache exists.

    Returns:
        dict with:
            "pair_arrays"
            "closest_partner0"
            "closest_partner1"
            "frame_indices"
            "cache_path"
    """


    cache_path = Path(cache_path)

    if cache_path.exists() and not force_recompute:
        out = load_pairing_arrays(cache_path)
        out["cache_path"] = str(cache_path)
        return out

    return compute_and_save_pairing_arrays(
        traj_file=traj_file,
        out_path=cache_path,
        cutoff=cutoff,
        exclude_chain=exclude_chain,
        start=start,
        stop=stop,
        stride=stride,
        compute_closest=compute_closest,
        pair_array_func=pair_array_func,
    )

# PAIRING STATS


def mean_stuckness(pair_arrays, min_union=3):
    """
    Compute mean frame-to-frame Jaccard similarity of pair identity, used as
    a "stuckness" metric.

    For each consecutive frame pair, the per-frame score is:
        |intersection of pair sets| / |union of pair sets|

    Frames with too few total unique pairs in the union can be ignored to
    reduce noise.

    Args:
        pair_arrays: np.ndarray of shape (T, N), where each row is a pair_array.
        min_union: Minimum union size required to include a frame-to-frame
            comparison.

    Returns:
        Mean stuckness across consecutive frame comparisons, or np.nan if
        no valid comparisons exist.
    """
    if len(pair_arrays) < 2:
        return float("nan")

    prev = pairs_from_pair_array(pair_arrays[0])
    vals = []

    for t in range(1, len(pair_arrays)):
        curr = pairs_from_pair_array(pair_arrays[t])
        union_size = len(prev | curr)
        if union_size >= min_union:
            vals.append(len(prev & curr) / union_size)
        prev = curr

    return float(np.mean(vals)) if vals else float("nan")


def switching_rate_from_pair_arrays(pair_arrays, ignore_unpaired=False):
    """
    Compute the average per-residue switching rate from frame to frame using
    cached partner arrays.

    A switch means the partner index changes between consecutive frames.

    Args:
        pair_arrays: np.ndarray of shape (T, N), where each row is a pair_array.
        ignore_unpaired:
            - False: paired<->unpaired transitions count as switches.
            - True: only paired->paired partner changes are counted, and
              transitions involving unpaired states are ignored.

    Returns:
        Mean switching rate across residues, or np.nan if fewer than two frames
        are available.
    """
    if len(pair_arrays) < 2:
        return float("nan")

    prev = pair_arrays[:-1]
    curr = pair_arrays[1:]

    if not ignore_unpaired:
        switches = (curr != prev)
        per_base = np.mean(switches, axis=0)
        return float(np.mean(per_base))

    valid = (curr >= 0) & (prev >= 0)
    switch_mask = (curr != prev) & valid

    switch_counts = np.sum(switch_mask, axis=0)
    denom = np.sum(valid, axis=0)

    per_base = np.divide(
        switch_counts,
        denom,
        out=np.zeros_like(switch_counts, dtype=float),
        where=denom > 0
    )
    return float(np.mean(per_base))


def canonical_pairs_from_expected(expected):
    """
    Convert an expected partner mapping into a set of unique unordered pairs.

    Args:
        expected: Iterable or np.ndarray of length N where expected[i] is the
            expected partner index for residue i, or -1 if unpaired.

    Returns:
        Set of tuples (i, j) with i < j.
    """
    return pairs_from_pair_array(np.asarray(expected))


def mean_canonical_Q_from_pair_arrays(pair_arrays, expected):
    """
    Compute the mean canonical pairing fraction Q from cached pair arrays.

    For each frame, Q is the fraction of expected canonical pairs that are
    present in that frame.

    Args:
        pair_arrays: np.ndarray of shape (T, N), where each row is a pair_array.
        expected: Expected canonical pairing specification, interpreted by
            canonical_pairs_from_expected().

    Returns:
        Mean Q across frames. Returns 0.0 if no expected canonical pairs exist.
        Returns np.nan if pair_arrays is empty.
    """
    if len(pair_arrays) == 0:
        return float("nan")

    exp_pairs = canonical_pairs_from_expected(expected)
    if not exp_pairs:
        return 0.0

    vals = []
    denom = len(exp_pairs)

    for row in pair_arrays:
        observed = pairs_from_pair_array(row)
        hits = sum((p in observed) for p in exp_pairs)
        vals.append(hits / denom)

    return float(np.mean(vals))


def mean_disagree_fraction_from_closest_arrays(closest_partner0, closest_partner1):
    """
    Compute the mean fraction of residues whose two brother/base-site closest
    partner assignments disagree, using cached closest-partner arrays.

    A residue contributes to disagreement in a frame only if both brother/base
    assignments are valid and the assigned partners differ.

    Args:
        closest_partner0: np.ndarray of shape (T, N)
        closest_partner1: np.ndarray of shape (T, N)

    Returns:
        Mean disagreement fraction across frames, or np.nan if no frames exist.
    """
    if len(closest_partner0) == 0:
        return float("nan")

    disagree = (
        (closest_partner0 >= 0) &
        (closest_partner1 >= 0) &
        (closest_partner0 != closest_partner1)
    )

    per_frame = np.mean(disagree, axis=1)
    return float(np.mean(per_frame))


def compute_pairing_summary_metrics(
    traj_file,
    cache_path,
    cutoff=1.5,
    exclude_chain=2,
    expected=None,
    start=None,
    stop=None,
    stride=1,
    min_union=3,
    force_recompute=False,
    compute_closest=True,
    pair_array_func=None,
):
    """
    Load or compute cached pairing arrays for a trajectory, then compute all
    scalar pairing-derived summary metrics from those arrays.

    Metrics returned:
    - mean_stuckness
    - switch_rate_all
    - switch_rate_paired
    - mean_disagree_frac
    - mean_Q (only if expected is provided, else np.nan)

    Args:
        traj_file: Path to trajectory .gsd file or an opened trajectory.
        cache_path: Path to the pairing-cache .npz file.
        cutoff: Pairing cutoff.
        exclude_chain: Sequence-distance exclusion for closest-partner search.
        expected: Expected canonical pairing specification for Q. If None,
            mean_Q is returned as np.nan.
        start: Optional start frame index.
        stop: Optional stop frame index (exclusive).
        stride: Frame stride.
        min_union: Minimum union size for stuckness frame comparisons.
        force_recompute: If True, recompute cache even if it already exists.

    Returns:
        dict of scalar metrics and the cache path used.
    """
    cached = load_or_compute_pairing_arrays(
        traj_file=traj_file,
        cache_path=cache_path,
        cutoff=cutoff,
        exclude_chain=exclude_chain,
        start=start,
        stop=stop,
        stride=stride,
        force_recompute=force_recompute,
        compute_closest=compute_closest,
        pair_array_func=pair_array_func,
    )

    pair_arrays = cached["pair_arrays"]

    closest_partner0 = cached.get("closest_partner0")
    closest_partner1 = cached.get("closest_partner1")

    results = {
        "paired_frac_last_0_2_mean": paired_fraction_last_mean_from_pair_arrays(
            pair_arrays, fraction_last=0.2
        ),
        "mean_stuckness": mean_stuckness(pair_arrays, min_union=min_union),
        "switch_rate_all": switching_rate_from_pair_arrays(
            pair_arrays, ignore_unpaired=False
        ),
        "switch_rate_paired": switching_rate_from_pair_arrays(
            pair_arrays, ignore_unpaired=True
        ),
        "mean_disagree_frac": (
            mean_disagree_fraction_from_closest_arrays(
                closest_partner0, closest_partner1
            )
            if closest_partner0 is not None and closest_partner1 is not None
            else np.nan
        ),
        "mean_Q": (
            mean_canonical_Q_from_pair_arrays(pair_arrays, expected)
            if expected is not None else np.nan
        ),
    }

    return results


# FOR PRINTING HBOND AND HBOND CROSS DISTANCES


def pbc_dist(box, a, b):
    dr = b - a
    dr = box.wrap(dr)          # minimum-image displacement
    return float(np.linalg.norm(dr))


def nt_identity(snapshot, nt_idx):
    """Returns (base_letter, bead_type_names_of_P_B0_B1) for nucleotide index."""
    types = snapshot.particles.types
    typeid = snapshot.particles.typeid

    # Base letter from your convention: types[typeid[1::3]][0]
    base_letter = types[typeid[1 + 3*nt_idx]][0]

    P_type = types[typeid[3*nt_idx + 0]]
    B0_type = types[typeid[3*nt_idx + 1]]
    B1_type = types[typeid[3*nt_idx + 2]]
    return base_letter, (P_type, B0_type, B1_type)


def pair_brother_distances(snapshot, pairs):
    """Print B0/B1 and P distances for each base pair (i, j).

    Args:
        snapshot: gsd.hoomd.Frame
        pairs: iterable of (i, j) nucleotide indices (e.g. from pairs_from_pair_array)
    """
    pos = snapshot.particles.position
    box = freud.box.Box.from_box(snapshot.configuration.box)

    for (i, j) in pairs:
        # bead indices (P, B0, B1)
        iP, iB0, iB1 = 3*i + 0, 3*i + 1, 3*i + 2
        jP, jB0, jB1 = 3*j + 0, 3*j + 1, 3*j + 2

        # identities
        i_base, i_types = nt_identity(snapshot, i)
        j_base, j_types = nt_identity(snapshot, j)

        # base-base distances (PBC)
        d00 = pbc_dist(box, pos[iB0], pos[jB0])  # B0-B0 (intended)
        d01 = pbc_dist(box, pos[iB0], pos[jB1])  # B0-B1 (wrong brother)
        d10 = pbc_dist(box, pos[iB1], pos[jB0])  # B1-B0 (wrong brother)
        d11 = pbc_dist(box, pos[iB1], pos[jB1])  # B1-B1
        dPP = pbc_dist(box, pos[iP],  pos[jP])   # P-P

        # phosphate-base cross distances (PBC)
        # i phosphate to j bases
        dP_i_B0j = pbc_dist(box, pos[iP], pos[jB0])   # iP - jB0
        dP_i_B1j = pbc_dist(box, pos[iP], pos[jB1])   # iP - jB1
        # j phosphate to i bases
        dP_j_B0i = pbc_dist(box, pos[jP], pos[iB0])   # jP - iB0
        dP_j_B1i = pbc_dist(box, pos[jP], pos[iB1])   # jP - iB1

        print(f"\nPair nt {i:2d}–{j:2d}   ({i_base}–{j_base})")
        print(f"  types i(P,B0,B1)={i_types}  j(P,B0,B1)={j_types}")
        print(
            f"  B0–B0: {d00:.4f}   B0–B1: {d01:.4f}   B1–B0: {d10:.4f}   B1–B1: {d11:.4f}")
        print(f"  P–P  : {dPP:.4f}")
        print(
            f"  P–B  : iP–jB0 {dP_i_B0j:.4f}   iP–jB1 {dP_i_B1j:.4f}   "
            f"jP–iB0 {dP_j_B0i:.4f}   jP–iB1 {dP_j_B1i:.4f}"
        )

# PLOTTING (needs some work still)

import numpy as np
import gsd.hoomd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_pairing_dynamics_4panel(traj_file, cutoff=1.5, exclude_chain=2, stride=1):
    """
    Plot a 4-panel interactive summary of pairing dynamics from a trajectory,
    recomputing frame-wise pairing information without saving anything.

    Panels:
    1) Paired fraction over time
    2) Disagreement fraction over time
    3) Switch count over time
    4) Stuckness over time (frame-to-frame Jaccard similarity)

    Args:
        traj_file: Path to .gsd file or an open gsd.hoomd trajectory object.
        cutoff: Pairing cutoff passed to pairing functions.
        exclude_chain: Sequence-distance exclusion for closest-partner search.
        stride: Use every `stride`-th frame.

    Returns:
        fig: plotly.graph_objects.Figure
        data: dict with computed arrays
    """
    opened_here = False
    if isinstance(traj_file, (str, Path)):
        traj = gsd.hoomd.open(traj_file, "r")
        opened_here = True
        title_base = traj_file
    else:
        traj = traj_file
        title_base = "trajectory"

    try:
        n_total = len(traj)
        frame_indices = np.arange(0, n_total, stride, dtype=int)

        if frame_indices.size == 0:
            raise ValueError("No frames selected.")

        # First frame determines N
        first_pair = get_pair_array(traj[int(frame_indices[0])], cutoff=cutoff)
        N = len(first_pair)
        T = len(frame_indices)

        pair_arrays = np.full((T, N), -1, dtype=np.int32)
        closest_partner0 = np.full((T, N), -1, dtype=np.int32)
        closest_partner1 = np.full((T, N), -1, dtype=np.int32)

        # Compute per-frame arrays
        for k, frame_idx in enumerate(frame_indices):
            snap = traj[int(frame_idx)]
            pair_arrays[k] = get_pair_array(snap, cutoff=cutoff)
            p0, p1 = get_closest_partner_arrays(
                snap, cutoff=cutoff, exclude_chain=exclude_chain
            )
            closest_partner0[k] = p0
            closest_partner1[k] = p1

        # 1) Paired fraction over time
        paired_fraction = np.mean(pair_arrays >= 0, axis=1)

        # 2) Disagreement fraction over time
        disagree_mask = (
            (closest_partner0 >= 0) &
            (closest_partner1 >= 0) &
            (closest_partner0 != closest_partner1)
        )
        disagreement_fraction = np.mean(disagree_mask, axis=1)

        # 3) Switch count over time (between consecutive sampled frames)
        if T >= 2:
            switch_count = np.sum(pair_arrays[1:] != pair_arrays[:-1], axis=1)
            switch_x = frame_indices[1:]
        else:
            switch_count = np.array([], dtype=int)
            switch_x = np.array([], dtype=int)

        # 4) Stuckness over time (frame-to-frame Jaccard)
        if T >= 2:
            stuckness = np.full(T - 1, np.nan, dtype=float)
            for i in range(T - 1):
                prev_pairs = pairs_from_pair_array(pair_arrays[i])
                curr_pairs = pairs_from_pair_array(pair_arrays[i + 1])
                union = prev_pairs | curr_pairs
                stuckness[i] = (
                    len(prev_pairs & curr_pairs) / len(union)
                    if len(union) > 0 else np.nan
                )
            stuckness_x = frame_indices[1:]
        else:
            stuckness = np.array([], dtype=float)
            stuckness_x = np.array([], dtype=int)

        # Build figure
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Paired Fraction Over Time",
                "Disagreement Fraction Over Time",
                "Switch Count Over Time",
                "Stuckness Over Time",
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.10,
        )

        fig.add_trace(
            go.Scatter(
                x=frame_indices,
                y=paired_fraction,
                mode="lines",
                name="Paired fraction",
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=frame_indices,
                y=disagreement_fraction,
                mode="lines",
                name="Disagreement fraction",
            ),
            row=1, col=2
        )

        fig.add_trace(
            go.Scatter(
                x=switch_x,
                y=switch_count,
                mode="lines",
                name="Switch count",
            ),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=stuckness_x,
                y=stuckness,
                mode="lines",
                name="Stuckness",
            ),
            row=2, col=2
        )

        # Axis labels
        fig.update_xaxes(title_text="Frame index", row=1, col=1)
        fig.update_yaxes(title_text="Fraction paired",
                         range=[0, 1], row=1, col=1)

        fig.update_xaxes(title_text="Frame index", row=1, col=2)
        fig.update_yaxes(title_text="Disagreement fraction",
                         range=[0, 1], row=1, col=2)

        fig.update_xaxes(title_text="Frame index", row=2, col=1)
        fig.update_yaxes(
            title_text="Number of residues switched", row=2, col=1)

        fig.update_xaxes(title_text="Frame index", row=2, col=2)
        fig.update_yaxes(title_text="Jaccard similarity",
                         range=[0, 1], row=2, col=2)

        fig.update_layout(
            title=f"Pairing Dynamics Summary ({title_base}, stride={stride})",
            height=800,
            width=1100,
            template="plotly_white",
            showlegend=False,
        )

        #fig.show()

        return fig

    finally:
        if opened_here:
            traj.close()

    """, {
            "frame_indices": frame_indices,
            "pair_arrays": pair_arrays,
            "closest_partner0": closest_partner0,
            "closest_partner1": closest_partner1,
            "paired_fraction": paired_fraction,
            "disagreement_fraction": disagreement_fraction,
            "switch_count": switch_count,
            "switch_x": switch_x,
            "stuckness": stuckness,
            "stuckness_x": stuckness_x,
        }
    """
