from __future__ import annotations
from collections.abc import Iterable
import freud
from pathlib import Path

import numpy as np
import gsd.hoomd
from numpy.typing import NDArray
from analysis_scripts.helpers import as_traj, load_npz, save_npz
import freud.box

# ============================================================================
# HELPERS
# ============================================================================


def pbc_distances_freud(
    box: freud.box.Box,
    ref_pos: NDArray[np.float64],
    other_pos: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute minimum-image distances from one reference position to many other positions.
    Returns an array of PBC-aware distances.
    """
    dr = other_pos - ref_pos
    dr = box.wrap(dr)
    return np.linalg.norm(dr, axis=1)


# ============================================================================
# STRICT PAIRING ANALYSIS: Both brothers must be paired ###
# ============================================================================


def get_mask(
    positions: NDArray[np.float64],
    idx: int,
    paired: NDArray[np.int_],
    type_mask: NDArray[np.bool_],
    exclude_chain: int = 2,
    curr_idx: int | None = None,
) -> NDArray[np.bool_]:
    """
    Return the mask of eligible partner indices for one nucleotide.
    """
    n_nt = len(positions) // 3
    neighbor_mask = np.abs(np.arange(n_nt) - idx) > exclude_chain
    checked_mask = paired == -2

    if curr_idx is not None:
        checked_mask[curr_idx] = True

    return type_mask & checked_mask & neighbor_mask


def find_single_partner(
    positions: NDArray[np.float64],
    box: freud.box.Box,
    idx: int,
    paired: NDArray[np.int_],
    type_mask: NDArray[np.bool_],
    base: int = 0,
    cutoff: float = 1.5,
    curr_idx: int | None = None,
    exclude_chain: int = 2,
) -> NDArray[np.int_]:
    """
    Find one eligible partner index for one base site. Returns an empty array if no partner is found.
    """
    if paired[idx] != -2:
        return np.array([], dtype=int)

    paired[idx] = -1
    mask = get_mask(
        positions,
        idx,
        paired,
        type_mask,
        exclude_chain=exclude_chain,
        curr_idx=curr_idx,
    )

    if not np.any(mask):
        return np.array([], dtype=int)

    filtered_positions = positions[base + 1 :: 3][mask]
    ref = positions[base + 1 + 3 * idx]
    distances = pbc_distances_freud(box, ref, filtered_positions)

    below_cutoff = np.nonzero(distances < cutoff)[0]
    original_idxs = np.where(mask)[0][below_cutoff]

    if len(original_idxs) > 1:
        closest_local_idx = np.argmin(distances[below_cutoff])
        return np.array([original_idxs[closest_local_idx]], dtype=int)

    if len(original_idxs) == 1:
        return np.array([original_idxs[0]], dtype=int)

    if curr_idx is not None:
        return np.array([curr_idx], dtype=int)

    return np.array([], dtype=int)


def find_mutual_pair(
    positions: NDArray[np.float64],
    box: freud.box.Box,
    idx: int,
    paired: NDArray[np.int_],
    self_mask: NDArray[np.bool_],
    complementary_mask: NDArray[np.bool_],
    base: int = 0,
    cutoff: float = 1.5,
    exclude_chain: int = 2,
) -> tuple[int, int] | None:
    """
    Find a mutual partner pair for one nucleotide and one base site. Returns a pair of nucleotide
    indices or None if no mutual pair is found.
    """
    partner_arr = find_single_partner(
        positions,
        box,
        idx,
        paired,
        complementary_mask,
        base=base,
        cutoff=cutoff,
        exclude_chain=exclude_chain,
    )

    if partner_arr.size == 0:
        return None

    partner = int(partner_arr[0])

    partner_pairing = find_single_partner(
        positions,
        box,
        partner,
        paired,
        self_mask,
        curr_idx=idx,
        base=base,
        cutoff=cutoff,
        exclude_chain=exclude_chain,
    )

    if partner_pairing.size > 0:
        return idx, partner

    return None


def get_pair_array(
    frame: gsd.hoomd.Frame,
    cutoff: float = 1.5,
    exclude_chain: int = 2,
) -> NDArray[np.int_]:
    """
    Compute the strict mutual pairing array for one frame. Returns an array where each entry is the
    partner index of one nucleotide, or -1 if unpaired.
    Note: only works for 3-bead model
    """
    complementary = {
        "A": "U",
        "U": "A",
        "G": "C",
        "C": "G",
    }

    positions = frame.particles.position
    box = freud.box.Box.from_box(frame.configuration.box)

    sequence = np.array(
        [frame.particles.types[x][0] for x in frame.particles.typeid[1::3]]
    )

    paired = np.full(len(sequence), -2, dtype=int)

    for base_type in ["A", "G"]:
        self_mask = sequence == base_type
        complementary_mask = sequence == complementary[base_type]
        check_indices = np.where(self_mask)[0]

        for idx in check_indices:
            pair = find_mutual_pair(
                positions,
                box,
                idx,
                paired,
                self_mask,
                complementary_mask,
                base=0,
                cutoff=cutoff,
                exclude_chain=exclude_chain,
            )

            if pair is None:
                continue

            paired[pair[0]], paired[pair[1]] = -2, -2

            brothers = find_mutual_pair(
                positions,
                box,
                idx,
                paired,
                self_mask,
                complementary_mask,
                base=1,
                cutoff=cutoff,
                exclude_chain=exclude_chain,
            )

            if brothers == (idx, pair[1]):
                paired[pair[0]], paired[pair[1]] = pair[1], pair[0]

    paired[paired == -2] = -1
    return paired


# ============================================================================
# NONSTRICT PAIRING ANALYSIS: Non-mutual pairs calculated
# ============================================================================


def get_mask_no_mutual(
    n_nt: int,
    idx: int,
    type_mask: NDArray[np.bool_],
    exclude_chain: int = 2,
) -> NDArray[np.bool_]:
    """
    Return the mask of eligible partner indices for one nucleotide without a mutual check.
    """
    neighbor_mask = np.abs(np.arange(n_nt) - idx) > exclude_chain
    return type_mask & neighbor_mask


def find_closest_partner_no_mutual(
    positions: NDArray[np.float64],
    box: freud.box.Box,
    idx: int,
    target_mask: NDArray[np.bool_],
    base: int = 0,
    cutoff: float = 1.5,
    exclude_chain: int = 2,
) -> int:
    """
    Find the closest eligible partner index for one base site without a mutual check.
    Returns -1 if no partner is found within the cutoff.
    """
    n_nt = len(positions) // 3
    mask = get_mask_no_mutual(
        n_nt,
        idx,
        target_mask,
        exclude_chain=exclude_chain,
    )

    if not np.any(mask):
        return -1

    candidates = positions[base + 1 :: 3][mask]
    ref = positions[base + 1 + 3 * idx]
    distances = pbc_distances_freud(box, ref, candidates)

    within = np.nonzero(distances < cutoff)[0]
    if within.size == 0:
        return -1

    local_best = within[np.argmin(distances[within])]
    partner = np.where(mask)[0][local_best]
    return int(partner)


def get_closest_partner_arrays(
    frame: gsd.hoomd.Frame,
    cutoff: float = 1.5,
    exclude_chain: int = 2,
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """
    Compute closest partner arrays for both base sites in one frame without a mutual check.
    Returns one partner array for base site 0 and one for base site 1.
    """
    complementary = {
        "A": "U",
        "U": "A",
        "G": "C",
        "C": "G",
    }

    positions = frame.particles.position
    box = freud.box.Box.from_box(frame.configuration.box)

    sequence = np.array(
        [frame.particles.types[x][0] for x in frame.particles.typeid[1::3]]
    )
    n_nt = len(sequence)

    partner0 = np.full(n_nt, -1, dtype=int)
    partner1 = np.full(n_nt, -1, dtype=int)

    masks = {base: sequence == base for base in ["A", "U", "G", "C"]}

    for base_type in ["A", "U", "G", "C"]:
        src_indices = np.where(masks[base_type])[0]
        target_mask = masks[complementary[base_type]]

        for idx in src_indices:
            partner0[idx] = find_closest_partner_no_mutual(
                positions,
                box,
                idx,
                target_mask,
                base=0,
                cutoff=cutoff,
                exclude_chain=exclude_chain,
            )
            partner1[idx] = find_closest_partner_no_mutual(
                positions,
                box,
                idx,
                target_mask,
                base=1,
                cutoff=cutoff,
                exclude_chain=exclude_chain,
            )

    return partner0, partner1


# ============================================================================
# MAIN PAIRING ANALYSIS
# ============================================================================


def compute_pairing_arrays(
    traj: gsd.hoomd.HOOMDTrajectory,
    cutoff: float = 1.5,
    exclude_chain: int = 2,
    stride: int = 1,
) -> dict[str, NDArray]:
    """
    Compute framewise pairing arrays for a trajectory in a single pass.
    Returns a dict suitable for saving with np.savez_compressed.
    """
    frame_indices = np.arange(0, len(traj), stride, dtype=np.int32)
    if frame_indices.size == 0:
        raise ValueError("No frames selected for pairing computation.")

    first_frame = traj[int(frame_indices[0])]
    first_pair = get_pair_array(
        first_frame,
        cutoff=cutoff,
        exclude_chain=exclude_chain,
    )
    first_closest0, first_closest1 = get_closest_partner_arrays(
        first_frame,
        cutoff=cutoff,
        exclude_chain=exclude_chain,
    )

    n_frames = len(frame_indices)
    n_nt = len(first_pair)

    pair_arrays = np.full((n_frames, n_nt), -1, dtype=np.int32)
    closest_partner0 = np.full((n_frames, n_nt), -1, dtype=np.int32)
    closest_partner1 = np.full((n_frames, n_nt), -1, dtype=np.int32)

    pair_arrays[0] = first_pair
    closest_partner0[0] = first_closest0
    closest_partner1[0] = first_closest1

    for f_idx, frame_idx in enumerate(frame_indices[1:], start=1):
        frame = traj[int(frame_idx)]

        pair_arrays[f_idx] = get_pair_array(
            frame,
            cutoff=cutoff,
            exclude_chain=exclude_chain,
        )
        p0, p1 = get_closest_partner_arrays(
            frame,
            cutoff=cutoff,
            exclude_chain=exclude_chain,
        )
        closest_partner0[f_idx] = p0
        closest_partner1[f_idx] = p1

    return {
        "frame": frame_indices,
        "pair_arrays": pair_arrays,
        "closest_partner0": closest_partner0,
        "closest_partner1": closest_partner1,
        "cutoff": np.array(cutoff, dtype=np.float64),
        "exclude_chain": np.array(exclude_chain, dtype=np.int32),
        "stride": np.array(stride, dtype=np.int32),
    }


def compute_and_save_pairing_arrays(
    traj_file: str | Path | gsd.hoomd.HOOMDTrajectory,
    out_path: Path,
    cutoff: float = 1.5,
    exclude_chain: int = 2,
    stride: int = 1,
    skip_if_exists: bool = True,
) -> dict[str, NDArray | str]:
    """
    Compute pairing arrays and save them to disk. If skip_if_exists is True and the file
    already exists, the cached file is loaded instead.
    """
    if skip_if_exists and out_path.exists():
        out = load_npz(out_path)
        out["cache_path"] = str(out_path)
        return out

    with as_traj(traj_file) as traj:
        out = compute_pairing_arrays(
            traj,
            cutoff=cutoff,
            exclude_chain=exclude_chain,
            stride=stride,
        )

    save_npz(out_path, out)
    out["cache_path"] = str(out_path)
    return out


###


def pairs_from_pair_array(
    pair_array: NDArray[np.int_],
) -> set[tuple[int, int]]:
    """
    Convert a partner array into a set of unique unordered pairs.
    """
    pairs: set[tuple[int, int]] = set()

    for i, j in enumerate(pair_array):
        if j >= 0:
            j = int(j)
            if i < j:
                pairs.add((i, j))

    return pairs


def canonical_pairs_from_expected(
    expected: Iterable[int] | NDArray[np.int_],
) -> set[tuple[int, int]]:
    """
    Convert an expected partner mapping into a set of unique unordered pairs.
    """
    return pairs_from_pair_array(np.asarray(expected, dtype=int))


def paired_strict_from_pair_arrays(
    pair_arrays: NDArray[np.int_],
) -> NDArray[np.float64]:
    """
    Compute the framewise strict paired fraction from pair arrays.
    """
    return np.mean(pair_arrays >= 0, axis=1)


def paired_nonstrict_from_closest_arrays(
    closest_partner0: NDArray[np.int_],
    closest_partner1: NDArray[np.int_],
) -> NDArray[np.float64]:
    """
    Compute the framewise non-strict paired fraction from closest-partner arrays.
    """
    paired_any = (closest_partner0 >= 0) | (closest_partner1 >= 0)
    return np.mean(paired_any, axis=1)


def disagree_from_closest_arrays(
    closest_partner0: NDArray[np.int_],
    closest_partner1: NDArray[np.int_],
) -> NDArray[np.float64]:
    """
    Compute the framewise disagreement fraction from closest-partner arrays.
    """
    disagree = (
        (closest_partner0 >= 0)
        & (closest_partner1 >= 0)
        & (closest_partner0 != closest_partner1)
    )
    return np.mean(disagree, axis=1)


def Q_from_pair_arrays(
    pair_arrays: NDArray[np.int_],
    expected: NDArray[np.int_],
) -> NDArray[np.float64]:
    """
    Compute the framewise canonical pairing fraction Q from pair arrays.
    """
    exp_pairs = canonical_pairs_from_expected(expected)
    if not exp_pairs:
        return np.zeros(len(pair_arrays), dtype=np.float64)

    denom = len(exp_pairs)
    out = np.empty(len(pair_arrays), dtype=np.float64)

    for t, row in enumerate(pair_arrays):
        observed = pairs_from_pair_array(row)
        out[t] = len(observed & exp_pairs) / denom

    return out
