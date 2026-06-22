import numpy as np
from collections import Counter, deque
from collections.abc import Iterable

from validation_analysis.molecule_graph import (
    count_molecule_contacts_from_array,
    filt_molecule_contacts_in_frame,
    molecule_contact_set_from_pair_counts,
)

from validation_analysis.cluster_histogram import get_cluster_sizes_from_molecule_pairs


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


def framewise_molecule_recurrence_rows_from_nlist_dict(
    nlist_dict: dict[str, np.ndarray],
    molecule_id: np.ndarray,
    recurrence_window: int = 25,
    min_intermol_bonds: int = 1,
) -> list[dict[str, float | int]]:
    """
    Build framewise molecule-contact recurrence rows.
    """
    rows = []
    recent_contact_sets: deque[set[tuple[int, int]]] = deque(
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

        recent_contact_sets.append(molecule_contacts)

        rows.append(
            {
                "frame": frame_idx,
                "molecule_contact_recurrence_window": recurrence_from_contact_sets(
                    recent_contact_sets
                ),
            }
        )

    return rows

# Legacy stuff

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


