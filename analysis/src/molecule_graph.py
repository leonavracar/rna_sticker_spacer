import numpy as np
from collections import Counter

# Rename: molecule graph NPZ → cdf cluster-size distribution

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
