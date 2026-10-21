"""Create weighted molecule-contact edges from filtered particle pairs."""

from collections import Counter
import numpy as np

def count_molecule_contacts_from_array(qi: np.ndarray,
    pi: np.ndarray, molecule_id: np.ndarray) -> Counter[tuple[int, int]]:
    """For each molecule pair in a frame, count the number of contacts between them."""
    pair_counts = Counter()
    for i, j in zip(qi, pi):
        mol_i = molecule_id[i]
        mol_j = molecule_id[j]
        if mol_i == mol_j:
            continue
        mol_a = min(mol_i, mol_j)
        mol_b = max(mol_i, mol_j)

        pair_counts[(mol_a, mol_b)] += 1
    return pair_counts



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
