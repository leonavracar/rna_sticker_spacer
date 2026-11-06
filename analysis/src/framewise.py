"""Framewise contact, molecular graph and cluster calculations."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .cluster_distribution import (
    connected_components, cluster_bin_fractions, ranked_cluster_rows,
)
from .recurrence import RollingRecurrence, pair_set


def molecule_lengths(molecule_lookup: np.ndarray) -> np.ndarray:
    ids = np.asarray(molecule_lookup, dtype=int)
    if not len(ids) or ids.min() < 0:
        raise ValueError("Invalid molecule lookup")
    lengths = np.bincount(ids)
    if np.any(lengths == 0):
        raise ValueError("Molecule lookup must contain every molecule ID")
    return lengths


def edge_frames(edge_npz: Mapping[str, np.ndarray] | np.lib.npyio.NpzFile) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Index precomputed weighted molecule edges by saved frame."""
    frame = np.asarray(edge_npz["frame"], dtype=int)
    a = np.asarray(edge_npz["mol_a"], dtype=int)
    b = np.asarray(edge_npz["mol_b"], dtype=int)
    weights = np.asarray(edge_npz["n_contacts"], dtype=int)
    order = np.argsort(frame, kind="stable")
    frame, a, b, weights = frame[order], a[order], b[order], weights[order]
    out = {}
    for f in np.asarray(edge_npz["frame_indices"], dtype=int):
        l = np.searchsorted(frame, f, side="left")
        r = np.searchsorted(frame, f, side="right")
        out[int(f)] = a[l:r], b[l:r], weights[l:r]
    return out


def component_frame_rows(
    frames: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    lengths: np.ndarray,
    min_contacts: int = 2,
    recurrence_window: int = 20,
    with_recurrence: bool = True,
    with_bins: bool = True,
    with_ranks: bool = False,
    species: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return graph, bin, ranked-cluster and component data for each frame."""
    graph_rows, bins_rows, ranks_rows = [], [], []
    n_molecules = len(lengths)
    recurrence = RollingRecurrence(recurrence_window) if with_recurrence else None
    for frame, (a, b, weights) in frames.items():
        mask = weights >= min_contacts
        a_f, b_f, w_f = a[mask], b[mask], weights[mask]
        comps = connected_components(n_molecules, a_f, b_f, w_f, min_contacts=1)
        edge_count = len(w_f)
        row = {
            "frame": int(frame),
            "n_edges": edge_count,
            "mean_edge_weight": float(np.mean(w_f)) if edge_count else np.nan,
            "mean_degree": 2 * edge_count / n_molecules,
        }
        if recurrence is not None:
            row["molecule_recurrence"] = recurrence.add(pair_set(a_f, b_f))
        graph_rows.append(row)
        if with_bins:
            for scheme, values in cluster_bin_fractions(comps, lengths).items():
                for label, fraction in values.items():
                    bins_rows.append({"frame": int(frame), "scheme": scheme,
                                      "bin": label, "fraction_particles": fraction})
        if with_ranks:
            for record in ranked_cluster_rows(comps, lengths, species):
                ranks_rows.append({"frame": int(frame), **record})
    return pd.DataFrame(graph_rows), pd.DataFrame(bins_rows), pd.DataFrame(ranks_rows)


def contact_frame_rows(pairing_npz: Mapping[str, np.ndarray] | np.lib.npyio.NpzFile, molecule_lookup: np.ndarray,
                       particle_recurrence: bool = False,
                       recurrence_window: int = 20) -> pd.DataFrame:
    """Count accepted particle pairs by type; optionally compute particle recurrence."""
    lookup = np.asarray(molecule_lookup, dtype=int)
    recurrence = RollingRecurrence(recurrence_window) if particle_recurrence else None
    rows = []
    for frame in pairing_npz["frame_indices"]:
        frame = int(frame)
        qi = np.asarray(pairing_npz[f"qi_{frame}"], dtype=int)
        pi = np.asarray(pairing_npz[f"pi_{frame}"], dtype=int)
        intra = int(np.count_nonzero(lookup[qi] == lookup[pi]))
        row = {"frame": frame, "n_base_pairs": len(qi),
               "n_intra_contacts": intra, "n_inter_contacts": len(qi) - intra}
        if recurrence:
            row["particle_recurrence"] = recurrence.add(pair_set(qi, pi))
        rows.append(row)
    return pd.DataFrame(rows)


def late_window(df: pd.DataFrame, fraction: float) -> pd.DataFrame:
    """Final fraction of *saved frame rows*; frame order is explicit."""
    if not 0 < fraction <= 1:
        raise ValueError("late fraction must lie within (0, 1]")
    if df.empty:
        return df
    data = df.sort_values("frame")
    n = max(1, int(np.ceil(len(data) * fraction)))
    return data.tail(n)


def summarize_numeric(df: pd.DataFrame, fraction: float,
                      metrics: list[str]) -> dict[str, float]:
    data = late_window(df, fraction)
    return {metric: float(data[metric].mean()) for metric in metrics if metric in data}


def summarize_bin_frames(df: pd.DataFrame, fraction: float) -> pd.DataFrame:
    if df.empty:
        return df
    n_frames = df["frame"].nunique()
    n_keep = max(1, int(np.ceil(n_frames * fraction)))
    frames = sorted(df["frame"].unique())[-n_keep:]
    selected = df[df["frame"].isin(frames)]
    return selected.groupby(["scheme", "bin"], as_index=False, sort=False)["fraction_particles"].mean()
