"""Aggregate selected framewise observables over a late trajectory window."""
from __future__ import annotations
import pandas as pd
from .framewise import summarize_numeric, summarize_bin_frames


def summarize_replicate(frame: pd.DataFrame, bins: pd.DataFrame,
                        fraction: float) -> tuple[dict[str, float], pd.DataFrame]:
    metrics = [c for c in (
        "n_base_pairs", "n_inter_contacts", "n_intra_contacts",
        "n_edges", "mean_edge_weight", "mean_degree",
        "particle_recurrence", "molecule_recurrence",
        "mfe_matches", "extra_pairs", "precision", "recall", "Rg_nm",
    ) if c in frame]
    return summarize_numeric(frame, fraction, metrics), summarize_bin_frames(bins, fraction)
