"""Single-chain conformation and ViennaRNA secondary-structure comparison."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import gsd.hoomd


def mfe_pairs_from_csv(path: str | Path) -> set[tuple[int, int]]:
    data = pd.read_csv(path)
    if not {"i","j"}.issubset(data):
        raise ValueError("MFE CSV requires columns i,j")
    return {tuple(sorted((int(a), int(b)))) for a,b in zip(data["i"],data["j"])}


def single_chain_framewise(pairing_npz: Mapping[str, np.ndarray] | np.lib.npyio.NpzFile, mfe_pairs: set[tuple[int,int]],
                           index_offset: int = 1) -> pd.DataFrame:
    rows = []
    for f in pairing_npz["frame_indices"]:
        frame = int(f)
        pairs = {tuple(sorted((int(a)+index_offset,int(b)+index_offset)))
                 for a,b in zip(pairing_npz[f"qi_{frame}"],pairing_npz[f"pi_{frame}"])}
        matched = len(pairs & mfe_pairs)
        extra = len(pairs - mfe_pairs)
        rows.append({"frame": frame, "mfe_matches": matched, "extra_pairs": extra,
                     "precision": matched/len(pairs) if pairs else 0.,
                     "recall": matched/len(mfe_pairs) if mfe_pairs else 0.})
    return pd.DataFrame(rows)


def radius_of_gyration(frame: gsd.hoomd.Frame, nm_per_sim_length_unit: float) -> float:
    """Return radius of gyration in nanometers for a periodic trajectory frame."""
    if nm_per_sim_length_unit <= 0:
        raise ValueError("nm_per_sim_length_unit must be positive")
    import freud
    box = freud.box.Box.from_box(frame.configuration.box)
    positions = box.unwrap(frame.particles.position, frame.particles.image)
    displacement = positions - positions.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum(displacement**2, axis=1))) * nm_per_sim_length_unit)


def add_gsd_geometry(
    df: pd.DataFrame, trajectory_path: str | Path, nm_per_sim_length_unit: float
) -> pd.DataFrame:
    import gsd.hoomd
    out = df.copy()
    with gsd.hoomd.open(str(trajectory_path), "r") as traj:
        out["Rg_nm"] = [radius_of_gyration(traj[int(i)], nm_per_sim_length_unit) for i in out["frame"]]
        out["step"] = [int(traj[int(i)].configuration.step) for i in out["frame"]]
    return out
