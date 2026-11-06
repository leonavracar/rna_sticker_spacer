"""Small IO helpers for parameter files, optional trajectories and NPZ arrays."""

from __future__ import annotations

from collections.abc import Sized
from contextlib import contextmanager
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import gsd.hoomd

@contextmanager
def as_traj(
    traj_file: str | Path | gsd.hoomd.HOOMDTrajectory,
) -> Iterator[gsd.hoomd.HOOMDTrajectory]:
    """
    Yield a trajectory. If a path is given, the trajectory is opened here and closed on exit.
    """
    if isinstance(traj_file, (str, Path)):
        # The NPZ-only analyze stage must not need to import GSD.
        import gsd.hoomd

        traj = gsd.hoomd.open(str(traj_file), "r")
        try:
            yield traj
        finally:
            traj.close()
    else:
        yield traj_file


def load_params_json(params_path: Path) -> dict[str, Any]:
    """Load params.json, tolerating extra concatenated JSON after the first object."""
    with open(params_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        raise ValueError(f"Empty params.json: {params_path}")

    decoder = json.JSONDecoder()
    data, end = decoder.raw_decode(content)

    extra = content[end:].strip()
    if extra:
        print(f"WARNING: Ignoring extra JSON data in {params_path} after char {end}")

    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object in {params_path}, got {type(data).__name__}")

    return data

def get_row_params_from_json(
    run_dir: Path,
    swept_param_keys: list[str],
) -> dict[str, object]:
    """
    Read the swept parameter values for one run folder from params.json.
    """
    params = load_params_json(run_dir / "params.json")

    row: dict[str, object] = {}
    for key in swept_param_keys:
        if key not in params:
            raise KeyError(f"Key '{key}' not found in {run_dir / 'params.json'}")
        row[key] = params[key]

    row["folder_name"] = run_dir.name
    return row

def save_npz(out_path: Path, data: dict[str, NDArray]) -> None:
    """Write a compressed NPZ archive."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **data)


def load_npz(cache_path: Path) -> dict[str, NDArray]:
    """Read all arrays from an NPZ archive."""
    with np.load(cache_path) as data:
        return {key: data[key] for key in data.files}


def get_frames(traj: Sized, fraction_last: float = 1.0) -> list[int]:
    """Return indices of the last `fraction_last` of frames (0<f<=1)."""
    if not 0 < fraction_last <= 1:
        raise ValueError("fraction_last must lie in (0, 1]")
    n = len(traj)
    start = int(n * (1.0 - fraction_last))
    return list(range(start, n))