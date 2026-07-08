from pathlib import Path
from typing import Any
import json
from contextlib import contextmanager
from typing import Iterator
import numpy as np
import gsd.hoomd
from numpy.typing import NDArray

@contextmanager
def as_traj(
    traj_file: str | Path | gsd.hoomd.HOOMDTrajectory,
) -> Iterator[gsd.hoomd.HOOMDTrajectory]:
    """
    Yield a trajectory. If a path is given, the trajectory is opened here and closed on exit.
    """
    if isinstance(traj_file, (str, Path)):
        traj = gsd.hoomd.open(str(traj_file), "r")
        try:
            yield traj
        finally:
            traj.close()
    else:
        yield traj_file

# Parameters from params.json

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

# npz files
def save_npz(out_path: Path, data: dict[str, NDArray]) -> None:
    """
    Save arrays to a compressed .npz file.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **data)


def load_npz(cache_path: Path) -> dict[str, NDArray]:
    """
    Load arrays from a compressed .npz file.
    """
    with np.load(cache_path) as data:
        return {key: data[key] for key in data.files}
    
def get_frames(traj, fraction_last: float = 1.0) -> list[int]:
    """Return indices of the last `fraction_last` of frames (0<f<=1)."""
    n = len(traj)
    start = int(n * (1.0 - fraction_last))
    return list(range(start, n))