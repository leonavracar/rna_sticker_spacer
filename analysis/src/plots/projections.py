"""Plot XY density projections from GSD trajectories."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

from ..utils.io import load_params_json

from .style import PlotStyle


def repeat_count(molecule_file: str) -> int:
    """Extract (CAG)n repeat number from filenames such as cag10x470.json."""
    name = Path(molecule_file).stem.lower()
    position = name.find("cag")
    if position < 0:
        raise ValueError(f"Cannot identify CAG repeat count in {molecule_file!r}")
    suffix = name[position + 3:].lstrip("_-")
    # Read the digits immediately following the CAG prefix.
    count = suffix[:len(suffix) - len(suffix.lstrip("0123456789"))]
    if not count:
        raise ValueError(f"Cannot identify CAG repeat count in {molecule_file!r}")
    return int(count)


def projection_xy(frame: Any, bins: int = 200, sigma: float = 0.0) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return final-frame XY particle-count histogram and its physical box edges."""
    if bins < 2 or sigma < 0:
        raise ValueError("bins must be >= 2 and sigma must be nonnegative")
    lengths = np.asarray(frame.configuration.box[:2], dtype=float)
    positions = np.asarray(frame.particles.position, dtype=float)
    hist, xa, ya = np.histogram2d(
        positions[:, 0], positions[:, 1], bins=bins,
        range=[[-lengths[0]/2, lengths[0]/2], [-lengths[1]/2, lengths[1]/2]],
    )
    if sigma:
        hist = gaussian_filter(hist, sigma=sigma)
    return hist, xa, ya


def _run_metadata(sweep_dir: Path, *, row_by: str, density: float | None) -> Iterator[tuple[Any, float, Path]]:
    for run_dir in sorted(sweep_dir.iterdir()):
        if not run_dir.is_dir() or not (run_dir / "params.json").is_file():
            continue
        params = load_params_json(run_dir / "params.json")
        if density is not None and not np.isclose(float(params["density"]), density):
            continue
        if row_by == "repeat_count":
            row_value = repeat_count(params.get("molecule_file", ""))
        elif row_by in params:
            row_value = params[row_by]
        else:
            raise KeyError(f"{row_by!r} missing in {run_dir}/params.json")
        yield row_value, float(params["E_att"]), run_dir


def plot_projection_sweep(
    sweep_dir: str | Path, *, row_by: str, style: PlotStyle,
    density: float | None = .01,
    attractions: tuple[float, ...] | None = None,
    trajectory: str = "trajectory_0.gsd",
    bins: int = 200,
    sigma: float = 0.0,
    gamma: float = 0.55,
    title: str | None = None,
):
    """Plot a grid of representative final GSD frames.

    For RQ1: row_by='angle_k_ppp' and columns E_att.
    For RQ2: row_by='repeat_count' and columns E_att.
    Figures 9b/13b use final frames, not an averaged trajectory.
    """
    import gsd.hoomd

    sweep_dir = Path(sweep_dir)
    if not sweep_dir.is_dir():
        raise FileNotFoundError(f"Sweep not found: {sweep_dir}")
    records = list(_run_metadata(sweep_dir, row_by=row_by, density=density))
    if attractions is not None:
        records = [r for r in records if any(np.isclose(r[1], e) for e in attractions)]
    if not records:
        raise ValueError("No simulations match the requested projection conditions")
    row_values = sorted(set(rec[0] for rec in records))
    x_values = sorted(set(rec[1] for rec in records))
    samples = {}
    for row_value, e_att, run in records:
        key = row_value, e_att
        if key in samples:
            raise ValueError(
                f"Duplicate projection condition {key}; filter conditions more narrowly"
            )
        path = run / trajectory
        if not path.is_file():
            raise FileNotFoundError(f"Missing representative trajectory: {path}")
        with gsd.hoomd.open(str(path), "r") as traj:
            samples[key] = projection_xy(traj[-1], bins=bins, sigma=sigma)
    vmax = max(np.max(arr[0]) for arr in samples.values())
    norm = PowerNorm(gamma=gamma, vmin=0, vmax=max(float(vmax), 1.))
    with plt.rc_context(style.mpl_rc()):
        fig, axes = plt.subplots(
            len(row_values), len(x_values), squeeze=False,
            figsize=(3.1*len(x_values)+.8, 3.0*len(row_values)),
            layout="constrained",
        )
    image = None
    for i, row_val in enumerate(row_values):
        for j, eat in enumerate(x_values):
            ax = axes[i, j]
            record = samples.get((row_val, eat))
            if record is None:
                ax.set_visible(False)
                continue
            hist, x_edges, y_edges = record
            image = ax.imshow(
                hist.T, origin="lower",
                extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                norm=norm, cmap=style.cmap, aspect="equal",
                interpolation="nearest",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(f"{style.label('E_att')} = {eat:g}")
            if j == 0:
                ax.set_ylabel(f"{style.label(row_by)} = {row_val}")
    if image is not None:
        fig.colorbar(image, ax=list(axes.flat), label="Projected particle count",
                     shrink=.7)
    fig.suptitle(title or f"Final-frame XY projections · {sweep_dir.name}")
    return fig, axes
