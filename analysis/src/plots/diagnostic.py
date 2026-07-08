# plot_utils.py
from __future__ import annotations
import numpy as np
import freud
import gsd.hoomd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import PowerNorm
from scipy.ndimage import gaussian_filter
from typing import Iterable
Axis = str  # "x" | "y" | "z"
_AX = {"x": 0, "y": 1, "z": 2}


# ---------- COMPUTE ----------

def compute_projection_2d(
    traj_or_frame,
    plane: str = "xy",
    bins: int = 200,
    sigma: float = 0.0,
    average: bool = False,
    fraction_last: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    2D histogram of particle positions projected onto `plane` ('xy'|'xz'|'yz').
    Accepts a single frame, or a trajectory (with average=True to average
    over the last `fraction_last` of frames).
    """
    # single frame path
    if not average:
        frame = traj_or_frame[-1] if hasattr(traj_or_frame, "__len__") else traj_or_frame
        a, b = plane[0], plane[1]
        ia, ib = _AX[a], _AX[b]
        pos = np.asarray(frame.particles.position)
        La, Lb = frame.configuration.box[ia], frame.configuration.box[ib]
        H, ea, eb = np.histogram2d(
            pos[:, ia], pos[:, ib],
            bins=bins,
            range=[[-La/2, La/2], [-Lb/2, Lb/2]],
        )
        if sigma > 0:
            H = gaussian_filter(H, sigma=sigma)
        return H, ea, eb

    # averaged path
    traj = traj_or_frame
    n = len(traj)
    frames = range(int(n * (1.0 - fraction_last)), n)
    H_sum, ea, eb = None, None, None
    for i in frames:
        H, ea, eb = compute_projection_2d(traj[i], plane=plane, bins=bins)
        H_sum = H if H_sum is None else H_sum + H
    H_avg = H_sum / len(frames)
    if sigma > 0:
        H_avg = gaussian_filter(H_avg, sigma=sigma)
    return H_avg, ea, eb


def compute_profile_1d(
    frame: gsd.hoomd.Frame,
    axis: Axis = "x",
    bins: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """1D density profile along one axis. Returns (bin_centers, counts)."""
    k = _AX[axis]
    pos = np.asarray(frame.particles.position)
    L = frame.configuration.box[k]
    counts, edges = np.histogram(pos[:, k], bins=bins, range=(-L/2, L/2))
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts


def compute_rdf(
    frame: gsd.hoomd.Frame,
    bins: int = 100,
    r_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Radial distribution g(r). r_max defaults to 0.95 * Lmin/2."""
    pos = np.asarray(frame.particles.position)
    box = freud.box.Box.from_box(frame.configuration.box)
    Lmin = min(frame.configuration.box[:3])
    if r_max is None:
        r_max = 0.95 * Lmin / 2
    rdf = freud.density.RDF(bins=bins, r_max=r_max)
    rdf.compute(system=(box, pos))
    return rdf.bin_centers, rdf.rdf



# ---------- AVERAGING OVER FRAMES ----------

def average_projection(
    traj: gsd.hoomd.HOOMDTrajectory,
    plane: str = "xy",
    frames: Iterable[int] | None = None,
    bins: int = 200,
    sigma: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Time-averaged 2D density projection over `frames` (default: all)."""
    frames = list(range(len(traj))) if frames is None else list(frames)
    H_sum, ea, eb = None, None, None
    for i in frames:
        H, ea, eb = compute_projection_2d(traj[i], plane=plane, bins=bins)
        H_sum = H if H_sum is None else H_sum + H
    H_avg = H_sum / len(frames)
    if sigma > 0:
        H_avg = gaussian_filter(H_avg, sigma=sigma)
    return H_avg, ea, eb


def average_profile_1d(
    traj: gsd.hoomd.HOOMDTrajectory,
    axis: Axis = "x",
    frames: Iterable[int] | None = None,
    bins: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Time-averaged 1D density profile along `axis`."""
    frames = list(range(len(traj))) if frames is None else list(frames)
    c_sum, centers = None, None
    for i in frames:
        centers, counts = compute_profile_1d(traj[i], axis=axis, bins=bins)
        c_sum = counts if c_sum is None else c_sum + counts
    return centers, c_sum / len(frames)


def average_rdf(
    traj: gsd.hoomd.HOOMDTrajectory,
    frames: Iterable[int] | None = None,
    bins: int = 100,
    r_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Time-averaged g(r) using freud's built-in accumulation."""
    frames = list(range(len(traj))) if frames is None else list(frames)
    if r_max is None:
        Lmin = min(traj[frames[0]].configuration.box[:3])
        r_max = 0.95 * Lmin / 2
    rdf = freud.density.RDF(bins=bins, r_max=r_max)
    for i in frames:
        f = traj[i]
        box = freud.box.Box.from_box(f.configuration.box)
        pos = np.asarray(f.particles.position)
        rdf.compute(system=(box, pos), reset=False)  # accumulate
    return rdf.bin_centers, rdf.rdf

# ---------- PLOT ----------

def plot_projection_2d(
    H: np.ndarray,
    ea: np.ndarray,
    eb: np.ndarray,
    ax: Axes | None = None,
    plane: str = "xy",
    gamma: float = 0.5,
    cmap: str = "magma",
    vmax: float | None = None,
    cbar: bool = True,
) -> Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(
        H.T, origin="lower",
        extent=(ea[0], ea[-1], eb[0], eb[-1]),
        cmap=cmap,
        norm=PowerNorm(gamma=gamma, vmin=0, vmax=(vmax if vmax is not None else H.max())),
        aspect="equal", interpolation="nearest",
    )
    ax.set_xlabel(plane[0]); ax.set_ylabel(plane[1])
    if cbar:
        plt.colorbar(im, ax=ax, shrink=0.75, label="counts")
    return ax


def plot_profile_1d(
    centers: np.ndarray,
    counts: np.ndarray,
    ax: Axes | None = None,
    axis: Axis = "x",
    color: str = "steelblue",
) -> Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3))
    ax.fill_between(centers, counts, color=color, alpha=0.5, step="mid")
    ax.plot(centers, counts, color=color, lw=1)
    ax.set_xlabel(axis); ax.set_ylabel("counts")
    return ax


def plot_rdf(r, g, ax=None, label=None, color=None, annotate=True):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.plot(r, g, label=label, color=color)
    ax.axhline(1.0, color="gray", lw=0.5, ls="--")
    ax.set_xlabel("r"); ax.set_ylabel("g(r)")

    if annotate:
        i = int(np.argmax(g))
        gmax, rmax = g[i], r[i]
        # crude "clustering index": how much the peak exceeds bulk
        cluster_idx = gmax - 1.0
        ax.plot(rmax, gmax, "o", color="red")
        ax.text(0.95, 0.95,
                f"g_max = {gmax:.2f}  @ r = {rmax:.2f}\n"
                f"clustering = g_max − 1 = {cluster_idx:.2f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9,
                bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))
    if label: ax.legend()
    return ax