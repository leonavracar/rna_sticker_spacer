"""Optional plots requiring FRAMEWISE trajectories (not used by curated notebooks).

The default analysis exports replicate and condition summaries, not framewise
CSVs. Keep these plotting functions available for diagnostic sessions without
introducing time-series dependencies into the four research notebooks.
"""
from __future__ import annotations

from collections.abc import Sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .style import PlotStyle
from .data import require_columns


def plot_framewise_replicates(
    df: pd.DataFrame, ax: plt.Axes, *, y: str,
    style: PlotStyle, x: str = "step", replicate: str = "traj_replicate",
    show_mean: bool = True,
) -> plt.Axes:
    """Thin trajectory traces plus their mean at matched sampled steps."""
    require_columns(df, [x, y, replicate])
    for _, run in df.dropna(subset=[y]).groupby(replicate):
        run = run.sort_values(x)
        ax.plot(run[x], run[y], lw=.8, alpha=.28, color="#547C9B")
    if show_mean:
        mean = df.groupby(x, as_index=False)[y].mean().sort_values(x)
        ax.plot(mean[x], mean[y], color="#134C7A", lw=style.mean_lw,
                label="Replicate mean")
        ax.legend(frameon=False)
    ax.set(xlabel=style.label(x), ylabel=style.label(y))
    if style.show_grid:
        ax.grid(alpha=style.grid_alpha)
    return ax


def plot_framewise_stacked(
    df: pd.DataFrame, ax: plt.Axes, *, columns: Sequence[str],
    style: PlotStyle, x: str = "step",
) -> plt.Axes:
    """Stacked time-dependent fractions for a single replicate only."""
    require_columns(df, [x, *columns])
    if "traj_replicate" in df and df["traj_replicate"].nunique() > 1:
        raise ValueError("Filter to one trajectory before stacking time series")
    d = df.sort_values(x)
    if not np.allclose(d[list(columns)].sum(axis=1), 1, atol=1e-3):
        raise ValueError("Fractions must sum to one in each frame")
    ax.stackplot(
        d[x], *[d[c] for c in columns],
        colors=style.stack_colors[:len(columns)],
        alpha=style.stack_alpha,
        labels=[style.label(c) for c in columns],
    )
    ax.set(xlabel=style.label(x), ylabel="Fraction", ylim=(0, 1))
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    return ax
