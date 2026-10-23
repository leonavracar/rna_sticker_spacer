"""Small axes-first multipanel utility; works naturally with functools.partial."""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .style import PlotStyle


def panels_by(
    df: pd.DataFrame, *,
    by: str | Sequence[str],
    plot_fn: Callable[[pd.DataFrame, plt.Axes], Any],
    style: PlotStyle,
    ncols: int = 3,
    width: float = 4.5,
    height: float = 3.4,
    sharex: bool = False,
    sharey: bool = False,
    title: str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Split data by one or more columns, then call plot_fn(subset, ax).

    Example:
        from functools import partial
        plot = partial(plot_metric_series, metric="n_edges", style=style)
        fig, axes = panels_by(summary, by="angle_k_ppp", plot_fn=plot, style=style)
    """
    if ncols < 1:
        raise ValueError("ncols must be positive")
    columns = [by] if isinstance(by, str) else list(by)
    if not columns or any(col not in df for col in columns):
        raise ValueError(f"Expected existing panel columns, got {columns}")
    groups = list(df.groupby(columns, sort=True, dropna=False))
    if not groups:
        raise ValueError("No data available to plot")
    ncols = min(ncols, len(groups))
    nrows = math.ceil(len(groups) / ncols)
    with plt.rc_context(style.mpl_rc()):
        fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                                sharex=sharex, sharey=sharey,
                                figsize=(width*ncols, height*nrows),
                                layout="constrained")
    for ax, (key, subset) in zip(axes.flat, groups):
        plot_fn(subset.copy(), ax)
        values = key if isinstance(key, tuple) else (key,)
        ax.set_title(", ".join(
            f"{style.label(col)} = {value:g}" if isinstance(value, (int, float, np.number))
            else f"{style.label(col)} = {value}"
            for col, value in zip(columns, values)
        ))
    for ax in list(axes.flat)[len(groups):]:
        ax.set_visible(False)
    if title:
        fig.suptitle(title)
    return fig, axes


def plot_panels(items: list, plot_fn, title_fn=str,
                n_cols: int = 3,
                figsize_per_panel: tuple[float, float] = (4., 4.),
                suptitle: str | None = None,
                sharex: bool = False, sharey: bool = False):
    """Backward-compatible diagnostic panel utility for arbitrary objects."""
    if not items or n_cols < 1:
        raise ValueError("Must supply items and n_cols >= 1")
    ncols = min(n_cols, len(items))
    nrows = math.ceil(len(items) / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                            figsize=(figsize_per_panel[0]*ncols,
                                     figsize_per_panel[1]*nrows),
                            sharex=sharex, sharey=sharey, squeeze=False,
                            layout="constrained")
    for ax, item in zip(axes.flat, items):
        plot_fn(item, ax)
        ax.set_title(title_fn(item))
    for ax in list(axes.flat)[len(items):]:
        ax.set_visible(False)
    if suptitle:
        fig.suptitle(suptitle)
    return fig, axes
