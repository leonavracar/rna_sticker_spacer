"""Arrange scientific plots into panels."""
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
    legend: str = "auto",
) -> tuple[plt.Figure, np.ndarray]:
    """Draw one panel for each group in the data."""
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
    if legend not in {"auto", "last", "none", "bottom"}:
        raise ValueError("legend must be 'auto', 'last', 'bottom', or 'none'")
    if legend != "auto":
        visible_axes = list(axes.flat)[:len(groups)]
        handles, labels = visible_axes[0].get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        labels = list(unique)
        handles = list(unique.values())
        labels = [style.label(label) if label in ("repeat", "scrambled")
                  else label for label in labels]
        for ax in visible_axes:
            original_legend = ax.get_legend()
            if original_legend is not None:
                original_legend.remove()
        if legend == "last" and handles:
            visible_axes[-1].legend(
                handles, labels, loc="upper left",
                bbox_to_anchor=(1.02, 1), frameon=False,
                fontsize=style.legend_size,
            )
        elif legend == "bottom" and handles:
            fig.set_layout_engine(None)
            fig.subplots_adjust(bottom=.25 if nrows == 1 else .16)
            fig.legend(
                handles, labels, loc="lower center",
                bbox_to_anchor=(.5, .035), ncol=min(len(handles), 7),
                frameon=False, fontsize=style.legend_size,
            )
    if title:
        fig.suptitle(title)
    return fig, axes


def plot_panels(items: list, plot_fn, title_fn=str,
                n_cols: int = 3,
                figsize_per_panel: tuple[float, float] = (4., 4.),
                suptitle: str | None = None,
                sharex: bool = False, sharey: bool = False):
    """Draw panels from a list of inputs."""
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
