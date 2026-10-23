"""Reusable, axes-first figures for wide per-replicate/condition tables.

Each function accepts (df, ax, *, ...) so functools.partial can fix plot
settings and panels_by can supply the current subset and Matplotlib Axes.
No function reads data, creates files, or runs simulation analysis.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from .data import require_columns
from .style import BIN_COLORS, PlotStyle


def _x_positions(df: pd.DataFrame, x: str) -> tuple[list, np.ndarray]:
    values = sorted(df[x].dropna().unique())
    return values, np.arange(len(values), dtype=float)


def _format_axis(ax: Axes, *, style: PlotStyle, xlabel: str, ylabel: str,
                 title: str | None = None) -> Axes:
    ax.set(xlabel=xlabel, ylabel=ylabel)
    if title:
        ax.set_title(title)
    if style.show_grid:
        ax.grid(axis="y", alpha=style.grid_alpha)
        ax.set_axisbelow(True)
    return ax


def plot_metric_series(
    df: pd.DataFrame, ax: Axes, *,
    metric: str, x: str = "E_att", style: PlotStyle,
    band: str = "std", color: str = "#2F76B7",
    label: str | None = None, show_points: bool = True,
) -> Axes:
    """Condition mean and variability from a *summary.csv* (not framewise)."""
    mean_col = f"{metric}_mean"
    lo_col = f"{metric}_p10"
    hi_col = f"{metric}_p90"
    if band not in ("std", "p10_p90", "none"):
        raise ValueError("band must be 'std', 'p10_p90', or 'none'")
    required = [x, mean_col]
    if band == "std":
        required.append(f"{metric}_std")
    elif band == "p10_p90":
        required += [lo_col, hi_col]
    require_columns(df, required)
    d = df.sort_values(x).dropna(subset=[x, mean_col])
    if d[x].duplicated().any():
        raise ValueError(f"Filter to one experimental series: {x} has duplicate values")
    xx = d[x].to_numpy(dtype=float)
    yy = d[mean_col].to_numpy(dtype=float)
    ax.plot(xx, yy, marker="o" if show_points else None, lw=style.mean_lw,
            color=color, label=label)
    if band != "none":
        if band == "std":
            deviation = d[f"{metric}_std"].fillna(0).to_numpy(dtype=float)
            low, high = yy - deviation, yy + deviation
        else:
            low = d[lo_col].to_numpy(dtype=float)
            high = d[hi_col].to_numpy(dtype=float)
        ax.fill_between(xx, low, high, color=color, alpha=style.band_alpha)
    if label is not None:
        ax.legend(loc=style.legend_loc, frameon=False)
    return _format_axis(ax, style=style, xlabel=style.label(x),
                        ylabel=style.label(metric))


def plot_metric_points(
    df: pd.DataFrame, ax: Axes, *,
    metric: str, x: str = "E_att", style: PlotStyle,
    hue: str | None = None, palette: Mapping | None = None,
    kind: str = "box",
) -> Axes:
    """Seaborn box/strip plot of independent replicate (not frame) measurements."""
    import seaborn as sns
    require_columns(df, [x, metric] + ([hue] if hue else []))
    if kind not in ("box", "strip"):
        raise ValueError("kind must be 'box' or 'strip'")
    kwargs = {"data": df, "x": x, "y": metric, "hue": hue, "ax": ax}
    if palette is not None and hue is not None:
        kwargs["palette"] = palette
    if kind == "box":
        sns.boxplot(**kwargs, showfliers=False, color=None if hue else "#DCE6EE",
                    linewidth=1, width=.65)
        sns.stripplot(**kwargs, dodge=bool(hue), color=None if hue else "#315577",
                      alpha=style.point_alpha, size=max(2, style.point_size ** .5 / 2),
                      jitter=style.jitter)
        # Strip points create a redundant legend if hue is present.
        if hue and ax.get_legend() is not None:
            handles, labels = ax.get_legend_handles_labels()
            n = df[hue].dropna().nunique()
            ax.legend(handles[:n], [style.label(str(v)) for v in labels[:n]],
                      title=style.label(hue), frameon=False,
                      loc=style.legend_loc)
    else:
        sns.stripplot(**kwargs, dodge=bool(hue), jitter=style.jitter,
                      alpha=style.point_alpha)
    return _format_axis(ax, style=style, xlabel=style.label(x),
                        ylabel=style.label(metric))


def plot_stacked_fractions(
    df: pd.DataFrame, ax: Axes, *,
    columns: Sequence[str], labels: Sequence[str], x: str = "E_att",
    style: PlotStyle, colors: Sequence[str] | None = None,
    annotation_columns: Sequence[str] | None = None,
    annotate_min_height: float = .055,
    legend: bool = True,
) -> Axes:
    """Stacked category means for bins, ranked material, or contact composition.

    Uses summary.csv columns directly. Each fraction is already averaged
    *within each trajectory* and then across independent replicates upstream.
    """
    if len(columns) != len(labels):
        raise ValueError("columns and labels must have equal length")
    if annotation_columns is not None and len(annotation_columns) != len(columns):
        raise ValueError("annotation_columns must match columns")
    required = [x, *columns, *(annotation_columns or [])]
    require_columns(df, required)
    d = df.sort_values(x).dropna(subset=[x])
    if d[x].duplicated().any():
        raise ValueError(f"Expected one row per {x}; filter the other parameters first")
    if d.empty:
        raise ValueError("No conditions available to plot")
    palette = list(colors or style.stack_colors)
    if len(palette) < len(columns):
        raise ValueError("Not enough colors for stacked categories")
    values = d[list(columns)].fillna(0).to_numpy(dtype=float)
    if np.any(values < -1e-9):
        raise ValueError("Fractions must not be negative")
    totals = values.sum(axis=1)
    if not np.allclose(totals, 1, atol=1e-3):
        raise ValueError(
            f"Stacked fractions do not sum to one: {np.round(totals, 3).tolist()}. "
            "Check omitted categories or incompatible denominators."
        )
    positions = np.arange(len(d), dtype=float)
    bottom = np.zeros(len(d), dtype=float)
    for i, (column, label) in enumerate(zip(columns, labels)):
        heights = values[:, i]
        ax.bar(positions, heights, bottom=bottom, width=.78,
               color=palette[i], alpha=style.stack_alpha, label=label)
        if annotation_columns is not None:
            counts = d[annotation_columns[i]].to_numpy(dtype=float)
            for xpos, height, base, count in zip(positions, heights, bottom, counts):
                if height >= annotate_min_height and np.isfinite(count):
                    ax.text(xpos, base + height / 2, f"{count:.0f}",
                            ha="center", va="center", fontsize=style.tick_size - 1)
        bottom += heights
    ax.set_xticks(positions, [f"{v:g}" if isinstance(v, (float, int, np.number)) else str(v)
                              for v in d[x]])
    ax.set_ylim(0, 1)
    if legend:
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    return _format_axis(ax, style=style, xlabel=style.label(x),
                        ylabel=style.label("fraction"))


def plot_cluster_bins(
    df: pd.DataFrame, ax: Axes, *, scheme: str,
    style: PlotStyle, x: str = "E_att",
) -> Axes:
    """Stacked RQ1 molecule-count or RQ2 particle-percentage bin fractions."""
    from .style import MOLECULE_BINS, PARTICLE_BINS
    if scheme not in ("molecule_count", "particle_percentage"):
        raise ValueError("Unknown cluster-bin scheme")
    categories = MOLECULE_BINS if scheme == "molecule_count" else PARTICLE_BINS
    columns = [f"cluster_bin_{scheme}_{c}_mean" for c in categories]
    return plot_stacked_fractions(
        df, ax, x=x, columns=columns,
        labels=[style.label(c) for c in categories],
        colors=style.stack_colors, style=style,
    )


def plot_ranked_clusters(
    df: pd.DataFrame, ax: Axes, *, style: PlotStyle, x: str = "E_att",
) -> Axes:
    """Pure-system ranked particle material; ranks together sum to one."""
    from .style import RANKS
    columns = [
        f"rank_{rank}_fraction_particles_mean" for rank in RANKS
    ]
    return plot_stacked_fractions(
        df, ax, x=x, columns=columns,
        labels=[style.label(rank) for rank in RANKS],
        colors=style.rank_colors, style=style,
    )


def plot_mixture_ranked_bars(
    df: pd.DataFrame, ax: Axes, *, style: PlotStyle,
    species: Sequence[str] = ("cag10", "cag47"),
    conditions: Sequence[float] | None = None,
    annotate_molecules: bool = False,
) -> Axes:
    """Mixture: group attraction conditions within rank, stack RNA species.

    Heights are fractions of ALL system particles, not of the rank alone.
    Sums over all rank and species combinations must equal 1 for each condition.
    """
    from .style import RANKS
    if conditions is None:
        conditions = sorted(df["E_att"].dropna().unique())
    d = df[df["E_att"].isin(conditions)].copy()
    require_columns(d, ["E_att"])
    if d["E_att"].duplicated().any():
        raise ValueError("Filter to one density and stiffness first")
    if len(d) != len(conditions):
        raise ValueError("One or more requested E_att conditions are absent")
    d = d.set_index("E_att").loc[list(conditions)]
    fraction_columns = [
        f"rank_{rank}_{sp}_fraction_total_particles_mean"
        for rank in RANKS for sp in species
    ]
    count_columns = [
        f"rank_{rank}_{sp}_n_molecules_mean"
        for rank in RANKS for sp in species
    ]
    require_columns(d, fraction_columns + (count_columns if annotate_molecules else []))
    if not np.allclose(d[fraction_columns].fillna(0).sum(axis=1), 1, atol=1e-3):
        raise ValueError("Rank/species material must sum to one per condition")

    centers = np.arange(len(RANKS), dtype=float)
    width = .78 / len(conditions)
    for ci, eat in enumerate(conditions):
        xpos = centers - .39 + (ci + .5) * width
        bottom = np.zeros(len(RANKS))
        for sp in species:
            values = np.asarray(
                [d.loc[eat, f"rank_{rank}_{sp}_fraction_total_particles_mean"]
                 for rank in RANKS], dtype=float
            )
            bars = ax.bar(xpos, values, width=width, bottom=bottom,
                          color=style.species_colors.get(sp), alpha=style.stack_alpha,
                          label=style.label(sp) if ci == 0 else None)
            if annotate_molecules:
                counts = [
                    d.loc[eat, f"rank_{rank}_{sp}_n_molecules_mean"]
                    for rank in RANKS
                ]
                for xloc, height, base, count in zip(xpos, values, bottom, counts):
                    if height > .055 and np.isfinite(count):
                        ax.text(xloc, base + height / 2, f"{count:.0f}",
                                ha="center", va="center",
                                fontsize=max(6, style.tick_size - 1))
            bottom += values

    ax.set_xticks(centers, [style.label(rank) for rank in RANKS], rotation=20)
    ax.set_ylim(0, 1)
    ax.legend(title="RNA species", frameon=False, loc="upper left",
              bbox_to_anchor=(1.01, 1))
    # Legend for attraction groups as a short descriptive panel caption.
    _format_axis(ax, style=style, xlabel="Ranked-cluster material",
                 ylabel=style.label("fraction"))
    ax.set_title("Grouped bars: " + ", ".join(f"E={float(e):g}" for e in conditions))
    return ax


def plot_largest_contact_composition(
    df: pd.DataFrame, ax: Axes, *, style: PlotStyle, x: str = "E_att",
    annotate_counts: bool = True,
) -> Axes:
    """RQ3 Fig. 17: intermolecular contacts INSIDE the largest component only."""
    kinds = ("cag10_cag10", "cag10_cag47", "cag47_cag47")
    columns = [
        f"largest_contacts_{kind}_fraction_inter_contacts_mean" for kind in kinds
    ]
    annotations = [
        f"largest_contacts_{kind}_n_inter_contacts_mean" for kind in kinds
    ] if annotate_counts else None
    return plot_stacked_fractions(
        df, ax, columns=columns, x=x,
        annotation_columns=annotations, style=style,
        labels=["10–10", "10–47", "47–47"],
        colors=[style.contact_colors[k] for k in kinds],
    )


def plot_precision_recall(
    df: pd.DataFrame, ax: Axes, *, style: PlotStyle,
    color_by: str = "E_att", threshold: float = .8,
) -> Axes:
    """Validation precision/recall quadrants; each point is one replicate."""
    require_columns(df, ["precision", "recall", color_by])
    art = ax.scatter(df["precision"], df["recall"], c=df[color_by],
                     cmap="viridis", s=style.point_size,
                     alpha=style.point_alpha, linewidths=0)
    ax.axvline(threshold, color="#888", linestyle="--", linewidth=1)
    ax.axhline(threshold, color="#888", linestyle="--", linewidth=1)
    ax.set(xlim=(-.02, 1.02), ylim=(-.02, 1.02))
    ax.figure.colorbar(art, ax=ax, label=style.label(color_by))
    return _format_axis(ax, style=style, xlabel="Precision", ylabel="Recall")


def make_figure(nrows: int = 1, ncols: int = 1, *,
                style: PlotStyle, width: float = 4.7, height: float = 3.6,
                sharex: bool = False, sharey: bool = False):
    """Convenient consistent Matplotlib panel canvas."""
    if nrows < 1 or ncols < 1:
        raise ValueError("nrows and ncols must be positive")
    with plt.rc_context(style.mpl_rc()):
        return plt.subplots(nrows, ncols, figsize=(width*ncols, height*nrows),
                            sharex=sharex, sharey=sharey, squeeze=False,
                            layout="constrained")
