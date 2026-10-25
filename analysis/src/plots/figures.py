"""Matplotlib figures for the analysis tables."""
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
    ax.set_xlabel(xlabel, fontsize=style.label_size)
    ax.set_ylabel(ylabel, fontsize=style.label_size)
    ax.tick_params(labelsize=style.tick_size)
    if title:
        ax.set_title(title, fontsize=style.panel_title_size)
    if style.show_grid:
        ax.grid(axis="y", alpha=style.grid_alpha)
        ax.set_axisbelow(True)
    return ax


def _categorical_axis(
    df: pd.DataFrame, x: str, order: Sequence | None = None,
) -> tuple[list, np.ndarray]:
    """Equally spaced positions for measured parameters, not numeric gaps."""
    values = list(order) if order is not None else sorted(df[x].dropna().unique())
    if not values:
        raise ValueError(f"No {x} values to plot")
    unknown = set(df[x].dropna().unique()) - set(values)
    if unknown:
        raise ValueError(f"x_order is missing observed values: {sorted(unknown)}")
    return values, np.arange(len(values), dtype=float)


def _set_categorical_ticks(ax: Axes, values: Sequence) -> None:
    ax.set_xticks(np.arange(len(values)), [
        f"{value:g}" if isinstance(value, (int, float, np.number))
        else str(value) for value in values
    ])
    ax.set_xlim(-.45, len(values) - .55)


def plot_metric_series(
    df: pd.DataFrame, ax: Axes, *,
    metric: str, x: str = "E_att", style: PlotStyle,
    band: str = "std", color: str = "#2F76B7",
    label: str | None = None, show_points: bool = True,
    x_order: Sequence | None = None,
) -> Axes:
    """Plot condition means with a variability band."""
    mean_col = f"{metric}_mean"
    lo_col, hi_col = f"{metric}_p10", f"{metric}_p90"
    if band not in ("std", "p10_p90", "none"):
        raise ValueError("band must be 'std', 'p10_p90', or 'none'")
    required = [x, mean_col]
    if band == "std":
        required.append(f"{metric}_std")
    elif band == "p10_p90":
        required += [lo_col, hi_col]
    require_columns(df, required)
    d = df.dropna(subset=[x, mean_col]).copy()
    if d[x].duplicated().any():
        raise ValueError(f"Filter to one experimental series: {x} has duplicate values")
    values, _ = _categorical_axis(d, x, x_order)
    locations = {v: i for i, v in enumerate(values)}
    d["_position"] = d[x].map(locations)
    d = d.sort_values("_position")
    xx = d["_position"].to_numpy(dtype=float)
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
    _set_categorical_ticks(ax, values)
    return _format_axis(ax, style=style, xlabel=style.label(x),
                        ylabel=style.label(metric))


def plot_contact_balance(
    df: pd.DataFrame, ax: Axes, *,
    style: PlotStyle, x: str = "E_att", x_order: Sequence | None = None,
    band: str = "std",
) -> Axes:
    """Thesis Fig. 10: inter-, intramolecular and total base-pair contacts."""
    metrics = (
        ("n_inter_contacts", "Intermolecular", "#CA4258"),
        ("n_intra_contacts", "Intramolecular", "#2798CC"),
        ("n_base_pairs", "Total", "#6D4AA1"),
    )
    for metric, label, color in metrics:
        plot_metric_series(df, ax, style=style, metric=metric, x=x,
                           x_order=x_order, color=color, label=label, band=band)
    ax.set_ylabel("Number of base-pair contacts")
    return ax


def plot_recurrence_summary(
    summary: pd.DataFrame, ax: Axes, *, replicates: pd.DataFrame,
    style: PlotStyle, x: str = "E_att",
    metric: str = "molecule_recurrence",
) -> Axes:
    """Thesis Fig. 12: individual replicate points, mean and SD."""
    values, _ = _categorical_axis(summary, x)
    positions = {v: i for i, v in enumerate(values)}
    require_columns(replicates, [x, metric])
    rng = np.random.default_rng(42)
    raw = replicates.dropna(subset=[x, metric])
    ax.scatter(raw[x].map(positions).to_numpy(dtype=float)
               + rng.uniform(-style.jitter, style.jitter, len(raw)),
               raw[metric], color="#75889B", s=style.point_size * .7,
               alpha=.55, zorder=2)
    plot_metric_series(summary, ax, metric=metric, x=x, x_order=values,
                       style=style, color="#235F92", band="std")
    ax.set_ylim(bottom=0)
    return ax


def plot_largest_cluster_metric(
    df: pd.DataFrame, ax: Axes, *, metric: str, annotation: str,
    style: PlotStyle, annotation_label: str,
    color: str = "#828282", x: str = "E_att",
) -> Axes:
    """Thesis Fig. 18: bars for largest-cluster material/degree with counts."""
    require_columns(df, [x, f"{metric}_mean", f"{annotation}_mean"])
    d = df.sort_values(x).dropna(subset=[x, f"{metric}_mean"])
    if d[x].duplicated().any():
        raise ValueError(f"Filter to one experimental series: {x} duplicated")
    positions = np.arange(len(d), dtype=float)
    values = d[f"{metric}_mean"].to_numpy(dtype=float)
    ax.bar(positions, values, width=.68, color=color, alpha=.88)
    for pos, (_, row) in zip(positions, d.iterrows()):
        count = row[f"{annotation}_mean"]
        if np.isfinite(count):
            text = (rf"$N_{{\mathrm{{molecules}}}} = {count:.1f}$"
                    if annotation_label in {"N_mol", "N_molecules"}
                    else f"{count:.0f} {annotation_label}")
            ax.annotate(
                text,
                (pos, row[f"{metric}_mean"]), xytext=(0, 6),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=style.tick_size, color="#383838",
            )
    _set_categorical_ticks(ax, d[x].tolist())
    ax.set_ylim(0, max(values) * 1.22 if len(values) and max(values) > 0 else 1)
    return _format_axis(ax, style=style, xlabel=style.label(x),
                        ylabel=style.label(metric))


def plot_metric_points(
    df: pd.DataFrame, ax: Axes, *,
    metric: str, x: str = "E_att", style: PlotStyle,
    hue: str | None = None, palette: Mapping | None = None,
    kind: str = "box",
) -> Axes:
    """Plot replicate distributions with optional group colors."""
    require_columns(df, [x, metric] + ([hue] if hue else []))
    if kind not in ("box", "strip"):
        raise ValueError("kind must be 'box' or 'strip'")
    d = df.dropna(subset=[x, metric])
    if d.empty:
        raise ValueError("No nonmissing replicate values to plot")

    if hue is None:
        categories = sorted(d[x].unique())
        arrays = [d.loc[d[x] == value, metric].to_numpy() for value in categories]
        positions = np.arange(len(categories))
        if kind == "box":
            ax.boxplot(
                arrays, positions=positions, widths=style.box_width,
                patch_artist=True, showfliers=False,
                boxprops={"facecolor": style.box_facecolor,
                          "edgecolor": style.box_edgecolor, "linewidth": 1},
                medianprops={"color": style.box_median_color, "linewidth": 1.25},
                whiskerprops={"color": style.box_whisker_color, "linewidth": 1},
                capprops={"color": style.box_whisker_color, "linewidth": 1},
            )
        rng = np.random.default_rng(42)
        for index, values in enumerate(arrays):
            ax.scatter(
                index + rng.uniform(-style.jitter, style.jitter, len(values)),
                values, s=style.box_point_size, color=style.box_point_color,
                alpha=style.box_point_alpha, linewidths=0, zorder=3,
            )
        _set_categorical_ticks(ax, categories)
    else:
        import seaborn as sns
        kwargs = {"data": d, "x": x, "y": metric, "hue": hue, "ax": ax}
        if palette is not None:
            kwargs["palette"] = palette
        if kind == "box":
            sns.boxplot(**kwargs, showfliers=False, linewidth=1, width=.65,
                        boxprops={"alpha": .65})
            sns.stripplot(**kwargs, dodge=True, jitter=style.jitter,
                          alpha=style.point_alpha, size=3,
                          linewidth=0)
            handles, labels = ax.get_legend_handles_labels()
            count = d[hue].nunique()
            if handles and ax.get_legend():
                ax.legend(handles[:count],
                          [style.label(str(v)) for v in labels[:count]],
                          title=style.label(hue), loc=style.legend_loc,
                          frameon=False)
        else:
            sns.stripplot(**kwargs, dodge=True, jitter=style.jitter,
                          alpha=style.point_alpha)
        _set_categorical_ticks(ax, sorted(d[x].unique()))
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
    """Plot stacked fractions and optional count labels."""
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
                    from matplotlib.colors import to_rgb
                    rgb = to_rgb(palette[i])
                    brightness = .2126*rgb[0] + .7152*rgb[1] + .0722*rgb[2]
                    ax.text(xpos, base + height / 2, f"{count:.0f}",
                            ha="center", va="center", fontsize=style.tick_size - 1,
                            color="white" if brightness < .52 else "#252525")
        bottom += heights
    ax.set_xticks(positions, [f"{v:g}" if isinstance(v, (float, int, np.number)) else str(v)
                              for v in d[x]])
    ax.set_ylim(0, 1)
    if legend:
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1),
                  fontsize=style.legend_size)
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
    annotate_molecules: bool = False,
) -> Axes:
    """Plot ranked cluster fractions with optional molecule counts."""
    from .style import RANKS
    columns = [f"rank_{rank}_fraction_particles_mean" for rank in RANKS]
    counts = ([f"rank_{rank}_n_molecules_mean" for rank in RANKS]
              if annotate_molecules else None)
    return plot_stacked_fractions(
        df, ax, x=x, columns=columns,
        labels=[style.label(rank) for rank in RANKS],
        annotation_columns=counts,
        colors=style.rank_colors, style=style,
    )


def plot_ranked_condition(
    df: pd.DataFrame, ax: Axes, *,
    style: PlotStyle, species: Sequence[str] | None = None,
    annotate_molecules: bool = False, color: str = "#4389BE",
) -> Axes:
    """Plot ranked-cluster particle fractions for one condition."""
    from .style import RANKS
    if len(df) != 1:
        raise ValueError("Filter to exactly one E_att, stiffness and density first")
    row = df.iloc[0]
    if species is None:
        columns = [f"rank_{r}_fraction_particles_mean" for r in RANKS]
    else:
        columns = [
            f"rank_{r}_{sp}_fraction_total_particles_mean"
            for r in RANKS for sp in species
        ]
    require_columns(df, columns)
    fractions = np.asarray([row[c] for c in columns], dtype=float)
    if not np.allclose(np.nansum(fractions), 1.0, atol=1e-3):
        raise ValueError("Ranked-cluster material fractions must sum to one")
    positions = np.arange(len(RANKS), dtype=float)
    bottom = np.zeros(len(RANKS))
    if species is None:
        ax.bar(positions, fractions, color=color, alpha=style.stack_alpha, width=.72)
        if annotate_molecules:
            require_columns(df, [f"rank_{r}_n_molecules_mean" for r in RANKS])
            for i, rank in enumerate(RANKS):
                count = row[f"rank_{rank}_n_molecules_mean"]
                if np.isfinite(count) and fractions[i] > .005:
                    ax.text(i, fractions[i] + .025, f"{count:.1f}",
                            ha="center", va="bottom", color=color,
                            fontsize=style.tick_size - 1)
    else:
        require_columns(
            df, [f"rank_{r}_{sp}_n_molecules_mean"
                 for r in RANKS for sp in species] if annotate_molecules else [],
        )
        for sp in species:
            values = np.asarray([
                row[f"rank_{r}_{sp}_fraction_total_particles_mean"] for r in RANKS
            ], dtype=float)
            ax.bar(positions, values, bottom=bottom,
                   color=style.species_colors[sp], alpha=style.stack_alpha,
                   width=.72, label=style.label(sp))
            if annotate_molecules:
                for i, rank in enumerate(RANKS):
                    count = row[f"rank_{rank}_{sp}_n_molecules_mean"]
                    if np.isfinite(count) and values[i] > .04:
                        ax.text(i, bottom[i] + values[i]/2,
                                f"{count:.0f}", ha="center", va="center",
                                fontsize=style.tick_size-1,
                                color="#202020")
            bottom += values
    ax.set_xticks(positions, [style.label(r) for r in RANKS], rotation=28)
    ax.set_ylim(0, 1.15 if annotate_molecules else 1.03)
    return _format_axis(ax, style=style, xlabel="", ylabel=style.label("fraction"))


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

    from matplotlib.patches import Patch
    hatches = ("", "///", "xx", "..", "++", "\\\\")
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
                          hatch=hatches[ci % len(hatches)], edgecolor="#E5E7EB",
                          linewidth=.45,
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
    species_legend = ax.legend(
        title="RNA species", frameon=False, loc="upper left",
        bbox_to_anchor=(1.01, 1), fontsize=style.legend_size,
    )
    ax.add_artist(species_legend)
    ax.legend(
        handles=[
            Patch(facecolor="#F1F4F7", edgecolor="#64748B",
                  hatch=hatches[i % len(hatches)],
                  label=f"E = {float(e):g}")
            for i, e in enumerate(conditions)
        ],
        title=style.label("E_att"), frameon=False, loc="upper left",
        bbox_to_anchor=(1.01, .55), fontsize=style.legend_size,
    )
    _format_axis(ax, style=style, xlabel="Ranked-cluster material",
                 ylabel=style.label("fraction"))
    return ax


def plot_largest_contact_composition(
    df: pd.DataFrame, ax: Axes, *, style: PlotStyle, x: str = "E_att",
    annotate_counts: bool = True, annotate_total: bool = True,
) -> Axes:
    """Thesis Fig. 17: contact classes inside the largest component."""
    kinds = ("cag10_cag10", "cag10_cag47", "cag47_cag47")
    columns = [
        f"largest_contacts_{kind}_fraction_inter_contacts_mean" for kind in kinds
    ]
    annotations = [
        f"largest_contacts_{kind}_n_inter_contacts_mean" for kind in kinds
    ] if annotate_counts else None
    plot_stacked_fractions(
        df, ax, columns=columns, x=x,
        annotation_columns=annotations, style=style,
        labels=["10–10", "10–47", "47–47"],
        colors=[style.contact_colors[k] for k in kinds],
    )
    if annotate_total:
        require_columns(df, [
            f"largest_contacts_{kind}_n_inter_contacts_mean" for kind in kinds
        ])
        for i, (_, row) in enumerate(df.sort_values(x).iterrows()):
            total = sum(row[f"largest_contacts_{k}_n_inter_contacts_mean"] for k in kinds)
            ax.text(i, 1.025, f"N={total:.0f}", ha="center", va="bottom",
                    fontsize=style.tick_size, clip_on=False)
        ax.set_ylim(0, 1.13)
    return ax


def plot_mfe_pairing_scatter(
    df: pd.DataFrame, ax: Axes, *,
    style: PlotStyle, x: str = "mfe_matches", y: str = "extra_pairs",
    attraction_order: Sequence | None = None,
) -> Axes:
    """Show replicate base-pair counts and the mean for each attraction."""
    from matplotlib.ticker import MaxNLocator

    require_columns(df, [x, y, "E_att"])
    strengths, _ = _categorical_axis(df, "E_att", attraction_order)
    palette = style.attraction_palette(strengths)
    for strength in strengths:
        group = df[df["E_att"] == strength].dropna(subset=[x, y])
        if group.empty:
            continue
        color = palette[strength]
        ax.scatter(group[x], group[y], s=30, color=color, alpha=.65,
                   edgecolor="#404040", linewidth=.45,
                   label=f"{strength:g}")
        mean_x, mean_y = group[[x, y]].mean()
        ax.scatter(mean_x, mean_y, s=90, marker="X", color=color,
                   edgecolor="#242424", linewidth=1.1, zorder=5)
        offset = (
            (5, 8) if strength == strengths[0]
            else (5, 23) if strength == strengths[1]
            else (-6, -13) if strength == strengths[3] and mean_x > 9
            else (-6, 9) if mean_x > 9
            else (6, 8)
        )
        ax.annotate(
            f"({mean_x:.1f}, {mean_y:.1f})",
            (mean_x, mean_y), xytext=offset,
            ha="right" if mean_x > 9 else "left",
            textcoords="offset points", fontsize=8, color="#303030",
        )
    ax.axhline(0, color="#53BCE7", linewidth=.9, linestyle="--")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    return _format_axis(ax, style=style, xlabel="MFE matches", ylabel="Extra pairs")


def plot_precision_recall(
    df: pd.DataFrame, ax: Axes, *, style: PlotStyle,
    color_by: str = "E_att", threshold: float = .8,
    attraction_order: Sequence | None = None,
) -> Axes:
    """Plot MFE recall against precision and mark the accuracy quadrants."""
    require_columns(df, ["precision", "recall", color_by])
    strengths, _ = _categorical_axis(df, color_by, attraction_order)
    palette = style.attraction_palette(strengths)
    for strength in strengths:
        group = df[df[color_by] == strength]
        ax.scatter(
            group["recall"], group["precision"], color=palette[strength],
            s=27, alpha=.7, edgecolor="#333333", linewidth=.4,
            label=f"{strength:g}",
        )
    for coordinate in ("vertical", "horizontal"):
        (ax.axvline if coordinate == "vertical" else ax.axhline)(
            threshold, linestyle="--", color="#53BCE7", linewidth=1,
        )
    ax.text(.05, .97, "partial\nbut specific", transform=ax.transAxes,
            va="top", ha="left", fontsize=8, color="#444444")
    ax.text(.97, .97, "complete\n+ specific", transform=ax.transAxes,
            va="top", ha="right", fontsize=8, color="#444444")
    ax.text(.05, .07, "missing\n+ noisy", transform=ax.transAxes,
            va="bottom", ha="left", fontsize=8, color="#444444")
    ax.text(.97, .07, "complete\nbut noisy", transform=ax.transAxes,
            va="bottom", ha="right", fontsize=8, color="#444444")
    ax.set(xlim=(0, 1.02), ylim=(0, 1.02))
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    return _format_axis(ax, style=style, xlabel="MFE recall",
                        ylabel="MFE precision")


def plot_rg_validation(
    summary: pd.DataFrame, ax: Axes, *, replicates: pd.DataFrame,
    style: PlotStyle, x: str = "E_att", x_order: Sequence | None = None,
    reference_nm: float | None = 1.47,
) -> Axes:
    """Plot per-condition Rg means, replicate values and percentile ranges."""
    require_columns(summary, [x, "Rg_nm_mean", "Rg_nm_p10", "Rg_nm_p90"])
    require_columns(replicates, [x, "Rg_nm"])
    strengths, _ = _categorical_axis(summary, x, x_order)
    positions = {value: i for i, value in enumerate(strengths)}
    palette = style.attraction_palette(strengths)
    rng = np.random.default_rng(42)

    for strength in strengths:
        raw = replicates.loc[replicates[x] == strength, "Rg_nm"].dropna()
        xpos = positions[strength]
        ax.scatter(
            xpos + rng.uniform(-.06, .06, len(raw)), raw,
            color=palette[strength], alpha=.55, s=20, zorder=3,
        )
        condition = summary.loc[summary[x] == strength]
        if condition.empty:
            continue
        row = condition.iloc[0]
        mean = float(row["Rg_nm_mean"])
        low, high = float(row["Rg_nm_p10"]), float(row["Rg_nm_p90"])
        ax.vlines(xpos, low, high, color="#9BD1ED", linewidth=6,
                  alpha=.5, zorder=2)
        ax.hlines(mean, xpos-.16, xpos+.16, color="black",
                  linewidth=2.3, zorder=4)
        ax.plot(xpos, mean, marker="o", color=palette[strength],
                markersize=6, zorder=5)
        ax.annotate(f"{mean:.2f}", (xpos, mean), xytext=(10, -2),
                    textcoords="offset points", va="center",
                    fontsize=8, color="#333333")

    if reference_nm is not None:
        ax.axhline(reference_nm, linestyle="--", linewidth=1.3,
                   color="#29B5E9")
        ax.text(.98, .96, f"ref Rg = {reference_nm:g}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8)
    _set_categorical_ticks(ax, strengths)
    return _format_axis(ax, style=style, xlabel=style.label(x),
                        ylabel="Radius of gyration")


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
