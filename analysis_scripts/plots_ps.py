import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from typing import Callable
import sys
from pathlib import Path
from typing import Any
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import Sequence
import pandas as pd
import numpy as np
import math
import os
from dataclasses import dataclass


# ============================================================================
# LABEL HANDLING
# ============================================================================

LABEL_MAP = {
    # ── Time axis ────────────────────────────────────────────────────────────
    "step":                 "Simulation Step",
    "frame":                "Frame",
    "time":                 "Time",
    "rna" : "RNA", # added column in processing of some dfs

    # ── Simulation parameters ─────────────────────────────────────────────
    "angle_k_ppp": r"$k_{\mathrm{bbb}}$",
    "E_att": r"$E_{\mathrm{att}}$",
    "density":              r"$\rho$",
    "molecule_file":        "RNA",

    # ── Logged simulation quantities ──────────────────────────────────────
    "ke_per_particle":      "KE/N",
    "E_pot":                "E_pot",
    "E_SQD":                "E_SQD",
    "E_bond":               "E_bond",
    "E_angle":              "E_angle",

    # ── Single-molecule conformation ──────────────────────────────────────
    "rg":                   "Rg",
    "end_to_end_distance":  "R",

    # ── Base-pair contacts ────────────────────────────────────────────────
    "particle_contacts":                    "N_bp",
    "distinct_molecule_pairs":              "N_mol-pair",
    "qualifying_molecule_pairs":            "N*_mol-pair",
    "inter_frac":                           "f_inter",
    "intra_frac":                           "f_intra",

    # ── Recurrence ────────────────────────────────────────────────────────
    # "contact_regions_particle":             "N_reg (exact)",
    # "contact_region_recurrence_particle":   "R_rec (exact)",
    # "contact_regions_fixed":                "N_reg (fixed bin)",
    # "contact_region_recurrence_fixed":      "R_rec (fixed bin)",
    # "contact_regions_repeat":               "N_reg (repeat unit)",
    # "contact_region_recurrence_repeat":     "R_rec (repeat unit)",
    # "contact_regions_repeat_window":        "N_reg (repeat window)",
    # "contact_region_recurrence_repeat_window": "R_rec (repeat window)",
    "molecule_contact_recurrence_window":     "recurrence (mol)",
    "particle_contact_recurrence_window":     "recurrence (particle)",

    # ── Cluster / phase fractions ─────────────────────────────────────────
    "mono_frac":                            "f_mono",
    "oligo_frac":                           "f_oligo",
    "drop_frac":                            "f_droplet",
    "largest_cluster":                      "largest cluster",
    "n_clusters":                           "N_clusters",
    "cluster_size": "Cluster size",
    "frac_mols": "Fraction of molecules",
    "n_clusters": "Number of clusters",
    "n_mols_in_clusters": "Molecules in clusters",
}

# ============================================================================
# Preparing framewise.csv for plotting
# ============================================================================

VALIDATION_ROOT = Path("/groups/goloborodko/projects/leona/validation")

def setup_validation_paths(sweep_dir_name: str) -> dict[str, Path]:
    """
    Create standard paths for loading paths of one validation sweep.
    """

    bulk_dir = VALIDATION_ROOT / "bulk"
    output_dir = bulk_dir / "output_dir"
    sweep_dir = output_dir / sweep_dir_name

    sys.path.extend([
        str(bulk_dir),
        str(VALIDATION_ROOT),
    ])

    return {
        "validation_dir": VALIDATION_ROOT,
        "bulk_dir": bulk_dir,
        "output_dir": output_dir,
        "sweep_dir": sweep_dir,
        "framewise_csv": sweep_dir / "analysis_framewise.csv",
        "parameters_txt": sweep_dir / "parameters.txt",
        "analysis_per_replicate_csv": sweep_dir / "analysis_per_replicate.csv",
        "error_rate": sweep_dir / "analysis_error_rate.csv",
        "clusters": sweep_dir / "analysis_clusters.csv",
        "graph": sweep_dir / "graph_framewise_summary.csv",
    }

def setup_npz_mp_paths(sweep_dir:str, run_dir_num:str, rep_num:int = 0) -> dict[str, Path]:
    """
    Create standard paths for loading pairing data (stored as npz with frame_indices, qi_{frame}, pi{frame}, distances, error_rates), 
    molecule lookup table mapping the molecule ID to particle indices and the respective trajectory.
    Inputs are sweep directory name (e.g. 41a_sweepx), number of the run_dir and number of replicate if present.
    """

    bulk_dir = VALIDATION_ROOT / "bulk"
    output_dir = bulk_dir / "output_dir"
    sweep_dir = output_dir / sweep_dir

    npz_mp_paths={}

    sys.path.extend([
        str(bulk_dir),
        str(VALIDATION_ROOT),
    ])
    
    for run_dir in sweep_dir.iterdir():
        if os.path.isdir(run_dir):
            if run_dir.name.startswith(f"{run_dir_num}_"):
                npz_mp_paths["run_dir"] = run_dir
                break

    
    npz_mp_paths["sweep_dir"] = sweep_dir
    npz_mp_paths["ml"] = f"{run_dir}/computed_metrics/pairing/molecule_lookup.npy"
    npz_mp_paths[f"traj_{rep_num}_npz"] = f"{run_dir}/computed_metrics/pairing/trajectory_{rep_num}_pairing.npz"
    npz_mp_paths["traj"] = f"{run_dir}/trajectory_{rep_num}.gsd"
    return npz_mp_paths



def get_unique_params_from_txt(path_to_parameters_txt: Path) -> dict[str, list[Any]]:
    """
    Read parameters.txt and return parameters with more than one unique value.

    Args:
        path_to_parameters_txt: Path to parameters.txt.

    Returns:
        Dictionary mapping varying parameter names to their unique values.
    """

    def parse_value(value: str) -> Any:
        try:
            return float(value)
        except ValueError:
            return value

    params: dict[str, set[Any]] = {}

    with path_to_parameters_txt.open("r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()

            if not parts:
                continue

            for i in range(0, len(parts), 2):
                key = parts[i].replace("--", "")
                value = parse_value(parts[i + 1])

                if key not in params:
                    params[key] = set()

                params[key].add(value)

    varying_params = {
        key: sorted(values)
        for key, values in params.items()
        if len(values) > 1
    }

    return varying_params

def framewise_steps_check(
    df: pd.DataFrame,
    params: dict[str, list[Any]],
    N_steps: int = 42_000_000,
    print_summary: bool = True,
) -> pd.DataFrame:
    """
    Return runs that did not reach the expected final step.

    Args:
        df: Framewise simulation DataFrame.
        params: Varying parameters from parameters.txt.
        N_steps: Expected final simulation step.
        print_summary: Whether to print a short summary.

    Returns:
        DataFrame with runs whose maximum step is below N_steps.
    """
    param_cols = list(params.keys())

    required_cols = ["run_id", "step", *param_cols]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    max_steps = (
        df.groupby("run_id", as_index=False)
        .agg(max_step=("step", "max"))
    )

    run_params = df[["run_id", *param_cols]].drop_duplicates()

    failed_runs = (
        max_steps
        .merge(run_params, on="run_id", how="left")
        .query("max_step < @N_steps")
        .assign(missing_steps=lambda x: N_steps - x["max_step"])
        .sort_values("max_step")
        .reset_index(drop=True)
    )

    if print_summary:
        n_runs = max_steps["run_id"].nunique()
        n_failed = failed_runs["run_id"].nunique()
        n_passed = n_runs - n_failed

        print(f"Expected final step: {N_steps}")
        print(f"Runs checked: {n_runs}")
        print(f"Runs reaching expected step: {n_passed}")
        print(f"Runs below expected step: {n_failed}")

        if n_failed > 0:
            print("\nFailed runs:")
            print(failed_runs)

    return failed_runs


def framewise_param_combis(
    df: pd.DataFrame,
    params: dict[str, list[Any]],
) -> pd.DataFrame:
    """
    Return one row per parameter combination with the number of replicate runs.

    Args:
        df: Framewise simulation DataFrame.
        params: Varying parameters from parameters.txt.

    Returns:
        DataFrame with parameter combinations and n_replicates.
    """
    param_cols = list(params.keys())

    if "run_id" not in df.columns:
        raise ValueError("DataFrame must contain a 'run_id' column.")

    missing_cols = [col for col in param_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing parameter columns in framewise DataFrame: {missing_cols}")

    param_combis = (
        df[param_cols + ["run_id"]]
        .drop_duplicates()
        .groupby(param_cols, as_index=False)
        .agg(n_replicates=("run_id", "nunique"))
        .sort_values(param_cols)
        .reset_index(drop=True)
    )

    return param_combis

# ============================================================================
# FILTERING (in practice based on value we want fixed)
# ============================================================================

def df_filter(
    df: pd.DataFrame,
    fixed: dict[str, Any | list[Any]] | None = None,
) -> pd.DataFrame:
    """
    Filter a DataFrame using fixed parameter values.

    Args:
        df: Input DataFrame.
        fixed: Dictionary with column names and allowed value(s).

    Returns:
        Filtered DataFrame.
    """
    if not fixed:
        return df.copy()
    mask = np.ones(len(df), dtype=bool)
    for col, val in fixed.items():
        if col not in df.columns:
            raise ValueError(f"Column not found: {col}")
        vals = val if isinstance(val, (list, tuple, set)) else [val]
        mask &= df[col].isin(vals).values
    return df.loc[mask].copy()

# Filtering final values

def final_fraction_df(
    df: pd.DataFrame,
    fraction: float = 0.2,
    run_col: str = "run_id",
    sort_col: str = "step",
) -> pd.DataFrame:
    """
    Return the final fraction of rows from each run.

    Args:
        df: Framewise DataFrame.
        fraction: Fraction of final rows to keep, e.g. 0.2 for final 20%.
        run_col: Column identifying replicate runs.
        sort_col: Column used to order frames.

    Returns:
        DataFrame containing only the final fraction of each run.
    """
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be between 0 and 1.")

    if run_col not in df.columns:
        raise ValueError(f"Missing run column: {run_col}")

    if sort_col not in df.columns:
        raise ValueError(f"Missing sort column: {sort_col}")

    final_parts = []

    for _, run_df in df.groupby(run_col):
        run_df = run_df.sort_values(sort_col)
        n_keep = max(1, math.ceil(len(run_df) * fraction))
        final_parts.append(run_df.tail(n_keep))

    return pd.concat(final_parts, ignore_index=True)

def melt_metrics(df, metrics, id_vars=("frame", "run_id", "molecule_file")):
    return df.melt(
        id_vars=list(id_vars),
        value_vars=metrics,
        var_name="metric",
        value_name="value",
    )

# ============================================================================
# BINNING (cluster df)
# ============================================================================

CLUSTER_ID_COLS = [
    "run_id",
    "folder_name",
    "traj_replicate",
    "frame",
    "step",
    "E_att",
    "angle_k_ppp",
    "density",
    "min_contacts",
    "molecule_file"
]


def make_binned_cluster_df(
    cdf: pd.DataFrame,
    bin_edges: list[int] | None = None,
    bin_labels: list[str] | None = None,
    id_cols: list[str] | None = None,
) -> pd.DataFrame:
    if bin_edges is None:
        bin_edges = [0, 1, 2, 3, 9, 19, 49, 100]

    if bin_labels is None:
        bin_labels = ["mono", "di", "tri", "4-9", "10-19", "20-49", "50+"]

    if id_cols is None:
        id_cols = CLUSTER_ID_COLS

    required_cols = [*id_cols, "cluster_size", "frac_mols"]
    missing_cols = [col for col in required_cols if col not in cdf.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = cdf.copy()

    df["cluster_bin"] = pd.cut(
        df["cluster_size"],
        bins=bin_edges,
        labels=bin_labels,
    )

    if df["cluster_bin"].isna().any():
        bad_sizes = sorted(df.loc[df["cluster_bin"].isna(), "cluster_size"].unique())
        raise ValueError(f"Some cluster sizes were not assigned to bins: {bad_sizes}")

    binned = (
        df.groupby(
            [*id_cols, "cluster_bin"],
            observed=True,
            as_index=False,
        )
        .agg(frac_mols=("frac_mols", "sum"))
    )

    wide = (
        binned.pivot_table(
            index=id_cols,
            columns="cluster_bin",
            values="frac_mols",
            fill_value=0,
            observed=True,
        )
        .reset_index()
    )

    for label in bin_labels:
        if label not in wide.columns:
            wide[label] = 0.0

    wide = wide[id_cols + bin_labels]

    frac_sum = wide[bin_labels].sum(axis=1)
    if not np.allclose(frac_sum, 1.0):
        bad = frac_sum.loc[~np.isclose(frac_sum, 1.0)]
        raise ValueError(
            f"Binned fractions do not sum to 1 for {len(bad)} rows. "
            f"Example sums: {bad.head().tolist()}"
        )

    return wide

def plot_stacked_replicate_bars_fraction(
    df,
    ax,
    y_cols,
    x_col,
    group_col,
    style,
    label_map=LABEL_MAP,
):
    required_cols = [x_col, group_col] + y_cols

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    x_values = sorted(df[x_col].dropna().unique())
    x_map = {value: i for i, value in enumerate(x_values)}

    groups = sorted(df[group_col].dropna().unique())
    n_groups = len(groups)

    bar_width = 0.75 / max(n_groups, 1)

    for j, group_value in enumerate(groups):
        d_group = df[df[group_col] == group_value].sort_values(x_col)

        x = d_group[x_col].map(x_map).to_numpy(dtype=float)
        x = x - 0.375 + (j + 0.5) * bar_width

        bottom = np.zeros(len(d_group))

        for i, col in enumerate(y_cols):
            label = get_label(col, label_map) if j == 0 else None

            ax.bar(
                x,
                d_group[col].to_numpy(),
                bottom=bottom,
                width=bar_width,
                color=style.stack_colors[i],
                alpha=style.stack_alpha,
                label=label,
                linewidth=0,
            )

            bottom += d_group[col].to_numpy()

    ax.set_xticks(range(len(x_values)))
    ax.set_xticklabels([
    f"{x:g}" if isinstance(x, (int, float, np.integer, np.floating)) else str(x)
    for x in x_values
    ])

    ax.set_xlabel(get_label(x_col, label_map))
    ax.set_ylabel("Fraction")
    ax.set_ylim(0, 1)

    if style.show_grid:
        ax.grid(True, axis="y", alpha=style.grid_alpha)

    return ax

def get_label(col: str, label_map: dict[str, str] | None = LABEL_MAP) -> str:
    """
    Return a display label for a column name.
    """
    if label_map is None:
        return col

    return label_map.get(col, col)

# ============================================================================
# PLOTS
# ============================================================================

STATE_BIN_LABELS = [
    "mono",
    "di",
    "tri",
    "4-9",
    "10-49",
    "50+",
]

STATE_BIN_EDGES = [0, 1, 2, 3, 9, 49, np.inf]

STATE_BIN_COLORS = (
    "#2166ac",  # mono
    "#67a9cf",  # di
    "#d1e5f0",  # tri
    "#fff176",  # 4-9
    "#f4a582",  # 10-49
    "#b2182b",  # 50+
)


TIME_BIN_LABELS = [
    "mono",
    "di",
    "tri",
    "4-9",
    "10-19",
    "20-49",
    "50+",
]

TIME_BIN_EDGES = [0, 1, 2, 3, 9, 19, 49, np.inf]

TIME_BIN_COLORS = (
    "#2166ac",  # mono
    "#67a9cf",  # di
    "#d1e5f0",  # tri
    "#fff176",  # 4-9
    "#f4a582",  # 10-19
    "#d6604d",  # 20-49
    "#b2182b",  # 50+
)


@dataclass
class PlotStyle:
    # Global
    cmap: str = "YlOrBr"
    show_grid: bool = True
    grid_alpha: float = 0.25
    legend_loc: str = "best"

    # Text
    font_size: float = 10
    title_size: float = 14
    panel_title_size: float = 12
    label_size: float = 11
    tick_size: float = 10
    legend_size: float = 10
    legend_title_size: float = 10

    # Scatter
    alpha: float = 0.4
    point_size: float = 8
    jitter: float = 0.04

    # Stacked area
    stack_alpha: float = 0.25
    stack_colors: tuple[str, ...] = TIME_BIN_COLORS

    # Lines + replicate bands
    mean_lw: float = 2.0
    rep_lw: float = 0.8
    rep_alpha: float = 0.25
    band_alpha: float = 0.2

    def mpl_rc(self) -> dict[str, Any]:
        return {
            "font.size": self.font_size,
            "axes.titlesize": self.panel_title_size,
            "axes.labelsize": self.label_size,
            "xtick.labelsize": self.tick_size,
            "ytick.labelsize": self.tick_size,
            "legend.fontsize": self.legend_size,
            "legend.title_fontsize": self.legend_title_size,
            "figure.titlesize": self.title_size,
        }

def make_panel_plot_fn(
    plot_fn: Callable[..., Any],
    handles: list[Any] | None = None,
    handle_index: int | None = None,
    **plot_kwargs: Any,
) -> Callable[[pd.DataFrame, plt.Axes], None]:
    """
    Wrap a single-panel plotting function so it can be used by plot_panels_by_group.

    Args:
        plot_fn: Plotting function that takes df and ax.
        handles: Optional list for storing returned plot objects, e.g. scatter handles.
        handle_index: Which item from the return value to store.
            Example: plot_scatter_summary returns (ax, sc), so handle_index=1 stores sc.
        **plot_kwargs: Fixed keyword arguments passed to plot_fn.

    Returns:
        Function with signature (d, ax) -> None.
    """
    def panel_plot_fn(d: pd.DataFrame, ax: plt.Axes) -> None:
        result = plot_fn(
            df=d,
            ax=ax,
            **plot_kwargs,
        )

        if handles is not None and handle_index is not None:
            handles.append(result[handle_index])

    return panel_plot_fn

def plot_scatter_summary(
    df: pd.DataFrame,
    ax: plt.Axes,
    x_col: str,
    y_col: str,
    color_col: str,
    style: PlotStyle,
    label_map: dict[str, str] | None = LABEL_MAP,
    vmin: float | None = None,
    vmax: float | None = None,
    run_col: str | None = "run_id",
    show_n: bool = True,
) -> tuple[plt.Axes, PathCollection]:
    """
    Plot framewise observations as a pooled scatter plot.

    Colors map color_col values via style.cmap. If vmin/vmax are given,
    the colormap is normalized globally so multiple panels share a meaningful
    colorbar. If omitted, matplotlib auto-scales per panel.
    """
    x_values = sorted(df[x_col].dropna().unique())
    x_map = {value: i for i, value in enumerate(x_values)}
    x_pos = df[x_col].map(x_map).astype(float)

    if style.jitter > 0:
        rng = np.random.default_rng(42)
        x_plot = x_pos + rng.normal(0, style.jitter, size=len(df))
    else:
        x_plot = x_pos

    sc = ax.scatter(
        x_plot,
        df[y_col],
        c=df[color_col],          # raw values, not normalized
        cmap=style.cmap,
        vmin=vmin,                # global min if provided
        vmax=vmax,                # global max if provided
        alpha=style.alpha,
        s=style.point_size,
        linewidths=0,
    )

    ax.set_xticks(range(len(x_values)))
    ax.set_xticklabels(x_values)
    ax.set_xlabel(get_label(x_col, label_map))
    ax.set_ylabel(get_label(y_col, label_map))

    if style.show_grid:
        ax.grid(True, alpha=style.grid_alpha)
    if show_n and run_col is not None and run_col in df.columns:
        n_per_x = df.groupby(x_col)[run_col].nunique()
        y_top = df[y_col].max()
        y_pad = 0.05 * (df[y_col].max() - df[y_col].min())
        for x_val, n in n_per_x.items():
            if x_val in x_map:
                ax.text(
                    x_map[x_val], y_top + y_pad,
                    f"n={n}",
                    ha="center", va="bottom",
                    fontsize=8, color="gray",
                )
    return ax, sc

def capture_handle(plot_fn: Callable[[pd.DataFrame, plt.Axes], None], handles:list, handle_idx:int, **kwargs):
    def wrapped(df, ax):
        result = plot_fn(df, ax)
        handles.append(result[handle_idx])
    return wrapped

def plot_box_summary(
    df: pd.DataFrame,
    ax: plt.Axes,
    x_col: str,
    y_col: str,
    style: PlotStyle,
    label_map: dict[str, str] | None = LABEL_MAP,
) -> plt.Axes:
    """
    Boxplot pooling all rows passed to this function.

    If df contains multiple replicates, replicates are pooled.
    If df was pre-filtered to the final fraction of each run, the boxes show
    the distribution of final-window framewise values.

    The x-axis is treated categorically, so numeric values like 1, 2, and 32
    are evenly spaced.
    """
    x_values = sorted(df[x_col].dropna().unique())

    groups = [
        df.loc[df[x_col] == value, y_col].dropna()
        for value in x_values
    ]

    ax.boxplot(
        groups,
        positions=range(len(x_values)),
        widths=0.6,
        patch_artist=True,
        showfliers=True,
    )

    ax.set_xticks(range(len(x_values)))
    ax.set_xticklabels(x_values)

    ax.set_xlabel(get_label(x_col, label_map))
    ax.set_ylabel(get_label(y_col, label_map))

    if style.show_grid:
        ax.grid(True, axis="y", alpha=style.grid_alpha)

    return ax

def plot_violin_by_group(
    df: pd.DataFrame,
    ax: plt.Axes,
    x_col: str,
    y_col: str,
    style: PlotStyle,
    color_map: dict[Any, str] | None = None,
    label_map: dict[str, str] | None = None,
    show_points: bool = True,
    show_means: bool = True,
    point_jitter: bool = True,
) -> plt.Axes:
    """
    Draw violin plots of y_col grouped by x_col on ax.

    Args:
        df: Long-format DataFrame. Each row contributes one observation.
        ax: Matplotlib axis to draw on.
        x_col: Categorical column for the x-axis (one violin per unique value).
        y_col: Numeric column whose distribution is shown.
        style: PlotStyle.
        color_map: Optional mapping x_col value -> color.
        label_map: Optional pretty-name mapping for axis labels and tick labels.
        show_points: If True, overlay individual data points on each violin.
        show_means: If True, mark the mean with a horizontal tick inside each violin.
        point_jitter: If True, jitter overlay points horizontally to reduce overlap.

    Returns:
        The same ax.
    """
    for col in (x_col, y_col):
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    df = df.dropna(subset=[x_col, y_col])
    categories = sorted(df[x_col].unique())
    positions = np.arange(len(categories))

    # Data per category
    data = [df.loc[df[x_col] == c, y_col].values for c in categories]

    # Color resolution
    default_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = [
        (color_map[c] if color_map and c in color_map
         else default_cycle[i % len(default_cycle)])
        for i, c in enumerate(categories)
    ]

    # Draw violins
    parts = ax.violinplot(
        data, positions=positions,
        showmeans=False, showmedians=False, showextrema=False,
        widths=0.8,
    )
    for body, c in zip(parts["bodies"], colors):
        body.set_facecolor(c)
        body.set_edgecolor(c)
        body.set_alpha(style.band_alpha + 0.2)   # a bit more opaque than line bands

    # Mean ticks
    if show_means:
        for pos, vals, c in zip(positions, data, colors):
            if len(vals) == 0:
                continue
            ax.hlines(
                np.mean(vals), pos - 0.25, pos + 0.25,
                color=c, lw=style.mean_lw,
            )

    # Overlay individual points
    if show_points:
        for pos, vals, c in zip(positions, data, colors):
            if len(vals) == 0:
                continue
            if point_jitter:
                x_jit = pos + np.random.uniform(-style.jitter, style.jitter, size=len(vals))
            else:
                x_jit = np.full_like(vals, pos, dtype=float)
            ax.scatter(
                x_jit, vals,
                s=style.point_size, color=c, alpha=style.alpha,
                edgecolors="none",
            )

    # Ticks and labels
    tick_labels = [
        (label_map.get(str(c), str(c)) if label_map else str(c))
        for c in categories
    ]
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel(get_label(x_col, label_map))
    ax.set_ylabel(get_label(y_col, label_map))

    if style.show_grid:
        ax.grid(True, axis="y", alpha=style.grid_alpha)

    return ax

def plot_stacked_replicates_fraction(
    df: pd.DataFrame,
    ax: plt.Axes,
    y_cols: list[str],
    x_col: str,
    group_col: str,
    style: PlotStyle,
    label_map: dict[str, str] | None =LABEL_MAP,
) -> plt.Axes:
    """
    Plot stacked fraction trajectories for multiple replicates.

    Each replicate is drawn as a semi-transparent stackplot. Where replicates
    agree, the stacked areas overlap strongly. Where replicates diverge, the
    edges become more spread out.

    This function does not filter or create panels. It assumes df has already
    been prepared by the caller or by plot_panels_by_group().
    """
    required_cols = [x_col, group_col] + y_cols

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    x_values = sorted(df[x_col].dropna().unique())
    x_map = {value: i for i, value in enumerate(x_values)}

    replicates = sorted(df[group_col].dropna().unique())

    for i, rep in enumerate(replicates):
        df_rep = (
            df.loc[df[group_col] == rep]
            .sort_values(x_col)
        )

        x = df_rep[x_col].map(x_map).to_numpy()

        labels = (
            [get_label(col, label_map) for col in y_cols]
            if i == 0
            else [None] * len(y_cols)
        )

        ax.stackplot(
            x,
            *[df_rep[col] for col in y_cols],
            labels=labels,
            alpha=style.stack_alpha,
            colors=style.stack_colors[:len(y_cols)],
        )

    ax.set_xticks(range(len(x_values)))
    ax.set_xticklabels([f"{x:g}" for x in x_values])

    ax.set_xlabel(get_label(x_col, label_map))
    ax.set_ylabel("Fraction")
    ax.set_ylim(0, 1)

    if style.show_grid:
        ax.grid(True, alpha=style.grid_alpha)

    # ax.legend(
    #     title=f"n = {len(replicates)}",
    #     loc="upper right",
    # )
    print(f"Replicates n = {len(replicates)}")
    return ax


def plot_stacked_replicates_over_time(
    df: pd.DataFrame,
    ax: plt.Axes,
    y_cols: list[str],
    x_col: str,
    group_col: str,
    style: PlotStyle,
    x_scale: float = 1e6,
    label_map: dict[str, str] | None = LABEL_MAP,
) -> plt.Axes:
    required_cols = [x_col, group_col, *y_cols]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    for i, (_, d_rep) in enumerate(df.groupby(group_col)):
        d_rep = d_rep.sort_values(x_col)

        labels = y_cols if i == 0 else [None] * len(y_cols)

        ax.stackplot(
            d_rep[x_col] / x_scale,
            *[d_rep[col] for col in y_cols],
            colors=style.stack_colors[:len(y_cols)],
            alpha=style.stack_alpha,
            labels=labels,
        )

    ax.set_xlabel("Simulation step (×10⁶)")
    ax.set_ylabel("Fraction")
    ax.set_ylim(0, 1)

    if style.show_grid:
        ax.grid(True, alpha=style.grid_alpha)

    return ax

# LINE PLOT

def plot_replicate_band(
    df: pd.DataFrame,
    ax: plt.Axes,
    x_col: str,
    y_col: str,
    style: PlotStyle,
    run_col: str = "run_id",
    color: str = "C0",
    label: str | None = None,
    band_mode: str = "replicates",   # "replicates" | "std" | "iqr" | "none"
    smoothing_window: int | None = None,
) -> plt.Axes:
    """
    Draw one group's mean trajectory with optional replicate band on ax.

    If smoothing_window is given, applies a centered rolling mean per replicate
    before aggregating. Useful for noisy trajectories like cluster counts.

    Args:
        df: Long-format DataFrame containing one group.
        ax: Matplotlib axis to draw on.
        x_col: Column for the x-axis.
        y_col: Column for the y-axis.
        style: PlotStyle.
        run_col: Column identifying replicates.
        color: Color for the mean line and band.
        label: Legend label for the mean line.
        band_mode: How to visualize replicate spread.
        smoothing_window: If not None, apply rolling mean of this window per
            replicate before plotting.

    Returns:
        The same ax.
    """
    if band_mode not in {"replicates", "std", "iqr", "none"}:
        raise ValueError(f"Unknown band_mode: {band_mode}")

    for col in (x_col, y_col, run_col):
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    # Optional smoothing — applied per replicate so runs don't bleed into each other
    if smoothing_window is not None and smoothing_window > 1:
        df = df.sort_values([run_col, x_col]).copy()
        df[y_col] = (
            df.groupby(run_col)[y_col]
              .transform(lambda s: s.rolling(smoothing_window, min_periods=1, center=True).mean())
        )

    # Replicate cloud
    if band_mode == "replicates":
        for _, rep_df in df.groupby(run_col):
            rep_df = rep_df.sort_values(x_col)
            ax.plot(
                rep_df[x_col], rep_df[y_col],
                color=color, lw=style.rep_lw, alpha=style.rep_alpha,
            )

    # Aggregate across replicates at each x
    grouped = df.groupby(x_col)[y_col]
    agg = pd.DataFrame({
        "mean": grouped.mean(),
        "std":  grouped.std(),
        "q25":  grouped.quantile(0.25),
        "q75":  grouped.quantile(0.75),
    }).sort_index()

    # Shaded band variants
    if band_mode == "std":
        ax.fill_between(
            agg.index,
            agg["mean"] - agg["std"],
            agg["mean"] + agg["std"],
            color=color, alpha=style.band_alpha, linewidth=0,
        )
    elif band_mode == "iqr":
        ax.fill_between(
            agg.index, agg["q25"], agg["q75"],
            color=color, alpha=style.band_alpha, linewidth=0,
        )

    # Bold mean on top
    ax.plot(
        agg.index, agg["mean"],
        color=color, lw=style.mean_lw, label=label,
    )

    return ax

def plot_lines_by_group(
    df: pd.DataFrame,
    ax: plt.Axes,
    x_col: str,
    y_col: str,
    group_col: str,
    style: PlotStyle,
    run_col: str = "run_id",
    color_map: dict[Any, str] | None = None,
    label_map: dict[str, str] | None = None,
    band_mode: str = "replicates",
    smoothing_window: int | None = None,
    show_n: bool = True,
    legend_loc: str | None = None,
) -> plt.Axes:
    """
    Draw one mean+band per group_col value on the same axis.

    For each unique value in group_col, calls plot_replicate_band with a
    distinct color. The legend optionally annotates the number of replicates
    per group (e.g. "unscrambled CAG, n=5").

    Args:
        df: Long-format DataFrame containing all groups to overlay.
        ax: Matplotlib axis to draw on.
        x_col: Column for the x-axis.
        y_col: Column for the y-axis.
        group_col: Column whose unique values become separate colored lines.
        style: PlotStyle.
        run_col: Column identifying replicates within each group.
        color_map: Mapping from group_col value to matplotlib color. If None,
            colors are taken from the default cycle.
        label_map: Mapping from raw values to display labels.
        band_mode: Passed to plot_replicate_band.
        show_n: If True, append ", n=K" to each legend label.
        legend_loc: Override style.legend_loc.

    Returns:
        The same ax.
    """
    required = [x_col, y_col, group_col, run_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    group_values = sorted(df[group_col].dropna().unique())
    default_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, gv in enumerate(group_values):
        sub = df[df[group_col] == gv]

        # Color resolution
        if color_map is not None and gv in color_map:
            color = color_map[gv]
        else:
            color = default_cycle[i % len(default_cycle)]

        # Label resolution
        base_label = (label_map.get(gv, str(gv))
                      if label_map is not None else str(gv))
        if show_n:
            n_reps = sub[run_col].nunique()
            label = f"{base_label}, n={n_reps}"
        else:
            label = base_label

        plot_replicate_band(
            df=sub, ax=ax,
            x_col=x_col, y_col=y_col,
            style=style,
            run_col=run_col,
            color=color, label=label,
            band_mode=band_mode,
            smoothing_window=smoothing_window,  
        )

    ax.set_xlabel(get_label(x_col, label_map))
    ax.set_ylabel(get_label(y_col, label_map))

    if style.show_grid:
        ax.grid(True, alpha=style.grid_alpha)

    ax.legend(loc=legend_loc or style.legend_loc)

    return ax

# PANEL PLOTTING

def make_panel_grid(
    n_panels: int,
    n_cols: int = 3,
    figsize_per_panel: tuple[float, float] = (5.0, 4.0),
    sharex: bool = False,
    sharey: bool = True,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Create a figure with n_panels axes laid out in a grid. Hides extras."""
    n_rows = math.ceil(n_panels / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        sharex=sharex, sharey=sharey,
    )
    axes = np.atleast_1d(axes).ravel().tolist()
    for ax in axes[n_panels:]:
        ax.set_visible(False)
    return fig, axes[:n_panels]


def prepare_panel_df(
    df: pd.DataFrame,
    fixed: dict | None = None,
    final_fraction: float | None = None,
    run_col: str = "run_id",
    sort_col: str = "step",
) -> pd.DataFrame:
    """Apply standard pre-plot filters: fixed-value subset, then final-fraction window."""
    df = df_filter(df, fixed)
    if final_fraction is not None:
        df = final_fraction_df(df, fraction=final_fraction,
                               run_col=run_col, sort_col=sort_col)
    return df


def make_panel_title(
    group_values: tuple | Any,
    panel_cols: list[str],
    label_map: dict[str, str] | None = None,
) -> str:
    """Format a panel title from group key values."""
    if not isinstance(group_values, tuple):
        group_values = (group_values,)
    return ", ".join(
        f"{get_label(col, label_map)} = {val}"
        for col, val in zip(panel_cols, group_values)
    )


def plot_panels_by_group(
    df: pd.DataFrame,
    panel_cols: list[str],
    plot_fn: Callable[[pd.DataFrame, plt.Axes], None],
    n_cols: int = 3,
    label_map: dict[str, str] | None = None,
    suptitle: str | None = None,
    figsize_per_panel: tuple[float, float] = (5.0, 4.0),
    sharex: bool = False,
    sharey: bool = True,
    fixed: dict | None = None,
    final_fraction: float | None = None,
    run_col: str = "run_id",
    sort_col: str = "step",
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Multi-panel figure, one panel per unique combo of panel_cols."""
    df = prepare_panel_df(df, fixed, final_fraction, run_col, sort_col)
    grouped = list(df.groupby(panel_cols, sort=True))

    fig, axes = make_panel_grid(
        n_panels=len(grouped), n_cols=n_cols,
        figsize_per_panel=figsize_per_panel,
        sharex=sharex, sharey=sharey,
    )

    for ax, (group_values, d_panel) in zip(axes, grouped):
        plot_fn(d_panel, ax)
        ax.set_title(make_panel_title(group_values, panel_cols, label_map))
    title_parts = []
    if suptitle:
        title_parts.append(suptitle)
    if final_fraction is not None:
        pct = int(round(final_fraction * 100))
        title_parts.append(f"last {pct}% of frames")

    if title_parts:
        fig.suptitle(" — ".join(title_parts))
    fig.tight_layout()
    return fig, axes


# # Equilibration plots: first individual plotting functions

# # Plot one timeseries
# def plot_one_timeseries(df:pd.DataFrame, ax, y_col:str,  x_col:str = "step", label=None) -> Axes:
#     plot_df = df.copy().sort_values(x_col) # sorted copy by x axis
#     ax.plot(plot_df[x_col], plot_df[y_col], label= label)
#     ax.set_xlabel(x_col)
#     ax.set_ylabel(y_col)
#     return ax

# # somehow we also need optional saving for all plots
# # Plot lines by group

# def plot_lines_by_group(
#     df: pd.DataFrame,
#     ax,
#     y_col: str,
#     group_col: str,
#     x_col: str = "step",
# ) -> Axes:
#     for group_value, group_df in df.groupby(group_col):
#         plot_one_timeseries(
#             df=group_df,
#             ax=ax,
#             y_col=y_col,
#             x_col=x_col,
#             label=f"{group_col}={group_value}",
#         )

#     ax.legend()
#     return ax

# def make_label(row: pd.Series, cols: Sequence[str]) -> str:
#     """
#     Make a readable label from one or more columns.
#     """
#     return ", ".join(f"{col}={row[col]}" for col in cols)

# def plot_panels_by_group(
#     df: pd.DataFrame,
#     y_col: str,
#     panel_cols: str | Sequence[str],
#     line_col: str,
#     x_col: str = "step",
#     n_cols: int = 2,
#     figsize_per_panel: tuple[float, float] = (6, 4),
# ) -> tuple[Figure, np.ndarray]:
#     """
#     Plot one panel per parameter combination, with one line per line_col value.

#     Args:
#         df: Input DataFrame.
#         y_col: Column to plot on the y-axis.
#         panel_cols: Column(s) used to create separate subplots.
#         line_col: Column used to create separate lines in each subplot.
#         x_col: Column to plot on the x-axis.
#         n_cols: Number of subplot columns.
#         figsize_per_panel: Size of each subplot.

#     Returns:
#         Figure and axes.
#     """
#     if isinstance(panel_cols, str):
#         panel_cols = [panel_cols]
#     else:
#         panel_cols = list(panel_cols)

#     required_cols = [x_col, y_col, line_col, *panel_cols]
#     missing_cols = [col for col in required_cols if col not in df.columns]

#     if missing_cols:
#         raise ValueError(f"Missing columns: {missing_cols}")

#     panel_combis = (
#         df[panel_cols]
#         .drop_duplicates()
#         .sort_values(panel_cols)
#         .reset_index(drop=True)
#     )

#     n_panels = len(panel_combis)
#     n_rows = math.ceil(n_panels / n_cols)

#     fig, axes = plt.subplots(
#         n_rows,
#         n_cols,
#         figsize=(
#             figsize_per_panel[0] * n_cols,
#             figsize_per_panel[1] * n_rows,
#         ),
#         sharex=True,
#         sharey=True,
#     )

#     axes = np.array(axes).reshape(-1)

#     for ax, (_, panel_values) in zip(axes, panel_combis.iterrows()):
#         panel_df = df.copy()

#         for col in panel_cols:
#             panel_df = panel_df[panel_df[col] == panel_values[col]]

#         plot_lines_by_group(
#             df=panel_df,
#             ax=ax,
#             y_col=y_col,
#             group_col=line_col,
#             x_col=x_col,
#         )

#         ax.set_title(make_label(panel_values, panel_cols))

#     for ax in axes[n_panels:]:
#         ax.set_visible(False)

#     fig.suptitle(f"{y_col} over {x_col}")
#     plt.tight_layout()

#     return fig, axes

# ============================================================================
# CUSTOM PLOT CONFIG
# ============================================================================
# Context manager (option 1)
# with plt.rc_context(PAPER_STYLE):
#     plot_panels_by_group(...)

# # style setup
# PAPER_STYLE = ...
# LABELS = ...
# COLORS = ...

# # helper functions
# get_label(...)
# make_label(...)

# # plotting functions
# plot_one_timeseries(...)
# plot_lines_by_group(...)
# plot_panels_by_group(...)

# Label control
# ax.set_xlabel(get_label(x_col, label_map))
# label=make_group_label(group_col, group_value, label_map, value_map)

# label_map = {
#     "temperature": "Temperature",
# }

# value_map = {
#     "rep1": "Replicate 1",
#     "rep2": "Replicate 2",
# }

# # COLORS
# color_map = {
#     "rep1": "...",
#     "rep2": "...",
#     "rep3": "...",
# }
# color = color_map.get(group_value)