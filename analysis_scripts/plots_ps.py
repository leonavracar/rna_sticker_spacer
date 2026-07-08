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
    stack_alpha: float = 0.8
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
    # Boxplots
    box_facecolor: str = "white"
    box_edgecolor: str = "black"
    box_alpha: float = 0.85
    box_lw: float = 1.0
    box_median_color: str = "black"
    box_median_lw: float = 1.5
    box_whisker_color: str = "black"
    box_point_color: str = "#666666"
    box_point_size: float = 28
    box_point_alpha: float = 0.8
    box_point_jitter: float = 0.06
    box_show_points: bool = True
    box_show_fliers: bool = False


# ============================================================================
# HELPER
# ============================================================================

def save_fig(fig, name, *, save=False, outdir=None, params=None, ext="png", dpi=200):
    """Show + optionally save. Filename = name + sorted params."""
    if save:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        parts = [name]
        if params:
            parts += [f"{k}={v}" for k, v in sorted(params.items())]

        path = outdir / (("__".join(parts)) + f".{ext}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"saved: {path}")

    plt.show()

# ============================================================================
# LABEL HANDLING
# ============================================================================


LABEL_MAP = {
    # ── Time axis ────────────────────────────────────────────────────────────
    "step": "Simulation Step",
    "frame": "Frame",
    "time": "Time",
    "rna": "RNA",

    # ── Simulation parameters ───────────────────────────────────────────────
    "angle_k_ppp": r"$k_{\mathrm{bbb}}$",
    "E_att": r"$E_{\mathrm{att}}$",
    "density": r"$\rho$",
    "molecule_file": "RNA",

    # ── Logged simulation quantities ────────────────────────────────────────
    "ke_per_particle": "KE/N",
    "E_pot": "E_pot",
    "E_SQD": "E_SQD",
    "E_bond": "E_bond",
    "E_angle": "E_angle",

    # ── Single-molecule conformation ────────────────────────────────────────
    "rg": "Rg",
    "end_to_end_distance": "R",

    # ── Base-pair contacts ──────────────────────────────────────────────────
    "particle_contacts": "N_bp",
    "distinct_molecule_pairs": "N_mol-pair",
    "qualifying_molecule_pairs": "N*_mol-pair",
    "inter_frac": "f_inter",
    "intra_frac": "f_intra",

    # ── Recurrence ──────────────────────────────────────────────────────────
    "molecule_contact_recurrence_window": "recurrence (mol)",
    "particle_contact_recurrence_window": "recurrence (particle)",

    # ── Cluster / phase fractions ───────────────────────────────────────────
    "mono_frac": "f_mono",
    "oligo_frac": "f_oligo",
    "drop_frac": "f_droplet",
    "largest_cluster": "largest cluster size (mol)",
    "n_clusters": "Number of clusters",
    "cluster_size": "Cluster size",
    "frac_mols": "Fraction of molecules",
    "n_mols_in_clusters": "Molecules in clusters",
    "sequence_class": "sequence class",
    "edge_strength_q90_qualifying": r"$q_{90}$ edge strength",
    "edge_density_qualifying": "Edge density",
    "mean_degree_qualifying": "Mean degree",

    # ── Cluster material percentage bins ─────────────────────────────────────
    "pct_mono": "mono",
    "pct_1-2": "1–2%",
    "pct_2-3": "2–3%",
    "pct_3-9": "3–9%",
    "pct_10-19": "10–19%",
    "pct_20-49": "20–49%",
    "pct_50+": "≥50%",

    # ── Cluster molecule-count bins ──────────────────────────────────────────
    "mol_mono": "mono",
    "mol_di": "di",
    "mol_tri": "tri",
    "mol_4-9": "4–9",
    "mol_10-19": "10–19",
    "mol_20-49": "20–49",
    "mol_50+": "50+",
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
        "clusters": sweep_dir / "analysis_clusters.csv",
        "graph": sweep_dir / "graph_framewise_summary.csv",
        "clusters_mix": sweep_dir / "analysis_cluster_resolved.csv",
        "binned_clusters": sweep_dir / "analysis_cluster_binned.csv"

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
    #"molecule_file"
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
    # missing_cols = [col for col in required_cols if col not in cdf.columns]

    # if missing_cols:
    #     raise ValueError(f"Missing required columns: {missing_cols}")

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


def plot_ranked_cluster_bars(
    df,
    ax,
    *,
    x_col,
    y_cols,
    mol_cols,
    colors,
    x_order,
    group_col=None,
    annotate_mols=False,
):
    d = df.copy()
    d[x_col] = pd.Categorical(d[x_col], categories=x_order, ordered=True)
    d = d.sort_values([x_col, group_col] if group_col else [x_col])

    x_base = np.arange(len(x_order)) * 1.35

    if group_col is None:
        bar_groups = [(x, None, g) for x, g in d.groupby(x_col, observed=True)]
        width = 0.75
    else:
        max_n = d.groupby(x_col, observed=True)[group_col].nunique().max()
        width = min(0.16, 1.05 / max_n)
        bar_groups = []

        for x, g in d.groupby(x_col, observed=True):
            reps = list(g.groupby(group_col, observed=True))
            offsets = (np.arange(len(reps)) - (len(reps) - 1) / 2) * width

            for offset, (_, r) in zip(offsets, reps):
                bar_groups.append((x, offset, r))

    for x, offset, g in bar_groups:
        i = x_order.index(x)
        xpos = x_base[i] if offset is None else x_base[i] + offset
        row = g.iloc[0]
        bottom = 0.0

        for y_col, mol_col, color in zip(y_cols, mol_cols, colors):
            height = row[y_col]

            ax.bar(
                xpos,
                height,
                bottom=bottom,
                width=width,
                color=color,
                linewidth=0,
            )

            if annotate_mols and height >= 0.055:
                ax.text(
                    xpos,
                    bottom + height / 2,
                    f"{row[mol_col]:.0f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

            bottom += height

    ax.set_xticks(x_base)
    ax.set_xticklabels(x_order, rotation=0)
    ax.set_ylim(0, 1)
    ax.set_xlabel(get_label(x_col, LABEL_MAP))
    ax.set_ylabel("Fraction")
    ax.grid(True, axis="y", alpha=0.25)

def plot_stacked_fraction_summary(
    df: pd.DataFrame,
    ax: plt.Axes,
    y_cols: list[str],
    x_col: str,
    group_col: str | None,
    style: PlotStyle,
    label_map: dict[str, str] | None = LABEL_MAP,
    aggfunc: str = "mean",
    ylabel: str = "Mean fraction",
    legend_title: str = "Cluster size bin",
    show_legend: bool = True,
) -> plt.Axes:
    """
    Plot stacked fraction bars summarized over groups.

    If group_col is provided:
        1. average y_cols within each group/x_col
        2. average those group-level values across groups for each x_col

    This is useful when df contains multiple frames per replicate and you want
    each replicate to contribute one value per x value before averaging.

    If group_col is None:
        directly average y_cols by x_col.
    """
    required_cols = [x_col, *y_cols]
    if group_col is not None:
        required_cols.append(group_col)

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    d = df.dropna(subset=[x_col]).copy()

    if group_col is not None:
        d_grouped = (
            d.groupby([group_col, x_col], as_index=False)[y_cols]
            .agg(aggfunc)
        )

        plot_df = (
            d_grouped
            .groupby(x_col, as_index=False)[y_cols]
            .agg(aggfunc)
            .sort_values(x_col)
        )
    else:
        plot_df = (
            d.groupby(x_col, as_index=False)[y_cols]
            .agg(aggfunc)
            .sort_values(x_col)
        )

    if plot_df.empty:
        raise ValueError("No data left to plot.")

    frac_sum = plot_df[y_cols].sum(axis=1)
    if not np.allclose(frac_sum, 1.0, atol=1e-5):
        print(
            "Warning: stacked fractions do not sum exactly to 1. "
            f"Range: {frac_sum.min():.4f}–{frac_sum.max():.4f}"
        )

    x_values = plot_df[x_col].to_list()
    x = np.arange(len(x_values))
    bottom = np.zeros(len(plot_df))

    for i, col in enumerate(y_cols):
        ax.bar(
            x,
            plot_df[col].to_numpy(),
            bottom=bottom,
            color=style.stack_colors[i],
            alpha=style.stack_alpha,
            label=get_label(col, label_map),
            width=0.75,
            linewidth=0,
        )
        bottom += plot_df[col].to_numpy()

    ax.set_xticks(x)
    ax.set_xticklabels([
        f"{value:g}" if isinstance(value, (int, float, np.integer, np.floating)) else str(value)
        for value in x_values
    ])

    ax.set_xlabel(get_label(x_col, label_map))
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)

    if style.show_grid:
        ax.grid(True, axis="y", alpha=style.grid_alpha)

    if show_legend:
        ax.legend(
            title=legend_title,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
        )

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
    color_map: dict[Any, str] | None = None,
    show_points: bool | None = None,
) -> plt.Axes:
    """
    Boxplot grouped by x_col, with optional individual points.

    This treats x_col categorically, so numeric values like 1, 2, 4, 8, 16
    are evenly spaced.
    """
    required_cols = [x_col, y_col]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    df_plot = df.dropna(subset=[x_col, y_col]).copy()

    x_values = sorted(df_plot[x_col].dropna().unique())
    positions = np.arange(len(x_values))

    groups = [
        df_plot.loc[df_plot[x_col] == value, y_col].dropna().to_numpy()
        for value in x_values
    ]

    if show_points is None:
        show_points = getattr(style, "box_show_points", True)

    bp = ax.boxplot(
        groups,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=getattr(style, "box_show_fliers", False),
        medianprops=dict(
            color=getattr(style, "box_median_color", "black"),
            linewidth=getattr(style, "box_median_lw", 1.5),
        ),
        whiskerprops=dict(
            color=getattr(style, "box_whisker_color", "black"),
            linewidth=getattr(style, "box_lw", 1.0),
        ),
        capprops=dict(
            color=getattr(style, "box_whisker_color", "black"),
            linewidth=getattr(style, "box_lw", 1.0),
        ),
        boxprops=dict(
            facecolor=getattr(style, "box_facecolor", "white"),
            edgecolor=getattr(style, "box_edgecolor", "black"),
            linewidth=getattr(style, "box_lw", 1.0),
            alpha=getattr(style, "box_alpha", 0.85),
        ),
    )

    # Optional individual points
    if show_points:
        rng = np.random.default_rng(42)

        default_point_color = getattr(style, "box_point_color", "#666666")
        point_size = getattr(style, "box_point_size", 28)
        point_alpha = getattr(style, "box_point_alpha", 0.8)
        jitter = getattr(style, "box_point_jitter", 0.06)

        for i, value in enumerate(x_values):
            vals = df_plot.loc[df_plot[x_col] == value, y_col].dropna().to_numpy()

            if len(vals) == 0:
                continue

            x_jit = i + rng.uniform(-jitter, jitter, size=len(vals))

            point_color = (
                color_map[value]
                if color_map is not None and value in color_map
                else default_point_color
            )

            ax.scatter(
                x_jit,
                vals,
                s=point_size,
                color=point_color,
                alpha=point_alpha,
                edgecolors="none",
                zorder=3,
            )

    ax.set_xticks(positions)
    ax.set_xticklabels([
        f"{x:g}" if isinstance(x, (int, float, np.integer, np.floating)) else str(x)
        for x in x_values
    ])

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
    label_map: dict[str, str] | None = LABEL_MAP,
    aggregate: bool = False,
    aggfunc: str = "mean",
) -> plt.Axes:
    """
    Plot stacked fractions for multiple groups/replicates.

    Args:
        df: Input dataframe.
        ax: Matplotlib axis.
        y_cols: Columns to stack. These become the colored layers.
        x_col: X-axis column.
        group_col: Column identifying replicate/group traces.
        style: PlotStyle.
        label_map: Optional display label map.
        aggregate: If True, collapse repeated rows per (group_col, x_col).
                   Useful after final_fraction filtering when multiple frames
                   remain per replicate and x value.
        aggfunc: Aggregation function for y_cols, usually "mean" or "sum".

    Returns:
        The same ax.
    """
    required_cols = [x_col, group_col, *y_cols]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    df_plot = df.copy()

    if aggregate:
        df_plot = (
            df_plot
            .groupby([group_col, x_col], as_index=False)[y_cols]
            .agg(aggfunc)
        )

    # Safety check: after optional aggregation, one row per group/x is expected
    dupes = df_plot.duplicated([group_col, x_col]).sum()
    if dupes:
        raise ValueError(
            f"Found {dupes} duplicate rows per ({group_col}, {x_col}). "
            "Use aggregate=True or pre-aggregate the dataframe."
        )

    x_values = sorted(df_plot[x_col].dropna().unique())
    x_map = {value: i for i, value in enumerate(x_values)}

    groups = sorted(df_plot[group_col].dropna().unique())

    for i, group_value in enumerate(groups):
        d_group = (
            df_plot.loc[df_plot[group_col] == group_value]
            .sort_values(x_col)
        )

        x = d_group[x_col].map(x_map).to_numpy()

        labels = (
            [get_label(col, label_map) for col in y_cols]
            if i == 0
            else [None] * len(y_cols)
        )

        ax.stackplot(
            x,
            *[d_group[col].to_numpy() for col in y_cols],
            labels=labels,
            alpha=style.stack_alpha,
            colors=style.stack_colors[:len(y_cols)],
        )

    ax.set_xticks(range(len(x_values)))
    ax.set_xticklabels([
        f"{x:g}" if isinstance(x, (int, float, np.integer, np.floating)) else str(x)
        for x in x_values
    ])

    ax.set_xlabel(get_label(x_col, label_map))
    ax.set_ylabel("Fraction")
    ax.set_ylim(0, 1)

    if style.show_grid:
        ax.grid(True, alpha=style.grid_alpha)

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
    band_mode: str = "replicates",
    smoothing_window: int | None = None,
    categorical_x: bool = False,
    x_values: Sequence[Any] | None = None,
    x_scale: float | None = None,
    x_label: str | None = None,
    show_points: bool = False,
    point_color: str = "#7a7a7a",
    point_alpha: float | None = None,
    point_size: float | None = None,
    point_jitter: float | None = None,
    marker: str | None = None,
    markersize: float = 5,
) -> plt.Axes:
    """
    Draw one group's mean trajectory with optional replicate band.

    If categorical_x=True, x_col values are mapped to evenly spaced positions.
    This is useful for parameter sweeps such as E_att = [2, 4, 6, 8, 12],
    where the visual spacing should be categorical rather than numeric.

    band_mode:
        "replicates" : draw one faint line per replicate
        "std"        : draw mean ± std
        "iqr"        : draw q25–q75 band
        "none"       : draw mean only
    """
    if band_mode not in {"replicates", "std", "iqr", "none"}:
        raise ValueError(f"Unknown band_mode: {band_mode}")

    for col in (x_col, y_col, run_col):
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    d = df.dropna(subset=[x_col, y_col]).copy()

    if d.empty:
        raise ValueError("No data left to plot after dropping NaNs.")

    # Optional smoothing — applied per replicate so runs do not bleed into each other.
    if smoothing_window is not None and smoothing_window > 1:
        d = d.sort_values([run_col, x_col]).copy()
        d[y_col] = (
            d.groupby(run_col)[y_col]
            .transform(
                lambda s: s.rolling(
                    smoothing_window,
                    min_periods=1,
                    center=True,
                ).mean()
            )
        )

    # Optional categorical/evenly spaced x-axis.
    if categorical_x:
        if x_values is None:
            x_values = sorted(d[x_col].dropna().unique())

        x_values = list(x_values)
        x_map = {value: i for i, value in enumerate(x_values)}

        observed_x = set(d[x_col].dropna().unique())
        missing_x = sorted(observed_x - set(x_values))

        if missing_x:
            raise ValueError(
                f"x_values does not include all observed {x_col} values. "
                f"Missing from x_values: {missing_x}"
            )

        d["_x_plot"] = d[x_col].map(x_map).astype(float)
        x_plot_col = "_x_plot"
    else:
        x_values = None
        x_map = None
        x_plot_col = x_col
    if x_scale is not None and not categorical_x:
        d["_x_plot_scaled"] = d[x_plot_col].astype(float) / x_scale
        x_plot_col = "_x_plot_scaled"

    # Optional replicate points.
    if show_points:
        rng = np.random.default_rng(42)

        if point_alpha is None:
            point_alpha = getattr(style, "box_point_alpha", 0.8)

        if point_size is None:
            point_size = getattr(style, "box_point_size", 28)

        if point_jitter is None:
            point_jitter = getattr(style, "box_point_jitter", 0.06)

        if categorical_x:
            x_points = d[x_plot_col].to_numpy(dtype=float)
            x_points = x_points + rng.uniform(
                -point_jitter,
                point_jitter,
                size=len(d),
            )
        else:
            x_points = d[x_plot_col].to_numpy(dtype=float)

        ax.scatter(
            x_points,
            d[y_col].to_numpy(),
            s=point_size,
            color=point_color,
            alpha=point_alpha,
            edgecolors="none",
            zorder=2,
        )

    # Replicate cloud.
    if band_mode == "replicates":
        for _, rep_df in d.groupby(run_col):
            rep_df = rep_df.sort_values(x_col)
            ax.plot(
                rep_df[x_plot_col],
                rep_df[y_col],
                color=color,
                lw=style.rep_lw,
                alpha=style.rep_alpha,
            )

    # Aggregate across replicates at each x.
    grouped = d.groupby(x_col)[y_col]

    agg = pd.DataFrame({
        "mean": grouped.mean(),
        "std": grouped.std(),
        "q25": grouped.quantile(0.25),
        "q75": grouped.quantile(0.75),
    }).sort_index()

    if categorical_x:
        x_agg = np.array([x_map[value] for value in agg.index], dtype=float)
    else:
        x_agg = agg.index.to_numpy(dtype=float)
        if x_scale is not None:
            x_agg = x_agg / x_scale

    # Shaded band variants.
    if band_mode == "std":
        ax.fill_between(
            x_agg,
            agg["mean"] - agg["std"],
            agg["mean"] + agg["std"],
            color=color,
            alpha=style.band_alpha,
            linewidth=0,
            zorder=1,
        )

    elif band_mode == "iqr":
        ax.fill_between(
            x_agg,
            agg["q25"],
            agg["q75"],
            color=color,
            alpha=style.band_alpha,
            linewidth=0,
            zorder=1,
        )

    # Bold mean on top.
    ax.plot(
        x_agg,
        agg["mean"],
        color=color,
        lw=style.mean_lw,
        label=label,
        marker=marker,
        markersize=markersize,
        zorder=3,
    )

    # Correct categorical ticks.
    if categorical_x:
        ax.set_xticks(np.arange(len(x_values)))
        ax.set_xticklabels([
            f"{value:g}" if isinstance(value, (int, float, np.integer, np.floating)) else str(value)
            for value in x_values
        ])
    if x_label is not None:
        ax.set_xlabel(x_label)
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
    x_scale: float | None = None,
    x_label: str | None = None,
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
                    df=sub,
                    ax=ax,
                    x_col=x_col,
                    y_col=y_col,
                    style=style,
                    run_col=run_col,
                    color=color,
                    label=label,
                    band_mode=band_mode,
                    smoothing_window=smoothing_window,
                    x_scale=x_scale,
                    x_label=x_label,
                )

    ax.set_xlabel(x_label if x_label is not None else get_label(x_col, label_map))
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
    title = suptitle or ""

    title_info = []

    if final_fraction is not None:
        pct = int(round(final_fraction * 100))
        title_info.append(f"last {pct}% of steps")

    if run_col in df.columns:
        n_reps = df[run_col].nunique()
        rep_word = "replicate" if n_reps == 1 else "replicates"
        title_info.append(f" {n_reps} {rep_word}")

    if title_info:
        title = f"{title} ({', '.join(title_info)})"

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, axes

