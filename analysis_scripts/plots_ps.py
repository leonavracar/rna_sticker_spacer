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
# ============================================================================
# Preparing framewise.csv for plotting
# ============================================================================

def setup_validation_paths(sweep_dir_name: str) -> dict[str, Path]:
    """
    Create standard paths for loading paths of one validation sweep.
    """
    validation_dir = Path("/groups/goloborodko/projects/leona/validation")
    bulk_dir = validation_dir / "bulk"
    output_dir = bulk_dir / "output_dir"
    sweep_dir = output_dir / sweep_dir_name

    sys.path.extend([
        str(bulk_dir),
        str(validation_dir),
    ])

    return {
        "validation_dir": validation_dir,
        "bulk_dir": bulk_dir,
        "output_dir": output_dir,
        "sweep_dir": sweep_dir,
        "framewise_csv": sweep_dir / "analysis_framewise.csv",
        "parameters_txt": sweep_dir / "parameters.txt",
        "analysis_per_replicate_csv": sweep_dir / "analysis_per_replicate.csv",
    }

def setup_npz_mp_paths(sweep_dir:str, run_dir_num:str, rep_num:int = 0) -> dict[str, Path]:
    """
    Create standard paths for loading pairing data (stored as npz with frame_indices, qi_{frame}, pi{frame}, distances, error_rates), 
    molecule lookup table mapping the molecule ID to particle indices and the respective trajectory.
    Inputs are sweep directory name (e.g. 41a_sweepx), number of the run_dir and number of replicate if present.
    """
    validation_dir = Path("/groups/goloborodko/projects/leona/validation")
    bulk_dir = validation_dir / "bulk"
    output_dir = bulk_dir / "output_dir"
    sweep_dir = output_dir / sweep_dir

    npz_mp_paths={}

    sys.path.extend([
        str(bulk_dir),
        str(validation_dir),
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
    if fixed is None:
        return df.copy()

    filtered_df = df.copy()

    for col, value in fixed.items():
        if col not in filtered_df.columns:
            raise ValueError(f"Column not found in DataFrame: {col}")

        if isinstance(value, list):
            filtered_df = filtered_df[filtered_df[col].isin(value)]
        else:
            filtered_df = filtered_df[filtered_df[col] == value]

    return filtered_df

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

# ============================================================================
# LABEL HANDLING
# ============================================================================

def get_label(col: str, label_map: dict[str, str] | None = None) -> str:
    """
    Return a display label for a column name.
    """
    if label_map is None:
        return col

    return label_map.get(col, col)

label_map = {
    # ── Time axis ────────────────────────────────────────────────────────────
    "step":                 "Simulation Step",
    "frame":                "Frame",
    "time":                 "Time",

    # ── Simulation parameters ─────────────────────────────────────────────
    "E_att":                "E_att",
    "angle_k_ppp":          "Chain Stiffness",
    "density":              "ρ",
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
    "contact_regions_particle":             "N_reg (exact)",
    "contact_region_recurrence_particle":   "R_rec (exact)",
    "contact_regions_fixed":                "N_reg (fixed bin)",
    "contact_region_recurrence_fixed":      "R_rec (fixed bin)",
    "contact_regions_repeat":               "N_reg (repeat unit)",
    "contact_region_recurrence_repeat":     "R_rec (repeat unit)",
    "contact_regions_repeat_window":        "N_reg (repeat window)",
    "contact_region_recurrence_repeat_window": "R_rec (repeat window)",
    "rec_mol":                              "R_mol",

    # ── Cluster / phase fractions ─────────────────────────────────────────
    "mono_frac":                            "f_mono",
    "oligo_frac":                           "f_oligo",
    "drop_frac":                            "f_droplet",
    "largest_cluster":                      "largest cluster",
    "n_clusters":                           "N_clusters",
}



# ============================================================================
# PLOTS
# ============================================================================

# Equilibration plots: first individual plotting functions

# Plot one timeseries
def plot_one_timeseries(df:pd.DataFrame, ax, y_col:str,  x_col:str = "step", label=None) -> Axes:
    plot_df = df.copy().sort_values(x_col) # sorted copy by x axis
    ax.plot(plot_df[x_col], plot_df[y_col], label= label)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    return ax

# somehow we also need optional saving for all plots
# Plot lines by group

def plot_lines_by_group(
    df: pd.DataFrame,
    ax,
    y_col: str,
    group_col: str,
    x_col: str = "step",
) -> Axes:
    for group_value, group_df in df.groupby(group_col):
        plot_one_timeseries(
            df=group_df,
            ax=ax,
            y_col=y_col,
            x_col=x_col,
            label=f"{group_col}={group_value}",
        )

    ax.legend()
    return ax

def make_label(row: pd.Series, cols: Sequence[str]) -> str:
    """
    Make a readable label from one or more columns.
    """
    return ", ".join(f"{col}={row[col]}" for col in cols)

def plot_panels_by_group(
    df: pd.DataFrame,
    y_col: str,
    panel_cols: str | Sequence[str],
    line_col: str,
    x_col: str = "step",
    n_cols: int = 2,
    figsize_per_panel: tuple[float, float] = (6, 4),
) -> tuple[Figure, np.ndarray]:
    """
    Plot one panel per parameter combination, with one line per line_col value.

    Args:
        df: Input DataFrame.
        y_col: Column to plot on the y-axis.
        panel_cols: Column(s) used to create separate subplots.
        line_col: Column used to create separate lines in each subplot.
        x_col: Column to plot on the x-axis.
        n_cols: Number of subplot columns.
        figsize_per_panel: Size of each subplot.

    Returns:
        Figure and axes.
    """
    if isinstance(panel_cols, str):
        panel_cols = [panel_cols]
    else:
        panel_cols = list(panel_cols)

    required_cols = [x_col, y_col, line_col, *panel_cols]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    panel_combis = (
        df[panel_cols]
        .drop_duplicates()
        .sort_values(panel_cols)
        .reset_index(drop=True)
    )

    n_panels = len(panel_combis)
    n_rows = math.ceil(n_panels / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(
            figsize_per_panel[0] * n_cols,
            figsize_per_panel[1] * n_rows,
        ),
        sharex=True,
        sharey=True,
    )

    axes = np.array(axes).reshape(-1)

    for ax, (_, panel_values) in zip(axes, panel_combis.iterrows()):
        panel_df = df.copy()

        for col in panel_cols:
            panel_df = panel_df[panel_df[col] == panel_values[col]]

        plot_lines_by_group(
            df=panel_df,
            ax=ax,
            y_col=y_col,
            group_col=line_col,
            x_col=x_col,
        )

        ax.set_title(make_label(panel_values, panel_cols))

    for ax in axes[n_panels:]:
        ax.set_visible(False)

    fig.suptitle(f"{y_col} over {x_col}")
    plt.tight_layout()

    return fig, axes