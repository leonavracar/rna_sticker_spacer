from pathlib import Path
import matplotlib.pyplot as plt
import itertools
import numpy as np
import operator
import json
from typing import Any

"""
Helper functions to:
- generate combinations for parameters.txt
- prepare sweep (make folder...)
"""

####### PREPARE SWEEP #######

def generate_combinations(**kwargs):
    keys = kwargs.keys()
    values = kwargs.values()

    all_combinations = list(itertools.product(*values))

    return all_combinations


def generate_formatted_combinations(combinations, keys):
    formatted_combinations = [
        " ".join(
            f"--{key} {np.format_float_positional(value, trim='-') if isinstance(value, (float, int)) else str(value)}"
            for key, value in zip(keys, combo)
        )
        for combo in combinations
    ]
    return formatted_combinations


def write_combinations_to_file(combinations, filename):
    with open(filename, 'w') as f:
        f.write('\n'.join(combinations))
    return


def format_row(row):
    return ' '.join([f"--{col} {row[col]}" for col in row.index])


def filt_angle_combis(all_combis):
    filt_combis = []
    for i in range(len(all_combis)):
        if all_combis[i][0] == all_combis[i][1] == all_combis[i][2]:
            filt_combis.append(all_combis[i])
    return filt_combis


def prepare_sweep(folder: str, params: dict, base_dir: str = "output_dir"):
    """
    Takes name of new folder to be made for this sweep, the params dictionary and the base output_dir.
    Makes dir, makes parameters.txt, prints number of combinations.
    """
    sweep_dir = Path(base_dir) / folder
    sweep_dir.mkdir(parents=True, exist_ok=True)

    combis = list(generate_combinations(**params))
    print(len(combis))

    formatted = generate_formatted_combinations(combis, params.keys())
    write_combinations_to_file(formatted, sweep_dir / "parameters.txt")

    return sweep_dir, combis, formatted

######## GENERATE SEQUENCES ########

def make_CGA_seq(nC, nG, nA):
    a = ["A"]
    c = ["C"]
    g = ["G"]
    seq = nC * c + nG * g + nA * a + nC * c + nG * g
    return seq


def make_GC_hairpin_seq(n_stem, n_loop):
    """
    n_stem means number of CG repeats on one side and on other
    """
    a = ["A"]
    seq = []
    # GC stem
    for _ in range(n_stem):
        seq += ["G", "C"]
    # loop
    seq += n_loop * a

    # GC stem
    for _ in range(n_stem):
        seq += ["G", "C"]

    return seq


def make_CAG_seq_of_given_repeat_number(n_CAG_repeats: int) -> list[str]:
    """
    Return a sequence as a list of bases for the given number of CAG repeats.

    Example:
        n_CAG_repeats=3 -> ["C", "A", "G", "C", "A", "G", "C", "A", "G"]
    """
    return ["C", "A", "G"] * n_CAG_repeats


def make_json(seq, count, suffix="_", save_dir="/groups/goloborodko/projects/leona/validation/bulk/sequences"):
    """Pass single sequence as a np array OR list, plus count to generate a json file"""
    # Convert np.array → list
    if isinstance(seq, list):
        seq_lst = seq
    else:
        # For numpy arrays
        seq_lst = seq.tolist()

    data = [{"sequence": seq_lst, "count": count}]

    with open(f"{save_dir}/{suffix}.json", "w") as f:
        json.dump(data, f)

    print(
        f"File seq {suffix}.json with seq of length {len(seq_lst)} saved to {save_dir}")

####### CORRELATION HEATMAP ########

def corr_heatmap_clean(
    df,
    method="spearman",
    figsize=(12, 10),
    vmax=1.0,
    min_non_nan=0.5,
    fmt=".2f",
    text_size=7,
    annotate_threshold=0.20,   # only annotate |corr| >= this
    hide_diagonal=False,       # optionally skip the 1.00 diagonal
):
    num = df.select_dtypes(include=[np.number]).copy()
    keep = (num.isna().mean() <= (1 - min_non_nan))
    num = num.loc[:, keep]
    num = num.loc[:, num.nunique(dropna=True) > 1]

    C = num.corr(method=method)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(C.values, vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax, label=f"{method} correlation")

    ax.set_xticks(range(len(C.columns)))
    ax.set_xticklabels(C.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(C.index)))
    ax.set_yticklabels(C.index, fontsize=7)

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            if hide_diagonal and i == j:
                continue
            val = C.iat[i, j]
            if np.isnan(val) or abs(val) < annotate_threshold:
                continue
            ax.text(j, i, format(val, fmt), ha="center",
                    va="center", fontsize=text_size)

    ax.set_title(f"{method} correlation (|r|≥{annotate_threshold})")
    fig.tight_layout()
    plt.show()
    return C


OPS = {
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
    ">=": operator.ge,
    ">": operator.gt
}


def find_dir_based_on_params(p: Path, *conditions: Any) -> None:
    """Takes path to sweep dir and conditions, returns folder names that satisfy them.

    Args:
        p (Path): _description_
        *conditions: tuples of format ("angle_branch_dir_ppb_t0", ">", 2).
    Example call:

    find_dir_based_on_params(p, ("molecule_file", "==", "AGUC30.json"), ...)
    """
    for dir in p.iterdir():
        if dir.is_dir():
            with open(dir / "params.json") as f:
                data = json.load(f)
                match = True
                for param, condi, val in conditions:
                    actual_val = data.get(param)
                    ops_get = OPS.get(condi)

                    if actual_val is None or not ops_get(actual_val, val):
                        match = False
                        break
                if match:
                    print(dir)

# if data["molecule_file"] == "AGUC30.json" and data["angle_branch_dir_ppb_t0"] > 2 and data["E_att"] == 50:
    # print(dir) # prints 0,2,4 for rep wc nondir 0, 10, 20

##### SEARCH OUTPUTS AND FIND MISSING ENTRIES OR SLURM IDX #####

DEFAULT_OUTDIR = "/groups/goloborodko/projects/leona/validation/bulk/output_dir"


def find_missing_combo(sweep_dir, output_dir=DEFAULT_OUTDIR, n_replicates=5):
    """Iterate over sweep_dir and check whether trajectory_{rep}.gsd exists in each combo folder.

    Assumes folders are named like: "<combo_idx>_molecule_file_..."

    Returns:
        list[tuple[int, int]]: list of (combo_idx, rep_idx) that are missing
    """
    p = Path(output_dir) / sweep_dir
    missing = []

    for d in p.iterdir():
        if not d.is_dir():
            continue

        # folder name like: 6_molecule_file_...
        parts = d.name.split("_", 1)
        try:
            combo_idx = int(parts[0])
        except ValueError:
            # skip folders that don't match expected naming
            continue

        for rep in range(n_replicates):
            traj = d / f"trajectory_{rep}.gsd"
            if not traj.is_file():
                missing.append((combo_idx, rep))

    missing.sort()
    return missing


def calc_slurm_array_idx(missing, n_replicates=5):
    """Convert (combo_idx, rep_idx) into slurm array indices."""
    slurm_indices = [combo * n_replicates + rep for combo, rep in missing]
    slurm_indices.sort()
    return slurm_indices


def format_slurm_array_idx(slurm_indices):
    """Print indices as '1,2,3' (no spaces, no trailing comma)."""
    print(",".join(str(idx) for idx in slurm_indices))


def print_missing_slurm_idx(sweep_dir, output_dir=DEFAULT_OUTDIR, n_replicates=5):
    missing = find_missing_combo(
        sweep_dir, output_dir=output_dir, n_replicates=n_replicates)
    slurm_indices = calc_slurm_array_idx(missing, n_replicates=n_replicates)
    format_slurm_array_idx(slurm_indices)


def print_missing_paths(sweep_dir, output_dir=DEFAULT_OUTDIR, n_replicates=5):
    """Print missing trajectory paths, one per line."""
    p = Path(output_dir) / sweep_dir
    missing_paths = []

    for d in p.iterdir():
        if not d.is_dir():
            continue
        for rep in range(n_replicates):
            traj = d / f"trajectory_{rep}.gsd"
            if not traj.is_file():
                missing_paths.append(str(traj))

    missing_paths.sort()
    if missing_paths:
        print("\n".join(missing_paths))


