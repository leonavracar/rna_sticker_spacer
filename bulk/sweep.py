#!/usr/bin/env python
# autopep8: off
#SBATCH --job-name=rna_sticker_spacer
#SBATCH --output=job_output/job_output_%A_%a.txt
#SBATCH --error=job_output/job_error_%A_%a.txt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --qos=g_medium
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0,8-11
# autopep8: on
"""Launch one SLURM array task for a parameter sweep.

Each line in the parameter file defines one parameter combination.
Each parameter combination is run for ``n_replicates`` replicates.

The global task id is:

    task_id = BASE + SLURM_ARRAY_TASK_ID

Then:

    combination_id = task_id // n_replicates
    replicate_id = task_id % n_replicates

This script creates one output directory per parameter combination and launches
``run_sim.py`` once for the selected replicate.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


PROJECT_DIR = Path(
    "/groups/goloborodko/projects/leona/rna_sticker_spacer/bulk")


def get_args() -> argparse.Namespace:
    """Parse sweep-launch arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dir",
        type=str,
        default="output",
        help="Directory containing all files associated with this sweep.",
    )
    parser.add_argument(
        "--paramfile",
        type=str,
        default="parameters.txt",
        help="Name of the parameter file inside --dir.",
    )
    parser.add_argument(
        "--n_replicates",
        type=int,
        default=3,
        help="Number of replicates per parameter set.",
    )
    parser.add_argument(
        "--restart_from_sweepdir",
        type=str,
        default=None,
        help=(
            "Sweep directory name to restart from. This directory should live in "
            "the parent of --dir. For example, if --dir is output/new_sweep, then "
            "restart_from_sweepdir should be the old sweep directory name inside output/."
        ),
    )

    return parser.parse_args()


def read_parameter_lines(sweep_dir: Path, paramfile: str) -> list[str]:
    """Read non-empty parameter lines from the sweep parameter file."""
    param_path = sweep_dir / paramfile

    if not param_path.exists():
        raise FileNotFoundError(f"Parameter file not found: {param_path}")

    with param_path.open() as handle:
        lines = [line.strip() for line in handle if line.strip()]

    if not lines:
        raise ValueError(f"Parameter file is empty: {param_path}")

    return lines


def get_task_ids(n_replicates: int, n_combinations: int) -> tuple[int, int, int]:
    """Return task_id, combination_id, and replicate_id for this SLURM task."""
    if n_replicates <= 0:
        raise ValueError(f"n_replicates must be positive, got {n_replicates}.")

    array_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    base = int(os.environ.get("BASE", "0"))

    task_id = base + array_id
    combination_id = task_id // n_replicates
    replicate_id = task_id % n_replicates

    if combination_id >= n_combinations:
        raise IndexError(
            "SLURM task maps beyond available parameter combinations. "
            f"task_id={task_id}, n_replicates={n_replicates}, "
            f"combination_id={combination_id}, "
            f"n_combinations={n_combinations}."
        )

    return task_id, combination_id, replicate_id


def make_folder_name(combination_id: int, parameter_line: str) -> str:
    """Create a readable output-folder name from a parameter line."""
    clean_line = parameter_line.replace(" ", "_").replace("--", "")
    return f"{combination_id}_{clean_line}"


def main() -> None:
    """Launch the selected replicate."""
    os.chdir(PROJECT_DIR)

    args = get_args()

    sweep_dir = Path(args.dir)
    parameter_lines = read_parameter_lines(
        sweep_dir=sweep_dir,
        paramfile=args.paramfile,
    )

    task_id, combination_id, replicate_id = get_task_ids(
        n_replicates=args.n_replicates,
        n_combinations=len(parameter_lines),
    )

    parameter_line = parameter_lines[combination_id]

    folder_name = make_folder_name(
        combination_id=combination_id,
        parameter_line=parameter_line,
    )
    outdir = sweep_dir / folder_name
    outdir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{replicate_id}"

    cmd = [
        "python",
        "-u",
        "run_sim.py",
        "--output_dir",
        str(outdir),
        "--suffix",
        suffix,
    ] + parameter_line.split()

    if args.restart_from_sweepdir is not None:
        parent_dir = sweep_dir.parent
        restart_root = parent_dir / args.restart_from_sweepdir
        restart_dir = restart_root / folder_name
        restart_file = restart_dir / f"trajectory{suffix}.gsd"

        if not restart_file.exists():
            raise FileNotFoundError(f"Restart file not found: {restart_file}")

        cmd += ["--restart_from", str(restart_file)]

    print("Launching sweep task:")
    print(f"  PROJECT_DIR: {PROJECT_DIR}")
    print(f"  sweep_dir: {sweep_dir}")
    print(f"  task_id: {task_id}")
    print(f"  combination_id: {combination_id}")
    print(f"  replicate_id: {replicate_id}")
    print(f"  suffix: {suffix}")
    print(f"  output_dir: {outdir}")
    print(f"  parameter_line: {parameter_line}")
    print(f"  command: {' '.join(cmd)}")

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
