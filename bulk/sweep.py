#!/usr/bin/env python
#SBATCH --job-name=50a
#SBATCH --output=job_output/job_output2_%A_%a.txt
#SBATCH --error=job_output/job_error2_%A_%a.txt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --qos=long
#SBATCH --time=12:00:00
#SBATCH --partition=g
#SBATCH --gres=gpu:1
#SBATCH --constraint=g4
#SBATCH --array=0-35


import argparse
import os
import subprocess
from pathlib import Path

# base_dir = Path(__file__).resolve().parent
os.chdir("/groups/goloborodko/projects/leona/validation/bulk")
# parse args
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dir",
    type=str,
    default="output",
    help="Directory containing all files associated with this simulation",
)
parser.add_argument(
    "--paramfile",
    type=str,
    default="parameters.txt",
    help="Name of the parameter file in dir",
)
parser.add_argument(
    "--n_replicates", type=int, default=3, help="Number of replicates per parameter set"
)
parser.add_argument(
    "--restart_from_sweepdir",
    type=str,
    default=None,
    help="Sweep dir name to restart sim from, lives in parent of --dir e.g. output_dir",
)

args = parser.parse_args()

# read param combinations
with open(f"{args.dir}/{args.paramfile}") as f:
    lines = f.readlines()


# get correct combination

array_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
base = int(os.environ.get("BASE", "0"))  # default 0 for first batch
task_id = base + array_id  # global task id


combination_id = task_id // args.n_replicates  # more than 1 replicate per condition
line = lines[combination_id].strip()

# create the folder
folder_name = line.replace(" ", "_").replace("--", "")
outdir = f"{args.dir}/{combination_id}_{folder_name}"
os.makedirs(outdir, exist_ok=True)

# get the replicate suffix
suffix = f"_{task_id%args.n_replicates}"

# make the command
cmd = [
    "python",
    "-u",
    "run_sim.py",
    "--output_dir",
    str(outdir),
    "--suffix",
    suffix,
] + line.split()

# get the .gsd file path (in case of running from state)

if args.restart_from_sweepdir is not None:
    parent_dir = Path(args.dir).parent
    restart_root = parent_dir / args.restart_from_sweepdir
    restart_dir = restart_root / f"{combination_id}_{folder_name}"

    restart_file = restart_dir / f"trajectory{suffix}.gsd"

    if not restart_file.exists():
        raise FileNotFoundError(f"Restart file not found: {restart_file}")

    cmd += ["--restart_from", str(restart_file)]

subprocess.run(cmd, check=True)
