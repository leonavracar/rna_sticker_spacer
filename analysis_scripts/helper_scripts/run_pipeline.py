#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime


LOG_DIR = Path("/groups/goloborodko/projects/leona/validation/bulk/pipeline_logs")


def run_command(cmd: list[str], log_file: Path) -> None:
    header = (
        "\n" + "=" * 80 + "\n"
        f"Started: {datetime.now()}\n"
        "Command: " + " ".join(cmd) + "\n"
        f"Log: {log_file}\n"
        + "=" * 80 + "\n"
    )

    print(header, flush=True)

    with log_file.open("a", buffering=1) as log:
        log.write(header)
        log.flush()

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None

        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()

        returncode = process.wait()

        footer = (
            f"\nFinished: {datetime.now()}\n"
            f"Return code: {returncode}\n"
        )

        print(footer, flush=True)
        log.write(footer)
        log.flush()

    if returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run compute → framewise → analysis sequentially."
    )

    parser.add_argument(
        "sweep_dirs",
        nargs="+",
        help="Normal sweep directories to process.",
    )

    parser.add_argument(
        "--mixture_sweep_dir",
        default=None,
        help="Optional mixture sweep directory to process with the mixture framewise script.",
    )

    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--exclude_chain", type=int, default=3)
    parser.add_argument("--contact_width", type=float, default=0.08)
    parser.add_argument("--min_intermol_bonds", type=int, default=2)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--min_num_frames", type=int, default=2)
    parser.add_argument("--recurrence_window", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument(
        "--run_ids",
        nargs="+",
        default=None,
        help="Optional run IDs to pass to scripts that support it.",
    )

    parser.add_argument(
        "--mixture_type_lengths",
        nargs="+",
        type=int,
        default=[30, 141],
        help="Molecule bead lengths for mixture type inference.",
    )

    parser.add_argument(
        "--mixture_type_labels",
        nargs="+",
        default=["cag10", "cag47"],
        help="Labels corresponding to mixture type lengths.",
    )

    parser.add_argument("--mixture_drop_threshold", type=int, default=5)

    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    here = Path(__file__).resolve().parent

    compute_script = here / "run_compute_ps.py"
    framewise_script = here / "run_time_df_ps.py"
    analysis_script = here / "run_analysis_ps.py"
    mixture_framewise_script = here / "run_time_df_ps_mix.py"

    required_scripts = [
        compute_script,
        framewise_script,
        analysis_script,
    ]

    if args.mixture_sweep_dir is not None:
        required_scripts.append(mixture_framewise_script)

    for script in required_scripts:
        if not script.exists():
            raise FileNotFoundError(f"Missing script: {script}")

    def add_run_ids(cmd: list[str]) -> list[str]:
        if args.run_ids is not None:
            cmd.extend(["--run_ids", *args.run_ids])
        return cmd

    def compute_cmd_for(sweep_dir: str) -> list[str]:
        cmd = [
            sys.executable,
            "-u",
            str(compute_script),
            sweep_dir,
            "--stride",
            str(args.stride),
            "--exclude_chain",
            str(args.exclude_chain),
            "--contact_width",
            str(args.contact_width),
        ]

        if args.overwrite:
            cmd.append("--overwrite")

        return add_run_ids(cmd)

    def framewise_cmd_for(sweep_dir: str) -> list[str]:
        cmd = [
            sys.executable,
            "-u",
            str(framewise_script),
            sweep_dir,
            "--min-intermol-bonds",
            str(args.min_intermol_bonds),
            "--recurrence-window",
            str(args.recurrence_window),
        ]

        return add_run_ids(cmd)

    def analysis_cmd_for(sweep_dir: str) -> list[str]:
        cmd = [
            sys.executable,
            "-u",
            str(analysis_script),
            sweep_dir,
            "--min-intermol-bonds",
            str(args.min_intermol_bonds),
            "--window",
            str(args.window),
            "--min-num-frames",
            str(args.min_num_frames),
            "--recurrence-window",
            str(args.recurrence_window),
        ]

        return add_run_ids(cmd)

    def mixture_framewise_cmd_for(sweep_dir: str) -> list[str]:
        cmd = [
            sys.executable,
            "-u",
            str(mixture_framewise_script),
            sweep_dir,
            "--min-intermol-bonds",
            str(args.min_intermol_bonds),
            "--drop-threshold",
            str(args.mixture_drop_threshold),
            "--type-lengths",
            *map(str, args.mixture_type_lengths),
            "--type-labels",
            *args.mixture_type_labels,
        ]

        return add_run_ids(cmd)

    for sweep_dir in args.sweep_dirs:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = LOG_DIR / f"{sweep_dir}_{stamp}.log"

        run_command(compute_cmd_for(sweep_dir), log_file)
        run_command(framewise_cmd_for(sweep_dir), log_file)
        run_command(analysis_cmd_for(sweep_dir), log_file)

    if args.mixture_sweep_dir is not None:
        sweep_dir = args.mixture_sweep_dir
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = LOG_DIR / f"{sweep_dir}_mixture_{stamp}.log"

        run_command(compute_cmd_for(sweep_dir), log_file)
        run_command(mixture_framewise_cmd_for(sweep_dir), log_file)

    print("\nPipeline finished successfully.")


if __name__ == "__main__":
    main()