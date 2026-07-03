#!/usr/bin/env python3

"""
Generate interactive summary plots for framewise CSVs in a sweep directory.

Usage:
    python run_time_plots.py <sweep_dir> [options]

Reads framewise CSVs from:
    BASE_DIR/<sweep_dir>/*/framewise_csvs/*_framewise.csv

Writes HTML plots into:
    BASE_DIR/<sweep_dir>/<run_folder>/framewise_plots/summary_<N>.html

where <N> matches the trajectory index (e.g. trajectory_0 -> summary_0).
"""

import pandas as pd
import argparse
from pathlib import Path
import sys

analysis_functions_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(analysis_functions_dir))

from analysis_scripts.analyze_leona import make_summary_figure

BASE_DIR = analysis_functions_dir / "bulk" / "output_dir"


def trajectory_index(csv_stem: str) -> str:
    """
    Extract the numeric index from a CSV stem.
    e.g. 'trajectory_0_framewise' -> '0'
         'trajectory_12_framewise' -> '12'
    Falls back to the full stem if parsing fails.
    """
    parts = csv_stem.split("_")
    for i, part in enumerate(parts):
        if part == "trajectory" and i + 1 < len(parts):
            idx = parts[i + 1]
            if idx.isdigit():
                return idx
    return csv_stem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "sweep_dir",
        help="Name of the sweep directory inside BASE_DIR",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=25,
        help="Rolling window for smoothing energy/structure signals (default: 25).",
    )
    parser.add_argument(
        "--score-smooth-window",
        type=int,
        default=100,
        help="Rolling window for smoothing the transition score (default: 100).",
    )
    parser.add_argument(
        "--height-quantile",
        type=float,
        default=0.87,
        help="Quantile threshold for transition peak detection (default: 0.87).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing HTML files (default: skip if already present).",
    )
    args = parser.parse_args()

    sweep_path = BASE_DIR / args.sweep_dir

    if not sweep_path.is_dir():
        raise FileNotFoundError(f"Sweep directory not found: {sweep_path}")

    print(f"Reading from: {sweep_path}")

    csv_paths = sorted(sweep_path.glob("*/framewise_csvs/*_framewise.csv"))
    if not csv_paths:
        raise FileNotFoundError(
            f"No *_framewise.csv files found under: {sweep_path}"
        )

    n_done = 0
    n_skipped = 0
    n_failed = 0

    for csv_path in csv_paths:
        run_folder = csv_path.parent.parent          # .../sweep/<run_folder>
        out_dir = run_folder / "framewise_plots"
        out_dir.mkdir(exist_ok=True)

        idx = trajectory_index(csv_path.stem)
        out_html = out_dir / f"summary_{idx}.html"

        if out_html.exists() and not args.overwrite:
            print(f"Skipping (already exists): {out_html}")
            n_skipped += 1
            continue

        try:
            print(f"Processing: {csv_path}")
            print(f"Writing to: {out_html}")

            df = pd.read_csv(csv_path)

            title = f"{run_folder.name} — trajectory {idx}"

            fig = make_summary_figure(
                df,
                smooth_window=args.smooth_window,
                score_smooth_window=args.score_smooth_window,
                height_quantile=args.height_quantile,
                title=title,
            )

            fig.write_html(str(out_html))
            n_done += 1

        except Exception as e:
            print(f"Failed on {csv_path}: {e}")
            n_failed += 1

    print(
        f"\nFinished. Generated: {n_done}  |  Skipped: {n_skipped}  |  Failed: {n_failed}"
    )


if __name__ == "__main__":
    main()
