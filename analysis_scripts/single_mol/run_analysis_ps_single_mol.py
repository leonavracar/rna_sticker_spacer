#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

analysis_functions_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(analysis_functions_dir))

from analysis_scripts.helpers import (
    get_row_params_from_json,
    get_swept_param_keys,
)
from analysis_scripts.phase_separation import add_metric_summary_stats


BASE_DIR = analysis_functions_dir / "bulk" / "output_dir"


FRAME_COLS = [
    # contact counts / recurrence
    "particle_contacts",
    "contact_regions_particle",
    "contact_region_recurrence_particle",
    "contact_regions_fixed",
    "contact_region_recurrence_fixed",
    "contact_regions_repeat",
    "contact_region_recurrence_repeat",
    "contact_regions_repeat_window",
    "contact_region_recurrence_repeat_window",

    # shape
    "rg",
    "end_to_end_distance",

    # logged quantities
    "ke_per_particle",
    "E_pot",
    "E_sqd",
    "E_bond",
    "E_angle",
]


def add_last_fraction_means(
    row: dict[str, object],
    df: pd.DataFrame,
    cols: list[str],
    fraction_last: float = 0.2,
) -> None:
    """
    Add mean over the final fraction of frames using simple column names.
    """
    if not 0 < fraction_last <= 1:
        raise ValueError("fraction_last must be in the interval (0, 1].")

    late_start_idx = int(len(df) * (1 - fraction_last))
    df_last = df.iloc[late_start_idx:]

    for col in cols:
        if col in df.columns:
            row[f"{col}_last_mean"] = df_last[col].mean()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build single-molecule analysis_per_replicate.csv from framewise CSVs."
    )

    parser.add_argument(
        "sweep_dir",
        help="Name of the sweep directory inside BASE_DIR.",
    )

    parser.add_argument(
        "--run_ids",
        nargs="+",
        default=None,
        help="Optional numeric folder prefixes to process, e.g. 251 252 253.",
    )

    parser.add_argument(
        "--summary-fraction-last",
        type=float,
        default=0.2,
        help="Final fraction of frames used for late/last summary statistics.",
    )

    args = parser.parse_args()

    sweep_path = BASE_DIR / args.sweep_dir

    if not sweep_path.is_dir():
        raise FileNotFoundError(f"Sweep directory not found: {sweep_path}")

    swept_param_keys = get_swept_param_keys(sweep_path / "parameters.txt")
    rows: list[dict[str, object]] = []

    if args.run_ids is None:
        csv_paths = sorted(sweep_path.glob("*/framewise/*_framewise_ps.csv"))
    else:
        csv_paths = []
        for run_id in args.run_ids:
            csv_paths.extend(
                sorted(sweep_path.glob(f"{run_id}_*/framewise/*_framewise_ps.csv"))
            )

    if not csv_paths:
        raise FileNotFoundError(f"No framewise CSVs found under: {sweep_path}")

    for csv_path in csv_paths:
        run_dir = csv_path.parent.parent

        row = get_row_params_from_json(
            run_dir=run_dir,
            swept_param_keys=swept_param_keys,
        )

        df = pd.read_csv(csv_path)

        row["folder_name"] = run_dir.name
        row["traj_replicate"] = (
            csv_path.stem
            .removesuffix("_framewise_ps")
            .removesuffix("_pairing")
        )
        row["n_saved_frames"] = len(df)
        row["summary_fraction_last"] = args.summary_fraction_last

        available_frame_cols = [
            col for col in FRAME_COLS
            if col in df.columns
        ]

        add_metric_summary_stats(
            row=row,
            df=df,
            cols=available_frame_cols,
            prefix="frame_",
            fraction_last=args.summary_fraction_last,
        )

        add_last_fraction_means(
            row=row,
            df=df,
            cols=[
                "rg",
                "end_to_end_distance",
            ],
            fraction_last=args.summary_fraction_last,
        )

        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_csv = sweep_path / "analysis_per_replicate.csv"
    out_df.to_csv(out_csv, index=False)

    print(f"Saved {len(out_df)} rows to {out_csv}")


if __name__ == "__main__":
    main()