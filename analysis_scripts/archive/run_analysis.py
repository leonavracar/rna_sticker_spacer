#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import sys
analysis_functions_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(analysis_functions_dir))
from analysis_scripts.helpers import (
    drop_all_nan_columns,
    get_row_params_from_json,
    get_swept_param_keys,
    load_npz,
)
from analysis_scripts.analysis_summary import (
    compute_group_summary,
    compute_replicate_analysis,
)

import argparse

import numpy as np
import pandas as pd



BASE_DIR = analysis_functions_dir / "bulk" / "output_dir"

EXPECTED_PAIRS_AGUC30 = np.array([
    29, 28, 27, 26, 25, 24, 23, 22, 21, 20,
    19, 18, 17, 16, 15, 14, 13, 12, 11, 10,
    9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
], dtype=int)


def get_expected_canonical(
    use_expected_canonical: bool,
) -> np.ndarray | None:
    """
    Return the default expected canonical partner map if requested.
    """
    if use_expected_canonical:
        return EXPECTED_PAIRS_AGUC30
    return None


def collect_replicate_rows(
    sweep_path: Path,
    expected: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Build the per-replicate analysis table from cached metric files in one sweep directory.
    """
    swept_param_keys = get_swept_param_keys(sweep_path / "parameters.txt")
    rows: list[dict[str, object]] = []

    for run_dir in sorted(p for p in sweep_path.iterdir() if p.is_dir()):
        params_path = run_dir / "params.json"
        if not params_path.exists():
            print(f"Skipping {run_dir.name}: missing {params_path}")
            continue

        pairing_dir = run_dir / "computed_metrics" / "pairing"
        geometry_dir = run_dir / "computed_metrics" / "geometry"
        observables_dir = run_dir / "computed_metrics" / "observables"
        energy_dir = run_dir / "computed_metrics" / "energy"

        if not pairing_dir.exists():
            print(f"Skipping {run_dir.name}: missing {pairing_dir}")
            continue
        if not geometry_dir.exists():
            print(f"Skipping {run_dir.name}: missing {geometry_dir}")
            continue
        if not observables_dir.exists():
            print(f"Skipping {run_dir.name}: missing {observables_dir}")
            continue

        pairing_paths = sorted(pairing_dir.glob("*_pairing.npz"))
        if not pairing_paths:
            print(f"Skipping {run_dir.name}: no pairing caches found")
            continue

        base_params = get_row_params_from_json(run_dir, swept_param_keys)

        for pairing_path in pairing_paths:
            stem = pairing_path.name.removesuffix("_pairing.npz")

            geometry_path = geometry_dir / f"{stem}_geometry.npz"
            observables_path = observables_dir / f"{stem}_observables.npz"
            energy_path = energy_dir / f"{stem}_energy.npz"

            if not geometry_path.exists():
                print(f"Skipping {stem}: missing {geometry_path.name}")
                continue
            if not observables_path.exists():
                print(f"Skipping {stem}: missing {observables_path.name}")
                continue

            print(f"Analyzing: {run_dir.name} / {stem}")

            pairing = load_npz(pairing_path)
            geometry = load_npz(geometry_path)
            observables = load_npz(observables_path)
            energy = load_npz(energy_path) if energy_path.exists() else None

            row = base_params.copy()
            row["traj_replicate"] = stem

            results = compute_replicate_analysis(
                pairing=pairing,
                geometry=geometry,
                observables=observables,
                energy=energy,
                expected=expected,
                fraction_last20=0.2,
                fraction_last5=0.05,
            )
            row.update(results)
            rows.append(row)

    return pd.DataFrame(rows)


def save_analysis_csvs(
    df_per_rep: pd.DataFrame,
    df_group: pd.DataFrame,
    sweep_path: Path,
) -> None:
    """
    Save per-replicate and grouped summary analysis tables.
    """
    per_rep_path = sweep_path / "analysis_per_replicate.csv"
    group_path = sweep_path / "analysis_group_summary.csv"

    df_per_rep.to_csv(per_rep_path, index=False)
    df_group.to_csv(group_path, index=False)

    print(f"Saved per-replicate analysis to: {per_rep_path}")
    print(f"Saved grouped summary to:       {group_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build analysis_per_replicate.csv and analysis_group_summary.csv from cached metric files."
    )
    parser.add_argument(
        "sweep_dir",
        help="Name of the sweep directory inside BASE_DIR.",
    )
    parser.add_argument(
        "--drop-nan-cols",
        action="store_true",
        help="Drop columns that are entirely NaN from the output CSVs.",
    )
    parser.add_argument(
        "--expected-canonical",
        action="store_true",
        help="Compute Q-based metrics using the default canonical expected partner map.",
    )
    args = parser.parse_args()

    sweep_path = BASE_DIR / args.sweep_dir
    if not sweep_path.is_dir():
        raise FileNotFoundError(f"Sweep directory not found: {sweep_path}")

    print(f"Reading from: {sweep_path}")

    expected = get_expected_canonical(args.expected_canonical)

    df_per_rep = collect_replicate_rows(
        sweep_path=sweep_path,
        expected=expected,
    )

    if df_per_rep.empty:
        print("No replicate rows were created. Nothing to save.")
        return

    group_cols = get_swept_param_keys(
        sweep_path / "parameters.txt") + ["folder_name"]

    df_group = compute_group_summary(
        df_per_rep=df_per_rep,
        group_cols=group_cols,
    )

    if args.drop_nan_cols:
        df_per_rep, dropped_per_rep = drop_all_nan_columns(df_per_rep)
        df_group, dropped_group = drop_all_nan_columns(df_group)

        if dropped_per_rep:
            print("Dropped all-NaN columns from analysis_per_replicate.csv:")
            for col in dropped_per_rep:
                print(f"  {col}")

        if dropped_group:
            print("Dropped all-NaN columns from analysis_group_summary.csv:")
            for col in dropped_group:
                print(f"  {col}")

    save_analysis_csvs(
        df_per_rep=df_per_rep,
        df_group=df_group,
        sweep_path=sweep_path,
    )

    print(df_per_rep.shape)
    print(df_group.shape)
    print(df_per_rep.memory_usage(deep=True).sum() / 1024**2, "MB")
    print(df_group.memory_usage(deep=True).sum() / 1024**2, "MB")
    print(
        f"Done. Wrote {len(df_per_rep)} replicate rows and {len(df_group)} grouped rows."
    )


if __name__ == "__main__":
    main()
