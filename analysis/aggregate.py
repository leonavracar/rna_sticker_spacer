#!/usr/bin/env python3
"""Rebuild the plotting tables from stored replicate checkpoints, without NPZ/GSD reads."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.results import EXPERIMENT_NAMES, export_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--experiment",
        choices=tuple(EXPERIMENT_NAMES),
        help="Re-export one experiment; omit to compile all four",
    )
    args = parser.parse_args()

    output_root = args.output_root.expanduser().resolve()
    db_path = output_root / "analysis.sqlite"
    if not db_path.is_file():
        parser.error(f"No analysis checkpoint database found: {db_path}")

    for experiment in (
        [args.experiment] if args.experiment else EXPERIMENT_NAMES
    ):
        destination, count = export_experiment(db_path, output_root, experiment)
        print(f"{experiment}: {count} metric rows -> {destination}")


if __name__ == "__main__":
    main()
