#!/usr/bin/env python3
"""Re-export plotting tables from sweep-specific SQLite checkpoints, without GSD/NPZ reads."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.results import export_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", type=Path, required=True,
        help="Root containing the mirrored input sweep directories",
    )
    parser.add_argument(
        "--sweep", type=str,
        help="Optional input sweep folder name, e.g. 52b_from_frame",
    )
    args = parser.parse_args()

    output_root = args.output_root.expanduser().resolve()
    if args.sweep:
        if Path(args.sweep).name != args.sweep or args.sweep in {".", ".."}:
            parser.error("--sweep must be a folder name, not a path")
        sweeps = [output_root / args.sweep]
    else:
        sweeps = sorted(
            path.parent for path in output_root.glob("*/analysis.sqlite")
            if path.is_file()
        )

    if not sweeps:
        parser.error(f"No sweep checkpoints found under {output_root}")

    for sweep in sweeps:
        db_path = sweep / "analysis.sqlite"
        if not db_path.is_file():
            parser.error(f"No analysis checkpoint found: {db_path}")
        destination, count = export_sweep(db_path, sweep)
        print(f"{sweep.name}: {count} metric rows -> {destination}")


if __name__ == "__main__":
    main()
