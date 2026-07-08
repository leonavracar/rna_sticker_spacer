#!/usr/bin/env python3
"""Build analysis_per_replicate.csv from analysis_framewise.csv."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

analysis_functions_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(analysis_functions_dir))

from analysis_scripts.helpers import (
    get_swept_param_keys,
    read_param,
)
from analysis_scripts.phase_separation import (
    calc_conc_in_mol_per_liter,
    calc_volume_in_liters,
    calc_volume,
    add_metric_summary_stats,
    ke_stats_from_series,
)

BASE_DIR = analysis_functions_dir / "bulk" / "output_dir"

FRAME_COLS = [
    "mono_frac",
    "oligo_frac",
    "drop_frac",
    "largest_cluster",
    "n_clusters",
    "particle_contacts",
    "distinct_molecule_pairs",
    "qualifying_molecule_pairs",
    "molecule_contact_recurrence_window",
    "inter_frac",
    "intra_frac",
]

STATIC_COLS = [
    "molecule_file",
    "dt",
    "E_att",
    "angle_k_ppp",
    "density",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build analysis_per_replicate.csv from analysis_framewise.csv."
    )
    parser.add_argument("sweep_dir", help="Name of the sweep directory inside BASE_DIR.")
    parser.add_argument("--fraction-last", type=float, default=0.1,
                        help="Fraction of last frames used for both KE and inst_* late/last summaries.")
    args = parser.parse_args()

    sweep_path = BASE_DIR / args.sweep_dir
    if not sweep_path.is_dir():
        raise FileNotFoundError(f"Sweep directory not found: {sweep_path}")

    framewise_csv = sweep_path / "analysis_framewise.csv"
    if not framewise_csv.is_file():
        raise FileNotFoundError(f"framewise CSV not found: {framewise_csv}")

    swept_param_keys = get_swept_param_keys(sweep_path / "parameters.txt")
    df = pd.read_csv(framewise_csv)

    static_cols = list(dict.fromkeys(swept_param_keys + STATIC_COLS))
    static_cols = [c for c in static_cols if c in df.columns]

    rows: list[dict[str, object]] = []

    for (folder_name, traj_replicate), g in df.groupby(
        ["folder_name", "traj_replicate"], sort=False
    ):
        g = g.sort_values("frame").reset_index(drop=True)
        first = g.iloc[0]

        row: dict[str, object] = {}
        for col in static_cols:
            row[col] = first[col]

        row["folder_name"] = folder_name
        row["traj_replicate"] = traj_replicate
        row["fraction_last"] = args.fraction_last

        # params.json extras
        params_path = sweep_path / folder_name / "params.json"
        if params_path.is_file():
            for k in ["N_steps", "N_write", "r_opt"]:
                v = read_param(params_path, k)
                if v is not None:
                    row[k] = v

        # Sizes / concentrations
        n_molecules = int(first["n_molecules_total"])
        length = int(first["mol_len"])
        N_particles = n_molecules * length
        density = float(first["density"])
        v = calc_volume(density, N_particles)
        vL = calc_volume_in_liters(v)

        row["n_saved_frames"] = int(len(g))
        row["n_molecules"] = n_molecules
        row["length"] = length
        row["conc_mol_per_L"] = calc_conc_in_mol_per_liter(n_molecules, vL)
        row["conc_nuc_per_L"] = calc_conc_in_mol_per_liter(N_particles, vL)

        # KE stats from framewise ke_per_particle
        if "ke_per_particle" in g.columns:
            row.update(ke_stats_from_series(g["ke_per_particle"], args.fraction_last))

        # inst_* summaries
        add_metric_summary_stats(
            row, g, FRAME_COLS, prefix="inst_",
            fraction_last=args.fraction_last,
        )

        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_path = sweep_path / "analysis_per_replicate.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved {len(out_df)} rows to {out_path}")


if __name__ == "__main__":
    main()