#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

analysis_functions_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(analysis_functions_dir))

from analysis_scripts.phase_separation import add_metric_summary_stats

BASE_DIR = analysis_functions_dir / "bulk" / "output_dir"


FRAME_COLS = [
    # original phase-separation metrics
    "mono_frac",
    "oligo_frac",
    "drop_frac",
    "largest_cluster",
    "n_clusters",
    "particle_contacts",
    "distinct_molecule_pairs",
    "qualifying_molecule_pairs",
    "inter_frac",
    "intra_frac",

    # mixture composition
    "n_mol_cag10",
    "frac_mol_cag10",
    "n_mol_cag47",
    "frac_mol_cag47",

    # molecule-level contact type metrics
    "mol_contacts_total",
    "frac_mol_contacts_cag10_cag10",
    "frac_mol_contacts_cag10_cag47",
    "frac_mol_contacts_cag47_cag47",
    "enrichment_mol_contacts_cag10_cag10",
    "enrichment_mol_contacts_cag10_cag47",
    "enrichment_mol_contacts_cag47_cag47",

    # type-specific mono / oligo / droplet fractions
    "mono_frac_cag10",
    "oligo_frac_cag10",
    "drop_frac_cag10",
    "mono_frac_cag47",
    "oligo_frac_cag47",
    "drop_frac_cag47",

    # largest-cluster composition
    "largest_cluster_size_mixture",
    "largest_cluster_n_cag10",
    "largest_cluster_frac_cag10",
    "largest_cluster_n_cag47",
    "largest_cluster_frac_cag47",
    "largest_cluster_purity",

    # mixed-cluster metrics
    "n_clusters_mixed",
    "frac_clusters_mixed",
    "largest_mixed_cluster",
    "n_clusters_pure_cag10",
    "n_clusters_pure_cag47",
    "frac_molecules_in_mixed_clusters",
    "frac_cag10_in_mixed_clusters",
    "frac_cag47_in_mixed_clusters",

    # energy / kinetic energy
    "E_pot",
    "E_sqd",
    "E_bond",
    "E_angle",
    "ke_per_particle",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build analysis_per_replicate.csv from mixture framewise CSV."
    )
    parser.add_argument("sweep_dir", help="Name of the sweep directory inside BASE_DIR.")
    parser.add_argument("--summary-fraction-last", type=float, default=0.2)
    args = parser.parse_args()

    sweep_path = BASE_DIR / args.sweep_dir
    if not sweep_path.is_dir():
        raise FileNotFoundError(f"Sweep directory not found: {sweep_path}")

    framewise_csv = sweep_path / "analysis_framewise.csv"
    if not framewise_csv.exists():
        raise FileNotFoundError(f"Framewise CSV not found: {framewise_csv}")

    df = pd.read_csv(framewise_csv)

    if "run_id" in df.columns:
        group_cols = ["run_id"]
    else:
        group_cols = ["folder_name", "traj_replicate"]

    missing_group_cols = [col for col in group_cols if col not in df.columns]
    if missing_group_cols:
        raise ValueError(f"Missing grouping columns: {missing_group_cols}")

    if "frame" in df.columns:
        df = df.sort_values(group_cols + ["frame"])

    metadata_cols = [
        "molecule_file",
        "E_att",
        "density",
        "angle_k_ppp",
        "folder_name",
        "traj_replicate",
        "min_intermol_bonds",
        "drop_threshold",
    ]

    frame_cols = [col for col in FRAME_COLS if col in df.columns]
    missing_frame_cols = [col for col in FRAME_COLS if col not in df.columns]

    if missing_frame_cols:
        print("Skipping missing framewise columns:")
        for col in missing_frame_cols:
            print(f"  {col}")

    rows: list[dict[str, object]] = []

    for _, rep_df in df.groupby(group_cols, sort=False):
        rep_df = rep_df.copy()

        if "frame" in rep_df.columns:
            rep_df = rep_df.sort_values("frame")

        row: dict[str, object] = {}

        for col in metadata_cols:
            if col in rep_df.columns:
                row[col] = rep_df[col].iloc[0]

        if "run_id" in rep_df.columns:
            row["run_id"] = rep_df["run_id"].iloc[0]

        row["summary_fraction_last"] = args.summary_fraction_last

        add_metric_summary_stats(
            row,
            rep_df,
            frame_cols,
            prefix="inst_",
            fraction_last=args.summary_fraction_last,
        )

        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_csv = sweep_path / "analysis_per_replicate.csv"
    out_df.to_csv(out_csv, index=False)

    print(f"Saved {len(out_df)} rows to {out_csv}")


if __name__ == "__main__":
    main()