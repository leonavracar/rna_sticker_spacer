#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

analysis_functions_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(analysis_functions_dir))

from analysis_scripts.helpers import (
    get_row_params_from_json,
    get_swept_param_keys,
    read_param,
    load_npz,
    as_traj,
)
from analysis_scripts.phase_separation import (
    count_molecule_contacts_from_all_arrays,
    framewise_ps_rows_from_nlist_dict,
    window_ps_rows_from_pair_counts_all_frames,
    calc_conc_in_mol_per_liter,
    calc_volume_in_liters,
    calc_volume,
    get_log_KE_stats,
    add_metric_summary_stats,
    framewise_molecule_recurrence_rows_from_nlist_dict,
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

WINDOW_COLS = [
    "mono_frac",
    "oligo_frac",
    "drop_frac",
    "largest_cluster",
    "n_clusters",
    "molecule_pairs",
]

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build analysis_per_replicate.csv from saved pairing .npz files."
    )
    parser.add_argument("sweep_dir", help="Name of the sweep directory inside BASE_DIR.")
    parser.add_argument(
    "--run_ids",
    nargs="+",
    default=None,
    help="Optional numeric folder prefixes to process, e.g. 251 252 253.",
)
    parser.add_argument("--min-intermol-bonds", type=int, default=2)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--min-num-frames", type=int, default=2)
    parser.add_argument("--ke-fraction-last", type=float, default=0.3)
    parser.add_argument("--summary-fraction-last", type=float, default=0.2)
    parser.add_argument("--recurrence-window", type=int, default=25)
    args = parser.parse_args()

    sweep_path = BASE_DIR / args.sweep_dir
    if not sweep_path.is_dir():
        raise FileNotFoundError(f"Sweep directory not found: {sweep_path}")

    swept_param_keys = get_swept_param_keys(sweep_path / "parameters.txt")
    rows: list[dict[str, object]] = []

    if args.run_ids is None: # entire sweep folder
        npz_paths = sorted(sweep_path.glob("*/computed_metrics/pairing/*_pairing.npz"))
    else: # selected directories
        npz_paths = []
        for run_id in args.run_ids:
            npz_paths.extend(
                sorted(sweep_path.glob(f"{run_id}_*/computed_metrics/pairing/*_pairing.npz"))
            )
    if not npz_paths:
        raise FileNotFoundError(f"No pairing .npz files found under: {sweep_path}")

    for npz_path in npz_paths:
        run_dir = npz_path.parent.parent.parent
        mol_id_path = npz_path.parent / "molecule_lookup.npy"

        if not mol_id_path.exists():
            print(f"Skipping {npz_path}: missing molecule_lookup.npy")
            continue

        row = get_row_params_from_json(run_dir=run_dir, swept_param_keys=swept_param_keys)
        row["folder_name"] = run_dir.name
        row["traj_replicate"] = npz_path.stem
        row["min_intermol_bonds"] = args.min_intermol_bonds
        row["window"] = args.window
        row["min_num_frames"] = args.min_num_frames
        row["summary_fraction_last"] = args.summary_fraction_last
        row["recurrence_window"] = args.recurrence_window
        data = load_npz(npz_path)
        molecule_id = np.load(mol_id_path)

        # Logs
        print(f"\nAnalyzing: {npz_path}")
        for key in ["r_opt", "contact_width"]:
            if key in data:
                row[key] = np.asarray(data[key]).item()
            else:
                print(f"{key}: not found")
        
        n_molecules = int(molecule_id[-1]) + 1
        length = len(molecule_id) // n_molecules # ASSUMES EQUAL LENGTH OF MOLECULES
        N_particles = len(molecule_id)

        row["n_saved_frames"] = len(data["frame_indices"])
        row["n_molecules"] = n_molecules
        row["length"] = length

        density = read_param(run_dir / "params.json", "density")
        v = calc_volume(density, N_particles)
        v_in_liters = calc_volume_in_liters(v)

        row["conc_mol_per_L"] = calc_conc_in_mol_per_liter(n_molecules, v_in_liters)
        row["conc_nuc_per_L"] = calc_conc_in_mol_per_liter(N_particles, v_in_liters)
        
        # Kinetic energy
        traj_stem = npz_path.stem.removesuffix("_pairing")
        traj_path = run_dir / f"{traj_stem}.gsd"

        try:
            with as_traj(traj_path) as traj:
                row.update(get_log_KE_stats(traj, fraction_last=args.ke_fraction_last))
        except Exception as exc:
            print(f"Skipping KE stats for {traj_path}: {exc}")
        # Turn npz into one row per frame
        frame_rows = framewise_ps_rows_from_nlist_dict(
            nlist_dict=data,
            molecule_id=molecule_id,
            min_intermol_bonds=args.min_intermol_bonds,
        )
        frame_df = pd.DataFrame(frame_rows)
        # add recurrence
        rec_rows = framewise_molecule_recurrence_rows_from_nlist_dict(
            nlist_dict=data,
            molecule_id=molecule_id,
            recurrence_window=args.recurrence_window,
            min_intermol_bonds=args.min_intermol_bonds,
        )
        rec_df = pd.DataFrame(rec_rows)

        frame_df = frame_df.join(
            rec_df.set_index("frame"),
            on="frame",
        )

        add_metric_summary_stats(
            row,
            frame_df,
            FRAME_COLS,
            prefix="inst_",
            fraction_last=args.summary_fraction_last,
        )

        pair_counts_all_frames = count_molecule_contacts_from_all_arrays(data, molecule_id)
        window_rows = window_ps_rows_from_pair_counts_all_frames(
            pair_counts_all_frames=pair_counts_all_frames,
            n_molecules=n_molecules,
            window=args.window,
            min_num_frames=args.min_num_frames,
            min_intermol_bonds=args.min_intermol_bonds,
        )
        window_df = pd.DataFrame(window_rows)
        add_metric_summary_stats(row, window_df, WINDOW_COLS, prefix="persistent_", fraction_last=args.summary_fraction_last)

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(sweep_path / "analysis_per_replicate.csv", index=False)
    print(f"Saved {len(df)} rows to {sweep_path / 'analysis_per_replicate.csv'}")


if __name__ == "__main__":
    main()





# #!/usr/bin/env python3

# # For now just a skeleton for per-traj csv with metrics
# # conc_g, conc_mol, mono_last, oligo_last, droplet_last, fraction_paired_of_all_bases_last
# # pct_inter_of_total, pct_intra_of_total

# # And finally a framewise.csv for now just with frame, mono, oligo, droplet so we can plot it

# from __future__ import annotations
# from pathlib import Path
# import sys

# analysis_functions_dir = Path(__file__).resolve().parent.parent.parent
# sys.path.append(str(analysis_functions_dir))

# from analysis_scripts.helpers import (
#     get_row_params_from_json,
#     get_swept_param_keys,
#     read_param,
# )
# from analysis_scripts.phase_separation import (
#     get_cluster_sizes_count,
#     get_cluster_sizes_from_array,
#     mono_oligo_drop_dict,
#     calc_conc_in_mol_per_liter,
#     calc_volume_in_liters,
#     calc_volume,
# )

# import argparse
# import numpy as np
# import pandas as pd

# BASE_DIR = analysis_functions_dir / "bulk" / "output_dir"


# def main() -> None:
#     parser = argparse.ArgumentParser(
#         description="Build analysis_per_replicate.csv from .npz files."
#     )

#     parser.add_argument(
#         "sweep_dir",
#         help="Name of the sweep directory inside BASE_DIR.",
#     )
#     args = parser.parse_args()
#     sweep_path = BASE_DIR / args.sweep_dir
#     if not sweep_path.is_dir():
#         raise FileNotFoundError(f"Sweep directory not found: {sweep_path}")

#     print(f"Reading from: {sweep_path}")

#     swept_param_keys = get_swept_param_keys(
#         sweep_path / "parameters.txt"
#     )  # swept params as rows in csv
#     rows = []

#     npz_paths = sorted(sweep_path.glob("*/computed_metrics/pairing/*_pairing.npz"))

#     for npz_path in npz_paths:
#         run_dir = npz_path.parent.parent.parent
#         mol_id_dir = npz_path.parent / "molecule_lookup.npy"  # molecule identity array

#         row = get_row_params_from_json(
#             run_dir=run_dir, swept_param_keys=swept_param_keys
#         )
#         row["folder_name"] = run_dir.name
#         row["traj_replicate"] = npz_path.stem

#         data = np.load(npz_path)
#         molecule_id = np.load(mol_id_dir)

#         # simple starting point = last frame calculations
#         frame_indices = data["frame_indices"]
#         last = frame_indices[-1]
#         qi = data[f"qi_{last}"]
#         pi = data[f"pi_{last}"]

#         # cluster distribution calculations (last frame)
#         cluster_sizes = get_cluster_sizes_from_array(
#             qi=qi, pi=pi, molecule_id=molecule_id
#         )
#         cluster_counts = get_cluster_sizes_count(cluster_sizes)
#         mod = mono_oligo_drop_dict(cluster_counts)

#         n_molecules = molecule_id[-1] + 1  # for a fraction of molecules
#         row["mono_frac_last"] = mod["mono"] / n_molecules  # fraction
#         row["oligo_frac_last"] = mod["oligo"] / n_molecules
#         row["drop_frac_last"] = mod["drop"] / n_molecules
#         row["largest_cluster_last"] = max(cluster_sizes)

#         # Inter vs intra
#         mol_i = molecule_id[qi]
#         mol_j = molecule_id[pi]
#         inter = np.sum(mol_i != mol_j)
#         intra = np.sum(mol_i == mol_j)
#         total_bonds = len(qi)
#         row["inter_frac_last"] = inter / total_bonds if total_bonds > 0 else 0
#         row["intra_frac_last"] = intra / total_bonds if total_bonds > 0 else 0
#         # Concentration
#         density = read_param(run_dir / "params.json", "density")
#         length = len(molecule_id) // n_molecules  # particles per molecule
#         N_particles = len(molecule_id)

#         v = calc_volume(density, N_particles)
#         v_in_liters = calc_volume_in_liters(v)

#         row["conc_mol_per_L"] = calc_conc_in_mol_per_liter(n_molecules, v_in_liters)
#         row["conc_nuc_per_L"] = calc_conc_in_mol_per_liter(
#             N_particles, v_in_liters
#         )  # per nucleotide
#         # Early frame snapshot (e.g. 10th sampled frame)
#         early_idx = min(
#             10, len(frame_indices) - 1
#         )  # in case there are fewer than 10 frames
#         early = frame_indices[early_idx]
#         qi_early = data[f"qi_{early}"]
#         pi_early = data[f"pi_{early}"]

#         cluster_sizes_early = get_cluster_sizes_from_array(
#             qi_early, pi_early, molecule_id
#         )
#         cluster_counts_early = get_cluster_sizes_count(cluster_sizes_early)
#         mod_early = mono_oligo_drop_dict(cluster_counts_early)

#         row["mono_frac_early"] = mod_early["mono"] / n_molecules
#         row["oligo_frac_early"] = mod_early["oligo"] / n_molecules
#         row["drop_frac_early"] = mod_early["drop"] / n_molecules
#         rows.append(row)
#         df = pd.DataFrame(rows)

#     df.to_csv(sweep_path / "analysis_per_replicate.csv", index=False)
#     print(f"Saved {len(df)} rows")


# if __name__ == "__main__":
#     main()
