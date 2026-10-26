#!/usr/bin/env python3
"""Run trajectory analysis and export sweep-level results."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Iterator, Literal

import numpy as np
import pandas as pd
import yaml

ANALYSIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ANALYSIS_DIR))

from src.concentration import concentrations_from_density
from src.results import export_sweep, metrics_from_analysis, save_checkpoint
from src.framewise import (
    component_frame_rows,
    contact_frame_rows,
    edge_frames,
    molecule_lengths,
    summarize_bin_frames,
    summarize_numeric,
)
from src.mixtures import identify_species, mixture_cluster_rows
from src.molecule_graph import (
    build_molecule_contact_edge_dict_from_pairing_npz,
    build_molecule_intra_contact_dict_from_pairing_npz,
)
from src.energy import energy_framewise
from src.single_chain import mfe_pairs_from_csv, single_chain_framewise
from src.utils.io import load_params_json


@dataclass(frozen=True)
class TrajectoryPaths:
    """Paths for one trajectory within its parameter run."""

    input_run: Path
    output_run: Path
    trajectory: Path

    @property
    def stem(self) -> str:
        return self.trajectory.stem


@dataclass
class AnalysisResult:
    """Per-replicate analysis tables and late-window summaries."""

    paths: TrajectoryPaths
    framewise: pd.DataFrame
    bins: pd.DataFrame
    summary: dict[str, Any]
    extra_tables: dict[str, pd.DataFrame]


def archive_path(
    run_dir: Path,
    kind: Literal["pairing", "lookup", "graph", "intramol"],
    stem: str,
) -> Path:
    """Resolve the established NPZ and molecule-lookup filenames."""
    base = run_dir / "computed_metrics"
    if kind == "pairing":
        return base / "pairing" / f"{stem}_pairing.npz"
    if kind == "lookup":
        return base / "pairing" / "molecule_lookup.npy"
    if kind == "graph":
        return base / "mol_graph" / f"{stem}_molecule_contact_edges.npz"
    if kind == "intramol":
        return base / "intramol_graph" / f"{stem}_intra_molecule_contacts.npz"
    raise ValueError(f"Unknown archive type: {kind}")


def find_archive(paths: TrajectoryPaths, kind: str) -> Path | None:
    """Locate a cached archive without mixing incompatible pairing generations."""
    output = archive_path(paths.output_run, kind, paths.stem)
    if output.is_file():
        return output

    # New pairing arrays require molecular-contact archives derived from them.
    output_pairing = archive_path(paths.output_run, "pairing", paths.stem)
    output_lookup = archive_path(paths.output_run, "lookup", paths.stem)
    if kind in {"graph", "intramol"} and output_pairing.is_file() and output_lookup.is_file():
        return None

    source = archive_path(paths.input_run, kind, paths.stem)
    return source if source.is_file() else None


def pairing_inputs(paths: TrajectoryPaths) -> tuple[Path, Path]:
    """Read pairing data and molecule IDs from the same run-directory tree."""
    for run_dir in (paths.output_run, paths.input_run):
        pairing = archive_path(run_dir, "pairing", paths.stem)
        lookup = archive_path(run_dir, "lookup", paths.stem)
        if pairing.is_file() and lookup.is_file():
            return pairing, lookup
    raise FileNotFoundError(
        f"No complete pairing archive and molecule lookup for {paths.trajectory}. "
        "Run --stage pairing or --stage all to generate them."
    )


def iter_trajectories(
    input_sweep: Path, output_sweep: Path, run_ids: list[str] | None
) -> Iterator[TrajectoryPaths]:
    """Yield trajectories from the selected parameter runs."""
    selected = set(run_ids) if run_ids else None
    for input_run in sorted(input_sweep.iterdir()):
        if not input_run.is_dir() or not (input_run / "params.json").is_file():
            continue
        if selected is not None and input_run.name.split("_", 1)[0] not in selected:
            continue
        for trajectory in sorted(input_run.glob("trajectory_*.gsd")):
            yield TrajectoryPaths(
                input_run=input_run,
                output_run=output_sweep / input_run.name,
                trajectory=trajectory,
            )


def generate_pairing(
    paths: TrajectoryPaths, config: dict[str, Any], overwrite: bool
) -> bool:
    """Compute missing particle-pair NPZ and molecule lookup files."""
    if not overwrite:
        try:
            pairing_inputs(paths)
            return False
        except FileNotFoundError:
            pass

    import gsd.hoomd
    from src.pairing import (
        compute_filtered_nlist_base_pairs_traj,
        make_molecule_lookup_table,
    )

    params = load_params_json(paths.input_run / "params.json")
    r_opt = (float(params["r_att_base"]) + float(params["r_rep_base"])) / 2.0

    with gsd.hoomd.open(str(paths.trajectory), "r") as traj:
        pairs = compute_filtered_nlist_base_pairs_traj(
            traj=traj,
            r_min=float(config["r_min"]),
            r_max=float(config["r_max"]),
            exclude_chain=int(config["exclude_chain"]),
            stride=int(config["stride"]),
            r_opt=r_opt,
            rg_nm_per_sim_length_unit=(
                float(config["nm_per_sim_length_unit"])
                if config["experiment"] == "single_chain" else None
            ),
        )
        lookup = make_molecule_lookup_table(traj[0])

    output_pairing = archive_path(paths.output_run, "pairing", paths.stem)
    output_lookup = archive_path(paths.output_run, "lookup", paths.stem)
    output_pairing.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_pairing, **pairs)
    np.save(output_lookup, lookup)
    return True


def generate_graphs(
    paths: TrajectoryPaths, config: dict[str, Any], overwrite: bool
) -> None:
    """Generate molecular-contact archives from particle-pairing archives."""
    if config["experiment"] == "single_chain":
        return

    want_graph = overwrite or find_archive(paths, "graph") is None
    want_intra = (
        config["experiment"] == "mixture"
        and (overwrite or find_archive(paths, "intramol") is None)
    )
    if not want_graph and not want_intra:
        return

    pairing_path, lookup_path = pairing_inputs(paths)
    lookup = np.load(lookup_path)
    with np.load(pairing_path) as pairing:
        if want_graph:
            graph_data = build_molecule_contact_edge_dict_from_pairing_npz(
                pairing, lookup
            )
            graph_path = archive_path(paths.output_run, "graph", paths.stem)
            graph_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(graph_path, **graph_data)

        if want_intra:
            intra_data = build_molecule_intra_contact_dict_from_pairing_npz(
                pairing, lookup
            )
            intra_path = archive_path(paths.output_run, "intramol", paths.stem)
            intra_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(intra_path, **intra_data)


def metadata(
    paths: TrajectoryPaths,
    lengths: np.ndarray,
    nm_per_sim_length_unit: float,
    experiment: str,
) -> dict[str, Any]:
    """Simulation parameters, replicate identifiers and physical concentrations."""
    params = load_params_json(paths.input_run / "params.json")
    run_number = paths.input_run.name.split("_", 1)[0]
    trajectory_id = paths.stem.replace("trajectory_", "traj_")
    density = float(params["density"])
    n_particles = int(lengths.sum())
    n_molecules = int(len(lengths))
    sweep_name = paths.input_run.parent.name
    molecule_file = str(params.get("molecule_file", ""))
    if experiment == "mixture":
        sequence_class = "mixed"
    elif experiment == "single_chain":
        sequence_class = "single_chain"
    elif experiment == "length_patterning" and (
        "rand" in sweep_name.lower()
        or "scram" in sweep_name.lower()
        or "_rep_" in molecule_file.lower()
    ):
        sequence_class = "scrambled"
    else:
        sequence_class = "repeat"
    unique_lengths = np.unique(lengths)
    repeat_count = (
        int(unique_lengths[0] // 3)
        if experiment in {"reentrance", "length_patterning"}
        and len(unique_lengths) == 1
        and unique_lengths[0] % 3 == 0
        else None
    )

    suffix = Path(molecule_file).stem.rsplit("_rep_", 1)[-1]
    scramble_id = (suffix if sequence_class == "scrambled"
                   and "_rep_" in Path(molecule_file).stem
                   and suffix.isdigit() else None)

    concentrations = concentrations_from_density(
        density=density,
        n_particles=n_particles,
        n_molecules=n_molecules,
        nm_per_sim_length_unit=nm_per_sim_length_unit,
    )
    return {
        "run_id": f"{run_number}__{trajectory_id}",
        "folder_name": paths.input_run.name,
        "traj_replicate": paths.stem,
        "source_sweep": sweep_name,
        "sequence_class": sequence_class,
        "repeat_count": repeat_count,
        "scramble_id": scramble_id,
        "E_att": params.get("E_att"),
        "angle_k_ppp": params.get("angle_k_ppp"),
        "density": density,
        "molecule_file": params.get("molecule_file"),
        "n_molecules": n_molecules,
        "n_particles": n_particles,
        "nm_per_sim_length_unit": nm_per_sim_length_unit,
        **concentrations,
    }

def summarize_table(
    framewise: pd.DataFrame,
    fraction: float,
    group_columns: list[str],
) -> pd.DataFrame:
    """Average numerical columns over the last fraction of sampled frames."""
    if framewise.empty:
        return framewise
    frames = np.sort(framewise["frame"].unique())
    n_keep = max(1, int(np.ceil(len(frames) * fraction)))
    selected = framewise[framewise["frame"].isin(frames[-n_keep:])]
    numeric = [
        name
        for name in selected.select_dtypes(include="number").columns
        if name not in {"frame", *group_columns}
    ]
    if not group_columns:
        return pd.DataFrame([selected[numeric].mean().to_dict()])
    if not numeric:
        return selected[group_columns].drop_duplicates().reset_index(drop=True)
    return selected.groupby(group_columns, as_index=False, dropna=False)[numeric].mean()


def analyze_trajectory(
    paths: TrajectoryPaths, input_sweep: Path, config: dict[str, Any]
) -> AnalysisResult:
    """Analyze a replicate using cached pairing and molecular-contact arrays."""
    experiment = str(config["experiment"])
    fraction = float(config["late_fraction"])
    min_contacts = int(config["min_contacts"])
    window = int(config["recurrence_window"])

    pairing_path, lookup_path = pairing_inputs(paths)
    lookup = np.load(lookup_path)
    lengths = molecule_lengths(lookup)

    with np.load(pairing_path) as pairing:
        step_lookup = (
            dict(zip(
                pairing["frame_indices"].astype(int),
                pairing["steps"].astype(int),
            ))
            if "steps" in pairing.files else None
        )
        framewise = contact_frame_rows(
            pairing,
            lookup,
            particle_recurrence=experiment == "length_patterning",
            recurrence_window=window,
        )
        bins = pd.DataFrame()
        extra_tables: dict[str, pd.DataFrame] = {}

        if experiment == "single_chain":
            if "Rg_nm" not in pairing.files:
                raise ValueError(
                    f"Single-chain pairing archive lacks Rg_nm: {pairing_path}. "
                    "Regenerate pairing with --stage all --overwrite-npz."
                )
            frame_indices = np.asarray(pairing["frame_indices"], dtype=int)
            rg_nm = np.asarray(pairing["Rg_nm"], dtype=float)
            if len(rg_nm) != len(frame_indices) or not np.all(np.isfinite(rg_nm)):
                raise ValueError(f"Invalid Rg_nm array in {pairing_path}")
            framewise = framewise.merge(
                pd.DataFrame({"frame": frame_indices, "Rg_nm": rg_nm}),
                on="frame", how="left", validate="one_to_one",
            )
            reference = Path(config["viennarna_reference_file"])
            if not reference.is_absolute():
                reference = input_sweep / reference
            if not reference.is_file():
                raise FileNotFoundError(f"ViennaRNA reference not found: {reference}")
            mfe = single_chain_framewise(
                pairing,
                mfe_pairs_from_csv(reference),
                int(config["pair_index_offset"]),
            )
            framewise = framewise.merge(mfe, on="frame", how="left")
        else:
            graph_path = find_archive(paths, "graph")
            if graph_path is None:
                raise FileNotFoundError(
                    f"Molecular-contact NPZ missing for {paths.trajectory}"
                )
            with np.load(graph_path) as graph:
                graph_frames, bins, ranked_frames = component_frame_rows(
                    edge_frames(graph),
                    lengths,
                    min_contacts=min_contacts,
                    recurrence_window=window,
                    with_recurrence=experiment == "reentrance",
                    with_bins=True,
                    with_ranks=experiment == "length_patterning",
                )
                framewise = framewise.merge(graph_frames, on="frame", how="inner")

                if experiment == "length_patterning":
                    extra_tables["ranked_clusters"] = summarize_table(
                        ranked_frames, fraction, ["rank"]
                    )
                elif experiment == "mixture":
                    species = identify_species(
                        lengths,
                        {
                            int(length): name
                            for length, name in config["species_by_length"].items()
                        },
                    )
                    intra_path = find_archive(paths, "intramol")
                    if intra_path is None:
                        raise FileNotFoundError(
                            f"Intramolecular-contact NPZ missing for {paths.trajectory}"
                        )
                    with np.load(intra_path) as intra:
                        ranks, contacts, largest, composition = mixture_cluster_rows(
                            graph, lengths, species, min_contacts, intra,
                            pairable_sites_by_species={
                                name: int(count)
                                for name, count in config["pairable_sites_by_species"].items()
                            },
                        )
                    framewise = framewise.merge(composition, on="frame", how="inner")
                    extra_tables = {
                        "ranked_clusters": summarize_table(
                            ranks, fraction, ["rank", "species"]
                        ),
                        "cluster_contacts": summarize_table(
                            contacts, fraction, ["type"]
                        ),
                        "largest_cluster": summarize_table(
                            largest, fraction, []
                        ),
                    }

    if step_lookup is not None and "step" not in framewise:
        framewise["step"] = framewise["frame"].map(step_lookup)
        if framewise["step"].isna().any():
            raise ValueError(f"Pairing NPZ has incomplete step metadata: {pairing_path}")

    info = metadata(
        paths,
        lengths,
        nm_per_sim_length_unit=float(config["nm_per_sim_length_unit"]),
        experiment=experiment,
    )
    for key, value in info.items():
        framewise[key] = value

    metric_columns = [
        name for name in (
            "n_base_pairs", "n_intra_contacts", "n_inter_contacts", "n_edges",
            "mean_edge_weight", "mean_degree", "molecule_recurrence",
            "particle_recurrence", "mfe_matches", "extra_pairs",
            "precision", "recall", "Rg_nm",
        )
        if name in framewise
    ]
    if experiment == "mixture":
        metric_columns.extend(
            name for name in composition.columns
            if name != "frame" and name not in metric_columns
        )
    summary = {
        **info,
        "late_fraction": fraction,
        "min_contacts": min_contacts,
        **summarize_numeric(framewise, fraction, metric_columns),
    }

    if not bins.empty:
        bin_scheme = config["cluster_binning"]
        bins = bins[bins["scheme"] == bin_scheme]
        bins = summarize_bin_frames(bins, fraction)
        bins["late_fraction"] = fraction
        bins["min_contacts"] = min_contacts
        for key, value in info.items():
            bins[key] = value

    for table in extra_tables.values():
        if not table.empty:
            table["late_fraction"] = fraction
            table["min_contacts"] = min_contacts
            for key, value in info.items():
                if key not in table:
                    table[key] = value

    return AnalysisResult(
        paths=paths,
        framewise=framewise,
        bins=bins,
        summary=summary,
        extra_tables=extra_tables,
    )


def save_framewise(result: AnalysisResult) -> None:
    """Optional per-replicate diagnostic time series (not generated by default)."""
    output = result.paths.output_run / "framewise" / f"{result.paths.stem}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    result.framewise.to_csv(output, index=False)


def checkpoint_energy(
    paths: TrajectoryPaths,
    config: dict[str, Any],
    db_path: Path,
) -> None:
    """Optional GSD pass; store only late-window energy/Rg averages in SQLite."""
    pairing_path, lookup_path = pairing_inputs(paths)
    with np.load(pairing_path) as pairing:
        frames = np.asarray(pairing["frame_indices"], dtype=np.int64)
    params = load_params_json(paths.input_run / "params.json")
    dt = float(params["dt"]) if params.get("dt") is not None else None
    data = energy_framewise(paths.trajectory, frames, dt)
    fields = ["E_pot", "E_sqd", "E_bond", "E_angle", "ke_per_particle"]
    available = [name for name in fields if name in data]
    values = summarize_numeric(data, float(config["late_fraction"]), available)
    info = metadata(
        paths,
        molecule_lengths(np.load(lookup_path)),
        nm_per_sim_length_unit=float(config["nm_per_sim_length_unit"]),
        experiment=str(config["experiment"]),
    )
    info["late_fraction"] = float(config["late_fraction"])
    info["min_contacts"] = int(config["min_contacts"])
    measurements = {
        name: float(value)
        for name, value in values.items()
        if np.isfinite(value)
    }
    save_checkpoint(
        db_path, str(config["experiment"]), paths.input_run.parent.name,
        info, measurements, part="energy",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, required=True, help="Input sweep directory")
    parser.add_argument(
        "--output-root", type=Path, required=True,
        help="Destination root; the sweep name is appended automatically",
    )
    parser.add_argument(
        "--experiment",
        choices=("single_chain", "reentrance", "length_patterning", "mixture"),
        required=True,
    )
    parser.add_argument(
        "--stage", choices=("pairing", "graphs", "analyze", "all", "energy"),
        default="analyze",
    )
    parser.add_argument("--run-ids", nargs="+", default=None)
    parser.add_argument("--overwrite-npz", action="store_true")
    parser.add_argument("--save-framewise", action="store_true")
    parser.add_argument(
        "--viennarna-reference", type=Path,
        help="ViennaRNA reference CSV, absolute or relative to the input sweep",
    )
    args = parser.parse_args()

    input_sweep = args.sweep.expanduser().resolve()
    if not input_sweep.is_dir():
        parser.error(f"Input sweep not found: {input_sweep}")

    output_root = args.output_root.expanduser().resolve()
    output_sweep = output_root / input_sweep.name
    if output_sweep == input_sweep or input_sweep in output_sweep.parents:
        parser.error("The output sweep must not be inside the input sweep")

    config_path = ANALYSIS_DIR / "configs" / f"{args.experiment}.yaml"
    config: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.viennarna_reference is not None:
        if args.experiment != "single_chain":
            parser.error("--viennarna-reference requires --experiment single_chain")
        config["viennarna_reference_file"] = str(args.viennarna_reference)

    trajectories = list(iter_trajectories(input_sweep, output_sweep, args.run_ids))
    if not trajectories:
        parser.error("No matching trajectory_*.gsd files found in the input sweep")

    db_path = output_sweep / "analysis.sqlite"
    processed_count = 0
    for paths in trajectories:
        print(f"{args.stage}: {paths.input_run.name}/{paths.trajectory.name}", flush=True)
        if args.stage == "energy":
            checkpoint_energy(paths, config, db_path)
            processed_count += 1
            print(f"Checkpointed energy: {paths.stem}", flush=True)
            continue
        pairing_changed = False
        if args.stage in ("pairing", "all"):
            pairing_changed = generate_pairing(paths, config, args.overwrite_npz)
        if args.stage in ("graphs", "all"):
            generate_graphs(paths, config, args.overwrite_npz or pairing_changed)
        if args.stage in ("analyze", "all"):
            result = analyze_trajectory(paths, input_sweep, config)
            measures = metrics_from_analysis(
                result.summary, result.bins, result.extra_tables
            )
            save_checkpoint(
                db_path, args.experiment, input_sweep.name,
                result.summary, measures,
            )
            if args.save_framewise:
                save_framewise(result)
            del result
            processed_count += 1
            print(f"Checkpointed replicate: {paths.stem}", flush=True)

    if processed_count:
        destination, n_rows = export_sweep(db_path, output_sweep)
        print(f"Exported {n_rows} metric rows to {destination}")
    else:
        print(f"Generated archives, if needed, under {output_sweep}")


if __name__ == "__main__":
    main()
