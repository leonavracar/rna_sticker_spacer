#!/usr/bin/env python3
"""Run RNA pairing, molecular graph and experiment-specific analyses.

Input trajectories, parameters and cached NPZ files are read from --sweep.
All newly generated files are written beneath --output-root / sweep_name.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Iterator

import numpy as np
import pandas as pd
import yaml

ANALYSIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ANALYSIS_DIR))

from src.framewise import (
    component_frame_rows,
    contact_frame_rows,
    edge_frames,
    molecule_lengths,
    summarize_bin_frames,
    summarize_numeric,
)
from src.mixtures import identify_species, mixture_cluster_rows
from src.single_chain import (
    add_gsd_geometry,
    mfe_pairs_from_csv,
    single_chain_framewise,
)
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


def archive_path(run_dir: Path, kind: str, stem: str) -> Path:
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
    """Select parameter runs; every trajectory file is an independent replicate."""
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

    from src.molecule_graph import (
        build_molecule_contact_edge_dict_from_pairing_npz,
        build_molecule_intra_contact_dict_from_pairing_npz,
    )

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


def metadata(paths: TrajectoryPaths) -> dict[str, Any]:
    """Simulation parameters and unambiguous replicate identifiers."""
    params = load_params_json(paths.input_run / "params.json")
    run_number = paths.input_run.name.split("_", 1)[0]
    trajectory_id = paths.stem.replace("trajectory_", "traj_")
    return {
        "run_id": f"{run_number}__{trajectory_id}",
        "folder_name": paths.input_run.name,
        "traj_replicate": paths.stem,
        "E_att": params.get("E_att"),
        "angle_k_ppp": params.get("angle_k_ppp"),
        "density": params.get("density"),
        "molecule_file": params.get("molecule_file"),
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
        framewise = contact_frame_rows(
            pairing,
            lookup,
            particle_recurrence=experiment == "length_patterning",
            recurrence_window=window,
        )
        bins = pd.DataFrame()
        extra_tables: dict[str, pd.DataFrame] = {}

        if experiment == "single_chain":
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
            framewise = add_gsd_geometry(
                framewise,
                paths.trajectory,
                nm_per_sim_length_unit=float(
                    config["rg"]["nm_per_sim_length_unit"]
                ),
            )
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
                        ranks, contacts, largest = mixture_cluster_rows(
                            graph, lengths, species, min_contacts, intra
                        )
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

    if "step" not in framewise:
        import gsd.hoomd

        with gsd.hoomd.open(str(paths.trajectory), "r") as traj:
            framewise["step"] = [
                int(traj[int(frame)].configuration.step)
                for frame in framewise["frame"]
            ]

    info = metadata(paths)
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
    summary = {
        **info,
        "late_fraction": fraction,
        **summarize_numeric(framewise, fraction, metric_columns),
    }

    if not bins.empty:
        bin_scheme = config["cluster_binning"]
        bins = bins[bins["scheme"] == bin_scheme]
        bins = summarize_bin_frames(bins, fraction)
        for key, value in info.items():
            bins[key] = value

    for table in extra_tables.values():
        if not table.empty:
            for key, value in info.items():
                table[key] = value

    return AnalysisResult(
        paths=paths,
        framewise=framewise,
        bins=bins,
        summary=summary,
        extra_tables=extra_tables,
    )


def write_results(
    output_sweep: Path,
    results: list[AnalysisResult],
    run_ids: list[str] | None,
    save_framewise: bool,
) -> Path:
    """Write sweep summaries and optional run-level replicate time series."""
    analysis_dir = output_sweep / "analysis"
    if run_ids:
        analysis_dir /= "selected_runs"
        analysis_dir /= "_".join(sorted(set(run_ids)))

    analysis_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result.summary for result in results]).to_csv(
        analysis_dir / "replicate_summary.csv", index=False
    )

    bins = [result.bins for result in results if not result.bins.empty]
    if bins:
        pd.concat(bins, ignore_index=True).to_csv(
            analysis_dir / "cluster_distribution.csv", index=False
        )

    for name in ("ranked_clusters", "cluster_contacts", "largest_cluster"):
        tables = [
            result.extra_tables[name]
            for result in results
            if name in result.extra_tables and not result.extra_tables[name].empty
        ]
        if tables:
            pd.concat(tables, ignore_index=True).to_csv(
                analysis_dir / f"{name}.csv", index=False
            )

    if save_framewise:
        for result in results:
            output = result.paths.output_run / "framewise" / f"{result.paths.stem}.csv"
            output.parent.mkdir(parents=True, exist_ok=True)
            result.framewise.to_csv(output, index=False)

    return analysis_dir


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
        "--stage", choices=("pairing", "graphs", "analyze", "all"),
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

    results: list[AnalysisResult] = []
    for paths in trajectories:
        print(f"{args.stage}: {paths.input_run.name}/{paths.trajectory.name}", flush=True)
        pairing_changed = False
        if args.stage in ("pairing", "all"):
            pairing_changed = generate_pairing(paths, config, args.overwrite_npz)
        if args.stage in ("graphs", "all"):
            generate_graphs(paths, config, args.overwrite_npz or pairing_changed)
        if args.stage in ("analyze", "all"):
            results.append(analyze_trajectory(paths, input_sweep, config))

    if results:
        destination = write_results(
            output_sweep, results, args.run_ids, args.save_framewise
        )
        print(f"Saved analysis tables to {destination}")
    else:
        print(f"Generated archives, if needed, under {output_sweep}")


if __name__ == "__main__":
    main()
