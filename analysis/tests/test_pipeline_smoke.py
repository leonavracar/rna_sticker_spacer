"""NPZ-only smoke tests for all four analysis profiles.

These deliberately use nonexistent .gsd paths. The analyze stage must not open
or even import a GSD file when pairing and graph NPZs are already available.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_pipeline
from src.results import metrics_from_analysis


def case(tmp_path, experiment):
    input_sweep = tmp_path / "input" / experiment
    input_run = input_sweep / "0_params"
    output_run = tmp_path / "output" / experiment / "0_params"
    input_run.mkdir(parents=True)
    output_run.mkdir(parents=True)

    if experiment == "mixture":
        lengths = [30, 141]
    elif experiment == "single_chain":
        lengths = [30]
    else:
        lengths = [30, 30, 30, 30]

    (input_run / "params.json").write_text(json.dumps({
        "density": .01, "E_att": 2, "angle_k_ppp": 2.5,
        "molecule_file": "cag10.json" if experiment != "mixture"
        else "cag10x1_cag47x1.json",
    }))
    paths = run_pipeline.TrajectoryPaths(
        input_run, output_run, input_run / "trajectory_0.gsd"
    )
    assert not paths.trajectory.exists()

    lookup = np.repeat(np.arange(len(lengths)), lengths)
    if experiment == "single_chain":
        qi, pi = np.array([0, 1]), np.array([9, 8])
    else:
        qi, pi = np.array([0, 1]), np.array([30, 31])

    pairing = run_pipeline.archive_path(output_run, "pairing", paths.stem)
    mol_lookup = run_pipeline.archive_path(output_run, "lookup", paths.stem)
    pairing.parent.mkdir(parents=True, exist_ok=True)
    np.savez(pairing, frame_indices=np.array([0]), steps=np.array([1000]),
             qi_0=qi, pi_0=pi)
    np.save(mol_lookup, lookup)

    config = {
        "experiment": experiment, "late_fraction": .10,
        "min_contacts": 2, "recurrence_window": 20,
        "nm_per_sim_length_unit": .6,
    }
    if experiment == "mixture":
        config.update(
            cluster_binning="particle_percentage",
            species_by_length={30: "cag10", 141: "cag47"},
            pairable_sites_by_species={"cag10": 20, "cag47": 94},
        )
    elif experiment in {"reentrance", "length_patterning"}:
        config["cluster_binning"] = (
            "molecule_count" if experiment == "reentrance"
            else "particle_percentage"
        )
    else:
        reference = input_sweep / "viennarna_reference_pairs.csv"
        pd.DataFrame([{"i": 1, "j": 10}]).to_csv(reference, index=False)
        config["viennarna_reference_file"] = reference.name
        config["pair_index_offset"] = 1

    run_pipeline.generate_graphs(paths, config, overwrite=False)
    return paths, input_sweep, config


@pytest.mark.parametrize(
    "experiment", ["single_chain", "reentrance", "length_patterning", "mixture"]
)
def test_analyze_uses_npzs_only_and_returns_wide_metrics(tmp_path, experiment):
    paths, sweep, config = case(tmp_path, experiment)
    result = run_pipeline.analyze_trajectory(paths, sweep, config)
    metrics = metrics_from_analysis(result.summary, result.bins, result.extra_tables)

    assert result.summary["late_fraction"] == .10
    assert result.summary["n_base_pairs"] == 2
    assert result.framewise.loc[0, "step"] == 1000
    assert metrics["n_base_pairs"] == 2

    if experiment == "single_chain":
        assert metrics["mfe_matches"] == 1
        assert metrics["extra_pairs"] == 1
        assert metrics["precision"] == .5
    elif experiment == "reentrance":
        assert metrics["n_edges"] == 1
        assert "molecule_recurrence" in result.framewise
        assert metrics["cluster_bin_molecule_count_dimer"] == .5
    elif experiment == "length_patterning":
        assert metrics["n_edges"] == 1
        assert "particle_recurrence" in result.framewise
        assert metrics["rank_largest_n_molecules"] == 2
    else:
        assert metrics["n_edges"] == 1
        assert metrics["rank_largest_cag10_n_molecules"] == 1
        assert metrics["rank_largest_cag47_n_molecules"] == 1
        assert metrics["largest_contacts_cag10_cag47_n_inter_contacts"] == 2
        assert metrics["largest_mean_degree"] == 1.0
