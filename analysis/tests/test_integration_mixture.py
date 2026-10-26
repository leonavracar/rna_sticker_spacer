"""Real-GSD integration regression for the checked-in short mixture example.

Run separately with: python -m pytest -m integration -q

The CSVs under analysis/tests/fixtures/mixture/expected/ are a recorded baseline generated from
the research pipeline, *not* an independently validated scientific truth.
Additional physical/conservation assertions guard against tautological checks.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.utils.io import load_params_json


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "analysis" / "tests" / "fixtures" / "mixture" / "input"
REFERENCE = ROOT / "analysis" / "tests" / "fixtures" / "mixture" / "expected"


def _compare_csv_tables(actual: Path, reference: Path) -> pd.DataFrame:
    """Compare all published columns, not merely the existence of CSV files."""
    expected_df = pd.read_csv(reference)
    actual_df = pd.read_csv(actual)
    assert not expected_df.empty, reference
    assert set(actual_df.columns) == set(expected_df.columns), (
        f"Changed exported columns in {actual.name}: "
        f"missing={set(expected_df) - set(actual_df)}, "
        f"new={set(actual_df) - set(expected_df)}"
    )
    # Stable identifiers permit deterministic comparison even if row order changes.
    order_by = [
        field for field in ("folder_name", "traj_replicate", "molecule_file",
                            "E_att", "angle_k_ppp", "density")
        if field in actual_df.columns
    ]
    if order_by:
        actual_df = actual_df.sort_values(order_by).reset_index(drop=True)
        expected_df = expected_df.sort_values(order_by).reset_index(drop=True)
    actual_df = actual_df[expected_df.columns]
    assert_frame_equal(
        actual_df, expected_df, check_dtype=False,
        rtol=1e-6, atol=1e-7,
        check_exact=False,
    )
    return actual_df


def test_checked_in_mixture_reference_matches_input_composition() -> None:
    """The numerical reference must describe the actual packaged molecules."""
    parameters = load_params_json(EXAMPLE / "params.json")
    lengths = [len(sequence) for sequence in parameters["sequences"]]
    assert lengths.count(30) == 235
    assert lengths.count(141) == 50
    assert len(lengths) == 285
    assert sum(lengths) == 14_100

    baseline = pd.read_csv(REFERENCE / "replicates.csv")
    assert len(baseline) == 1
    assert baseline.loc[0, "n_molecules"] == len(lengths)
    assert baseline.loc[0, "n_particles"] == sum(lengths)
    assert baseline.loc[0, "E_att"] == pytest.approx(parameters["E_att"])
    assert baseline.loc[0, "density"] == pytest.approx(parameters["density"])


@pytest.mark.integration
def test_mixture_gsd_end_to_end(tmp_path: Path) -> None:
    """Rebuild pairing, graph, checkpoints and CSVs from real trajectory frames."""
    if importlib.util.find_spec("gsd") is None or importlib.util.find_spec("freud") is None:
        pytest.skip("Install analysis/requirements-trajectory.txt (gsd and freud)")
    for filename in ("params.json", "trajectory_0.gsd"):
        assert (EXAMPLE / filename).is_file(), f"Missing tracked fixture: {filename}"
    for filename in ("replicates.csv", "summary.csv"):
        assert (REFERENCE / filename).is_file(), f"Missing golden CSV: {filename}"

    # The published example holds files flat in eatt2/, whereas the CLI
    # expects a sweep directory containing one or more run subdirectories.
    sweep = tmp_path / "mixture_example"
    run = sweep / "eatt2"
    run.mkdir(parents=True)
    for filename in ("params.json", "trajectory_0.gsd"):
        shutil.copy2(EXAMPLE / filename, run / filename)

    output_root = tmp_path / "results"
    command = [
        sys.executable,
        str(ROOT / "analysis" / "run_pipeline.py"),
        "--sweep", str(sweep),
        "--output-root", str(output_root),
        "--experiment", "mixture",
        "--stage", "all",
        "--overwrite-npz",
    ]
    run_result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert run_result.returncode == 0, (
        f"Pipeline failed ({run_result.returncode})\n"
        f"STDOUT:\n{run_result.stdout}\nSTDERR:\n{run_result.stderr}"
    )

    generated = output_root / sweep.name
    assert (generated / "analysis.sqlite").is_file()
    assert (generated / "eatt2" / "computed_metrics" / "pairing" /
            "trajectory_0_pairing.npz").is_file()
    assert (generated / "eatt2" / "computed_metrics" / "mol_graph" /
            "trajectory_0_molecule_contact_edges.npz").is_file()

    replicates = _compare_csv_tables(
        generated / "replicates.csv", REFERENCE / "replicates.csv"
    )
    summary = _compare_csv_tables(
        generated / "summary.csv", REFERENCE / "summary.csv"
    )
    assert len(replicates) == len(summary) == 1
    row = replicates.iloc[0]
    assert row["n_molecules"] == 285  # 235 CAG10 + 50 CAG47
    assert row["n_particles"] == 235 * 30 + 50 * 141
    assert row["E_att"] == pytest.approx(2.0)
    assert row["late_fraction"] == pytest.approx(0.1)
    assert row["min_contacts"] == 2
    assert row["n_base_pairs"] >= row["n_inter_contacts"] >= 0
    assert row["n_base_pairs"] >= row["n_intra_contacts"] >= 0
    assert row["n_base_pairs"] == pytest.approx(
        row["n_intra_contacts"] + row["n_inter_contacts"]
    )
    assert row["n_edges"] >= 0
    assert row["mean_degree"] >= 0
    assert summary.iloc[0]["n"] == 1
    assert summary.iloc[0]["n_trajectories"] == 1
    # All particles, irrespective of RNA species, must be accounted for.
    assert row["frac_mol_cag10"] + row["frac_mol_cag47"] == pytest.approx(1.0)
    assert row["largest_fraction_total_particles"] <= 1.0
    for metric in ("n_base_pairs", "n_edges", "n_inter_contacts", "n_intra_contacts"):
        assert np.isfinite(row[metric]), metric

    # Independently reconstruct late-window contact means from cached arrays.
    # This cross-check does not use analyze_trajectory or SQLite aggregation.
    pairing_file = (
        generated / "eatt2" / "computed_metrics" / "pairing" /
        "trajectory_0_pairing.npz"
    )
    molecule_ids = np.load(
        generated / "eatt2" / "computed_metrics" / "pairing" /
        "molecule_lookup.npy"
    )
    with np.load(pairing_file) as pairing:
        all_frames = np.asarray(pairing["frame_indices"], dtype=int)
        assert len(all_frames) >= 10, "Fixture must cover the late-window average"
        n_keep = max(1, int(np.ceil(len(all_frames) * 0.10)))
        last_frames = all_frames[-n_keep:]
        pair_counts: list[int] = []
        intra_counts: list[int] = []
        for frame in last_frames:
            left = np.asarray(pairing[f"qi_{frame}"], dtype=int)
            right = np.asarray(pairing[f"pi_{frame}"], dtype=int)
            pair_counts.append(len(left))
            intra_counts.append(int(np.count_nonzero(
                molecule_ids[left] == molecule_ids[right]
            )))
    assert row["n_base_pairs"] == pytest.approx(np.mean(pair_counts))
    assert row["n_intra_contacts"] == pytest.approx(np.mean(intra_counts))
    assert row["n_inter_contacts"] == pytest.approx(
        np.mean(np.subtract(pair_counts, intra_counts))
    )

    graph_file = (
        generated / "eatt2" / "computed_metrics" / "mol_graph" /
        "trajectory_0_molecule_contact_edges.npz"
    )
    with np.load(graph_file) as graph:
        edge_frames = np.asarray(graph["frame"], dtype=int)
        contact_weights = np.asarray(graph["n_contacts"], dtype=int)
        qualified_counts = [
            int(np.count_nonzero(
                (edge_frames == frame) & (contact_weights >= 2)
            ))
            for frame in last_frames
        ]
    assert row["n_edges"] == pytest.approx(np.mean(qualified_counts))
    assert row["mean_degree"] == pytest.approx(
        2.0 * np.mean(qualified_counts) / 285
    )
